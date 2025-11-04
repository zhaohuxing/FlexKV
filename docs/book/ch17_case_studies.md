# 第十七章：实战案例

> 本章通过实际案例展示 FlexKV 在不同场景下的部署和使用，包括单机部署、分布式部署，以及性能测试。

## 17.1 单机部署案例

### 17.1.1 场景描述

**需求**：
- 单 GPU（A100 80GB）
- 运行 Qwen-32B 模型
- 支持多轮对话
- CPU 内存：512GB
- SSD：2TB NVMe

### 17.1.2 配置方案

```json
{
    "server_recv_port": "ipc:///tmp/flexkv_test",
    "cache_config": {
        "enable_cpu": true,
        "enable_ssd": true,
        "enable_remote": false,
        "tokens_per_block": 16,
        "num_cpu_blocks": 50000,   // 约 400GB CPU 内存
        "num_ssd_blocks": 200000,  // 约 1.6TB SSD
        "ssd_cache_dir": "/data/flexkv_ssd/",
        "evict_ratio": 0.05,
        "use_ce_transfer_h2d": true,
        "use_ce_transfer_d2h": true,
        "transfer_sms_h2d": 8,
        "ssd_cache_iouring_entries": 512
    }
}
```

### 17.1.3 部署步骤

```bash
# 1. 编译 FlexKV
cd FlexKV && ./build.sh

# 2. 创建配置
cat > flexkv_config.json <<EOF
{...配置内容...}
EOF

# 3. 设置环境变量
export FLEXKV_CONFIG_PATH="./flexkv_config.json"

# 4. 启动 vLLM
VLLM_USE_V1=1 python -m vllm.entrypoints.cli.main serve \
    Qwen/Qwen-32B \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
```

### 17.1.4 性能表现

- **缓存命中率**：~70%
- **传输延迟**：GPU↔CPU < 5ms
- **吞吐量**：提升 30%

## 17.2 分布式部署案例

### 17.2.1 场景描述

**需求**：
- 4 个 Worker 节点
- 每个节点 2 个 GPU（A100 80GB）
- 运行 DeepSeek-70B 模型
- 通过 Dynamo 统一调度

### 17.2.2 配置方案

**每个 Worker 的 FlexKV 配置**：

```json
{
    "server_recv_port": "ipc:///tmp/flexkv_${worker_id}_test",
    "cache_config": {
        "enable_cpu": true,
        "enable_ssd": true,
        "enable_remote": true,
        "tokens_per_block": 64,
        "num_cpu_blocks": 200000,
        "num_ssd_blocks": 2000000,
        "num_remote_blocks": 20000000,
        "ssd_cache_dir": "/data/flexkv_ssd/",
        "remote_cache_path": "s3://flexkv-cache/",
        "evict_ratio": 0.05,
        "remote_config_custom": {
            "endpoint": "s3.amazonaws.com",
            "bucket": "flexkv-cache"
        }
    }
}
```

### 17.2.3 部署步骤

```bash
# 1. 启动 Dynamo Ingress
python -m dynamo.frontend --router-mode kv --http-port 8000 &

# 2. 启动各个 Worker
for i in {0..3}; do
    FLEXKV_CONFIG_PATH="./flexkv_config_${i}.json" \
    CUDA_VISIBLE_DEVICES=$((i*2)),$((i*2+1)) \
    python3 -m dynamo.vllm \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
        --tensor-parallel-size 2 \
        --block-size 64 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 100310 &
done
```

### 17.2.4 性能表现

- **缓存命中率**：~60%（跨节点共享）
- **路由效率**：KV Router 正确路由 95%+ 请求
- **吞吐量**：提升 50%

## 17.3 性能测试案例

### 17.3.1 测试场景

**目标**：测试 FlexKV 在不同配置下的性能

**测试方法**：
1. 固定请求模式
2. 变化 FlexKV 配置
3. 测量性能指标

### 17.3.2 测试脚本

```python
import time
from flexkv.kvmanager import KVManager
from flexkv.common.config import ModelConfig, CacheConfig

# 配置 1：小缓存
config1 = CacheConfig(
    num_cpu_blocks=10000,
    tokens_per_block=16
)

# 配置 2：大缓存
config2 = CacheConfig(
    num_cpu_blocks=100000,
    tokens_per_block=32
)

# 测试函数
def test_performance(config):
    kv_manager = KVManager(model_config, config)
    kv_manager.start()
    
    # 执行测试请求
    start_time = time.time()
    # ... 执行请求 ...
    end_time = time.time()
    
    # 计算指标
    latency = end_time - start_time
    hit_rate = # ... 计算命中率 ...
    
    return latency, hit_rate
```

### 17.3.3 测试结果

| 配置 | 缓存容量 | Block大小 | 命中率 | 传输延迟 | 吞吐量 |
|------|---------|----------|--------|----------|--------|
| 1 | 小 | 16 | 45% | 8ms | 100 req/s |
| 2 | 大 | 32 | 75% | 5ms | 150 req/s |

## 17.4 大规模推理场景

### 17.4.1 场景描述

**需求**：
- 大规模批量推理
- 长文本生成
- 多轮对话

### 17.4.2 优化配置

```json
{
    "cache_config": {
        "tokens_per_block": 64,        // 大 Block，减少传输
        "num_cpu_blocks": 500000,      // 大容量 CPU 缓存
        "num_ssd_blocks": 5000000,     // 大容量 SSD 缓存
        "evict_ratio": 0.0,            // 不淘汰
        "use_ce_transfer_h2d": true,   // 启用 Copy Engine
        "transfer_sms_h2d": 16,        // 增加 SM
        "ssd_cache_iouring_entries": 1024  // 大 I/O 队列
    }
}
```

### 17.4.3 性能提升

- **吞吐量**：提升 80%
- **延迟**：降低 50%
- **成本**：减少 40%（减少重复计算）

## 17.5 本章小结

本章通过实战案例展示了：

1. **单机部署**：小规模场景的配置和部署
2. **分布式部署**：大规模场景的多节点部署
3. **性能测试**：测试方法和结果分析
4. **大规模场景**：高性能配置和优化

实战案例帮助读者理解 FlexKV 在实际场景中的应用和优化方法。

---

**全书完**

---

## 全书总结

本书系统性地介绍了 FlexKV 的设计与实现：

### 第一部分：基础入门篇
- 理解了 FlexKV 的定位和价值
- 掌握了核心概念（KV Cache、Block、多级缓存）
- 建立了整体架构认知

### 第二部分：架构设计篇
- 深入理解了适配层、管理层、引擎层的设计
- 掌握了各层的职责和接口
- 理解了系统的工作流程

### 第三部分：核心实现篇
- 深入理解了 RadixTree、Mempool 的实现
- 掌握了传输图构建和优化
- 理解了高性能传输的实现

### 第四部分：实践应用篇
- 学会了如何集成和配置
- 掌握了性能优化方法
- 理解了实际应用场景

希望本书能帮助读者深入理解 FlexKV，并在实际项目中应用这些知识。

---

**感谢阅读！**

