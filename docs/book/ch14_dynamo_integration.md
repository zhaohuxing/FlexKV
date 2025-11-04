# 第十四章：FlexKV 与 Dynamo 集成

> 本章详细介绍 FlexKV 与 NVIDIA Dynamo 框架的集成，包括 Dynamo KV Router 机制、集成配置，以及多 Worker 部署。

## 14.1 Dynamo 框架概述

### 14.1.1 Dynamo 简介

Dynamo 是 NVIDIA 开发的**大规模分布式推理框架**，支持：

- **多个后端引擎**：TensorRT-LLM、vLLM、SGLang
- **KV Router**：智能请求路由
- **分布式部署**：跨节点推理

### 14.1.2 KV Router 机制

KV Router 是 Dynamo 的核心组件：

```
┌─────────────────────────────────┐
│     Dynamo Frontend             │
│     (KV Router)                  │
└───────────┬─────────────────────┘
            │ 智能路由
            ↓
┌─────────────────────────────────┐
│  Worker 0  │  Worker 1  │ ...   │
│  (vLLM)    │  (vLLM)    │       │
│  + FlexKV  │  + FlexKV  │       │
└─────────────────────────────────┘
```

**工作原理**：
1. KV Router 维护每个 Worker 的 KV Cache 索引
2. 接收新请求时，查找最合适的 Worker（有最多匹配的 KV Cache）
3. 将请求路由到该 Worker

## 14.2 FlexKV 集成方式

### 14.2.1 集成架构

FlexKV 通过 **vLLM 后端** 集成到 Dynamo：

```
Dynamo KV Router
    ↓ (通过 vLLM 配置)
vLLM (后端引擎)
    ↓ (通过 KV Connector)
FlexKV
```

**关键理解**：
- FlexKV **不直接**与 Dynamo 交互
- 通过 vLLM 的 KV Connector 接口间接集成
- Dynamo 通过 vLLM 的 Event 机制了解 FlexKV 状态

### 14.2.2 集成步骤

1. **修改 Dynamo 配置**：指定使用 FlexKV Connector
2. **应用 vLLM Patch**：让 vLLM 支持 FlexKV
3. **配置 FlexKV**：设置缓存参数
4. **启动 Dynamo**：自动使用 FlexKV

## 14.3 配置修改

### 14.3.1 Dynamo 配置

修改 Dynamo 镜像内的配置：

```python
# /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/args.py
# 修改 245-248 行

kv_transfer_config = KVTransferConfig(
    kv_connector="FlexKVConnectorV1",
    kv_role="kv_both"
)
logger.info("Using FlexKVConnectorV1 configuration")
```

**关键点**：
- `kv_connector="FlexKVConnectorV1"`：指定使用 FlexKV
- `kv_role="kv_both"`：Scheduler 和 Worker 都使用

### 14.3.2 CPU Offloading 配置

当 FlexKV 启用 CPU offloading 时，需要修改 vLLM：

```python
# 删除 vLLM 中的 BlockRemove
# 让 FlexKV 通过 CPU 缓存所有 KV Block
# 这样 KV Router 的索引能反映 FlexKV 的真实状态
```

**原因**：
- Dynamo KV Router 通过接收 Worker 发送的 Event 更新索引
- 如果 vLLM 删除 Block，KV Router 的索引会不准确
- FlexKV 通过 CPU 缓存，确保所有 Block 都保留

## 14.4 启动脚本

### 14.4.1 完整启动脚本

```bash
#!/bin/bash
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# 1. 启动 NATS 和 etcd（Dynamo 依赖）
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379 \
     --data-dir /tmp/etcd &

sleep 3

# 2. 启动 Ingress（KV Router 模式）
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 &

# 3. 定义 Worker 数量
NUM_WORKERS=4

# 4. 为每个 Worker 创建 FlexKV 配置
for i in $(seq 0 $((NUM_WORKERS-1))); do
    cat <<EOF > ./flexkv_config_${i}.json
{
    "enable_flexkv": true,
    "server_recv_port": "ipc:///tmp/flexkv_${i}_test",
    "cache_config": {
        "enable_cpu": true,
        "enable_ssd": false,
        "enable_remote": false,
        "use_gds": false,
        "enable_trace": false,
        "ssd_cache_iouring_entries": 512,
        "tokens_per_block": 64,
        "num_cpu_blocks": 10240,
        "num_ssd_blocks": 256000,
        "ssd_cache_dir": "/data/flexkv_ssd/",
        "evict_ratio": 0.05,
        "index_accel": true
    },
    "num_log_interval_requests": 200
}
EOF
done

# 5. 启动 Worker 节点
for i in $(seq 0 $((NUM_WORKERS-1))); do
    GPU_START=$((i*2))
    GPU_END=$((i*2+1))
    
    if [ $i -lt $((NUM_WORKERS-1)) ]; then
        FLEXKV_CONFIG_PATH="./flexkv_config_${i}.json" \
        CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} \
        python3 -m dynamo.vllm \
            --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
            --tensor-parallel-size 2 \
            --block-size 64 \
            --gpu-memory-utilization 0.9 \
            --max-model-len 100310 &
    else
        # 最后一个 Worker 在前台运行
        FLEXKV_CONFIG_PATH="./flexkv_config_${i}.json" \
        CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} \
        python3 -m dynamo.vllm \
            --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
            --tensor-parallel-size 2 \
            --block-size 64 \
            --gpu-memory-utilization 0.9 \
            --max-model-len 100310
    fi
done
```

**关键配置**：
- **server_recv_port**：每个 Worker 使用不同的端口
- **NUM_WORKERS**：Worker 数量
- **GPU 分配**：每个 Worker 使用不同的 GPU

### 14.4.2 端口配置

**重要**：多个 Worker 时，FlexKV 端口必须不同：

```json
{
    "server_recv_port": "ipc:///tmp/flexkv_0_test",  // Worker 0
    "server_recv_port": "ipc:///tmp/flexkv_1_test",  // Worker 1
    "server_recv_port": "ipc:///tmp/flexkv_2_test",  // Worker 2
    // ...
}
```

如果端口相同，会在 `flexkv init` 步骤卡住。

## 14.5 KV Router 工作原理

### 14.5.1 Event 机制

Dynamo KV Router 通过接收 Worker 发送的 Event 更新索引：

```
Worker (vLLM + FlexKV)
    ↓ 发送 Event
Dynamo KV Router
    ↓ 更新索引
KV Cache 索引
```

**Event 内容**：
- 新添加的 KV Cache（Block 信息）
- 删除的 KV Cache（Block 信息）
- Worker 状态更新

### 14.5.2 路由决策

当新请求到达时，KV Router：

1. **查找匹配**：在所有 Worker 中查找匹配的 KV Cache
2. **计算匹配度**：计算每个 Worker 的匹配 Block 数量
3. **考虑负载**：结合 Worker 当前负载
4. **选择 Worker**：选择最合适的 Worker

### 14.5.3 FlexKV 的影响

FlexKV 通过 CPU 缓存确保：

1. **索引准确性**：KV Router 的索引反映 FlexKV 的真实状态
2. **完整缓存**：所有 KV Block 都保留在 CPU，不被删除
3. **快速恢复**：从 CPU 快速恢复到 GPU

## 14.6 验证与测试

### 14.6.1 服务验证

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7
  }'
```

### 14.6.2 性能监控

- **缓存命中率**：查看 FlexKV 统计信息
- **路由效率**：查看 Dynamo KV Router 日志
- **传输延迟**：查看 FlexKV 追踪文件

## 14.7 本章小结

本章介绍了 FlexKV 与 Dynamo 的集成：

1. **集成方式**：通过 vLLM 后端间接集成
2. **配置修改**：Dynamo 和 vLLM 配置
3. **启动脚本**：多 Worker 部署
4. **KV Router**：Event 机制和路由决策
5. **验证测试**：服务验证和性能监控

FlexKV 与 Dynamo 的集成为大规模分布式推理提供了强大的 KV Cache 管理能力。

---

**下一章预告**：第十五章将详细介绍配置优化，包括模型配置、缓存配置，以及传输配置的优化方法。

