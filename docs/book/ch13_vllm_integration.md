# 第十三章：FlexKV 与 vLLM 集成

> 本章详细介绍如何将 FlexKV 集成到 vLLM 中，包括 Patch 应用、配置设置，以及集成后的使用流程。

## 13.1 集成概述

### 13.1.1 集成方式

FlexKV 通过**Patch** 方式集成到 vLLM：

1. **应用 Patch**：修改 vLLM 代码，注册 FlexKV Connector
2. **配置环境变量**：设置 FlexKV 配置
3. **启动 vLLM**：自动使用 FlexKV

### 13.1.2 版本兼容性

- **FlexKV >= 1.0.0**：使用当前版本 API，Patch 位于 `examples/vllm_adaption/`
- **vLLM >= 0.8.5**：原则上都支持，示例基于 vLLM 0.10.1.1

## 13.2 Patch 应用

### 13.2.1 获取 Patch

```bash
# FlexKV 项目中的 Patch 文件
examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch
```

### 13.2.2 应用 Patch

```bash
cd /path/to/vllm
git apply /path/to/FlexKV/examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch
```

### 13.2.3 Patch 内容

Patch 主要修改以下部分：

1. **KVConnectorFactory**：注册 `FlexKVConnectorV1`
2. **Scheduler**：在调度循环中调用 Connector 接口
3. **Worker**：在 Forward 过程中调用 Connector 接口

## 13.3 配置设置

### 13.3.1 配置文件

创建 `flexkv_config.json`：

```json
{
    "server_recv_port": "ipc:///tmp/flexkv_test",
    "cache_config": {
        "enable_cpu": true,
        "enable_ssd": false,
        "enable_remote": false,
        "num_cpu_blocks": 10240,
        "tokens_per_block": 16,
        "evict_ratio": 0.05
    },
    "num_log_interval_requests": 200
}
```

### 13.3.2 环境变量

```bash
export FLEXKV_CONFIG_PATH="./flexkv_config.json"
```

### 13.3.3 vLLM 启动参数

```bash
VLLM_USE_V1=1 python -m vllm.entrypoints.cli.main serve \
    Qwen3/Qwen3-32B \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --port 30001 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --max_model_len 8192 \
    --gpu-memory-utilization 0.8 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
```

**关键参数**：
- `--kv-transfer-config`：指定使用 FlexKV Connector
- `--enable-prefix-caching`：启用前缀缓存
- `--gpu-memory-utilization`：GPU 显存利用率

## 13.4 集成验证

### 13.4.1 离线测试

```bash
# vLLM 提供的测试脚本
python examples/offline_inference/prefix_caching_flexkv.py
```

### 13.4.2 在线服务测试

启动服务后，使用 HTTP API 测试：

```bash
curl http://localhost:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-32B",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## 13.5 接口对接细节

### 13.5.1 Scheduler 端对接

vLLM Scheduler 在以下时机调用 FlexKV：

1. **调度前**：`get_num_new_matched_tokens()` - 查询可用的缓存
2. **块分配后**：`update_state_after_alloc()` - 更新状态，准备传输
3. **调度步骤**：`build_connector_meta()` - 启动/取消任务
4. **请求完成**：`request_finished()` - 保存 KV Cache

### 13.5.2 Worker 端对接

vLLM Worker 在以下时机调用 FlexKV：

1. **初始化**：`register_kv_caches()` - 注册 GPU KV Cache
2. **Forward 前**：`start_load_kv()` - 开始加载（可选）
3. **Forward 中**：`wait_for_layer_load()` - 等待加载完成（可选）
4. **Forward 后**：`save_kv_layer()` - 保存一层 KV Cache

## 13.6 常见问题与调试

### 13.6.1 初始化失败

**问题**：FlexKV 初始化超时

**排查**：
1. 检查配置文件路径
2. 检查端口是否被占用
3. 查看日志：`flexkv_logger`

### 13.6.2 传输失败

**问题**：数据传输失败

**排查**：
1. 检查 GPU 块是否已注册
2. 检查 Slot Mapping 是否正确
3. 检查存储空间是否足够

### 13.6.3 性能问题

**问题**：性能没有提升

**排查**：
1. 检查缓存命中率
2. 检查传输延迟
3. 调整配置参数（Block 大小、缓存容量等）

## 13.7 本章小结

本章介绍了 FlexKV 与 vLLM 的集成：

1. **集成方式**：通过 Patch 修改 vLLM 代码
2. **配置设置**：环境变量和 JSON 配置文件
3. **接口对接**：Scheduler 和 Worker 端的接口调用
4. **问题排查**：常见问题和调试方法

FlexKV 与 vLLM 的集成相对简单，通过 Patch 和配置即可完成，无需修改 FlexKV 代码。

---

**下一章预告**：第十四章将介绍 FlexKV 与 Dynamo 框架的集成，包括 Dynamo KV Router 的配置和使用。

