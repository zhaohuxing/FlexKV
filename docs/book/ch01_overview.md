# 第一章：FlexKV 概述

## 1.1 什么是 FlexKV

FlexKV 是腾讯云 TACO 团队和社区合作开发推出的面向**超大规模 LLM 推理场景**的分布式 KV Store 与多级缓存管理系统。

### 核心定位

FlexKV 是一个**KV Cache 管理器**，它不参与模型推理计算，而是专注于：

1. **KV Cache 存储管理**：在 GPU 显存、CPU 内存、SSD 和远程存储之间管理 KV Cache
2. **多级缓存优化**：通过多级缓存减少显存占用，避免 KV Cache 被丢弃后重新计算
3. **高性能传输**：优化 GPU↔CPU↔SSD↔Remote 之间的数据传输效率
4. **分布式支持**：支持跨节点的 KV Cache 共享

### 与推理引擎的关系

```
┌─────────────────┐
│  vLLM / SGLang  │  推理引擎（负责模型计算）
└────────┬────────┘
         │ 读写 KV Cache
         ↓
┌─────────────────┐
│    FlexKV       │  KV Cache 管理器（负责存储和传输）
└────────┬────────┘
         │ 管理存储
         ↓
┌─────────────────┐
│  GPU / CPU / SSD │  存储介质
└─────────────────┘
```

FlexKV **不负责**：
- ❌ 模型推理计算
- ❌ Token 生成
- ❌ 注意力计算

FlexKV **负责**：
- ✅ KV Cache 的存储
- ✅ KV Cache 的传输
- ✅ KV Cache 的匹配和查找
- ✅ 多级缓存的协调

## 1.2 为什么需要 KV Cache 管理

### 1.2.1 LLM 推理中的 KV Cache 问题

在 LLM 推理过程中，KV Cache 面临以下挑战：

#### 问题 1：显存消耗巨大

对于一个大模型（如 Qwen-32B），KV Cache 的大小计算：

```
单个 token 的 KV Cache 大小 = 
  num_layers × num_kv_heads × head_size × 2 (K和V) × dtype.itemsize

例如：
  32层 × 32头 × 128维 × 2 × 2字节 = 524,288 字节 ≈ 512 KB

如果序列长度为 8192 tokens：
  8192 × 512 KB = 4 GB

批量大小 = 8：
  8 × 4 GB = 32 GB（仅 KV Cache）
```

**问题**：GPU 显存有限（如 A100 80GB），KV Cache 占用大量显存。

#### 问题 2：KV Cache 被频繁丢弃

当 GPU 显存不足时，推理引擎可能会：
1. 丢弃旧的 KV Cache
2. 需要时重新计算（Prefill 阶段）

**成本**：重新计算 Prefill 的 KV Cache 需要大量计算资源。

#### 问题 3：重复计算的浪费

在多轮对话场景中：
- 用户输入："你好，请介绍一下自己"
- 模型回答："我是..."
- 用户继续："你有哪些功能？"

如果第二轮的输入包含第一轮的某些 token，这些 token 的 KV Cache 已经被丢弃，需要重新计算。

### 1.2.2 FlexKV 的解决方案

FlexKV 通过**多级缓存**解决这些问题：

```
GPU 显存（昂贵、快速、容量小）
    ↓ 溢出
CPU 内存（中等成本、中等速度、容量大）
    ↓ 溢出
SSD（便宜、较慢、容量很大）
    ↓ 溢出
Remote 存储（非常便宜、较慢、容量极大）
```

**核心思想**：
1. 将 KV Cache 从 GPU 溢出到更便宜的存储
2. 需要时再从下级存储传输回 GPU
3. 通过前缀匹配，避免重复计算

**效果**：
- ✅ 减少 GPU 显存占用
- ✅ 避免 KV Cache 被丢弃
- ✅ 减少重复计算
- ✅ 提升推理吞吐量

## 1.3 FlexKV 的设计目标

### 1.3.1 高性能

- **低延迟**：传输延迟最小化
- **高吞吐**：支持高并发请求
- **异步传输**：传输与计算重叠

### 1.3.2 可扩展性

- **多级存储**：支持 CPU、SSD、Remote 三级缓存
- **分布式**：支持跨节点共享 KV Cache
- **可配置**：灵活的配置选项

### 1.3.3 易集成

- **标准接口**：实现 KV Connector 标准接口
- **框架适配**：支持 vLLM、SGLang 等推理框架
- **最小侵入**：通过 Patch 方式集成

### 1.3.4 高可靠性

- **数据一致性**：保证 KV Cache 的正确性
- **错误处理**：完善的错误处理机制
- **可观测性**：丰富的日志和追踪

## 1.4 适用场景与价值

### 1.4.1 适用场景

1. **长文本生成**
   - 场景：长文档生成、代码生成
   - 价值：减少 Prefill 阶段的重复计算

2. **多轮对话**
   - 场景：ChatBot、对话系统
   - 价值：复用历史对话的 KV Cache

3. **批量推理**
   - 场景：大批量文本处理
   - 价值：通过缓存提升吞吐量

4. **分布式推理**
   - 场景：跨节点推理
   - 价值：共享 KV Cache，减少网络传输

### 1.4.2 业务价值

1. **降低成本**
   - 减少 GPU 显存需求
   - 使用更便宜的存储（CPU 内存、SSD）

2. **提升性能**
   - 减少重复计算
   - 提升推理吞吐量

3. **增强能力**
   - 支持更长的序列长度
   - 支持更大的批量大小

## 1.5 快速开始

### 1.5.1 安装依赖

```bash
# 安装系统依赖
apt install liburing-dev      # io_uring 支持
apt install libxxhash-dev     # Hash 计算
```

### 1.5.2 编译 FlexKV

```bash
git clone https://github.com/taco-project/FlexKV
cd FlexKV
./build.sh
```

### 1.5.3 基本使用

#### 方式 1：作为库使用（单进程）

```python
from flexkv.kvmanager import KVManager
from flexkv.common.config import ModelConfig, CacheConfig
import torch

# 配置模型
model_config = ModelConfig(
    num_layers=32,
    num_kv_heads=32,
    head_size=128,
    dtype=torch.float16
)

# 配置缓存
cache_config = CacheConfig(
    tokens_per_block=16,
    enable_cpu=True,
    num_cpu_blocks=10000,
    enable_ssd=False,
    enable_remote=False
)

# 创建 KVManager
kv_manager = KVManager(model_config, cache_config)
kv_manager.start()

# 使用 KVManager
token_ids = torch.tensor([1, 2, 3, ...], dtype=torch.long)
slot_mapping = torch.tensor([0, 1, 2, ...], dtype=torch.long)

# GET 操作
task_id = kv_manager.get_async(token_ids, slot_mapping)
kv_manager.launch([task_id], [slot_mapping])
response = kv_manager.wait([task_id])

# PUT 操作
task_id = kv_manager.put_async(token_ids, slot_mapping)
kv_manager.launch([task_id], [slot_mapping])
response = kv_manager.wait([task_id])
```

#### 方式 2：与 vLLM 集成（推荐）

参考：[docs/vllm_adapter/README_zh.md](../../docs/vllm_adapter/README_zh.md)

### 1.5.4 示例代码

FlexKV 提供了多个示例：

- `examples/run_server.py`：独立服务器模式示例
- `examples/scheduler_server_example.py`：Scheduler Server 示例

## 1.6 本章小结

本章介绍了：

1. **FlexKV 是什么**：一个 KV Cache 管理器
2. **为什么需要**：解决 LLM 推理中的 KV Cache 存储和传输问题
3. **设计目标**：高性能、可扩展、易集成、高可靠性
4. **适用场景**：长文本生成、多轮对话、批量推理、分布式推理
5. **快速开始**：安装、编译、基本使用

在下一章中，我们将深入探讨 FlexKV 的核心概念，包括 KV Cache、Block、多级缓存等基础概念。

---

**下一章预告**：第二章将详细介绍 FlexKV 的核心概念，包括 Token 与 KV Cache 的关系、Block 抽象设计、以及多级缓存机制。

