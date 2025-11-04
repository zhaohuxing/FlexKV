# 第十五章：配置优化

> 本章详细介绍 FlexKV 的配置参数，以及如何根据实际场景优化配置，以获得最佳性能。

## 15.1 模型配置优化

### 15.1.1 ModelConfig 参数

```python
@dataclass
class ModelConfig:
    num_layers: int          # Transformer 层数
    num_kv_heads: int        # KV 注意力头数
    head_size: int           # 每个头的维度
    use_mla: bool = False    # 是否使用 MLA
    dtype: torch.dtype       # 数据类型
    tp_size: int = 1         # 张量并行大小
    dp_size: int = 1         # 数据并行大小
```

**优化建议**：
- `num_layers`, `num_kv_heads`, `head_size`：从模型配置自动读取
- `dtype`：建议使用 `bfloat16` 或 `float16`（平衡精度和性能）
- `tp_size`, `dp_size`：根据实际部署设置

### 15.1.2 token_size 计算

```python
@property
def token_size_in_bytes(self) -> int:
    kv_dim = 1 if self.use_mla else 2
    return self.num_layers * self.num_kv_heads * self.head_size * \
           kv_dim * self.dtype.itemsize
```

**示例**（Qwen-32B）：
- `num_layers = 64`
- `num_kv_heads = 8`
- `head_size = 256`
- `dtype = bfloat16` (2 bytes)
- `kv_dim = 2` (K 和 V)

```
token_size = 64 × 8 × 256 × 2 × 2 = 524,288 字节 ≈ 512 KB
```

## 15.2 缓存配置优化

### 15.2.1 tokens_per_block

```python
tokens_per_block: int = 16  # 默认 16
```

**优化建议**：
- **16**：适合大多数场景，平衡传输效率和内存利用率
- **32**：适合长序列，减少传输次数
- **64**：适合超长序列，但内存利用率可能降低

**选择原则**：
- 序列长度越长，Block 越大越好
- 必须为 2 的幂

### 15.2.2 多级缓存容量规划

#### CPU 缓存容量

```python
num_cpu_blocks: int = 1000000  # 默认 100 万 Block
```

**计算示例**（512 KB/token, 16 tokens/block）：
```
1 Block = 16 tokens × 512 KB/token = 8 MB
1000000 Blocks = 1000000 × 8 MB = 8 TB
```

**优化建议**：
- 根据可用 CPU 内存设置
- 建议：至少能缓存常用前缀（如 10-100 个常用前缀）

#### SSD 缓存容量

```python
num_ssd_blocks: int = 10000000  # 默认 1000 万 Block
```

**计算示例**：
```
10000000 Blocks = 10000000 × 8 MB = 80 TB
```

**优化建议**：
- 根据 SSD 容量设置
- 建议：10-100 倍 CPU 缓存容量

#### Remote 缓存容量

```python
num_remote_blocks: Optional[int] = None  # 无限制
```

**优化建议**：
- 根据实际需求设置
- Remote 存储通常容量较大，可以设置较大的值

### 15.2.3 淘汰策略

```python
evict_ratio: float = 0.0  # 默认不淘汰
```

**优化建议**：
- **0.0**：不淘汰，适合容量充足的情况
- **0.05-0.1**：轻度淘汰，适合容量紧张的情况
- **> 0.1**：积极淘汰，可能影响缓存命中率

### 15.2.4 存储布局

```python
cpu_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE
ssd_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE
remote_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE
gpu_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.LAYERWISE
```

**优化建议**：
- CPU/SSD/Remote：使用 `BLOCKWISE`（适合 Block 级别传输）
- GPU：使用 `LAYERWISE`（适合 Layer 并行计算）
- **不要修改**：这些是经过优化的默认值

## 15.3 传输配置优化

### 15.3.1 GPU↔CPU 传输

```python
use_ce_transfer_h2d: bool = False  # 使用 Copy Engine (CPU → GPU)
use_ce_transfer_d2h: bool = False  # 使用 Copy Engine (GPU → CPU)
transfer_sms_h2d: int = 8          # H2D 的 SM 数量
transfer_sms_d2h: int = 8          # D2H 的 SM 数量
```

**优化建议**：
- `use_ce_transfer_*`：启用 Copy Engine 可以提升传输速度
- `transfer_sms_*`：根据 GPU 型号调整（A100: 8, H100: 16）

### 15.3.2 CPU↔SSD 传输

```python
ssd_cache_iouring_entries: int = 512  # io_uring 队列大小
```

**优化建议**：
- **512**：适合大多数场景
- **1024-2048**：适合高并发场景
- 过大会占用过多内存

### 15.3.3 并发传输

FlexKV 支持多线程并行传输，通过以下方式优化：

1. **多个传输 Worker**：每个 DP Rank 有独立的 Worker
2. **异步传输**：传输与计算重叠
3. **批量传输**：一次传输多个 Block

## 15.4 部署模式配置

### 15.4.1 单进程模式

```python
dp_size: int = 1  # 单进程模式
```

**特点**：
- 所有组件在同一进程
- 延迟最低
- 适合单卡或小规模部署

### 15.4.2 多进程模式

```python
dp_size: int > 1  # 多进程模式
```

**配置**：
```json
{
    "server_recv_port": "ipc:///tmp/flexkv_server",
    "cache_config": {
        "enable_cpu": true,
        "num_cpu_blocks": 10240
    }
}
```

**特点**：
- Scheduler 和 Worker 分离
- 支持多 DP Rank
- 适合大规模分布式部署

## 15.5 性能调优指南

### 15.5.1 缓存命中率优化

1. **增加缓存容量**：提高 `num_cpu_blocks`、`num_ssd_blocks`
2. **优化 Block 大小**：调整 `tokens_per_block`
3. **减少淘汰**：降低 `evict_ratio`

### 15.5.2 传输延迟优化

1. **启用 Copy Engine**：`use_ce_transfer_h2d = True`
2. **增加 SM 数量**：`transfer_sms_h2d = 16`
3. **使用 Pin Memory**：CPU 内存自动使用 Pin Memory

### 15.5.3 内存使用优化

1. **合理设置缓存容量**：避免过度分配
2. **使用 SSD 缓存**：减少 CPU 内存占用
3. **启用 Remote 缓存**：进一步减少本地内存

## 15.6 配置示例

### 15.6.1 小规模部署（单卡）

```json
{
    "server_recv_port": "ipc:///tmp/flexkv_test",
    "cache_config": {
        "enable_cpu": true,
        "enable_ssd": false,
        "enable_remote": false,
        "tokens_per_block": 16,
        "num_cpu_blocks": 10000,
        "evict_ratio": 0.05,
        "use_ce_transfer_h2d": true,
        "use_ce_transfer_d2h": true,
        "transfer_sms_h2d": 8
    }
}
```

### 15.6.2 中规模部署（多卡）

```json
{
    "server_recv_port": "ipc:///tmp/flexkv_server",
    "cache_config": {
        "enable_cpu": true,
        "enable_ssd": true,
        "enable_remote": false,
        "tokens_per_block": 32,
        "num_cpu_blocks": 100000,
        "num_ssd_blocks": 1000000,
        "ssd_cache_dir": "/data/flexkv_ssd/",
        "evict_ratio": 0.1,
        "ssd_cache_iouring_entries": 1024
    }
}
```

### 15.6.3 大规模部署（分布式）

```json
{
    "server_recv_port": "ipc:///tmp/flexkv_server",
    "cache_config": {
        "enable_cpu": true,
        "enable_ssd": true,
        "enable_remote": true,
        "tokens_per_block": 64,
        "num_cpu_blocks": 1000000,
        "num_ssd_blocks": 10000000,
        "num_remote_blocks": 100000000,
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

## 15.7 本章小结

本章介绍了 FlexKV 的配置优化：

1. **模型配置**：从模型自动读取，确保准确性
2. **缓存配置**：根据容量和场景优化
3. **传输配置**：启用 Copy Engine 等优化
4. **部署模式**：单进程 vs 多进程
5. **性能调优**：缓存命中率、传输延迟、内存使用

合理配置是获得最佳性能的关键，需要根据实际场景进行调优。

---

**下一章预告**：第十六章将介绍性能优化与调优的具体方法，包括性能指标分析、调优实践，以及故障排查技巧。

