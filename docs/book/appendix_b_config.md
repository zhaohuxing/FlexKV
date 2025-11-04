# 附录 B：配置参考

## B.1 ModelConfig

```python
@dataclass
class ModelConfig:
    num_layers: int                # Transformer 层数
    num_kv_heads: int              # KV 注意力头数
    head_size: int                 # 每个头的维度
    use_mla: bool = False          # 是否使用 MLA
    dtype: torch.dtype             # 数据类型 (float16/bfloat16)
    max_req_tokens: int = 163840   # 最大请求 token 数
    tp_size: int = 1               # 张量并行大小
    dp_size: int = 1               # 数据并行大小
```

## B.2 CacheConfig

### B.2.1 基础配置

```python
@dataclass
class CacheConfig:
    tokens_per_block: int = 16     # 每个 Block 的 token 数量（必须是 2 的幂）
    enable_cpu: bool = True        # 启用 CPU 缓存
    enable_ssd: bool = False       # 启用 SSD 缓存
    enable_remote: bool = False    # 启用 Remote 缓存
    use_gds: bool = False          # 使用 GDS (GPU Direct Storage)
    index_accel: bool = False      # 使用加速索引（C++ 实现）
```

### B.2.2 存储布局

```python
gpu_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.LAYERWISE
cpu_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE
ssd_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE
remote_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE
```

### B.2.3 缓存容量

```python
num_cpu_blocks: int = 1000000      # CPU Block 数量
num_ssd_blocks: int = 10000000     # SSD Block 数量
num_remote_blocks: Optional[int] = None  # Remote Block 数量（无限制）
```

### B.2.4 传输配置

```python
use_ce_transfer_h2d: bool = False  # 使用 Copy Engine (CPU → GPU)
use_ce_transfer_d2h: bool = False  # 使用 Copy Engine (GPU → CPU)
transfer_sms_h2d: int = 8          # H2D 的 SM 数量
transfer_sms_d2h: int = 8          # D2H 的 SM 数量
```

### B.2.5 SSD 配置

```python
max_blocks_per_file: int = 32000   # 每个文件的最大 Block 数
ssd_cache_dir: Optional[Union[str, List[str]]] = None  # SSD 缓存目录
ssd_cache_iouring_entries: int = 0  # io_uring 队列大小
ssd_cache_iouring_flags: int = 0    # io_uring 标志
```

### B.2.6 Remote 配置

```python
remote_cache_size_mode: str = "file_size"  # 容量模式：file_size 或 block_num
remote_file_size: Optional[int] = None    # 文件大小（file_size 模式）
remote_file_num: Optional[int] = None     # 文件数量（block_num 模式）
remote_file_prefix: Optional[str] = None   # 文件前缀
remote_cache_path: Optional[Union[str, List[str]]] = None  # Remote 缓存路径
remote_config_custom: Optional[Dict[str, Any]] = None      # 自定义配置
```

### B.2.7 其他配置

```python
evict_ratio: float = 0.0           # 淘汰比例
enable_trace: bool = True          # 启用追踪
trace_file_path: str = "./flexkv_trace.log"  # 追踪文件路径
```

## B.3 完整配置示例

### B.3.1 最小配置

```json
{
    "server_recv_port": "ipc:///tmp/flexkv_test",
    "cache_config": {
        "enable_cpu": true,
        "num_cpu_blocks": 10000
    }
}
```

### B.3.2 完整配置

```json
{
    "server_recv_port": "ipc:///tmp/flexkv_test",
    "cache_config": {
        "enable_cpu": true,
        "enable_ssd": true,
        "enable_remote": false,
        "tokens_per_block": 16,
        "num_cpu_blocks": 100000,
        "num_ssd_blocks": 1000000,
        "ssd_cache_dir": "/data/flexkv_ssd/",
        "evict_ratio": 0.05,
        "use_ce_transfer_h2d": true,
        "use_ce_transfer_d2h": true,
        "transfer_sms_h2d": 8,
        "transfer_sms_d2h": 8,
        "ssd_cache_iouring_entries": 512,
        "enable_trace": true,
        "trace_file_path": "./flexkv_trace.log"
    },
    "num_log_interval_requests": 200
}
```

---

**详细配置说明请参考 [`docs/flexkv_config_reference/README_zh.md`](../../flexkv_config_reference/README_zh.md)**

