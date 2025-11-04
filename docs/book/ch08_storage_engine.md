# 第八章：存储引擎设计

> 存储引擎负责管理多级存储空间，包括存储初始化、布局管理，以及 Block 级别的存储访问。本章详细介绍 StorageEngine 的设计和实现。

## 8.1 StorageEngine 架构

### 8.1.1 核心职责

StorageEngine 负责：

1. **存储空间初始化**：为 CPU、SSD、Remote 分配存储空间
2. **存储布局管理**：管理不同的存储布局（LAYERWISE/BLOCKWISE）
3. **存储句柄管理**：提供统一的存储访问接口
4. **GPU 块注册**：接收并管理 GPU KV Cache 的注册

### 8.1.2 初始化流程

```python
class StorageEngine:
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig):
        self._storage_handles: Dict[Tuple[DeviceType, int], StorageHandle] = {}
        
        # 初始化 CPU 存储
        if cache_config.enable_cpu:
            self._cpu_layout = KVCacheLayout(
                type=cache_config.cpu_kv_layout_type,
                num_layer=model_config.num_layers,
                num_block=cache_config.num_cpu_blocks,
                tokens_per_block=cache_config.tokens_per_block,
                num_head=model_config.num_kv_heads,
                head_size=model_config.head_size,
                is_mla=model_config.use_mla
            )
            self.allocate(
                device_type=DeviceType.CPU,
                layout=self._cpu_layout,
                dtype=model_config.dtype,
            )
        
        # 初始化 SSD 存储
        if cache_config.enable_ssd:
            # SSD 布局必须与 CPU 一致
            assert cache_config.ssd_kv_layout_type == self._cpu_layout.type
            self._ssd_layout = KVCacheLayout(...)
            self.allocate(...)
        
        # 初始化 Remote 存储
        if cache_config.enable_remote:
            # Remote 布局必须与 CPU 一致
            assert cache_config.remote_kv_layout_type == self._cpu_layout.type
            self._remote_layout = KVCacheLayout(...)
            self.allocate(...)
```

## 8.2 KVCacheLayout

### 8.2.1 布局类型

FlexKV 支持两种存储布局：

#### LAYERWISE（按层组织）

**形状**：`[layer, block, token, head, head_dim]`

**特点**：
- 同一层的所有 Block 连续存储
- 适合 GPU（Layer 并行计算）
- 默认用于 GPU

**内存布局**：

```
Layer 0: Block 0, Block 1, ..., Block N
Layer 1: Block 0, Block 1, ..., Block N
...
Layer L: Block 0, Block 1, ..., Block N
```

#### BLOCKWISE（按块组织）

**形状**：`[block, layer, token, head, head_dim]`

**特点**：
- 同一 Block 的所有层连续存储
- 适合 CPU/SSD（Block 为单位传输）
- 默认用于 CPU/SSD/Remote

**内存布局**：

```
Block 0: Layer 0, Layer 1, ..., Layer L
Block 1: Layer 0, Layer 1, ..., Layer L
...
Block N: Layer 0, Layer 1, ..., Layer L
```

### 8.2.2 布局计算

```python
class KVCacheLayout:
    def _compute_kv_shape(self) -> None:
        """计算 KV Cache 的形状"""
        if self.type == KVCacheLayoutType.LAYERWISE:
            # [layer, block, token, head, head_dim]
            self._kv_shape = torch.Size([
                self.num_layer,
                self.num_block,
                self.tokens_per_block,
                self.num_head,
                self.head_size
            ])
        elif self.type == KVCacheLayoutType.BLOCKWISE:
            # [block, layer, token, head, head_dim]
            self._kv_shape = torch.Size([
                self.num_block,
                self.num_layer,
                self.tokens_per_block,
                self.num_head,
                self.head_size
            ])
    
    def get_block_stride(self) -> int:
        """获取 Block 之间的步长（字节）"""
        if self.type == KVCacheLayoutType.LAYERWISE:
            return self.kv_shape[3:].numel() * dtype.itemsize
        elif self.type == KVCacheLayoutType.BLOCKWISE:
            return self.kv_shape[1:].numel() * dtype.itemsize
    
    def get_layer_stride(self) -> int:
        """获取 Layer 之间的步长（字节）"""
        if self.type == KVCacheLayoutType.LAYERWISE:
            return self.kv_shape[1:].numel() * dtype.itemsize
        elif self.type == KVCacheLayoutType.BLOCKWISE:
            return self.kv_shape[2:].numel() * dtype.itemsize
```

### 8.2.3 Block 偏移计算

```python
def get_block_offset(block_id: int, layout: KVCacheLayout, dtype: torch.dtype) -> int:
    """计算 Block 的存储偏移量（字节）"""
    block_stride = layout.get_block_stride()
    return block_id * block_stride
```

## 8.3 存储分配器

### 8.3.1 CPUAllocator

```python
class CPUAllocator(BaseStorageAllocator):
    @classmethod
    def allocate(cls, layout: KVCacheLayout, dtype: torch.dtype, 
                 pin_memory: bool = False) -> StorageHandle:
        """分配 CPU 内存"""
        # 1. 计算总大小
        total_elements = layout.get_total_elements()
        
        # 2. 分配内存
        tensor = torch.zeros(
            total_elements,
            dtype=dtype,
            device='cpu',
            pin_memory=pin_memory  # Pin Memory 用于加速 GPU 传输
        )
        
        # 3. 创建 StorageHandle
        return StorageHandle(
            allocator=cls(tensor, layout),
            layout=layout,
            dtype=dtype
        )
```

**特点**：
- 使用 `torch.zeros` 分配连续内存
- 支持 Pin Memory 选项
- 固定大小，预分配

### 8.3.2 SSDAllocator

```python
class SSDAllocator(BaseStorageAllocator):
    @classmethod
    def allocate(cls, layout: KVCacheLayout, dtype: torch.dtype,
                 cache_dir: str, max_blocks_per_file: int = 32000) -> StorageHandle:
        """分配 SSD 存储"""
        num_blocks = layout.num_block
        num_files = (num_blocks + max_blocks_per_file - 1) // max_blocks_per_file
        
        file_list = []
        for file_id in range(num_files):
            file_path = os.path.join(cache_dir, f"flexkv_ssd_cache_{file_id}.bin")
            
            # 计算文件大小
            blocks_in_file = min(max_blocks_per_file, num_blocks - file_id * max_blocks_per_file)
            file_size = blocks_in_file * layout.get_block_stride()
            
            # 创建文件（使用 mmap）
            fd = os.open(file_path, os.O_RDWR | os.O_CREAT)
            os.ftruncate(fd, file_size)
            
            file_list.append((file_path, fd))
        
        return StorageHandle(
            allocator=cls(file_list, layout, max_blocks_per_file),
            layout=layout,
            dtype=dtype
        )
```

**特点**：
- 使用文件系统存储
- 支持多个文件（避免单个文件过大）
- 使用 `mmap` 进行内存映射

### 8.3.3 RemoteAllocator

```python
class RemoteAllocator(BaseStorageAllocator):
    @classmethod
    def allocate(cls, layout: KVCacheLayout, dtype: torch.dtype,
                 file_path: Union[str, List[str]],
                 remote_config_custom: Dict[str, Any]) -> StorageHandle:
        """分配 Remote 存储"""
        # 根据 remote_config_custom 选择存储后端
        # 支持 S3、Azure Blob、GCS 等
        # ...
```

**特点**：
- 支持多种远程存储后端
- 通过 `remote_config_custom` 配置
- 网络传输，延迟较高

### 8.3.4 GPUAllocator

```python
class GPUAllocator(BaseStorageAllocator):
    @classmethod
    def from_raw_data(cls, data: List[torch.Tensor], layout: KVCacheLayout,
                     dtype: torch.dtype, device_id: int) -> StorageHandle:
        """从 GPU 张量创建"""
        # GPU 块由 Worker 注册，不需要分配
        # 只需要保存张量引用
        return StorageHandle(
            allocator=cls(data, layout, device_id),
            layout=layout,
            dtype=dtype
        )
```

**特点**：
- GPU 块不由 FlexKV 分配
- 由 Worker 注册 GPU KV Cache
- 只保存张量引用

## 8.4 GPU 块注册

### 8.4.1 注册流程

```python
def register_gpu_blocks(
    self,
    gpu_blocks: List[TensorSharedHandle],
    gpu_layout: KVCacheLayout,
    device_id: int = 0,
    dtype: torch.dtype = torch.float16
) -> None:
    """注册 GPU KV Cache"""
    self.allocate(
        device_type=DeviceType.GPU,
        layout=gpu_layout,
        dtype=dtype,
        device_id=device_id,
        raw_data=gpu_blocks  # GPU 张量列表
    )
```

### 8.4.2 存储句柄访问

```python
def get_storage_handle(
    self,
    device_type: DeviceType,
    device_id: int = 0
) -> StorageHandle:
    """获取存储句柄"""
    key = (device_type, device_id)
    if key not in self._storage_handles:
        raise ValueError(f"Storage handle not found: {device_type}, {device_id}")
    return self._storage_handles[key]
```

**使用示例**：

```python
# 获取 CPU 存储句柄
cpu_handle = storage_engine.get_storage_handle(DeviceType.CPU)

# 获取张量（用于传输）
cpu_tensor = cpu_handle.get_tensor()

# 获取 Block 的数据指针
block_id = 100
block_offset = layout.get_block_offset(block_id, dtype)
block_ptr = cpu_tensor.data_ptr() + block_offset
```

## 8.5 本章小结

本章详细介绍了存储引擎的设计：

1. **StorageEngine**：
   - 多级存储初始化
   - 存储句柄管理
   - GPU 块注册

2. **KVCacheLayout**：
   - LAYERWISE vs BLOCKWISE
   - 布局计算
   - Block 偏移计算

3. **存储分配器**：
   - CPUAllocator：CPU 内存分配
   - SSDAllocator：SSD 文件分配
   - RemoteAllocator：远程存储分配
   - GPUAllocator：GPU 块注册

存储引擎作为 FlexKV 的存储层，负责管理多级存储空间，提供统一的存储访问接口。

---

**下一章预告**：第九章将深入探讨 RadixTree 的实现细节，包括数据结构设计、匹配算法的优化，以及节点管理的技巧。

