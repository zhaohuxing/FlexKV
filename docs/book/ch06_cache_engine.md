# 第六章：缓存引擎设计

> 缓存引擎是 FlexKV 的控制面，负责前缀匹配决策和多级缓存协调。本章详细介绍 GlobalCacheEngine 的架构、RadixTree 的前缀匹配算法，以及 Mempool 的内存池管理。

## 6.1 GlobalCacheEngine 架构

### 6.1.1 核心职责

GlobalCacheEngine 是 FlexKV 的**控制面**，负责：

1. **前缀匹配决策**：决定哪些 Block 可以从缓存中获取
2. **多级缓存协调**：协调 CPU、SSD、Remote 三级缓存
3. **传输图构建**：根据匹配结果构建数据传输图
4. **缓存更新**：在 PUT 操作时更新缓存

### 6.1.2 初始化

```python
class GlobalCacheEngine:
    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig):
        self.cache_config = cache_config
        self.model_config = model_config
        self.tokens_per_block = cache_config.tokens_per_block
        
        # 为每个存储级别创建 CacheEngine
        self.cpu_cache_engine = None
        self.ssd_cache_engine = None
        self.remote_cache_engine = None
        
        if cache_config.enable_cpu:
            if cache_config.index_accel:
                # 使用加速版本的索引（C++ 实现）
                self.cpu_cache_engine = CacheEngineAccel(
                    DeviceType.CPU,
                    cache_config.num_cpu_blocks,
                    cache_config.tokens_per_block,
                    cache_config.evict_ratio
                )
            else:
                # 使用 Python 版本的索引
                self.cpu_cache_engine = CacheEngine(
                    DeviceType.CPU,
                    cache_config.num_cpu_blocks,
                    cache_config.tokens_per_block,
                    cache_config.evict_ratio
                )
        
        # 类似地初始化 SSD 和 Remote CacheEngine
        # ...
```

**关键点**：
- 每个存储级别都有独立的 `CacheEngine`
- 支持加速索引（`index_accel=True`）：使用 C++ 实现的 RadixTree

### 6.1.3 CacheEngine 结构

每个 `CacheEngine` 包含两个核心组件：

```python
class CacheEngine:
    def __init__(self, device_type, num_total_blocks, tokens_per_block, evict_ratio):
        # 1. RadixTree 索引：用于前缀匹配
        if index_accel:
            self.index = CRadixTreeIndex(tokens_per_block, num_total_blocks)
        else:
            self.index = RadixTreeIndex(tokens_per_block=tokens_per_block)
        
        # 2. Mempool：用于 Block 分配和回收
        self.mempool = Mempool(num_total_blocks=num_total_blocks)
        
        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
```

## 6.2 RadixTree 前缀匹配

### 6.2.1 RadixTree 数据结构

RadixTree 是一种**前缀树**（Trie），用于存储和匹配 Block 序列。

**节点结构**：

```python
@dataclass
class RadixNode:
    block_hashes: np.ndarray      # Block Hash 数组
    physical_blocks: np.ndarray  # 物理 Block ID 数组
    
    is_ready: bool               # 是否已准备好（数据传输完成）
    lock_cnt: int                # 锁定计数（正在使用中）
    last_access_time: float      # 最后访问时间（用于 LRU）
    
    parent: Optional['RadixNode'] = None
    children: Dict[HashType, 'RadixNode'] = field(default_factory=dict)
```

**树结构示例**：

```
Root
├── Block 0 Hash: 0x1234
│   └── Block 1 Hash: 0x5678
│       └── Block 2 Hash: 0x9ABC  (叶子节点)
│           - physical_blocks: [100, 101, 102]
│           - is_ready: True
└── Block 0 Hash: 0xABCD
    └── Block 1 Hash: 0xEF01  (叶子节点)
        - physical_blocks: [200, 201]
        - is_ready: False
```

### 6.2.2 前缀匹配算法

#### match_prefix 实现

```python
def match_prefix(
    self,
    sequence_meta: SequenceMeta,
    update_cache_info: bool = True
) -> MatchResult:
    """前缀匹配"""
    result = MatchResult()
    
    # 1. 生成 Block Hash
    if not sequence_meta.has_hashes():
        sequence_meta.gen_hashes()
    
    block_hashes = sequence_meta.block_hashes
    num_blocks = len(block_hashes)
    
    # 2. 从根节点开始匹配
    current_node = self.root
    
    for block_idx in range(num_blocks):
        block_hash = HashType(int(block_hashes[block_idx]))
        
        # 3. 查找子节点
        if block_hash not in current_node.children:
            # 没有匹配的子节点，停止匹配
            break
        
        child_node = current_node.children[block_hash]
        
        # 4. 检查节点是否准备好
        if not child_node.is_ready:
            # 节点还未准备好（数据传输未完成）
            result.last_ready_node = current_node
            result.num_ready_matched_blocks = block_idx
            break
        
        # 5. 更新当前节点
        current_node = child_node
        
        if update_cache_info:
            current_node.last_access_time = time.time()
        
        # 6. 累积匹配的 Block
        result.num_matched_blocks = block_idx + 1
    
    # 7. 收集物理 Block IDs
    if result.num_matched_blocks > 0:
        matched_physical_blocks = []
        current_node = self.root
        
        for block_idx in range(result.num_matched_blocks):
            block_hash = HashType(int(block_hashes[block_idx]))
            current_node = current_node.children[block_hash]
            matched_physical_blocks.append(current_node.physical_blocks[0])
        
        result.physical_blocks = np.array(matched_physical_blocks, dtype=np.int64)
    
    result.last_node = current_node
    result.last_ready_node = current_node if current_node.is_ready else result.last_ready_node
    
    return result
```

**匹配示例**：

假设缓存中有序列 `[Block 0, Block 1, Block 2]`，查询序列 `[Block 0, Block 1, Block 3]`：

```
1. Block 0 Hash → 匹配 ✓ (num_matched_blocks = 1)
2. Block 1 Hash → 匹配 ✓ (num_matched_blocks = 2)
3. Block 3 Hash → 不匹配 ✗ (停止)
   
结果：num_matched_blocks = 2
      physical_blocks = [100, 101]  # Block 0 和 Block 1 的物理 ID
```

### 6.2.3 插入算法

#### insert 实现

```python
def insert(
    self,
    sequence_meta: SequenceMeta,
    physical_block_ids: np.ndarray,
    num_insert_blocks: int = -1,
    is_ready: bool = True,
    match_result: Optional[MatchResult] = None
) -> Optional[RadixNode]:
    """插入新的序列"""
    # 1. 生成 Block Hash
    if not sequence_meta.has_hashes():
        sequence_meta.gen_hashes()
    
    block_hashes = sequence_meta.block_hashes
    
    # 2. 如果没有提供匹配结果，先匹配
    if match_result is None:
        match_result = self.match_prefix(sequence_meta, update_cache_info=False)
    
    num_matched = match_result.num_matched_blocks
    num_insert = len(block_hashes) if num_insert_blocks == -1 else num_insert_blocks
    
    # 3. 如果没有匹配，直接插入根节点
    if num_matched == 0:
        # 创建新节点
        new_node = RadixNode(
            block_hashes=block_hashes[:num_insert],
            physical_blocks=physical_block_ids[:num_insert],
            is_ready=is_ready,
            lock_cnt=0,
            last_access_time=time.time(),
            parent=self.root
        )
        self.root.children[block_hashes[0]] = new_node
        return new_node
    
    # 4. 如果有部分匹配，需要分裂节点
    current_node = match_result.last_node
    
    if num_matched < current_node.size():
        # 需要分裂当前节点
        current_node = current_node.split(num_matched)
    
    # 5. 插入剩余的 Block
    if num_insert > num_matched:
        remaining_hashes = block_hashes[num_matched:num_insert]
        remaining_blocks = physical_block_ids[num_matched:num_insert]
        
        # 创建新节点或更新现有节点
        # ...
    
    return current_node
```

**插入示例**：

假设已有序列 `[Block 0, Block 1]`，插入序列 `[Block 0, Block 1, Block 2]`：

```
1. 匹配：找到 [Block 0, Block 1] 匹配 (num_matched = 2)
2. 检查：current_node 有 2 个 Block，完全匹配
3. 插入：在 current_node 下添加 Block 2
   
结果：树扩展为 [Block 0, Block 1, Block 2]
```

### 6.2.4 节点锁定机制

为了确保数据一致性，RadixTree 实现了节点锁定机制：

```python
def lock(self, node: RadixNode) -> None:
    """锁定节点（增加锁定计数）"""
    node.lock_cnt += 1
    
    # 递归锁定父节点
    current = node.parent
    while current is not None:
        current.lock_cnt += 1
        current = current.parent

def unlock(self, node: RadixNode) -> None:
    """解锁节点（减少锁定计数）"""
    node.lock_cnt -= 1
    
    # 递归解锁父节点
    current = node.parent
    while current is not None:
        current.lock_cnt -= 1
        current = current.parent
```

**锁定规则**：
- 正在传输的节点需要锁定（防止被淘汰）
- 锁定会传播到父节点
- 锁定的节点不能被淘汰

### 6.2.5 LRU 淘汰机制

```python
def evict(self, num_blocks_to_evict: int) -> List[int]:
    """淘汰最久未使用的节点"""
    evicted_blocks = []
    
    # 1. 收集可淘汰的节点（叶子节点、未锁定、未使用中）
    evictable_nodes = []
    self._collect_evictable_nodes(self.root, evictable_nodes)
    
    # 2. 按最后访问时间排序（LRU）
    evictable_nodes.sort(key=lambda n: n.last_access_time)
    
    # 3. 淘汰节点
    for node in evictable_nodes:
        if len(evicted_blocks) >= num_blocks_to_evict:
            break
        
        if node.evictable():
            # 收集物理 Block IDs
            evicted_blocks.extend(node.physical_blocks.tolist())
            
            # 从树中删除节点
            self._remove_node(node)
    
    return evicted_blocks

def _collect_evictable_nodes(self, node: RadixNode, result: List[RadixNode]):
    """收集可淘汰的节点"""
    if node.evictable():
        result.append(node)
    
    for child in node.children.values():
        self._collect_evictable_nodes(child, result)
```

## 6.3 Mempool 内存池

### 6.3.1 设计目标

Mempool 负责 Block 的分配和回收，使用位图（Bitmap）高效管理。

**核心特性**：
1. **快速分配**：O(1) 时间复杂度
2. **快速回收**：O(1) 时间复杂度
3. **空间高效**：使用位图，每个 Block 只需 1 bit

### 6.3.2 数据结构

```python
class Mempool:
    def __init__(self, num_total_blocks: int):
        self.num_total_blocks = num_total_blocks
        
        # 位图：True 表示空闲，False 表示已分配
        self._free_mask = np.ones(self.num_total_blocks, dtype=np.bool_)
        self._num_free = num_total_blocks
        
        # 空闲 Block ID 数组（用于快速分配）
        self._free_ids = self._free_mask.nonzero()[0]
        self._free_ids_offset = 0
```

**示例**（8 个 Block）：

```
_free_mask:  [True, True, False, True, False, False, True, True]
              ↑     ↑     ↑      ↑     ↑      ↑      ↑     ↑
Block ID:     0     1     2      3     4      5      6     7

_free_ids:   [0, 1, 3, 6, 7]  # 空闲的 Block IDs
_num_free:    5
```

### 6.3.3 分配算法

```python
def allocate_blocks(self, num: int) -> np.ndarray:
    """分配 num 个 Block"""
    if num > self._num_free:
        raise NotEnoughSpaceError(...)
    
    # 1. 如果空闲 ID 数组不够，更新
    if num > len(self._free_ids) - self._free_ids_offset:
        self._update_free_ids()
    
    # 2. 从空闲数组中选择
    free_ids = self._free_ids[
        self._free_ids_offset:self._free_ids_offset+num
    ]
    self._free_ids_offset += num
    
    # 3. 更新位图
    self._free_mask[free_ids] = False
    self._num_free -= num
    
    return free_ids
```

**时间复杂度**：O(1)（平均情况）

### 6.3.4 回收算法

```python
def recycle_blocks(self, block_ids: np.ndarray) -> None:
    """回收 Block"""
    # 1. 验证
    if self._free_mask[block_ids].any():
        raise ValueError("Cannot recycle free block_ids repeatedly")
    
    # 2. 更新位图
    self._free_mask[block_ids] = True
    self._num_free += len(block_ids)
    
    # 3. 更新空闲数组（延迟，在下次分配时更新）
```

**时间复杂度**：O(n)，n 为回收的 Block 数量

## 6.4 GET 操作流程

### 6.4.1 整体流程

```python
def get(
    self,
    request_id: int,
    token_ids: np.ndarray,
    token_mask: np.ndarray,
    slot_mapping: np.ndarray,
    layer_num: int = -1,
    layer_granularity: int = -1,
    dp_id: int = 0
) -> Tuple[TransferOpGraph, np.ndarray, Callable, int]:
    # 1. 对齐到 Block 边界
    aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
    aligned_token_ids = token_ids[:aligned_length]
    
    # 2. 构建 SequenceMeta
    sequence_meta = SequenceMeta(
        token_ids=aligned_token_ids,
        tokens_per_block=self.cache_config.tokens_per_block
    )
    
    # 3. 根据是否启用 Remote 选择实现
    if not self.cache_config.enable_remote:
        transfer_graph, ... = self._get_impl_local(...)
    else:
        transfer_graph, ... = self._get_impl_global(...)
    
    # 4. 构建返回 mask
    return_mask = np.zeros_like(token_mask, dtype=np.bool_)
    return_mask[block_start_idx * self.tokens_per_block:
                (block_start_idx + num_gpu_blocks_to_transfer) * self.tokens_per_block] = True
    
    return transfer_graph, return_mask, callback, task_end_op_id
```

### 6.4.2 本地匹配 (_get_impl_local)

只匹配 CPU 和 SSD：

```python
def _get_impl_local(...):
    # 1. 匹配 CPU 和 SSD
    cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta)
    
    # 2. 计算 Fragment
    # Fragment 1: CPU 匹配的部分
    fragment1_num_blocks = len(cpu_matched_blocks)
    
    # Fragment 2: SSD 匹配但 CPU 未匹配的部分
    fragment2_num_blocks = max(len(ssd_matched_blocks) - len(cpu_matched_blocks), 0)
    
    # 3. 构建传输图
    transfer_graph = TransferOpGraph()
    
    # Fragment 2: SSD → CPU → GPU
    if fragment2_num_blocks > 0:
        # 从 SSD 传输到 CPU
        op_disk2h = TransferOp(
            transfer_type=TransferType.DISK2H,
            src_block_ids=fragment2_ssd_blocks,
            dst_block_ids=fragment23_cpu_blocks[...],
            ...
        )
        transfer_graph.add_transfer_op(op_disk2h)
        
        # 从 CPU 传输到 GPU
        op_h2d = TransferOp(
            transfer_type=TransferType.H2D,
            src_block_ids=fragment23_cpu_blocks[...],
            dst_block_ids=fragment23_gpu_blocks,
            ...
        )
        transfer_graph.add_transfer_op(op_h2d)
        op_h2d.add_predecessor(op_disk2h.op_id)
    
    # Fragment 1: CPU → GPU
    if fragment1_num_blocks > 0:
        op_h2d = TransferOp(
            transfer_type=TransferType.H2D,
            src_block_ids=fragment1_cpu_blocks,
            dst_block_ids=fragment1_gpu_blocks,
            ...
        )
        transfer_graph.add_transfer_op(op_h2d)
    
    return transfer_graph, ...
```

**传输模式**：

```
GPU: (need)  | fragment1 | fragment2      |
              ↑           ↑
CPU: (cached) | fragment1 | fragment2(new) ← (from SSD)
                                    ↑
SSD: (cached) | ...       | fragment2      |
```

### 6.4.3 全局匹配 (_get_impl_global)

匹配 CPU、SSD 和 Remote：

```python
def _get_impl_global(...):
    # 1. 匹配所有级别
    cpu_matched_result, ssd_matched_result, remote_matched_result = \
        self.match_all(sequence_meta)
    
    # 2. 计算 Fragment
    # Fragment 1: CPU 匹配
    fragment1_num_blocks = len(cpu_matched_blocks)
    
    # Fragment 2: SSD 匹配但 CPU 未匹配
    fragment2_num_blocks = max(len(ssd_matched_blocks) - len(cpu_matched_blocks), 0)
    
    # Fragment 3: Remote 匹配但 CPU/SSD 未匹配
    fragment3_num_blocks = max(
        len(remote_matched_blocks) - 
        max(len(cpu_matched_blocks), len(ssd_matched_blocks)), 
        0
    )
    
    # 3. 构建传输图
    # Fragment 3: Remote → CPU → GPU
    # Fragment 2: SSD → CPU → GPU
    # Fragment 1: CPU → GPU
```

**传输模式**：

```
GPU: (need)  | fragment1 | fragment2      | fragment3      |
              ↑           ↑                ↑
CPU: (cached) | fragment1 | fragment2(new) | fragment3(new) ← (from Remote)
                                    ↑                ↓
SSD: (cached) | ...       | fragment2      | fragment3(new)
                                                      ↓
Remote: (cached) | ...   | ...            | fragment3      |
```

## 6.5 PUT 操作流程

### 6.5.1 整体流程

```python
def put(
    self,
    request_id: int,
    token_ids: np.ndarray,
    token_mask: np.ndarray,
    slot_mapping: np.ndarray,
    layer_num: int = -1,
    dp_id: int = 0
) -> Tuple[TransferOpGraph, np.ndarray, Callable, int, int]:
    # 1. 对齐到 Block 边界（丢弃不完整的 Block）
    aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
    aligned_token_ids = token_ids[:aligned_length]
    
    # 2. 构建 SequenceMeta
    sequence_meta = SequenceMeta(...)
    
    # 3. 匹配（找出已存在的部分）
    # 4. 构建传输图（只传输未匹配的部分）
    # 5. 插入缓存（更新 RadixTree）
```

### 6.5.2 匹配与插入

```python
def _put_impl_local(...):
    # 1. 匹配 CPU 和 SSD
    cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta)
    
    # 2. 计算需要传输的部分
    num_skipped_blocks = len(cpu_matched_blocks)  # CPU 已存在，跳过
    fragment12_num_blocks = len(gpu_block_ids) - num_skipped_blocks  # 需要传输
    
    # 3. 分配 CPU Block（如果需要）
    if fragment12_num_blocks > 0:
        fragment12_cpu_blocks = self.cpu_cache_engine.take(
            num_required_blocks=fragment12_num_blocks,
            protected_node=cpu_matched_result.last_node,
            strict=False
        )
    
    # 4. 构建传输图
    # GPU → CPU
    op_d2h = TransferOp(
        transfer_type=TransferType.D2H,
        src_block_ids=fragment12_gpu_blocks,
        dst_block_ids=fragment12_cpu_blocks,
        ...
    )
    
    # CPU → SSD（如果需要）
    if self.cache_config.enable_ssd:
        op_h2disk = TransferOp(
            transfer_type=TransferType.H2DISK,
            src_block_ids=fragment2_cpu_blocks,
            dst_block_ids=fragment2_ssd_blocks,
            ...
        )
        op_h2disk.add_predecessor(op_d2h.op_id)
    
    # 5. 插入缓存（在传输完成后）
    # 通过 callback 实现
    callback = partial(self._transfer_callback, ...)
    
    return transfer_graph, ...
```

**传输模式**：

```
GPU: (skipped)  | fragment1      | fragment2      |
                    ↓                ↓
CPU: (cached)   | fragment1(new) | fragment2(new) |
                                         ↓
SSD: (cached)   | ...            | fragment2(new) |
```

### 6.5.3 缓存更新

PUT 操作的缓存更新通过 callback 实现：

```python
def _transfer_callback(
    self,
    node_to_unlock: Dict,
    buffer_to_free: Dict,
    ...
):
    """传输完成后的回调"""
    # 1. 解锁节点
    for device_type, (node, size) in node_to_unlock.items():
        self.cache_engines[device_type].unlock_node(node)
    
    # 2. 释放缓冲区
    for device_type, blocks in buffer_to_free.items():
        self.cache_engines[device_type].recycle_blocks(blocks)
    
    # 3. 更新节点状态（标记为 ready）
    # （在传输操作完成时自动更新）
```

## 6.6 本章小结

本章详细介绍了缓存引擎的设计：

1. **GlobalCacheEngine**：
   - 多级缓存协调
   - 传输图构建
   - GET/PUT 操作流程

2. **RadixTree**：
   - 前缀匹配算法
   - 插入算法
   - 节点锁定机制
   - LRU 淘汰

3. **Mempool**：
   - 位图管理
   - 快速分配和回收

缓存引擎作为 FlexKV 的控制面，负责匹配决策和缓存管理，是整个系统的核心。

---

**下一章预告**：第七章将详细介绍传输引擎的设计，包括传输调度、依赖图执行，以及各种传输 Worker 的实现。

