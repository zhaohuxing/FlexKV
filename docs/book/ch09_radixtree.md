# 第九章：RadixTree 实现详解

> 本章深入探讨 RadixTree 的实现细节，包括数据结构设计、前缀匹配算法的优化，以及节点管理的技巧。

## 9.1 RadixTree 数据结构

### 9.1.1 节点设计

RadixTree 的节点设计考虑了以下需求：

1. **快速匹配**：通过 Hash 值快速查找
2. **内存效率**：最小化内存占用
3. **并发安全**：支持锁定机制
4. **淘汰支持**：LRU 淘汰需要访问时间

```python
@dataclass
class RadixNode:
    block_hashes: np.ndarray      # Block Hash 数组
    physical_blocks: np.ndarray  # 物理 Block ID 数组
    
    is_ready: bool               # 是否准备好（数据传输完成）
    lock_cnt: int                # 锁定计数
    last_access_time: float      # 最后访问时间
    
    parent: Optional['RadixNode'] = None
    children: Dict[HashType, 'RadixNode'] = field(default_factory=dict)
```

### 9.1.2 树的结构

RadixTree 是一个**前缀树**，每个节点代表一个 Block 序列的前缀：

```
Root (空)
├── Hash(Block0) = 0x1234
│   └── Hash(Block1) = 0x5678
│       └── Hash(Block2) = 0x9ABC  (叶子节点)
│           - block_hashes: [0x1234, 0x5678, 0x9ABC]
│           - physical_blocks: [100, 101, 102]
│           - is_ready: True
└── Hash(Block0) = 0xABCD
    └── Hash(Block1) = 0xEF01  (叶子节点)
        - block_hashes: [0xABCD, 0xEF01]
        - physical_blocks: [200, 201]
```

## 9.2 前缀匹配算法

### 9.2.1 匹配流程

```python
def match_prefix(
    self,
    sequence_meta: SequenceMeta,
    update_cache_info: bool = True
) -> MatchResult:
    """前缀匹配算法"""
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
            break  # 没有匹配
        
        child_node = current_node.children[block_hash]
        
        # 4. 检查节点是否准备好
        if not child_node.is_ready:
            result.last_ready_node = current_node
            result.num_ready_matched_blocks = block_idx
            break
        
        # 5. 更新当前节点
        current_node = child_node
        
        if update_cache_info:
            current_node.last_access_time = time.time()
        
        # 6. 更新匹配计数
        result.num_matched_blocks = block_idx + 1
    
    # 7. 收集物理 Block IDs
    if result.num_matched_blocks > 0:
        matched_blocks = []
        current_node = self.root
        
        for block_idx in range(result.num_matched_blocks):
            block_hash = HashType(int(block_hashes[block_idx]))
            current_node = current_node.children[block_hash]
            matched_blocks.append(current_node.physical_blocks[0])
        
        result.physical_blocks = np.array(matched_blocks, dtype=np.int64)
    
    result.last_node = current_node
    result.last_ready_node = current_node if current_node.is_ready else result.last_ready_node
    
    return result
```

### 9.2.2 匹配优化

**优化 1：提前终止**

如果节点未准备好（`is_ready = False`），立即停止匹配，避免等待。

**优化 2：批量 Hash 计算**

一次性计算所有 Block 的 Hash，避免重复计算。

**优化 3：索引加速**

使用 C++ 实现的 `CRadixTreeIndex`，进一步提升匹配速度。

## 9.3 插入算法

### 9.3.1 插入流程

```python
def insert(
    self,
    sequence_meta: SequenceMeta,
    physical_block_ids: np.ndarray,
    num_insert_blocks: int = -1,
    is_ready: bool = True,
    match_result: Optional[MatchResult] = None
) -> Optional[RadixNode]:
    """插入新序列"""
    # 1. 生成 Hash
    if not sequence_meta.has_hashes():
        sequence_meta.gen_hashes()
    
    block_hashes = sequence_meta.block_hashes
    num_insert = len(block_hashes) if num_insert_blocks == -1 else num_insert_blocks
    
    # 2. 匹配（找出已有前缀）
    if match_result is None:
        match_result = self.match_prefix(sequence_meta, update_cache_info=False)
    
    num_matched = match_result.num_matched_blocks
    
    # 3. 如果没有匹配，直接插入根节点
    if num_matched == 0:
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
    
    # 4. 如果有部分匹配，处理节点分裂
    current_node = match_result.last_node
    
    # 如果当前节点包含的 Block 数 > 匹配数，需要分裂
    if num_matched < current_node.size():
        current_node = current_node.split(num_matched)
    
    # 5. 插入剩余的 Block
    if num_insert > num_matched:
        # 创建新节点或更新现有节点
        # ...
    
    return current_node
```

### 9.3.2 节点分裂

当插入的序列与现有节点部分匹配时，需要分裂节点：

```python
def split(self, prefix_length: int) -> 'RadixNode':
    """分裂节点"""
    assert prefix_length < self.size()
    assert prefix_length > 0
    
    # 1. 创建新节点（前缀部分）
    new_node = RadixNode(
        block_hashes=self.block_hashes[:prefix_length],
        physical_blocks=self.physical_blocks[:prefix_length],
        is_ready=self.is_ready,
        lock_cnt=0,
        last_access_time=self.last_access_time,
        parent=self.parent
    )
    
    # 2. 更新当前节点（后缀部分）
    self.block_hashes = self.block_hashes[prefix_length:]
    self.physical_blocks = self.physical_blocks[prefix_length:]
    self.parent = new_node
    
    # 3. 更新父节点的子节点引用
    if self.parent:
        parent_hash = self.head_hash()
        self.parent.children[parent_hash] = new_node
    
    return new_node
```

## 9.4 节点锁定机制

### 9.4.1 锁定原理

节点锁定用于防止正在使用的节点被淘汰：

```python
def lock(self, node: RadixNode) -> None:
    """锁定节点"""
    node.lock_cnt += 1
    
    # 递归锁定父节点
    current = node.parent
    while current is not None:
        current.lock_cnt += 1
        current = current.parent

def unlock(self, node: RadixNode) -> None:
    """解锁节点"""
    node.lock_cnt -= 1
    
    # 递归解锁父节点
    current = node.parent
    while current is not None:
        current.lock_cnt -= 1
        current = current.parent
```

**锁定时机**：
- GET 操作：锁定匹配到的节点
- PUT 操作：锁定插入的节点
- 传输完成：解锁节点

## 9.5 LRU 淘汰

### 9.5.1 淘汰算法

```python
def evict(self, num_blocks_to_evict: int) -> List[int]:
    """LRU 淘汰"""
    evicted_blocks = []
    
    # 1. 收集可淘汰的节点
    evictable_nodes = []
    self._collect_evictable_nodes(self.root, evictable_nodes)
    
    # 2. 按最后访问时间排序（LRU）
    evictable_nodes.sort(key=lambda n: n.last_access_time)
    
    # 3. 淘汰节点
    for node in evictable_nodes:
        if len(evicted_blocks) >= num_blocks_to_evict:
            break
        
        if node.evictable():
            evicted_blocks.extend(node.physical_blocks.tolist())
            self._remove_node(node)
    
    return evicted_blocks

def _collect_evictable_nodes(self, node: RadixNode, result: List[RadixNode]):
    """收集可淘汰的节点"""
    if node.evictable():
        result.append(node)
    
    for child in node.children.values():
        self._collect_evictable_nodes(child, result)

def evictable(self) -> bool:
    """检查节点是否可淘汰"""
    return (not self.is_root() and 
            self.is_leaf() and 
            not self.in_use())

def in_use(self) -> bool:
    """检查节点是否正在使用"""
    return self.lock_cnt > 0 or not self.is_ready
```

## 9.6 本章小结

本章详细介绍了 RadixTree 的实现：

1. **数据结构**：节点设计和树结构
2. **匹配算法**：前缀匹配和优化技巧
3. **插入算法**：节点分裂和插入流程
4. **锁定机制**：节点锁定和解锁
5. **LRU 淘汰**：淘汰算法和可淘汰性判断

RadixTree 是 FlexKV 缓存匹配的核心，高效的实现直接影响系统性能。

---

**下一章预告**：第十章将详细介绍 Mempool 的实现，包括位图管理、快速分配算法，以及回收机制。

