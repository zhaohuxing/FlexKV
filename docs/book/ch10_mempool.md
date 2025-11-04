# 第十章：Mempool 实现详解

> 本章深入探讨 Mempool 的实现细节，包括位图管理、快速分配算法，以及 Block 回收机制。

## 10.1 Mempool 设计目标

### 10.1.1 核心需求

Mempool 需要满足以下需求：

1. **快速分配**：O(1) 平均时间复杂度
2. **快速回收**：O(1) 时间复杂度
3. **空间高效**：最小化内存开销
4. **线程安全**：支持并发访问（如果需要）

### 10.1.2 设计选择

FlexKV 选择了**位图（Bitmap）**方案：

- **优点**：空间效率高（每 Block 1 bit），查找快速
- **缺点**：需要维护空闲 ID 数组

## 10.2 数据结构

### 10.2.1 核心数据结构

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

**内存占用**：
- 位图：`num_total_blocks / 8` 字节（每 Block 1 bit）
- 空闲数组：`num_free_blocks * 8` 字节（每 ID 8 字节）

### 10.2.2 状态示例

```
_free_mask:  [True, True, False, True, False, False, True, True]
              ↑     ↑     ↑      ↑     ↑      ↑      ↑     ↑
Block ID:     0     1     2      3     4      5      6     7

_free_ids:   [0, 1, 3, 6, 7]  # 空闲的 Block IDs
_free_ids_offset: 0
_num_free: 5
```

## 10.3 分配算法

### 10.3.1 allocate_blocks 实现

```python
def allocate_blocks(self, num: int) -> np.ndarray:
    """分配 num 个 Block"""
    # 1. 检查空间是否足够
    if num > self._num_free:
        raise NotEnoughSpaceError(
            "Not enough free blocks",
            required=num,
            available=self._num_free
        )
    
    # 2. 如果空闲数组不够，更新
    available_in_array = len(self._free_ids) - self._free_ids_offset
    if num > available_in_array:
        self._update_free_ids()
    
    # 3. 从空闲数组中选择
    free_ids = self._free_ids[
        self._free_ids_offset:self._free_ids_offset+num
    ]
    self._free_ids_offset += num
    
    # 4. 更新位图
    self._free_mask[free_ids] = False
    self._num_free -= num
    
    return free_ids
```

**时间复杂度**：
- **平均**：O(1)（从数组中选择）
- **最坏**：O(n)（需要更新数组，n 为总 Block 数）

### 10.3.2 _update_free_ids 实现

```python
def _update_free_ids(self) -> None:
    """更新空闲 ID 数组"""
    # 重新计算所有空闲 Block IDs
    self._free_ids = self._free_mask.nonzero()[0]
    self._free_ids_offset = 0
```

**调用时机**：
- 分配时发现数组不够用
- 触发频率较低（通常只在回收后首次分配时）

## 10.4 回收算法

### 10.4.1 recycle_blocks 实现

```python
def recycle_blocks(self, block_ids: np.ndarray) -> None:
    """回收 Block"""
    # 1. 验证输入
    if block_ids.ndim != 1 or block_ids.dtype != np.int64:
        raise ValueError("block_ids must be a 1D tensor of int64")
    
    # 2. 检查是否重复回收
    if self._free_mask[block_ids].any():
        free_ids = block_ids[self._free_mask[block_ids]]
        raise ValueError(f"Cannot recycle free block_ids repeatedly: {free_ids}")
    
    # 3. 更新位图
    self._free_mask[block_ids] = True
    self._num_free += len(block_ids)
    
    # 注意：不立即更新 _free_ids，延迟到需要时更新
```

**关键设计**：
- **延迟更新**：不立即更新 `_free_ids`，延迟到分配时
- **避免重复回收**：检查并拒绝重复回收
- **原子操作**：位图更新是原子操作，线程安全

### 10.4.2 延迟更新的优势

**优势 1：减少计算**

如果回收后立即不使用，更新数组是浪费。

**优势 2：批量更新**

多个回收操作后，一次更新数组，减少开销。

**优势 3：自然去重**

如果同一个 Block 被回收多次，第二次会触发错误。

## 10.5 性能分析

### 10.5.1 分配性能

**最佳情况**（数组足够）：
- 时间复杂度：O(1)
- 操作：数组切片 + 位图更新

**最坏情况**（需要更新数组）：
- 时间复杂度：O(n)
- 操作：重新计算所有空闲 ID

**平均情况**：
- 时间复杂度：O(1)（更新数组的频率很低）

### 10.5.2 回收性能

**时间复杂度**：O(n)，n 为回收的 Block 数量

**操作**：
1. 验证：O(n)
2. 位图更新：O(n)
3. 计数更新：O(1)

## 10.6 与淘汰机制的集成

### 10.6.1 淘汰触发分配

当空间不足时，Mempool 会触发淘汰：

```python
def allocate_blocks(self, num: int) -> np.ndarray:
    if num > self._num_free:
        # 触发淘汰
        num_to_evict = num - self._num_free + some_extra
        evicted_blocks = self.cache_engine.evict(num_to_evict)
        
        # 回收被淘汰的 Block
        self.recycle_blocks(evicted_blocks)
        
        # 重新尝试分配
        # ...
```

### 10.6.2 保护机制

```python
def take(
    self,
    num_required_blocks: int,
    protected_node: Optional[RadixNode] = None,
    strict: bool = False
) -> np.ndarray:
    """分配 Block，但保护某些节点不被淘汰"""
    # 1. 尝试分配
    try:
        return self.allocate_blocks(num_required_blocks)
    except NotEnoughSpaceError:
        # 2. 空间不足，触发淘汰（排除保护节点）
        # 3. 重新分配
        # ...
```

## 10.7 本章小结

本章详细介绍了 Mempool 的实现：

1. **数据结构**：位图 + 空闲数组
2. **分配算法**：快速分配，延迟更新数组
3. **回收算法**：验证 + 位图更新
4. **性能分析**：O(1) 平均分配，O(n) 回收
5. **与淘汰集成**：淘汰触发分配，保护机制

Mempool 是 FlexKV 内存管理的核心，高效的实现直接影响系统的内存利用率。

---

**下一章预告**：第十一章将详细介绍传输图的构建过程，包括 GET 和 PUT 操作的传输图构建，以及依赖关系的管理。

