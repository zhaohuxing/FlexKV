# 第十一章：传输图构建

> 本章详细介绍传输图的构建过程，包括 GET 和 PUT 操作的传输图构建，依赖关系的管理，以及传输路径的优化。

## 11.1 传输图基础

### 11.1.1 TransferOp

单个传输操作：

```python
@dataclass
class TransferOp:
    op_id: int                    # 操作 ID
    graph_id: int                 # 传输图 ID
    transfer_type: TransferType   # 传输类型
    src_block_ids: np.ndarray     # 源 Block IDs
    dst_block_ids: np.ndarray     # 目标 Block IDs
    layer_id: int                 # 层 ID（起始层）
    layer_granularity: int        # 层粒度（传输多少层）
    
    predecessors: List[int] = field(default_factory=list)  # 前置操作
    successors: List[int] = field(default_factory=list)     # 后续操作
```

### 11.1.2 TransferOpGraph

传输操作图：

```python
class TransferOpGraph:
    def __init__(self):
        self.graph_id = get_next_graph_id()
        self.ops: List[TransferOp] = []
    
    def add_transfer_op(self, op: TransferOp):
        """添加传输操作"""
        self.ops.append(op)
    
    def add_dependency(self, op_id1: int, op_id2: int):
        """添加依赖：op_id1 完成后才能执行 op_id2"""
        op1 = self.get_op(op_id1)
        op2 = self.get_op(op_id2)
        op2.predecessors.append(op_id1)
        op1.successors.append(op_id2)
```

## 11.2 GET 传输图构建

### 11.2.1 本地模式 (_get_impl_local)

只匹配 CPU 和 SSD：

```python
def _get_impl_local(...):
    # 1. 匹配 CPU 和 SSD
    cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta)
    
    cpu_matched_blocks = cpu_matched_result.physical_blocks[...]
    ssd_matched_blocks = ssd_matched_result.physical_blocks[...]
    
    # 2. 计算 Fragment
    fragment1_num_blocks = len(cpu_matched_blocks)  # CPU 匹配
    fragment2_num_blocks = max(
        len(ssd_matched_blocks) - len(cpu_matched_blocks), 0
    )  # SSD 匹配但 CPU 未匹配
    
    # 3. 分配 CPU Block（如果需要）
    if fragment2_num_blocks > 0:
        fragment23_cpu_blocks = self.cpu_cache_engine.take(...)
    
    # 4. 构建传输图
    transfer_graph = TransferOpGraph()
    
    # Fragment 2: SSD → CPU → GPU
    if fragment2_num_blocks > 0:
        # SSD → CPU
        op_disk2h = TransferOp(
            transfer_type=TransferType.DISK2H,
            src_block_ids=fragment2_ssd_blocks,
            dst_block_ids=fragment23_cpu_blocks[fragment1_num_blocks:...],
            layer_id=0,
            layer_granularity=layer_num
        )
        transfer_graph.add_transfer_op(op_disk2h)
        
        # CPU → GPU
        op_h2d = TransferOp(
            transfer_type=TransferType.H2D,
            src_block_ids=fragment23_cpu_blocks[fragment1_num_blocks:...],
            dst_block_ids=fragment2_gpu_blocks,
            layer_id=0,
            layer_granularity=layer_num
        )
        transfer_graph.add_transfer_op(op_h2d)
        op_h2d.add_predecessor(op_disk2h.op_id)  # 添加依赖
    
    # Fragment 1: CPU → GPU
    if fragment1_num_blocks > 0:
        op_h2d = TransferOp(
            transfer_type=TransferType.H2D,
            src_block_ids=fragment1_cpu_blocks,
            dst_block_ids=fragment1_gpu_blocks,
            layer_id=0,
            layer_granularity=layer_num
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

### 11.2.2 全局模式 (_get_impl_global)

匹配 CPU、SSD 和 Remote：

```python
def _get_impl_global(...):
    # 1. 匹配所有级别
    cpu_matched_result, ssd_matched_result, remote_matched_result = \
        self.match_all(sequence_meta)
    
    # 2. 计算 Fragment
    fragment1_num_blocks = len(cpu_matched_blocks)
    fragment2_num_blocks = max(len(ssd_matched_blocks) - len(cpu_matched_blocks), 0)
    fragment3_num_blocks = max(
        len(remote_matched_blocks) - 
        max(len(cpu_matched_blocks), len(ssd_matched_blocks)),
        0
    )
    
    # 3. 分配 CPU Block
    if fragment23_num_blocks > 0:
        fragment23_cpu_blocks = self.cpu_cache_engine.take(...)
    
    # 4. 构建传输图
    # Fragment 3: Remote → CPU → GPU
    if fragment3_num_blocks > 0:
        op_remote2h = TransferOp(...)  # Remote → CPU
        op_h2d = TransferOp(...)       # CPU → GPU
        op_h2d.add_predecessor(op_remote2h.op_id)
    
    # Fragment 2: SSD → CPU → GPU
    if fragment2_num_blocks > 0:
        op_disk2h = TransferOp(...)    # SSD → CPU
        op_h2d = TransferOp(...)       # CPU → GPU
        op_h2d.add_predecessor(op_disk2h.op_id)
    
    # Fragment 1: CPU → GPU
    if fragment1_num_blocks > 0:
        op_h2d = TransferOp(...)
    
    return transfer_graph, ...
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

## 11.3 PUT 传输图构建

### 11.3.1 本地模式 (_put_impl_local)

```python
def _put_impl_local(...):
    # 1. 匹配 CPU 和 SSD
    cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta)
    
    cpu_matched_blocks = cpu_matched_result.physical_blocks[...]
    ssd_matched_blocks = ssd_matched_result.physical_blocks[...]
    
    # 2. 计算需要传输的部分
    num_skipped_blocks = len(cpu_matched_blocks)  # CPU 已存在，跳过
    fragment12_num_blocks = len(gpu_block_ids) - num_skipped_blocks
    
    # 3. 分配 CPU Block
    if fragment12_num_blocks > 0:
        fragment12_cpu_blocks = self.cpu_cache_engine.take(...)
    
    # 4. 构建传输图
    transfer_graph = TransferOpGraph()
    
    # GPU → CPU
    op_d2h = TransferOp(
        transfer_type=TransferType.D2H,
        src_block_ids=fragment12_gpu_blocks,
        dst_block_ids=fragment12_cpu_blocks,
        layer_id=0,
        layer_granularity=layer_num
    )
    transfer_graph.add_transfer_op(op_d2h)
    
    # CPU → SSD（如果需要）
    if self.cache_config.enable_ssd and fragment2_num_blocks > 0:
        fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
        fragment2_ssd_blocks = self.ssd_cache_engine.take(...)
        
        op_h2disk = TransferOp(
            transfer_type=TransferType.H2DISK,
            src_block_ids=fragment2_cpu_blocks,
            dst_block_ids=fragment2_ssd_blocks,
            layer_id=0,
            layer_granularity=layer_num
        )
        transfer_graph.add_transfer_op(op_h2disk)
        op_h2disk.add_predecessor(op_d2h.op_id)  # 依赖 GPU → CPU
    
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

## 11.4 依赖关系管理

### 11.4.1 依赖类型

**数据依赖**：
- SSD → CPU → GPU：必须先完成 SSD → CPU，才能执行 CPU → GPU

**资源依赖**：
- 多个操作使用同一 CPU Block：需要串行执行

### 11.4.2 依赖解析

TransferScheduler 负责解析依赖：

```python
class TransferScheduler:
    def _can_execute(self, op: TransferOp) -> bool:
        """检查操作是否可执行"""
        # 所有前置操作都已完成
        for pred_id in op.predecessors:
            if pred_id not in self.completed_ops:
                return False
        return True
```

## 11.5 传输路径优化

### 11.5.1 并行传输

**优化 1：独立 Fragment 并行**

```
Fragment 1: CPU → GPU  (可以并行)
Fragment 2: SSD → CPU → GPU
```

**优化 2：不同层并行**

```
Layer 0: CPU → GPU  (可以并行)
Layer 1: CPU → GPU
...
```

### 11.5.2 传输合并

**优化：合并小传输**

如果多个小传输可以合并为一个，减少系统调用开销。

## 11.6 本章小结

本章详细介绍了传输图的构建：

1. **GET 传输图**：
   - 本地模式：CPU 和 SSD 匹配
   - 全局模式：CPU、SSD 和 Remote 匹配
   - Fragment 划分和传输路径

2. **PUT 传输图**：
   - 跳过已匹配的部分
   - 只传输未匹配的部分
   - 多级缓存写入

3. **依赖关系**：
   - 数据依赖
   - 资源依赖
   - 依赖解析

传输图构建是 FlexKV 性能的关键，合理的传输路径可以显著提升传输效率。

---

**下一章预告**：第十二章将详细介绍高性能传输的实现，包括 CUDA 传输优化、io_uring 异步 I/O，以及网络传输的实现。

