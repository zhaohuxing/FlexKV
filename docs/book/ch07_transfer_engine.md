# 第七章：传输引擎设计

> 传输引擎是 FlexKV 的数据面，负责执行实际的数据传输操作。本章详细介绍 TransferEngine 的架构、传输调度算法，以及各种传输 Worker 的实现。

## 7.1 TransferEngine 架构

### 7.1.1 核心职责

TransferEngine 是 FlexKV 的**数据面**，负责：

1. **数据传输执行**：执行 GPU↔CPU↔SSD↔Remote 之间的数据传输
2. **异步并行传输**：支持多线程并行传输
3. **依赖图调度**：根据传输图的依赖关系调度传输任务

### 7.1.2 架构设计

```
┌─────────────────────────────────────┐
│      TransferEngine                  │
│  ┌──────────────────────────────┐   │
│  │  TransferScheduler            │   │
│  │  - 依赖图解析                  │   │
│  │  - 任务调度                    │   │
│  └──────────────────────────────┘   │
│           │                          │
│           ↓                          │
│  ┌──────────────────────────────┐   │
│  │  TransferWorker                │   │
│  │  ├── GPUCPUTransferWorker      │   │
│  │  ├── CPUSSDDiskTransferWorker  │   │
│  │  └── CPURemoteTransferWorker   │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 7.1.3 初始化

```python
class TransferEngine:
    def __init__(self,
        gpu_handles: List[StorageHandle],
        model_config: ModelConfig,
        cache_config: CacheConfig,
        cpu_handle: Optional[StorageHandle] = None,
        ssd_handle: Optional[StorageHandle] = None,
        remote_handle: Optional[StorageHandle] = None):
        
        # 1. 创建调度器
        self.scheduler = TransferScheduler()
        
        # 2. 创建任务队列
        self.task_queue: Queue[TransferOpGraph] = Queue()
        self.completed_queue: Queue[Tuple[int, int]] = Queue()
        self.finished_ops_queue: MPQueue[int] = MPQueue()
        
        # 3. 初始化 Worker（延迟到 start() 时）
        self._worker_map: Dict[TransferType, Union[WorkerHandle, List[WorkerHandle]]] = {}
```

## 7.2 传输调度

### 7.2.1 调度器主循环

```python
def _scheduler_loop(self) -> None:
    """调度器主循环"""
    while self._running:
        # 1. 接收新的传输图
        new_graphs_num = 0
        while True:
            try:
                transfer_graph = self.task_queue.get_nowait()
                self.scheduler.add_transfer_graph(transfer_graph)
                new_graphs_num += 1
            except queue.Empty:
                break
        
        # 2. 收集完成的传输操作
        finished_ops: List[TransferOp] = []
        while True:
            try:
                op_id = self.finished_ops_queue.get_nowait()
                op = self.op_id_to_op[op_id]
                free_op_from_buffer(op, self.pin_buffer)
                self.completed_queue.put((op.graph_id, op.op_id))
                finished_ops.append(op)
                del self.op_id_to_op[op_id]
            except queue.Empty:
                break
        
        # 3. 调度下一个操作
        if finished_ops or new_graphs_num > 0:
            completed_graph_ids, next_ops = self.scheduler.schedule(finished_ops)
            
            # 4. 分发新操作到 Worker
            for op in next_ops:
                if op.transfer_type == TransferType.VIRTUAL:
                    self.completed_queue.put((op.graph_id, op.op_id))
                else:
                    self.op_id_to_op[op.op_id] = op
                    register_op_to_buffer(op, self.pin_buffer)
                    self._assign_op_to_worker(op)
            
            # 5. 处理完成的传输图
            for graph_id in completed_graph_ids:
                self.completed_queue.put((graph_id, -1))
        
        time.sleep(0.001)  # 防止忙等待
```

### 7.2.2 依赖图调度

TransferScheduler 负责解析传输图的依赖关系：

```python
class TransferScheduler:
    def schedule(self, finished_ops: List[TransferOp]) -> Tuple[List[int], List[TransferOp]]:
        """调度下一个可执行的操作"""
        # 1. 更新完成的操作
        for op in finished_ops:
            self._mark_completed(op)
        
        # 2. 查找可执行的操作（所有前置操作都已完成）
        next_ops = []
        for graph in self.active_graphs.values():
            for op in graph.ops:
                if op.op_id not in self.completed_ops:
                    if self._can_execute(op):
                        next_ops.append(op)
        
        # 3. 检查是否有传输图完成
        completed_graph_ids = []
        for graph_id, graph in self.active_graphs.items():
            if self._is_graph_completed(graph):
                completed_graph_ids.append(graph_id)
        
        return completed_graph_ids, next_ops
    
    def _can_execute(self, op: TransferOp) -> bool:
        """检查操作是否可执行"""
        # 所有前置操作都已完成
        for pred_id in op.predecessors:
            if pred_id not in self.completed_ops:
                return False
        return True
```

## 7.3 传输 Worker

### 7.3.1 GPU↔CPU 传输

#### GPUCPUTransferWorker

```python
class GPUCPUTransferWorker(TransferWorkerBase):
    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        ...
    ):
        """GPU 和 CPU 之间的传输"""
        if transfer_type == TransferType.D2H:
            # GPU → CPU
            transfer_kv_blocks(
                src_ptrs=src_layer_ptrs,
                dst_ptr=dst_tensor.data_ptr(),
                src_block_ids=src_block_ids,
                dst_block_ids=dst_block_ids,
                ...
            )
        elif transfer_type == TransferType.H2D:
            # CPU → GPU
            transfer_kv_blocks(
                src_ptr=src_tensor.data_ptr(),
                dst_ptrs=dst_layer_ptrs,
                src_block_ids=src_block_ids,
                dst_block_ids=dst_block_ids,
                ...
            )
```

**关键技术**：
- **CUDA 异步内存拷贝**：使用 `cudaMemcpyAsync`
- **Pin Memory**：CPU 内存固定，加速传输
- **CUDA Stream**：并行传输

### 7.3.2 CPU↔SSD 传输

#### CPUSSDDiskTransferWorker

```python
class CPUSSDDiskDiskTransferWorker(TransferWorkerBase):
    def _transfer_impl(...):
        """CPU 和 SSD 之间的传输"""
        if transfer_type == TransferType.H2DISK:
            # CPU → SSD
            transfer_kv_blocks_ssd(
                cpu_tensor_ptr=cpu_tensor.data_ptr(),
                cpu_block_ids=cpu_block_ids,
                ssd_block_ids=ssd_block_ids,
                file_descriptors=file_descriptors,
                is_read=False,  # 写操作
                ...
            )
        elif transfer_type == TransferType.DISK2H:
            # SSD → CPU
            transfer_kv_blocks_ssd(
                cpu_tensor_ptr=cpu_tensor.data_ptr(),
                cpu_block_ids=cpu_block_ids,
                ssd_block_ids=ssd_block_ids,
                file_descriptors=file_descriptors,
                is_read=True,  # 读操作
                ...
            )
```

**关键技术**：
- **io_uring**：Linux 异步 I/O 接口
- **批量 I/O**：一次提交多个 I/O 操作
- **直接 I/O**：绕过页缓存

### 7.3.3 CPU↔Remote 传输

#### CPURemoteTransferWorker

```python
class CPURemoteTransferWorker(TransferWorkerBase):
    def _transfer_impl(...):
        """CPU 和 Remote 之间的传输"""
        # 使用网络协议（如 S3、HTTP）进行传输
        # 支持断点续传、重试等机制
```

## 7.4 Pin Memory 优化

### 7.4.1 Pin Memory 的作用

Pin Memory（固定内存）是 CUDA 优化的关键技术：

```python
def cudaHostRegister(tensor: torch.Tensor) -> None:
    """注册 CPU tensor 为 Pin Memory"""
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    ret = cudart.cudaHostRegister(
        ctypes.c_void_p(ptr), 
        ctypes.c_size_t(size), 
        1  # cudaHostRegisterPortable
    )
```

**优势**：
- GPU 可以直接访问 CPU 内存
- 传输速度提升 2-3 倍
- 支持异步传输

### 7.4.2 SharedOpPool

FlexKV 使用 `SharedOpPool` 管理 Pin Memory：

```python
class SharedOpPool:
    def __init__(self, num_slots: int, max_blocks_per_slot: int):
        # 预分配 Pin Memory
        self.buffer = torch.zeros(
            (num_slots, max_blocks_per_slot),
            dtype=torch.int64,
            pin_memory=True
        )
    
    def allocate_slot(self, block_ids: np.ndarray) -> int:
        """分配一个 Slot，将 Block IDs 复制到 Pin Memory"""
        slot_id = self._find_free_slot()
        self.buffer[slot_id, :len(block_ids)] = torch.from_numpy(block_ids)
        return slot_id
```

## 7.5 本章小结

本章详细介绍了传输引擎的设计：

1. **TransferEngine**：
   - 传输调度
   - 依赖图执行
   - 异步并行传输

2. **传输 Worker**：
   - GPU↔CPU：CUDA 异步传输
   - CPU↔SSD：io_uring 异步 I/O
   - CPU↔Remote：网络传输

3. **性能优化**：
   - Pin Memory
   - 批量传输
   - 并行执行

传输引擎作为 FlexKV 的数据面，负责高效执行数据传输，是整个系统性能的关键。

---

**下一章预告**：第八章将详细介绍存储引擎的设计，包括存储初始化、布局管理，以及各种存储分配器的实现。

