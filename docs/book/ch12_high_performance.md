# 第十二章：高性能传输实现

> 本章详细介绍 FlexKV 的高性能传输优化技术，包括 CUDA 传输优化、io_uring 异步 I/O，以及网络传输的实现细节。

## 12.1 CUDA 传输优化

### 12.1.1 异步内存拷贝

FlexKV 使用 CUDA 异步内存拷贝进行 GPU↔CPU 传输：

```cpp
// C++ 实现 (transfer.cu)
void transfer_kv_blocks(
    void* src_ptr,
    void* dst_ptr,
    const int64_t* src_block_ids,
    const int64_t* dst_block_ids,
    int num_blocks,
    size_t block_size,
    cudaStream_t stream
) {
    // 使用 cudaMemcpyAsync 异步传输
    for (int i = 0; i < num_blocks; i++) {
        void* src = (char*)src_ptr + src_block_ids[i] * block_size;
        void* dst = (char*)dst_ptr + dst_block_ids[i] * block_size;
        
        cudaMemcpyAsync(
            dst, src, block_size,
            cudaMemcpyDeviceToHost,  // 或 cudaMemcpyHostToDevice
            stream
        );
    }
}
```

**优势**：
- **异步执行**：不阻塞 CPU
- **流并行**：多个传输可以并行
- **与计算重叠**：传输与计算可以同时进行

### 12.1.2 Copy Engine 加速

Copy Engine 是 NVIDIA GPU 的专用硬件，可以加速内存传输：

```python
use_ce_transfer_h2d: bool = False  # 启用 Copy Engine (CPU → GPU)
use_ce_transfer_d2h: bool = False  # 启用 Copy Engine (GPU → CPU)
```

**启用 Copy Engine**：

```cpp
// 使用 Copy Engine 进行传输
cudaMemcpy3DPeerAsync(
    &copy_params,
    stream
);
```

**优势**：
- **硬件加速**：专用硬件，速度更快
- **卸载 CPU**：减少 CPU 负担

### 12.1.3 Pin Memory

Pin Memory（固定内存）是 CUDA 传输的关键优化：

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
- **直接访问**：GPU 可以直接访问 CPU 内存
- **传输加速**：速度提升 2-3 倍
- **DMA 传输**：使用 DMA，不经过 CPU

**使用场景**：
- CPU 缓存的内存自动使用 Pin Memory
- 传输缓冲区使用 Pin Memory

## 12.2 io_uring 异步 I/O

### 12.2.1 io_uring 基础

io_uring 是 Linux 5.1+ 引入的异步 I/O 接口：

```cpp
// C++ 实现 (transfer_ssd.cpp)
void transfer_kv_blocks_ssd(
    void* cpu_tensor_ptr,
    const int64_t* cpu_block_ids,
    const int64_t* ssd_block_ids,
    int num_blocks,
    int* file_descriptors,
    int num_files_per_device,
    bool is_read,
    ...
) {
    // 使用 io_uring 进行异步 I/O
    for (int bid = 0; bid < num_blocks; bid++) {
        int cpu_block_id = cpu_block_ids[bid];
        int ssd_block_id = ssd_block_ids[bid];
        int fd = file_descriptors[ssd_block_id % num_files_per_device];
        
        // 计算文件偏移
        off_t offset = calculate_offset(ssd_block_id, ...);
        
        // 提交 I/O 请求
        struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
        if (is_read) {
            io_uring_prep_read(sqe, fd, cpu_ptr, size, offset);
        } else {
            io_uring_prep_write(sqe, fd, cpu_ptr, size, offset);
        }
    }
    
    // 提交所有请求
    io_uring_submit(ring);
    
    // 等待完成
    io_uring_wait_cqe(ring, &cqe);
}
```

### 12.2.2 批量 I/O

io_uring 支持批量提交多个 I/O 请求：

```cpp
// 一次提交多个 I/O 操作
for (int i = 0; i < num_blocks; i++) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    io_uring_prep_read(sqe, ...);
}
io_uring_submit(ring);  // 批量提交
```

**优势**：
- **减少系统调用**：批量提交，减少系统调用次数
- **提升吞吐量**：并行执行多个 I/O

### 12.2.3 配置优化

```python
ssd_cache_iouring_entries: int = 512  # io_uring 队列大小
```

**优化建议**：
- **512**：适合大多数场景
- **1024-2048**：适合高并发场景
- 过大会占用过多内存

## 12.3 网络传输

### 12.3.1 Remote 存储传输

FlexKV 支持通过网络访问 Remote 存储（如 S3）：

```python
class CPURemoteTransferWorker(TransferWorkerBase):
    def _transfer_impl(...):
        """CPU 和 Remote 之间的传输"""
        # 使用网络协议（如 S3 SDK）进行传输
        if transfer_type == TransferType.H2REMOTE:
            # CPU → Remote
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=cpu_data
            )
        elif transfer_type == TransferType.REMOTE2H:
            # Remote → CPU
            response = s3_client.get_object(
                Bucket=bucket,
                Key=key
            )
            cpu_data = response['Body'].read()
```

### 12.3.2 断点续传

Remote 传输支持断点续传：

```python
# 支持 Range 请求
s3_client.get_object(
    Bucket=bucket,
    Key=key,
    Range=f'bytes={start}-{end}'
)
```

### 12.3.3 重试机制

网络传输实现重试机制：

```python
max_retries = 3
for attempt in range(max_retries):
    try:
        # 传输操作
        break
    except Exception as e:
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # 指数退避
        else:
            raise
```

## 12.4 并行传输优化

### 12.4.1 多线程传输

FlexKV 使用多线程并行传输：

```python
class TransferEngine:
    def _init_workers(self):
        # 为每个 DP Rank 创建 Worker
        self.gpucpu_workers = [
            GPUCPUTransferWorker.create_worker(...)
            for i in range(self.dp_size)
        ]
```

**优势**：
- **并行传输**：多个传输同时进行
- **资源利用**：充分利用多核 CPU

### 12.4.2 传输调度

TransferScheduler 负责调度传输任务：

```python
class TransferScheduler:
    def schedule(self, finished_ops: List[TransferOp]):
        # 1. 查找可执行的操作（依赖已满足）
        next_ops = []
        for op in self.active_ops:
            if self._can_execute(op):
                next_ops.append(op)
        
        # 2. 按优先级排序
        next_ops.sort(key=lambda op: op.priority)
        
        return next_ops
```

## 12.5 性能指标

### 12.5.1 传输带宽

**GPU↔CPU**：
- **理论带宽**：PCIe 4.0 ≈ 64 GB/s
- **实际带宽**：通常 40-50 GB/s（受限于实现）

**CPU↔SSD**：
- **理论带宽**：NVMe SSD ≈ 3-7 GB/s
- **实际带宽**：取决于 SSD 性能

**CPU↔Remote**：
- **理论带宽**：取决于网络带宽
- **实际带宽**：通常较低（网络延迟）

### 12.5.2 传输延迟

**GPU↔CPU**：
- **延迟**：微秒级（μs）
- **优化**：使用 Pin Memory、Copy Engine

**CPU↔SSD**：
- **延迟**：毫秒级（ms）
- **优化**：使用 io_uring、批量 I/O

**CPU↔Remote**：
- **延迟**：10-100 毫秒级
- **优化**：并发传输、压缩

## 12.6 本章小结

本章详细介绍了高性能传输的实现：

1. **CUDA 传输**：
   - 异步内存拷贝
   - Copy Engine 加速
   - Pin Memory 优化

2. **io_uring I/O**：
   - 异步 I/O 接口
   - 批量 I/O 操作
   - 配置优化

3. **网络传输**：
   - Remote 存储访问
   - 断点续传
   - 重试机制

4. **并行优化**：
   - 多线程传输
   - 传输调度

高性能传输是 FlexKV 性能的关键，各种优化技术的组合可以显著提升传输效率。

---

**下一章预告**：第十三章将详细介绍 FlexKV 与 vLLM 的集成，包括 Patch 应用、配置设置，以及接口对接细节。

