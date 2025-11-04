# 第三章：架构概览

> 本章从宏观角度介绍 FlexKV 的整体架构和代码组织，帮助读者建立对系统的整体认识。通过理解架构层次和目录结构，读者可以快速掌握 FlexKV 的"骨架"，为深入理解各个模块的实现细节打下基础。

## 3.1 整体架构分层

FlexKV 采用清晰的三层架构设计，从上到下分别是：**适配层**、**管理层**和**引擎层**。

```
┌─────────────────────────────────────────────┐
│  第一层：适配层 (Integration Layer)           │
│  - KVConnector                               │
│  - 外部框架接口适配                           │
│  - 配置转换与封装                             │
└───────────────┬─────────────────────────────┘
                │ 接口适配、配置转换
                ↓
┌─────────────────────────────────────────────┐
│  第二层：管理层 (Management Layer)            │
│  - KVManager (统一管理接口)                  │
│  - KVTaskEngine (任务管理)                   │
│  - Server/Client (进程间通信)                │
└───────────────┬─────────────────────────────┘
                │ 任务调度、生命周期管理
                ↓
┌─────────────────────────────────────────────┐
│  第三层：引擎层 (Engine Layer)                │
│  ├── GlobalCacheEngine (缓存引擎)            │
│  │   - 前缀匹配决策                           │
│  │   - 多级缓存协调                           │
│  ├── TransferEngine (传输引擎)              │
│  │   - 数据传输执行                           │
│  │   - 异步并行传输                           │
│  └── StorageEngine (存储引擎)                │
│      - 存储空间管理                           │
│      - Block 级别存储                         │
└─────────────────────────────────────────────┘
```

### 3.1.1 数据流向

一个典型的 GET 请求在 FlexKV 中的处理流程：

```
用户请求 (vLLM)
    ↓
适配层: KVConnector 接收请求
    ↓
管理层: KVManager.get_match() 
    ↓
管理层: KVTaskEngine 创建任务
    ↓
引擎层: GlobalCacheEngine.match() 匹配缓存
    ↓
引擎层: GlobalCacheEngine 构建传输图
    ↓
引擎层: TransferEngine 执行数据传输
    ↓
引擎层: StorageEngine 提供存储访问
    ↓
管理层: 任务完成，返回结果
    ↓
适配层: KVConnector 返回给 vLLM
```

## 3.2 适配层 (Integration Layer)

### 3.2.1 职责与定位

适配层是 FlexKV 与外部推理框架（如 vLLM）交互的桥梁，负责：

1. **接口适配**：实现外部框架定义的 KV Connector 接口
2. **配置转换**：将外部框架的配置转换为 FlexKV 内部配置
3. **任务封装**：将外部框架的请求封装为 FlexKV 的任务
4. **角色分离**：区分 Scheduler 和 Worker 两种角色

### 3.2.2 核心组件

#### FlexKVConnectorV1Impl

这是适配层的入口类，实现了 vLLM 的 KV Connector 接口：

```python
class FlexKVConnectorV1Impl:
    def __init__(self, vllm_config: "VllmConfig", role: "KVConnectorRole"):
        self.role = role
        flexkv_config = FlexKVConfig.from_env()
        flexkv_config.post_init_from_vllm_config(vllm_config)
        
        if role == KVConnectorRole.SCHEDULER:
            self.connector = FlexKVSchedulerConnector(flexkv_config)
        elif role == KVConnectorRole.WORKER:
            self.connector = FlexKVWorkerConnector(flexkv_config)
```

**关键特性**：
- 根据角色（Scheduler/Worker）创建不同的 Connector
- 从环境变量加载配置
- 将 vLLM 配置转换为 FlexKV 配置

#### FlexKVSchedulerConnector

Scheduler 端的 Connector，负责：

1. **匹配决策**：`get_num_new_matched_tokens()` - 查询有多少新的 token 可以从缓存中获取
2. **任务管理**：`launch_tasks()`, `cancel_tasks()` - 管理任务的启动和取消
3. **状态更新**：`update_state_after_alloc()` - 在块分配后更新状态
4. **请求完成**：`request_finished()` - 处理请求完成后的 KV Cache 保存

```python
class FlexKVSchedulerConnector:
    def __init__(self, flexkv_config: FlexKVConfig):
        # 初始化配置
        self.model_config = ModelConfig(...)
        self.cache_config = CacheConfig(...)
        
        # 创建 KVManager（进入管理层）
        self.flexkv_manager = KVManager(...)
        self.flexkv_manager.start()
        
        # 任务管理
        self.get_tasks: dict[int, FlexKVGetTask] = {}
        self.put_tasks: dict[int, FlexKVPutTask] = {}
        self.tasks_to_launch: dict[int, FlexKVTask] = {}
```

#### FlexKVWorkerConnector

Worker 端的 Connector，负责：

1. **GPU 块注册**：`register_kv_caches()` - 将 GPU 的 KV Cache 注册到 FlexKV
2. **与 Server 通信**：通过 `KVTPClient` 与 Scheduler 端的 Server 通信

```python
class FlexKVWorkerConnector:
    def __init__(self, flexkv_config: FlexKVConfig):
        current_device_id = torch.cuda.current_device()
        # 创建 TP Client（用于与 Server 通信）
        self.tp_client = KVTPClient(flexkv_config.server_recv_port, 0, current_device_id)
    
    def register_to_server(self, kv_caches: dict[str, torch.Tensor]):
        # 解析 GPU KV Cache 的布局信息
        gpu_layout = KVCacheLayout(...)
        # 注册到 Server
        self.tp_client.register_to_server(gpu_blocks, gpu_layout)
```

### 3.2.3 目录位置

适配层代码位于：
```
flexkv/integration/
├── __init__.py
├── config.py              # FlexKVConfig 配置类
├── stats.py              # 统计信息
├── utils.py              # 工具函数
└── vllm/
    ├── __init__.py
    └── vllm_v1_adapter.py  # 核心适配实现
```

## 3.3 管理层 (Management Layer)

### 3.3.1 职责与定位

管理层是 FlexKV 的核心协调层，负责：

1. **统一接口**：提供统一的 API（get/put/launch/cancel/wait）
2. **任务管理**：管理任务的创建、调度、生命周期
3. **进程通信**：在多进程部署时，协调 Scheduler 和 Worker 之间的通信

### 3.3.2 核心组件

#### KVManager

KVManager 是 FlexKV 的统一管理接口，提供了简洁的 API：

```python
class KVManager:
    def __init__(self, model_config, cache_config, ...):
        self.server_client_mode = model_config.dp_size > 1
        
        if self.server_client_mode:
            # 多进程模式：创建 Server 或 Client
            if dp_client_id == 0:
                self.server_handle = KVServer.create_server(...)
            else:
                self.dp_client = KVDPClient(...)
        else:
            # 单进程模式：直接创建 KVTaskEngine
            self.kv_task_engine = KVTaskEngine(...)
    
    # 核心 API
    def get_match(self, token_ids, token_mask=None):
        """匹配 KV Cache，返回匹配的 token mask"""
        
    def put_match(self, token_ids, token_mask=None):
        """匹配 KV Cache，返回未匹配的 token mask"""
        
    def launch(self, task_ids, slot_mappings):
        """启动传输任务"""
        
    def cancel(self, task_ids):
        """取消任务"""
        
    def wait(self, task_ids):
        """等待任务完成"""
```

**两种部署模式**：

1. **单进程模式**（`dp_size == 1`）：
   - 直接使用 `KVTaskEngine`
   - 所有组件在同一进程

2. **多进程模式**（`dp_size > 1`）：
   - Scheduler 进程：创建 `KVServer`
   - Worker 进程：创建 `KVDPClient` 连接 Server
   - 通过 ZMQ 进行进程间通信

#### KVTaskEngine

KVTaskEngine 负责任务的生命周期管理：

```python
class KVTaskEngine(KVTaskManager):
    def __init__(self, model_config, cache_config, ...):
        super().__init__(...)
        
        # 创建缓存引擎（进入引擎层）
        self.cache_engine = GlobalCacheEngine(cache_config, model_config)
        
        # 创建传输管理器（进入引擎层）
        self.transfer_handle = TransferManagerHandle(...)
        
        # 任务管理
        self.tasks: ExpiringDict[int, KVTask] = ExpiringDict(...)
        self.graph_to_task: Dict[int, int] = {}
    
    def get_async(self, token_ids, slot_mapping, token_mask=None):
        """异步 GET 操作"""
        # 1. 创建任务
        task_id = self.create_get_task(...)
        # 2. 调用缓存引擎匹配
        # 3. 构建传输图
        # 4. 启动任务
        self._launch_task(task_id)
        return task_id, return_mask
    
    def put_async(self, token_ids, slot_mapping, token_mask=None):
        """异步 PUT 操作"""
        # 类似流程...
```

**任务生命周期**：

```
创建任务 (CREATED)
    ↓
匹配缓存，构建传输图 (READY)
    ↓
启动传输 (RUNNING)
    ↓
传输完成 (COMPLETED)
```

#### Server/Client 进程通信

在多进程部署时，需要 Server 和 Client 协调工作：

**KVServer**：
- 在 Scheduler 进程运行
- 接收来自多个 DP Client 的请求
- 协调 GPU 块的注册（来自 TP Client）

**KVDPClient**：
- 在 Worker 进程运行（数据并行）
- 向 Server 发送请求
- 接收 Server 的响应

**KVTPClient**：
- 在 Worker 进程运行（张量并行）
- 将 GPU KV Cache 注册到 Server

### 3.3.3 目录位置

管理层代码位于：
```
flexkv/
├── kvmanager.py          # KVManager 统一接口
├── kvtask.py             # KVTaskEngine 任务管理
└── server/
    ├── server.py          # KVServer 服务器
    ├── client.py          # KVDPClient, KVTPClient
    ├── request.py         # 请求/响应类型
    └── utils.py           # 工具函数
```

## 3.4 引擎层 (Engine Layer)

引擎层是 FlexKV 的核心功能层，包含三个关键引擎。

### 3.4.1 缓存引擎 (GlobalCacheEngine)

#### 职责

GlobalCacheEngine 是 FlexKV 的**控制面**，负责：

1. **前缀匹配决策**：决定哪些 Block 可以从缓存中获取
2. **多级缓存协调**：协调 CPU、SSD、Remote 三级缓存
3. **传输图构建**：根据匹配结果构建数据传输图

#### 核心组件

```python
class GlobalCacheEngine:
    def __init__(self, cache_config, model_config):
        # 为每个存储级别创建 CacheEngine
        if cache_config.enable_cpu:
            self.cpu_cache_engine = CacheEngine(
                DeviceType.CPU,
                cache_config.num_cpu_blocks,
                cache_config.tokens_per_block,
                cache_config.evict_ratio
            )
        if cache_config.enable_ssd:
            self.ssd_cache_engine = CacheEngine(...)
        if cache_config.enable_remote:
            self.remote_cache_engine = CacheEngine(...)
    
    def match_all(self, sequence_meta: SequenceMeta):
        """在所有存储级别匹配序列"""
        cpu_result = self.cpu_cache_engine.match(sequence_meta)
        ssd_result = self.ssd_cache_engine.match(sequence_meta)
        remote_result = self.remote_cache_engine.match(sequence_meta)
        return cpu_result, ssd_result, remote_result
    
    def get(self, token_ids, token_mask, slot_mapping, layer_num=-1):
        """GET 操作：构建传输图，从缓存中获取数据"""
        # 1. 匹配缓存
        # 2. 构建传输图（GPU ← CPU ← SSD ← Remote）
        # 3. 返回传输图
```

**RadixTree 前缀匹配**：

- 使用 RadixTree 索引 Block，支持前缀匹配
- 每个 Block 有唯一的 Hash 值
- 匹配时查找最长公共前缀

**Mempool 内存池**：

- 管理 Block 的分配和回收
- 使用位图跟踪 Block 的使用状态
- 支持 LRU 淘汰策略

#### 匹配流程示例

假设有一个序列 `[token1, token2, token3, token4, ..., token16, token17]`（17个 token，tokens_per_block=16）：

1. **对齐到 Block**：丢弃最后一个不完整的 Block，处理 `[token1...token16]`（1个 Block）
2. **计算 Block Hash**：为 `[token1...token16]` 计算 Hash
3. **在各级缓存中匹配**：
   - CPU 缓存：找到匹配 → 命中
   - SSD 缓存：找到匹配 → 需要从 SSD 传输
   - Remote 缓存：未匹配 → 不需要传输
4. **构建传输图**：根据匹配结果，决定传输路径

### 3.4.2 传输引擎 (TransferEngine)

#### 职责

TransferEngine 是 FlexKV 的**数据面**，负责：

1. **数据传输执行**：执行实际的数据传输操作
2. **异步并行传输**：支持多线程并行传输
3. **依赖图调度**：根据传输图的依赖关系调度传输任务

#### 核心组件

```python
class TransferEngine:
    def __init__(self, gpu_handles, model_config, cache_config, 
                 cpu_handle=None, ssd_handle=None, remote_handle=None):
        # 创建传输调度器
        self.scheduler = TransferScheduler()
        
        # 创建各种传输 Worker
        self.gpucpu_workers = [
            GPUCPUTransferWorker.create_worker(...)
            for i in range(dp_size)
        ]
        self.cpussd_read_worker = CPUSSDDiskTransferWorker.create_worker(...)
        self.cpussd_write_worker = CPUSSDDiskTransferWorker.create_worker(...)
        # ...
        
        # Worker 映射
        self._worker_map = {
            TransferType.H2D: self.gpucpu_workers,
            TransferType.D2H: self.gpucpu_workers,
            TransferType.DISK2H: self.cpussd_read_worker,
            TransferType.H2DISK: self.cpussd_write_worker,
            # ...
        }
    
    def submit_transfer_graph(self, transfer_graph: TransferOpGraph):
        """提交传输图执行"""
        self.task_queue.put(transfer_graph)
    
    def _scheduler_loop(self):
        """调度器循环：持续处理传输任务"""
        while self._running:
            # 1. 接收新的传输图
            # 2. 收集完成的传输操作
            # 3. 调度下一个操作
            # 4. 分配给对应的 Worker
```

**传输类型**：

- `H2D` (Host to Device)：CPU → GPU
- `D2H` (Device to Host)：GPU → CPU
- `DISK2H`：SSD → CPU
- `H2DISK`：CPU → SSD
- `REMOTE2H`：Remote → CPU
- `H2REMOTE`：CPU → Remote

**传输 Worker**：

- `GPUCPUTransferWorker`：使用 CUDA 异步内存拷贝
- `CPUSSDDiskTransferWorker`：使用 io_uring 异步 I/O
- `CPURemoteTransferWorker`：网络传输

### 3.4.3 存储引擎 (StorageEngine)

#### 职责

StorageEngine 负责：

1. **存储空间初始化**：为 CPU、SSD、Remote 分配存储空间
2. **Block 级别管理**：以 Block 为单位管理存储
3. **存储布局管理**：管理不同的存储布局（LAYERWISE/BLOCKWISE）

#### 核心组件

```python
class StorageEngine:
    def __init__(self, model_config, cache_config):
        self._storage_handles: Dict[Tuple[DeviceType, int], StorageHandle] = {}
        
        # 初始化 CPU 存储
        if cache_config.enable_cpu:
            cpu_layout = KVCacheLayout(
                type=cache_config.cpu_kv_layout_type,
                num_layer=model_config.num_layers,
                num_block=cache_config.num_cpu_blocks,
                tokens_per_block=cache_config.tokens_per_block,
                # ...
            )
            self.allocate(DeviceType.CPU, cpu_layout, model_config.dtype)
        
        # 初始化 SSD 存储
        if cache_config.enable_ssd:
            # ...
        
        # 初始化 Remote 存储
        if cache_config.enable_remote:
            # ...
    
    def allocate(self, device_type, layout, dtype):
        """分配存储空间"""
        if device_type == DeviceType.CPU:
            allocator = CPUAllocator(layout, dtype)
        elif device_type == DeviceType.SSD:
            allocator = SSDAllocator(layout, dtype, ...)
        # ...
        handle = StorageHandle(allocator, layout, dtype)
        self._storage_handles[(device_type, device_id)] = handle
```

**存储布局类型**：

1. **LAYERWISE**：按层组织，适合 GPU
   ```
   Shape: [layer, block, token, head, head_dim]
   ```

2. **BLOCKWISE**：按块组织，适合 CPU/SSD/Remote
   ```
   Shape: [block, layer, token, head, head_dim]
   ```

**存储分配器**：

- `CPUAllocator`：在 CPU 内存中分配连续空间
- `SSDAllocator`：在 SSD 上创建文件，使用 mmap
- `RemoteAllocator`：在远程存储上创建文件

### 3.4.4 目录位置

引擎层代码位于：
```
flexkv/
├── cache/
│   ├── cache_engine.py    # GlobalCacheEngine, CacheEngine
│   ├── radixtree.py       # RadixTree 索引
│   ├── mempool.py         # Mempool 内存池
│   └── transfer_pattern.py
├── transfer/
│   ├── transfer_engine.py # TransferEngine
│   ├── scheduler.py       # TransferScheduler
│   └── worker.py          # 各种 TransferWorker
└── storage/
    ├── storage_engine.py  # StorageEngine
    └── allocator.py       # 各种 Allocator
```

## 3.5 公共组件

FlexKV 定义了一些公共组件，供各层使用：

### 3.5.1 Block 抽象

**SequenceMeta**：序列元数据

```python
@dataclass
class SequenceMeta:
    token_ids: np.ndarray          # Token ID 数组
    tokens_per_block: int          # 每个 Block 的 token 数量
    block_hashes: np.ndarray       # Block Hash 数组
    
    @property
    def num_blocks(self) -> int:
        return len(self.token_ids) // self.tokens_per_block
    
    def gen_hashes(self) -> None:
        """生成所有 Block 的 Hash"""
        self.block_hashes = gen_hashes(self.token_ids, self.tokens_per_block)
```

### 3.5.2 配置管理

**ModelConfig**：模型配置

```python
@dataclass
class ModelConfig:
    num_layers: int                # Transformer 层数
    num_kv_heads: int              # KV 注意力头数
    head_size: int                 # 每个头的维度
    use_mla: bool = False          # 是否使用 MLA
    dtype: torch.dtype             # 数据类型
    
    @property
    def token_size_in_bytes(self) -> int:
        """计算一个 token 的 KV Cache 大小"""
        kv_dim = 1 if self.use_mla else 2
        return self.num_layers * self.num_kv_heads * self.head_size * \
               kv_dim * self.dtype.itemsize
```

**CacheConfig**：缓存配置

```python
@dataclass
class CacheConfig:
    tokens_per_block: int = 16     # 每个 Block 的 token 数量
    enable_cpu: bool = True        # 是否启用 CPU 缓存
    enable_ssd: bool = False       # 是否启用 SSD 缓存
    enable_remote: bool = False    # 是否启用 Remote 缓存
    num_cpu_blocks: int            # CPU Block 数量
    num_ssd_blocks: int            # SSD Block 数量
    evict_ratio: float = 0.0       # 淘汰比例
    # ...
```

### 3.5.3 传输图抽象

**TransferOp**：单个传输操作

```python
@dataclass
class TransferOp:
    op_id: int
    graph_id: int
    transfer_type: TransferType    # 传输类型
    src_block_ids: np.ndarray      # 源 Block IDs
    dst_block_ids: np.ndarray      # 目标 Block IDs
    layer_id: int                  # 层 ID
    layer_granularity: int         # 层粒度
    successors: List[int]          # 后续操作 IDs
```

**TransferOpGraph**：传输操作图

```python
class TransferOpGraph:
    def __init__(self):
        self.graph_id = get_next_graph_id()
        self.ops: List[TransferOp] = []
    
    def add_transfer_op(self, op: TransferOp):
        """添加传输操作"""
        self.ops.append(op)
    
    def add_dependency(self, op_id1: int, op_id2: int):
        """添加依赖关系：op_id1 完成后才能执行 op_id2"""
        # ...
```

### 3.5.4 目录位置

公共组件代码位于：
```
flexkv/common/
├── block.py           # SequenceMeta, Block 抽象
├── config.py          # ModelConfig, CacheConfig
├── storage.py         # KVCacheLayout, StorageHandle
├── transfer.py        # TransferOp, TransferOpGraph
├── request.py         # KVRequest, KVResponse
├── hash_utils.py      # Hash 计算工具
├── memory_handle.py   # 内存句柄
├── ring_buffer.py     # 环形缓冲区
├── tracer.py          # 追踪工具
└── debug.py           # 调试工具
```

## 3.6 完整的目录结构

```
flexkv/
├── __init__.py
│
├── integration/              # 【适配层】
│   ├── config.py            # FlexKVConfig
│   ├── stats.py             # 统计信息
│   ├── utils.py             # 工具函数
│   └── vllm/
│       └── vllm_v1_adapter.py  # KVConnector 实现
│
├── kvmanager.py             # 【管理层】KVManager 统一接口
├── kvtask.py                # 【管理层】KVTaskEngine 任务管理
│
├── server/                   # 【管理层】进程通信
│   ├── server.py            # KVServer
│   ├── client.py            # KVDPClient, KVTPClient
│   ├── request.py           # 请求/响应类型
│   └── utils.py            # 工具函数
│
├── cache/                    # 【引擎层】缓存引擎
│   ├── cache_engine.py      # GlobalCacheEngine, CacheEngine
│   ├── radixtree.py         # RadixTree 索引
│   ├── mempool.py           # Mempool 内存池
│   └── transfer_pattern.py # 传输模式
│
├── transfer/                 # 【引擎层】传输引擎
│   ├── transfer_engine.py   # TransferEngine
│   ├── scheduler.py         # TransferScheduler
│   └── worker.py            # 各种 TransferWorker
│
├── storage/                  # 【引擎层】存储引擎
│   ├── storage_engine.py    # StorageEngine
│   └── allocator.py         # 各种 Allocator
│
└── common/                   # 【公共组件】
    ├── block.py             # SequenceMeta
    ├── config.py            # ModelConfig, CacheConfig
    ├── storage.py           # KVCacheLayout
    ├── transfer.py          # TransferOp, TransferOpGraph
    ├── request.py           # KVRequest, KVResponse
    ├── hash_utils.py        # Hash 工具
    ├── memory_handle.py     # 内存句柄
    ├── ring_buffer.py       # 环形缓冲区
    ├── tracer.py            # 追踪
    └── debug.py             # 调试
```

## 3.7 总结

本章从宏观角度介绍了 FlexKV 的三层架构：

1. **适配层**：连接外部框架，实现接口适配
2. **管理层**：统一 API，管理任务和进程通信
3. **引擎层**：三个核心引擎协同工作
   - **缓存引擎**：决策"要不要传输、从哪里传输"
   - **传输引擎**：执行"如何传输"
   - **存储引擎**：管理"存储在哪里"

理解这个架构层次关系，有助于：
- 快速定位代码位置
- 理解数据流向
- 掌握系统设计思路

在接下来的章节中，我们将深入探讨每一层的设计和实现细节。

---

**下一章预告**：第四章将详细介绍适配层的设计，包括如何实现 KV Connector 接口，以及与 vLLM 的集成机制。

