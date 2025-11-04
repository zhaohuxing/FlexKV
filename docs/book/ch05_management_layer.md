# 第五章：管理层设计

> 管理层是 FlexKV 的核心协调层，负责统一接口、任务管理和进程通信。本章详细介绍 KVManager、KVTaskEngine 以及 Server/Client 的设计与实现。

## 5.1 KVManager 统一接口

### 5.1.1 设计目标

KVManager 是 FlexKV 的**统一管理接口**，提供简洁、一致的 API，隐藏底层实现细节。

**核心职责**：
1. **统一 API**：提供 get/put/launch/cancel/wait 等统一接口
2. **部署模式选择**：根据配置选择单进程或多进程模式
3. **接口适配**：适配不同的底层实现（KVTaskEngine 或 Server/Client）

### 5.1.2 两种部署模式

#### 单进程模式（dp_size == 1）

```python
class KVManager:
    def __init__(self, model_config, cache_config, ...):
        self.server_client_mode = model_config.dp_size > 1
        
        if not self.server_client_mode:
            # 单进程模式：直接创建 KVTaskEngine
            self.kv_task_engine = KVTaskEngine(
                model_config, 
                cache_config, 
                gpu_register_port
            )
    
    def start(self):
        if not self.server_client_mode:
            self.kv_task_engine.start()
```

**特点**：
- 所有组件在同一进程
- 直接调用，无需进程通信
- 延迟最低

#### 多进程模式（dp_size > 1）

```python
class KVManager:
    def __init__(self, model_config, cache_config, ...):
        self.server_client_mode = model_config.dp_size > 1
        
        if self.server_client_mode:
            if dp_client_id == 0:
                # DP Rank 0：创建 Server
                self.server_handle = KVServer.create_server(...)
            else:
                # 其他 DP Rank：创建 Client
                self.dp_client = KVDPClient(...)
    
    def start(self):
        if self.server_client_mode:
            if self.dp_client_id == 0:
                # Server 会自动启动
                pass
            else:
                # Client 发送启动请求
                self.dp_client.start_server_and_register()
```

**特点**：
- Scheduler 进程运行 Server
- Worker 进程运行 Client
- 通过 ZMQ 进行进程间通信
- 支持多 DP Rank 并行

### 5.1.3 核心 API

#### get_match

```python
def get_match(
    self, 
    token_ids: Union[torch.Tensor, np.ndarray],
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    layer_granularity: int = -1,
    dp_id: int = 0,
) -> Tuple[int, np.ndarray]:
    """匹配 KV Cache，返回匹配的 token mask"""
    # 类型转换
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.numpy()
    if isinstance(token_mask, torch.Tensor):
        token_mask = token_mask.numpy()
    
    # 根据模式分发
    if self.server_client_mode:
        task_id, mask = self.dp_client.get_match(...)
    else:
        task_id, mask = self.kv_task_engine.get_match(...)
    
    return task_id, mask
```

**返回值**：
- `task_id`：任务 ID，用于后续操作
- `mask`：布尔数组，标记哪些 token 已匹配

#### put_match

```python
def put_match(
    self,
    token_ids: Union[torch.Tensor, np.ndarray],
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    dp_id: int = 0,
) -> Tuple[int, np.ndarray]:
    """匹配 KV Cache，返回未匹配的 token mask"""
    # 类似 get_match 的实现
```

**返回值**：
- `task_id`：任务 ID
- `mask`：布尔数组，标记哪些 token **未匹配**（需要保存）

#### launch

```python
def launch(
    self,
    task_ids: Union[int, List[int]],
    slot_mappings: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]]
) -> None:
    """启动传输任务"""
    # 类型转换和标准化
    if isinstance(task_ids, int):
        task_ids = [task_ids]
    if not isinstance(slot_mappings, List):
        slot_mappings = [slot_mappings]
    if isinstance(slot_mappings[0], torch.Tensor):
        slot_mappings = [sm.numpy() for sm in slot_mappings]
    
    # 根据模式分发
    if self.server_client_mode:
        self.dp_client.launch_tasks(task_ids, slot_mappings)
    else:
        self.kv_task_engine.launch_tasks(task_ids, slot_mappings)
```

**关键点**：
- `slot_mapping`：每个 token 对应的 GPU Block Slot ID
- 支持批量启动多个任务

#### cancel

```python
def cancel(self, task_ids: Union[int, List[int]]) -> None:
    """取消任务"""
    if isinstance(task_ids, int):
        task_ids = [task_ids]
    
    if self.server_client_mode:
        self.dp_client.cancel_tasks(task_ids)
    else:
        self.kv_task_engine.cancel_tasks(task_ids)
```

#### wait 和 try_wait

```python
def wait(
    self,
    task_ids: Union[int, List[int]],
    timeout: float = 20.0,
    completely: bool = False
) -> Dict[int, KVResponse]:
    """阻塞等待任务完成"""
    
def try_wait(
    self, 
    task_ids: Union[int, List[int]]
) -> Dict[int, KVResponse]:
    """非阻塞查询任务状态"""
```

**区别**：
- `wait`：阻塞等待，直到任务完成或超时
- `try_wait`：立即返回，不阻塞

## 5.2 KVTaskEngine 任务管理

### 5.2.1 任务生命周期

KVTaskEngine 管理任务的完整生命周期：

```
CREATED → READY → RUNNING → COMPLETED
              ↓
         CANCELLED
```

**状态定义**：

```python
class TaskStatus(Enum):
    CREATED = "created"        # 已创建
    READY = "ready"           # 准备就绪，可以启动
    RUNNING = "running"       # 正在运行
    COMPLETED = "completed"   # 已完成
    CANCELLED = "cancelled"   # 已取消
```

### 5.2.2 任务创建

#### create_get_task

```python
def create_get_task(
    self,
    task_id: int,
    token_ids: np.ndarray,
    slot_mapping: np.ndarray,
    token_mask: Optional[np.ndarray] = None,
    layer_granularity: int = -1,
    dp_id: int = 0,
    is_fake_slot_mapping: bool = False,
) -> None:
    """创建 GET 任务"""
    # 1. 调用缓存引擎匹配，构建传输图
    graph, return_mask, callback, task_end_op_id = self.cache_engine.get(
        task_id,
        token_ids,
        token_mask,
        slot_mapping,
        self.model_config.num_layers,
        layer_granularity,
        dp_id
    )
    
    # 2. 创建任务对象
    self.tasks[task_id] = KVTask(
        task_id=task_id,
        task_type=TaskType.GET,
        task_end_op_id=task_end_op_id,
        task_end_op_finished=False,
        status=TaskStatus.UNREADY if is_fake_slot_mapping else TaskStatus.READY,
        token_ids=token_ids,
        slot_mapping=slot_mapping,
        token_mask=token_mask,
        dp_id=dp_id,
        graph=graph,
        return_mask=return_mask,
        callback=callback
    )
    
    # 3. 建立图到任务的映射
    self.graph_to_task[graph.graph_id] = task_id
```

**关键点**：
- 调用 `cache_engine.get()` 匹配缓存，构建传输图
- 任务初始状态：如果有 fake slot_mapping，状态为 `UNREADY`，否则为 `READY`

#### create_put_task

```python
def create_put_task(
    self,
    task_id: int,
    token_ids: np.ndarray,
    slot_mapping: np.ndarray,
    token_mask: Optional[np.ndarray] = None,
    dp_id: int = 0,
    is_fake_slot_mapping: bool = False,
) -> None:
    """创建 PUT 任务"""
    # 类似 create_get_task，但调用 cache_engine.put()
    graph, return_mask, callback, task_end_op_id = self.cache_engine.put(...)
    # ...
```

### 5.2.3 任务启动

```python
def _launch_task(self, task_id: int) -> None:
    """启动任务"""
    task = self.tasks[task_id]
    
    # 1. 检查任务状态
    if task.is_completed():
        return
    if task.status != TaskStatus.READY:
        raise ValueError(f"Task {task_id} status is {task.status}, cannot launch")
    
    # 2. 更新状态
    transfer_graph = task.graph
    task.status = TaskStatus.RUNNING
    
    # 3. 提交传输图
    if transfer_graph.num_ops > 0:
        self.transfer_handle.submit(transfer_graph)
```

**关键点**：
- 只有 `READY` 状态的任务才能启动
- 将传输图提交给 `TransferHandle`

### 5.2.4 任务状态更新

```python
def _update_tasks(self, timeout: float = 0.001) -> None:
    """更新任务状态"""
    # 1. 获取完成的传输操作
    completed_ops = self._get_completed_ops(timeout)
    
    # 2. 更新任务状态
    for completed_graph_id, completed_op_id in completed_ops:
        if completed_graph_id not in self.graph_to_task:
            continue
        
        task_id = self.graph_to_task[completed_graph_id]
        task = self.tasks[task_id]
        
        if completed_op_id == -1:
            # 整个传输图完成
            self._mark_completed(task_id)
        elif completed_op_id == task.task_end_op_id:
            # 任务结束操作完成
            self.tasks[task_id].task_end_op_finished = True
```

**关键理解**：
- `completed_op_id == -1`：表示整个传输图完成
- `completed_op_id == task.task_end_op_id`：表示任务的关键操作完成

### 5.2.5 任务取消

```python
def _cancel_task(self, task_id: int) -> None:
    """取消任务"""
    task = self.tasks[task_id]
    
    # 检查任务状态
    if task.is_completed():
        flexkv_logger.warning(f"Task {task_id} is already completed")
        return
    if task.status == TaskStatus.RUNNING:
        flexkv_logger.warning(f"Task {task_id} is running, cannot cancel")
        return
    if task.status == TaskStatus.CANCELLED:
        flexkv_logger.warning(f"Task {task_id} is already cancelled")
        return
    
    # 更新状态
    task.status = TaskStatus.CANCELLED
    self.graph_to_task.pop(task.graph.graph_id, None)
```

**限制**：
- 只有 `CREATED` 或 `READY` 状态的任务可以取消
- `RUNNING` 状态的任务不能取消（已提交传输）

### 5.2.6 任务完成检查

```python
def check_completed(self, task_id: int, completely: bool = False) -> bool:
    """检查任务是否完成"""
    # 更新任务状态
    self._update_tasks()
    
    task = self.tasks.get(task_id)
    if task is None:
        return False
    
    if completely:
        # 完全完成：传输图完成 + 任务结束操作完成
        return (task.status == TaskStatus.COMPLETED and 
                task.task_end_op_finished)
    else:
        # 部分完成：只需传输图完成
        return task.status == TaskStatus.COMPLETED
```

## 5.3 Server/Client 进程通信

### 5.3.1 架构设计

在多进程模式下，FlexKV 使用 **Server-Client 架构**：

```
┌─────────────────────────────────┐
│  Scheduler 进程 (DP Rank 0)     │
│  ┌──────────────────────────┐   │
│  │      KVServer            │   │
│  │  - 接收 Client 请求      │   │
│  │  - 管理 KVTaskEngine     │   │
│  │  - 协调 GPU 块注册       │   │
│  └──────────────────────────┘   │
└───────────┬─────────────────────┘
            │ ZMQ 通信
            ↓
┌─────────────────────────────────┐
│  Worker 进程 (DP Rank 1, 2, ...)│
│  ┌──────────────────────────┐   │
│  │    KVDPClient            │   │
│  │  - 发送请求到 Server    │   │
│  │  - 接收 Server 响应     │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
```

### 5.3.2 KVServer 设计

#### 初始化

```python
class KVServer:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        gpu_register_port: str,
        server_recv_port: str
    ):
        # 1. 初始化 ZMQ
        self.context = zmq.Context(2)
        self.recv_from_client = get_zmq_socket(
            self.context, 
            zmq.SocketType.PULL, 
            server_recv_port, 
            True
        )
        
        # 2. 客户端管理
        self.client_manager = ClientManager(max_num_dp_client=model_config.dp_size)
        
        # 3. 创建 KVTaskEngine
        self.kv_task_engine = KVTaskEngine(
            model_config, 
            cache_config, 
            gpu_register_port, 
            False  # 不使用独立进程
        )
        
        # 4. 请求处理器映射表
        self.request_handlers = {
            StartRequest: self._handle_start_request,
            RegisterDPClientRequest: self._handle_register_dp_client_request,
            GetRequest: self._handle_get_request,
            PutRequest: self._handle_put_request,
            LaunchTaskRequest: self._handle_launch_task_request,
            CancelTaskRequest: self._handle_cancel_task_request,
            WaitRequest: self._handle_wait_request,
            # ...
        }
```

#### 请求处理循环

```python
def run(self) -> None:
    """Server 主循环"""
    self.start_server()
    self._running = True
    
    while self._running:
        try:
            # 1. 接收请求
            request = self.recv_from_client.recv_pyobj(zmq.NOBLOCK)
            
            # 2. 分发处理
            request_type = type(request)
            if request_type in self.request_handlers:
                handler = self.request_handlers[request_type]
                handler(request)
            else:
                logger.error(f"Unknown request type: {request_type}")
                
        except zmq.Again:
            # 没有请求，继续循环
            time.sleep(0.001)
        except Exception as e:
            logger.error(f"Error handling request: {e}")
```

#### 请求处理示例

```python
def _handle_get_request(self, req: GetRequest) -> None:
    """处理 GET 请求"""
    task_id, return_mask = self.kv_task_engine.get_match(
        token_ids=req.token_ids,
        token_mask=req.token_mask,
        layer_granularity=req.layer_granularity,
        dp_id=req.dp_client_id
    )
    
    # 发送响应
    response = GetResponse(
        request_id=req.request_id,
        task_id=task_id,
        return_mask=return_mask
    )
    self.client_manager.send_response(req.dp_client_id, response)

def _handle_launch_task_request(self, req: LaunchTaskRequest) -> None:
    """处理 Launch 请求"""
    self.kv_task_engine.launch_tasks(
        task_ids=req.task_ids,
        slot_mappings=req.slot_mappings
    )
    
    response = LaunchTaskResponse(request_id=req.request_id)
    self.client_manager.send_response(req.dp_client_id, response)
```

### 5.3.3 KVDPClient 设计

#### 初始化

```python
class KVDPClient:
    def __init__(
        self,
        server_recv_port: str,
        model_config: ModelConfig,
        dp_client_id: int
    ):
        self.dp_client_id = dp_client_id
        self.model_config = model_config
        
        # 创建 ZMQ Socket
        self.context = zmq.Context(2)
        self.send_to_server = get_zmq_socket(
            self.context,
            zmq.SocketType.PUSH,
            server_recv_port,
            True
        )
        
        # 接收响应的线程
        self.response_thread = threading.Thread(target=self._response_loop)
        self.response_thread.daemon = True
        self.response_thread.start()
```

#### 请求发送

```python
def get_match(
    self,
    token_ids: np.ndarray,
    token_mask: Optional[np.ndarray] = None,
    layer_granularity: int = -1
) -> Tuple[int, np.ndarray]:
    """发送 GET 匹配请求"""
    request_id = self._get_next_request_id()
    
    # 构建请求
    req = GetMatchRequest(
        dp_client_id=self.dp_client_id,
        request_id=request_id,
        token_ids=token_ids,
        token_mask=token_mask,
        layer_granularity=layer_granularity
    )
    
    # 发送请求
    self.send_to_server.send_pyobj(req, flags=zmq.NOBLOCK)
    
    # 等待响应
    response = self._wait_for_response(request_id, timeout=10.0)
    return response.task_id, response.return_mask
```

#### 响应接收

```python
def _response_loop(self) -> None:
    """响应接收循环"""
    recv_socket = get_zmq_socket(
        self.context,
        zmq.SocketType.PULL,
        self.response_port,
        True
    )
    
    while self._running:
        try:
            # 接收响应
            response = recv_socket.recv_pyobj(zmq.NOBLOCK)
            
            # 存入响应队列
            request_id = response.request_id
            self.response_queue.put((request_id, response))
            
        except zmq.Again:
            time.sleep(0.001)
        except Exception as e:
            logger.error(f"Error receiving response: {e}")

def _wait_for_response(self, request_id: int, timeout: float) -> Any:
    """等待特定请求的响应"""
    end_time = time.time() + timeout
    
    while time.time() < end_time:
        try:
            # 检查响应队列
            resp_id, response = self.response_queue.get(timeout=0.1)
            if resp_id == request_id:
                return response
            else:
                # 不是这个请求的响应，放回去
                self.response_queue.put((resp_id, response))
        except queue.Empty:
            continue
    
    raise TimeoutError(f"Timeout waiting for response {request_id}")
```

### 5.3.4 KVTPClient 设计

KVTPClient 用于 Worker 进程将 GPU KV Cache 注册到 Server。

```python
class KVTPClient:
    def __init__(
        self,
        server_recv_port: str,
        dp_client_id: int,
        device_id: int
    ):
        self.dp_client_id = dp_client_id
        self.device_id = device_id
        
        # 创建 Socket（用于注册）
        self.context = zmq.Context(2)
        self.send_to_server = get_zmq_socket(...)
    
    def register_to_server(
        self,
        gpu_blocks: List[torch.Tensor],
        gpu_layout: KVCacheLayout
    ) -> None:
        """注册 GPU KV Cache 到 Server"""
        # 1. 构建注册请求
        req = RegisterTPClientRequest(
            dp_client_id=self.dp_client_id,
            device_id=self.device_id,
            gpu_blocks=gpu_blocks,
            gpu_layout=gpu_layout
        )
        
        # 2. 发送请求
        self.send_to_server.send_pyobj(req, flags=zmq.NOBLOCK)
        
        # 3. 等待确认
        # （Server 会通过另一个通道发送确认）
```

## 5.4 通信协议

### 5.4.1 请求类型

```python
@dataclass
class StartRequest:
    dp_client_id: int

@dataclass
class GetMatchRequest:
    dp_client_id: int
    request_id: int
    token_ids: np.ndarray
    token_mask: Optional[np.ndarray]
    layer_granularity: int

@dataclass
class PutMatchRequest:
    dp_client_id: int
    request_id: int
    token_ids: np.ndarray
    token_mask: Optional[np.ndarray]

@dataclass
class LaunchTaskRequest:
    dp_client_id: int
    request_id: int
    task_ids: List[int]
    slot_mappings: List[np.ndarray]
```

### 5.4.2 响应类型

```python
@dataclass
class GetMatchResponse:
    request_id: int
    task_id: int
    return_mask: np.ndarray

@dataclass
class LaunchTaskResponse:
    request_id: int
```

### 5.4.3 通信流程

```
Client                    Server
  │                         │
  │─── GetMatchRequest ────>│
  │                         │─── get_match() ──> KVTaskEngine
  │                         │<─── task_id, mask ───
  │<─── GetMatchResponse ───│
  │                         │
  │─── LaunchTaskRequest ──>│
  │                         │─── launch_tasks() ──> KVTaskEngine
  │<─── LaunchTaskResponse ─│
```

## 5.5 本章小结

本章详细介绍了管理层的设计：

1. **KVManager**：
   - 统一管理接口
   - 支持单进程和多进程两种模式
   - 提供简洁的 API（get/put/launch/cancel/wait）

2. **KVTaskEngine**：
   - 任务生命周期管理
   - 任务创建、启动、更新、取消
   - 与缓存引擎和传输引擎协调

3. **Server/Client**：
   - 多进程部署的通信机制
   - ZMQ 消息传递
   - 请求/响应模式

管理层作为 FlexKV 的协调中心，连接适配层和引擎层，负责任务的统一管理和进程间通信。

---

**下一章预告**：第六章将详细介绍缓存引擎的设计，包括 GlobalCacheEngine 的匹配决策、RadixTree 的前缀匹配算法，以及 Mempool 的内存池管理。

