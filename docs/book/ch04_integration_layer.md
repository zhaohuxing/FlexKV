# 第四章：适配层设计

> 适配层是 FlexKV 与外部推理框架（如 vLLM）交互的桥梁。本章详细介绍适配层的设计思路、接口实现，以及如何与 vLLM 进行集成。

## 4.1 KV Connector 接口规范

### 4.1.1 vLLM KV Connector 接口定义

vLLM 定义了标准的 KV Connector 接口，允许外部系统管理 KV Cache。FlexKV 通过实现这个接口与 vLLM 集成。

**核心接口**（Scheduler 端）：

```python
class KVConnectorV1(ABC):
    # Scheduler 端接口
    @abstractmethod
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """获取可以从外部缓存加载的新 token 数量"""
        
    @abstractmethod
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        """在块分配后更新状态"""
        
    @abstractmethod
    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, Optional[dict]]:
        """请求完成时的回调"""
        
    @abstractmethod
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """构建连接器元数据"""
        
    @abstractmethod
    def update_connector_output(
        self, connector_output: KVConnectorOutput
    ):
        """更新连接器输出状态"""
```

**核心接口**（Worker 端）：

```python
class KVConnectorV1(ABC):
    # Worker 端接口
    @abstractmethod
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """注册 GPU KV Cache"""
        
    @abstractmethod
    def start_load_kv(self, forward_context: ForwardContext, **kwargs):
        """开始加载 KV Cache"""
        
    @abstractmethod
    def wait_for_layer_load(self, layer_name: str):
        """等待特定层的 KV Cache 加载完成"""
        
    @abstractmethod
    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, ...):
        """保存一层 KV Cache"""
        
    @abstractmethod
    def wait_for_save(self):
        """等待所有保存操作完成"""
```

### 4.1.2 FlexKV 的实现策略

FlexKV 的实现分为三个层次：

1. **FlexKVConnectorV1Impl**：实现 vLLM 接口，根据角色分发
2. **FlexKVSchedulerConnector**：Scheduler 端的实现
3. **FlexKVWorkerConnector**：Worker 端的实现

```
FlexKVConnectorV1Impl (实现 vLLM 接口)
    ├── FlexKVSchedulerConnector (Scheduler 端)
    │   └── KVManager (管理层)
    └── FlexKVWorkerConnector (Worker 端)
        └── KVTPClient (进程通信)
```

## 4.2 Scheduler Connector 实现

### 4.2.1 初始化流程

```python
class FlexKVSchedulerConnector:
    def __init__(self, flexkv_config: FlexKVConfig):
        # 1. 配置转换
        self.model_config = ModelConfig(
            num_layers=flexkv_config.num_layers,
            num_kv_heads=flexkv_config.num_kv_heads,
            head_size=flexkv_config.head_size,
            use_mla=flexkv_config.use_mla,
            dtype=flexkv_config.dtype,
            tp_size=flexkv_config.tp_size,
        )
        self.cache_config = CacheConfig(
            tokens_per_block=flexkv_config.block_size,
            **flexkv_config.cache_config,
        )
        
        # 2. 创建 KVManager（进入管理层）
        self.flexkv_manager = KVManager(
            model_config=self.model_config,
            cache_config=self.cache_config,
            gpu_register_port=flexkv_config.server_recv_port
        )
        self.flexkv_manager.start()
        
        # 3. 任务管理
        self.req_id_to_task_dict: dict[str, int] = {}
        self.get_tasks: dict[int, FlexKVGetTask] = {}
        self.put_tasks: dict[int, FlexKVPutTask] = {}
        self.tasks_to_launch: dict[int, FlexKVTask] = {}
        self.tasks_to_cancel: dict[int, FlexKVTask] = {}
        
        # 4. 等待初始化完成
        while not self.is_ready():
            time.sleep(5)
```

**关键点**：
- 将 vLLM 配置转换为 FlexKV 配置
- 创建 KVManager，这是连接管理层的入口
- 初始化任务管理数据结构

### 4.2.2 get_num_new_matched_tokens 实现

这个方法用于查询有多少新的 token 可以从 FlexKV 缓存中获取。

```python
def get_num_new_matched_tokens(
    self, request: "Request", num_computed_tokens: int
) -> tuple[int, bool]:
    """
    Args:
        request: vLLM 的请求对象
        num_computed_tokens: 已经在 GPU 上计算过的 token 数量
        
    Returns:
        tuple[int, bool]: (新匹配的 token 数量, 是否需要获取)
    """
    # 1. 调用 _get_match 进行匹配
    task_id, num_new_matched_tokens = self._get_match(
        request=request,
        num_computed_tokens=num_computed_tokens
    )
    
    # 2. 记录统计信息
    self.flexkv_stats.record_get(
        num_prompt_tokens=request.num_tokens,
        num_gpu_matched_tokens=num_computed_tokens,
        num_flexkv_matched_tokens=num_new_matched_tokens
    )
    
    # 3. 判断是否需要获取
    if not self._need_to_get(
        num_prompt_tokens=request.num_tokens,
        num_computed_tokens=num_computed_tokens,
        num_new_matched_tokens=num_new_matched_tokens
    ):
        return 0, False
    
    return num_new_matched_tokens, True
```

**内部实现 _get_match**：

```python
def _get_match(self, request: "Request", num_computed_tokens: int = 0) -> tuple[int, int]:
    # 1. 对齐到 Block 边界
    num_tokens_to_get = (request.num_tokens // self.block_size) * self.block_size
    token_ids = request.all_token_ids[:num_tokens_to_get]
    
    # 2. 构建 token_mask（标记哪些 token 需要获取）
    np_token_ids = np.array(token_ids)
    np_token_mask = np.ones_like(np_token_ids, dtype=bool)
    np_token_mask[:num_computed_tokens] = False  # 已计算的 token 不需要获取
    
    # 3. 调用 KVManager 匹配
    task_id, matched_mask = self.flexkv_manager.get_match(
        token_ids=np_token_ids,
        token_mask=np_token_mask
    )
    
    # 4. 计算新匹配的 token 数量
    num_new_matched_tokens = matched_mask.sum().item()
    
    # 5. 创建任务（但不立即启动）
    if num_new_matched_tokens > 0:
        self.req_id_to_task_dict[request.request_id] = task_id
        self.tasks_to_cancel[task_id] = FlexKVGetTask(...)
    
    return task_id, num_new_matched_tokens
```

**关键理解**：
- `num_computed_tokens`：已经在 GPU 上计算过的 token，不需要从 FlexKV 获取
- `num_new_matched_tokens`：在 FlexKV 缓存中找到的新 token
- 任务创建后，存储在 `tasks_to_cancel` 中，等待后续启动

### 4.2.3 update_state_after_alloc 实现

在 vLLM 分配 GPU 块后，需要更新 FlexKV 的任务状态。

```python
def update_state_after_alloc(
    self, request: "Request", blocks: "KVCacheBlocks", num_new_matched_tokens: int
) -> None:
    """在块分配后更新状态，准备启动任务"""
    if num_new_matched_tokens == 0:
        return
    
    # 1. 获取任务
    task_id = self.req_id_to_task_dict[request.request_id]
    task: FlexKVGetTask = self.tasks_to_cancel.pop(task_id)
    self.tasks_to_launch[task_id] = task
    
    # 2. 计算 slot_mapping
    num_computed_blocks = task.num_computed_tokens // self.block_size
    num_blocks_to_get = num_new_matched_tokens // self.block_size
    all_block_ids = blocks.get_block_ids()[0]
    block_ids_to_get = all_block_ids[
        num_computed_blocks:num_computed_blocks+num_blocks_to_get
    ]
    
    # 3. 构建 slot_mapping（每个 token 对应的 slot）
    task.slot_mapping = np.array(block_ids_to_get).repeat(self.block_size) * self.block_size
```

**关键点**：
- 任务从 `tasks_to_cancel` 移到 `tasks_to_launch`
- 计算需要传输的 GPU Block IDs
- 构建 slot_mapping，用于数据传输时的地址映射

### 4.2.4 launch_tasks 和 cancel_tasks

这两个方法在 `build_connector_meta` 中被调用：

```python
def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> "KVConnectorMetadata":
    """构建连接器元数据，在每个调度步骤调用"""
    # 1. 取消不需要的任务
    self.cancel_tasks()
    
    # 2. 启动准备好的任务
    self.launch_tasks()
    
    return KVConnectorMetadata()

def cancel_tasks(self) -> None:
    """取消任务"""
    for task in self.tasks_to_cancel.values():
        del self.req_id_to_task_dict[task.request.request_id]
        logger.info(f"FlexKV Cancel task: {task}")
    
    self.flexkv_manager.cancel(task_ids=list(self.tasks_to_cancel.keys()))
    self.tasks_to_cancel.clear()

def launch_tasks(self) -> None:
    """启动任务"""
    task_ids: list[int] = []
    slot_mappings: list[np.ndarray] = []
    
    for task_id, task in self.tasks_to_launch.items():
        logger.info(f"FlexKV Launch task: {task}")
        task_ids.append(task_id)
        slot_mappings.append(task.slot_mapping)
        
        if isinstance(task, FlexKVGetTask):
            self.get_tasks[task_id] = task
        else:
            self.put_tasks[task_id] = task
    
    # 调用 KVManager 启动任务
    self.flexkv_manager.launch(task_ids=task_ids, slot_mappings=slot_mappings)
    self.tasks_to_launch.clear()
```

### 4.2.5 request_finished 实现

当请求完成时，需要将 GPU 上的 KV Cache 保存到 FlexKV。

```python
def request_finished(
    self, request: "Request", block_ids: list[int]
) -> bool:
    """请求完成时的回调"""
    # 1. 检查是否有未完成的任务
    if request.request_id in self.req_id_to_task_dict:
        return True  # 还有未完成的任务，不能释放 Block
    
    # 2. 检查请求是否正常完成
    if not (request.is_finished() and request.get_finished_reason() < 2):
        return False  # 异常完成，不保存
    
    # 3. 匹配 PUT 操作
    task_id, num_matched_tokens, num_unmatched_tokens = self._put_match(request=request)
    
    # 4. 判断是否需要 PUT
    if not self._need_to_put(...):
        return False
    
    # 5. 准备启动 PUT 任务
    task: FlexKVPutTask = self.tasks_to_cancel.pop(task_id)
    self.tasks_to_launch[task_id] = task
    
    # 6. 计算 slot_mapping（只 PUT 未匹配的部分）
    num_matched_blocks = num_matched_tokens // self.block_size
    num_unmatched_blocks = num_unmatched_tokens // self.block_size
    block_ids_to_put = block_ids[num_matched_blocks:num_matched_blocks+num_unmatched_blocks]
    task.slot_mapping = np.array(block_ids_to_put).repeat(self.block_size) * self.block_size
    
    return True  # 返回 True，表示 Block 不能立即释放
```

### 4.2.6 query_finished_task 实现

查询已完成的任务，更新状态。

```python
def query_finished_task(self) -> tuple[set[str], set[str]]:
    """查询已完成的任务"""
    task_ids = list(self.get_tasks.keys()) + list(self.put_tasks.keys())
    responses_from_manager = self.flexkv_manager.try_wait(task_ids)
    
    finished_sending = set()  # 已完成的 PUT 任务
    finished_recving = set()   # 已完成的 GET 任务
    
    for task_id, response in responses_from_manager.items():
        success = (response.status == KVResponseStatus.SUCCESS)
        
        if task_id in self.get_tasks:
            task = self.get_tasks.pop(task_id)
            finished_recving.add(task.request.request_id)
        else:
            task = self.put_tasks.pop(task_id)
            finished_sending.add(task.request.request_id)
        
        del self.req_id_to_task_dict[task.request.request_id]
    
    return finished_sending, finished_recving
```

## 4.3 Worker Connector 实现

### 4.3.1 初始化流程

```python
class FlexKVWorkerConnector:
    def __init__(self, flexkv_config: FlexKVConfig):
        current_device_id = torch.cuda.current_device()
        # 创建 TP Client，用于与 Server 通信
        self.tp_client = KVTPClient(
            flexkv_config.server_recv_port, 
            0, 
            current_device_id
        )
```

**关键点**：
- Worker Connector 不直接创建 KVManager
- 通过 `KVTPClient` 与 Scheduler 端的 Server 通信

### 4.3.2 register_kv_caches 实现

将 GPU 的 KV Cache 注册到 FlexKV Server。

```python
def register_to_server(self, kv_caches: dict[str, torch.Tensor]):
    """注册 GPU KV Cache"""
    gpu_blocks = list(kv_caches.values())
    num_layer = len(kv_caches)
    
    # 1. 解析 GPU KV Cache 的布局信息
    if self.flexkv_config.use_mla:
        # MLA 格式：3D tensor [num_blocks, block_size, head_dim]
        num_blocks = gpu_blocks[0].shape[0]
        block_size = gpu_blocks[0].shape[1]
        num_kv_heads = 1
        head_size = gpu_blocks[0].shape[2]
    else:
        # 标准格式：5D tensor [kv_dim, num_blocks, block_size, num_heads, head_size]
        num_blocks = gpu_blocks[0].shape[1]
        block_size = gpu_blocks[0].shape[2]
        num_kv_heads = gpu_blocks[0].shape[3]
        head_size = gpu_blocks[0].shape[4]
    
    # 2. 创建 GPU 布局
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=num_layer,
        num_block=num_blocks,
        tokens_per_block=block_size,
        num_head=num_kv_heads,
        head_size=head_size,
        is_mla=self.flexkv_config.use_mla,
    )
    
    # 3. 通过 TP Client 注册
    self.tp_client.register_to_server(gpu_blocks, gpu_layout)
```

**关键理解**：
- GPU KV Cache 的布局是 `LAYERWISE`：`[layer, block, token, head, head_dim]`
- 需要解析布局信息，构建 `KVCacheLayout`
- 通过 `KVTPClient` 将 GPU 块信息发送到 Server

## 4.4 配置转换与适配

### 4.4.1 FlexKVConfig

FlexKVConfig 负责从环境变量和 vLLM 配置中加载配置：

```python
class FlexKVConfig:
    @classmethod
    def from_env(cls) -> 'FlexKVConfig':
        """从环境变量加载配置"""
        config_path = os.getenv("FLEXKV_CONFIG_PATH")
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        return cls()
    
    def post_init_from_vllm_config(self, vllm_config: "VllmConfig"):
        """从 vLLM 配置补充信息"""
        # 从 vLLM 配置中提取模型信息
        self.num_layers = vllm_config.model_config.hf_config.num_hidden_layers
        self.num_kv_heads = vllm_config.model_config.hf_config.num_key_value_heads
        self.head_size = vllm_config.model_config.hf_config.head_dim
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        # ...
```

### 4.4.2 配置转换流程

```
vLLM 配置 (VllmConfig)
    ↓
FlexKVConfig.from_env() + post_init_from_vllm_config()
    ↓
FlexKVConfig
    ↓
ModelConfig + CacheConfig
    ↓
KVManager 初始化
```

**关键点**：
- 环境变量配置：通过 `FLEXKV_CONFIG_PATH` 指定 JSON 配置文件
- vLLM 配置补充：从 vLLM 的模型配置中提取参数
- 配置验证：确保配置的正确性和一致性

## 4.5 与 vLLM 的集成流程

### 4.5.1 集成步骤

1. **应用 Patch**：修改 vLLM 代码，注册 FlexKV Connector
2. **配置环境变量**：设置 `FLEXKV_CONFIG_PATH`
3. **启动 vLLM**：vLLM 会自动创建 FlexKVConnectorV1Impl
4. **自动使用**：vLLM 通过 KV Connector 接口调用 FlexKV

### 4.5.2 Patch 内容

vLLM 的 Patch 主要修改：

1. **KVConnectorFactory**：注册 `FlexKVConnectorV1`
2. **Scheduler**：在调度循环中调用 Connector 接口
3. **Worker**：在 Forward 过程中调用 Connector 接口

### 4.5.3 数据流向

```
vLLM Scheduler
    ↓ get_num_new_matched_tokens()
FlexKVSchedulerConnector
    ↓ get_match()
KVManager
    ↓
GlobalCacheEngine (匹配缓存)
    ↓
返回匹配结果
    ↓
vLLM 分配 Block
    ↓ update_state_after_alloc()
FlexKVSchedulerConnector
    ↓ launch_tasks()
KVManager
    ↓
TransferEngine (传输数据)
```

## 4.6 本章小结

本章详细介绍了适配层的设计：

1. **接口适配**：实现 vLLM 的 KV Connector 接口
2. **Scheduler Connector**：
   - 匹配决策（get_num_new_matched_tokens）
   - 任务管理（launch_tasks, cancel_tasks）
   - 状态更新（update_state_after_alloc）
   - 请求完成（request_finished）
3. **Worker Connector**：
   - GPU 块注册（register_kv_caches）
   - 与 Server 通信（KVTPClient）
4. **配置转换**：从 vLLM 配置转换为 FlexKV 配置

适配层作为 FlexKV 与外部框架的桥梁，负责接口转换和任务协调，是整个系统的重要入口。

---

**下一章预告**：第五章将详细介绍管理层的设计，包括 KVManager 的统一接口、KVTaskEngine 的任务管理，以及 Server/Client 的进程通信机制。

