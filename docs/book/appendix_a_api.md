# 附录 A：API 参考

## A.1 KVManager API

### A.1.1 初始化

```python
KVManager(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    gpu_register_port: Optional[str] = None,
    server_recv_port: Optional[str] = None,
    dp_client_id: int = 0
)
```

### A.1.2 核心方法

#### get_match

```python
def get_match(
    self,
    token_ids: Union[torch.Tensor, np.ndarray],
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    layer_granularity: int = -1,
    dp_id: int = 0,
) -> Tuple[int, np.ndarray]:
    """
    匹配 KV Cache，返回匹配的 token mask
    
    Returns:
        task_id: 任务 ID
        matched_mask: 匹配的 token mask
    """
```

#### put_match

```python
def put_match(
    self,
    token_ids: Union[torch.Tensor, np.ndarray],
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    dp_id: int = 0,
) -> Tuple[int, np.ndarray]:
    """
    匹配 KV Cache，返回未匹配的 token mask
    
    Returns:
        task_id: 任务 ID
        unmatched_mask: 未匹配的 token mask
    """
```

#### launch

```python
def launch(
    self,
    task_ids: Union[int, List[int]],
    slot_mappings: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]]
) -> None:
    """启动传输任务"""
```

#### wait

```python
def wait(
    self,
    task_ids: Union[int, List[int]],
    timeout: float = 20.0,
    completely: bool = False
) -> Dict[int, KVResponse]:
    """阻塞等待任务完成"""
```

## A.2 GlobalCacheEngine API

### A.2.1 get

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
    """
    GET 操作：构建传输图
    
    Returns:
        transfer_graph: 传输操作图
        return_mask: 返回的 token mask
        callback: 传输完成后的回调
        task_end_op_id: 任务结束操作 ID
    """
```

### A.2.2 put

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
    """
    PUT 操作：构建传输图
    
    Returns:
        transfer_graph: 传输操作图
        return_mask: 返回的 token mask
        callback: 传输完成后的回调
        task_end_op_id: 任务结束操作 ID
        skipped_gpu_blocks: 跳过的 GPU Block 数量
    """
```

## A.3 TransferEngine API

### A.3.1 submit_transfer_graph

```python
def submit_transfer_graph(self, transfer_graph: TransferOpGraph) -> None:
    """提交传输图执行"""
```

### A.3.2 get_completed_graphs_and_ops

```python
def get_completed_graphs_and_ops(
    self,
    timeout: Optional[float] = None
) -> List[Tuple[int, int]]:
    """
    获取已完成的传输图和操作
    
    Returns:
        List of (graph_id, op_id) tuples
    """
```

---

**完整 API 文档请参考代码注释和官方文档。**

