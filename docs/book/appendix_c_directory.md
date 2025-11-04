# 附录 C：代码目录说明

## C.1 目录结构

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

## C.2 关键文件说明

### C.2.1 适配层

- `integration/vllm/vllm_v1_adapter.py`：KVConnector 实现，与 vLLM 集成

### C.2.2 管理层

- `kvmanager.py`：统一管理接口
- `kvtask.py`：任务管理引擎
- `server/server.py`：服务器实现
- `server/client.py`：客户端实现

### C.2.3 引擎层

- `cache/cache_engine.py`：缓存引擎核心
- `cache/radixtree.py`：RadixTree 索引
- `cache/mempool.py`：内存池
- `transfer/transfer_engine.py`：传输引擎核心
- `transfer/worker.py`：传输 Worker
- `storage/storage_engine.py`：存储引擎核心
- `storage/allocator.py`：存储分配器

### C.2.4 公共组件

- `common/block.py`：Block 抽象
- `common/config.py`：配置类
- `common/storage.py`：存储布局
- `common/transfer.py`：传输图

## C.3 C++ 扩展

```
csrc/
├── bindings.cpp            # Python 绑定
├── hash.cpp                # Hash 计算
├── radix_tree.cpp          # RadixTree C++ 实现
├── transfer.cu              # CUDA 传输内核
├── transfer_ssd.cpp         # SSD 传输
└── tp_transfer_thread_group.cpp  # TP 传输线程组
```

## C.4 代码阅读建议

1. **入门**：从 `kvmanager.py` 开始，理解统一接口
2. **缓存**：阅读 `cache/cache_engine.py`，理解匹配逻辑
3. **传输**：阅读 `transfer/transfer_engine.py`，理解传输调度
4. **存储**：阅读 `storage/storage_engine.py`，理解存储管理

