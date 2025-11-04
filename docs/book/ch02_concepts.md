# 第二章：核心概念

> 在深入理解 FlexKV 的架构和实现之前，我们需要掌握一些核心概念。这些概念是理解 FlexKV 设计思路的基础，包括 KV Cache 的本质、Block 抽象的设计原因、以及多级缓存的机制。

## 2.1 KV Cache 基础

### 2.1.1 什么是 KV Cache

在 Transformer 模型的注意力机制中，每个 token 在通过每一层时都会生成：

- **K (Key) 向量**：用于计算注意力权重
- **V (Value) 向量**：用于加权求和生成输出

这些 K 和 V 向量被缓存起来，就是 **KV Cache**。

### 2.1.2 Token 与 KV Cache 的关系

**关键理解**：**一个 Token 对应一个 KV Cache**

但这个 KV Cache 包含该 token 在**所有层**上的 K 和 V 向量。

```
Token 0 ("你")
    ↓
通过 Layer 0 → 生成 Layer 0 的 K 和 V
通过 Layer 1 → 生成 Layer 1 的 K 和 V
...
通过 Layer 31 → 生成 Layer 31 的 K 和 V
    ↓
完整的 KV Cache = 32 层的 K + 32 层的 V
```

**在代码中的体现**：

```python
@property
def token_size_in_bytes(self) -> int:
    kv_dim = 1 if self.use_mla else 2  # K 和 V = 2
    return self.num_layers * self.num_kv_heads * self.head_size * \
           kv_dim * self.dtype.itemsize
```

**计算示例**（Qwen-32B）：
- `num_layers = 32`
- `num_kv_heads = 32`
- `head_size = 128`
- `kv_dim = 2` (K 和 V)
- `dtype = bfloat16` (2 字节)

```
一个 token 的 KV Cache 大小 = 
  32 × 32 × 128 × 2 × 2 = 524,288 字节 ≈ 512 KB
```

### 2.1.3 KV Cache 的动态性

KV Cache 在推理过程中是**动态增长**的：

**预填充阶段（Prefill）**：
```
输入: ["你好", "世界"]
  ↓
为每个 token 生成 KV Cache
  ↓
Token 0: 512 KB 的 KV Cache
Token 1: 512 KB 的 KV Cache
  ↓
总 KV Cache = 2 × 512 KB = 1 MB
```

**解码生成阶段（Decode）**：
```
生成 Token 2:
  ↓
只计算 Token 2 的 KV Cache（512 KB）
  ↓
追加到已有 KV Cache
  ↓
总 KV Cache = 3 × 512 KB = 1.5 MB
```

**关键点**：
- 每个 token 的 KV Cache **结构**是固定的（由模型参数决定）
- KV Cache 的**总数量**是动态增长的（随序列长度增加）

### 2.1.4 KV Cache 在 GPU 中的存储

在 GPU 显存中，KV Cache 通常以**分页**（Paged）方式存储：

```
GPU 显存中的 KV Cache 布局：
┌──────────────────────────────────────┐
│ Block 0: Token 0-15 的 KV Cache       │
│ Block 1: Token 16-31 的 KV Cache      │
│ Block 2: Token 32-47 的 KV Cache      │
│ ...                                   │
└──────────────────────────────────────┘
```

每个 Block 包含多个 token 的 KV Cache，这样便于：
- 内存管理（固定大小的 Block）
- I/O 操作（以 Block 为单位传输）
- 缓存匹配（以 Block 为单位匹配）

## 2.2 Block 抽象

### 2.2.1 为什么需要 Block

在 FlexKV 中，**Block 是存储和管理的最小单位**，而不是单个 Token。这是为什么？

#### 原因 1：I/O 效率

如果以 Token 为单位传输（512 KB），会有以下问题：
- 传输次数过多（每次传输都需要系统调用）
- 传输开销大（小批量传输效率低）
- 难以并行化

以 Block 为单位传输（默认 16 tokens = 8 MB）：
- 减少传输次数
- 提高 I/O 吞吐量
- 便于并行传输

#### 原因 2：缓存匹配效率

前缀匹配时需要：
1. 计算 Hash
2. 查找索引
3. 比较匹配

如果以 Token 为单位：
- Hash 计算次数过多
- 索引查找开销大

以 Block 为单位：
- 一个 Block 计算一个 Hash
- 减少索引查找次数

#### 原因 3：内存管理

固定大小的 Block 便于：
- 预分配内存（避免动态分配）
- 内存池管理（位图跟踪）
- LRU 淘汰（以 Block 为单位）

### 2.2.2 Block 的设计

#### tokens_per_block 配置

```python
@dataclass
class CacheConfig:
    tokens_per_block: int = 16  # 默认 16 个 token 一个 Block
```

**设计约束**：
- 必须是 **2 的幂**（便于对齐和地址计算）

```python
if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
    raise InvalidConfigError(
        f"tokens_per_block must be a power of 2"
    )
```

**常见配置**：
- 16 tokens/block：适合大多数场景
- 32 tokens/block：适合长序列
- 64 tokens/block：适合超长序列

#### Block 大小计算

```
一个 Block 的 KV Cache 大小 = 
  tokens_per_block × token_size_in_bytes

例如（16 tokens, Qwen-32B）：
  16 × 512 KB = 8 MB
```

#### SequenceMeta：序列的 Block 表示

```python
@dataclass
class SequenceMeta:
    token_ids: np.ndarray          # Token ID 数组
    tokens_per_block: int          # 每个 Block 的 token 数量
    
    @property
    def num_blocks(self) -> int:
        return len(self.token_ids) // self.tokens_per_block
```

**示例**：
```python
token_ids = [1, 2, 3, ..., 17]  # 17 个 token
tokens_per_block = 16

sequence_meta = SequenceMeta(token_ids, tokens_per_block)
sequence_meta.num_blocks  # = 1（整数除法，丢弃最后一个不完整的 Block）
```

### 2.2.3 不完整 Block 的处理

当序列长度不是 `tokens_per_block` 的倍数时，会出现**不完整的 Block**。

**示例**：17 个 token，`tokens_per_block = 16`

```
完整 Block: Token 0-15 (16 tokens)
不完整 Block: Token 16 (1 token)  ← 被丢弃
```

**代码实现**：

```python
# 对齐到 Block 边界
aligned_length = (token_ids.shape[0] // self.tokens_per_block) * \
                 self.tokens_per_block
aligned_token_ids = token_ids[:aligned_length]  # 丢弃不完整的 Block
```

**设计原因**：
1. **简化实现**：不需要处理部分 Block
2. **保证对齐**：所有 Block 大小一致
3. **性能考虑**：不完整 Block 的传输开销相对较大

**浪费分析**：
- 17 个 token：浪费 1/17 ≈ 5.88%
- 33 个 token：浪费 1/33 ≈ 3.03%
- 浪费比例随序列长度增加而降低

### 2.2.4 Block Hash 计算

为了支持前缀匹配，FlexKV 为每个 Block 计算 Hash：

```python
def gen_hashes(token_ids: np.ndarray, tokens_per_block: int) -> np.ndarray:
    """生成所有 Block 的 Hash"""
    num_blocks = len(token_ids) // tokens_per_block
    block_hashes = np.zeros(num_blocks, dtype=np.uint64)
    
    for i in range(num_blocks):
        block_token_ids = token_ids[i * tokens_per_block:(i + 1) * tokens_per_block]
        block_hashes[i] = hash_array(block_token_ids)
    
    return block_hashes
```

**Hash 算法**：使用 XXHash（快速非加密 Hash）

**Hash 的作用**：
1. **快速比较**：Hash 值相同 → Block 内容相同
2. **索引查找**：通过 Hash 快速定位 Block
3. **前缀匹配**：匹配时比较 Hash 值

## 2.3 多级缓存

### 2.3.1 三级缓存架构

FlexKV 采用三级缓存架构：

```
┌─────────────────────────────────┐
│  Level 1: GPU 显存              │  ← 最快、最贵、容量最小
│  (推理引擎直接使用)              │
└─────────────┬───────────────────┘
              │ 溢出
              ↓
┌─────────────────────────────────┐
│  Level 2: CPU 内存               │  ← 第一级外部缓存
│  (第一级外部缓存)                │
└─────────────┬───────────────────┘
              │ 溢出
              ↓
┌─────────────────────────────────┐
│  Level 3: SSD 磁盘               │  ← 第二级持久化缓存
│  (第二级持久化缓存)              │
└─────────────┬───────────────────┘
              │ 溢出
              ↓
┌─────────────────────────────────┐
│  Level 4: Remote 存储            │  ← 第三级分布式缓存
│  (第三级分布式缓存)             │
└─────────────────────────────────┘
```

### 2.3.2 各级缓存的特点

#### Level 1: GPU 显存

- **特点**：推理引擎直接使用，无需传输
- **容量**：通常 40GB-80GB（A100/H100）
- **速度**：最快（无需传输）
- **成本**：最昂贵

#### Level 2: CPU 内存

- **特点**：第一级外部缓存，需要通过 PCIe 传输
- **容量**：通常 256GB-2TB
- **速度**：较快（PCIe 4.0 ≈ 64 GB/s）
- **成本**：中等

**传输方式**：
- GPU → CPU：CUDA `memcpyAsync`
- CPU → GPU：CUDA `memcpyAsync`

#### Level 3: SSD 磁盘

- **特点**：第二级持久化缓存，数据持久保存
- **容量**：通常 2TB-10TB
- **速度**：较慢（NVMe SSD ≈ 3-7 GB/s）
- **成本**：便宜

**传输方式**：
- CPU → SSD：使用 `io_uring` 异步 I/O
- SSD → CPU：使用 `io_uring` 异步 I/O

#### Level 4: Remote 存储

- **特点**：第三级分布式缓存，支持跨节点共享
- **容量**：几乎无限
- **速度**：最慢（网络传输）
- **成本**：最便宜

**传输方式**：
- CPU → Remote：网络传输
- Remote → CPU：网络传输

### 2.3.3 缓存层次与性能权衡

| 存储层级 | 容量 | 延迟 | 带宽 | 成本 | 持久化 |
|---------|------|------|------|------|--------|
| GPU | 小 | 最低 | 最高 | 最高 | ❌ |
| CPU | 中 | 低 | 高 | 中 | ❌ |
| SSD | 大 | 中 | 中 | 低 | ✅ |
| Remote | 极大 | 高 | 低 | 最低 | ✅ |

**设计原则**：
1. **热点数据**在 GPU（最快访问）
2. **温数据**在 CPU（快速恢复）
3. **冷数据**在 SSD（持久保存）
4. **归档数据**在 Remote（跨节点共享）

### 2.3.4 LRU 淘汰机制

当缓存空间不足时，FlexKV 使用 **LRU (Least Recently Used)** 策略淘汰 Block。

#### 逻辑淘汰 vs 物理淘汰

**逻辑淘汰**：
- 只更新索引，标记 Block 为"可复用"
- **不触发物理数据传输**
- 快速、低开销

**物理淘汰**：
- 实际删除数据
- 可能需要数据传输
- 慢、高开销

FlexKV 主要使用**逻辑淘汰**：

```python
# 当空间不足时，Mempool 会触发淘汰
if free_blocks < required_blocks:
    # 逻辑淘汰：标记 Block 为可复用
    evicted_blocks = self.index.evict(num_blocks_to_evict)
    # 不删除数据，只是更新索引
```

#### LRU 实现

```python
class RadixNode:
    last_access_time: float  # 最后访问时间
    
    def update_access_time(self):
        self.last_access_time = time.time()
```

淘汰时选择 `last_access_time` 最早的 Block。

### 2.3.5 多级缓存的匹配策略

当 GET 请求到达时，FlexKV 会在**所有级别**的缓存中匹配：

```
请求: Token 0-15 (Block 0)

匹配流程：
  1. 检查 GPU：❌ 未命中
  2. 检查 CPU：✅ 命中！→ 从 CPU 传输到 GPU
  3. 检查 SSD：✅ 也命中（但 CPU 已经命中，优先用 CPU）
  4. 检查 Remote：✅ 也命中（但 CPU 已经命中，优先用 CPU）

决策：从 CPU 传输到 GPU
```

**匹配原则**：
1. **优先使用最接近 GPU 的缓存**
2. **并行匹配所有级别**（不串行）
3. **选择最优传输路径**

## 2.4 本章小结

本章介绍了 FlexKV 的核心概念：

1. **KV Cache 基础**：
   - 一个 Token 对应一个 KV Cache（包含所有层的 K 和 V）
   - KV Cache 大小由模型参数决定
   - KV Cache 数量随序列长度动态增长

2. **Block 抽象**：
   - Block 是存储和管理的最小单位
   - 设计原因：I/O 效率、匹配效率、内存管理
   - `tokens_per_block` 必须是 2 的幂
   - 不完整的 Block 会被丢弃

3. **多级缓存**：
   - GPU → CPU → SSD → Remote 四级缓存
   - 各级缓存有不同的特点和使用场景
   - LRU 淘汰机制（逻辑淘汰为主）
   - 多级并行匹配，选择最优路径

理解这些核心概念，是深入理解 FlexKV 架构和实现的基础。

---

**下一章预告**：第三章将从宏观角度介绍 FlexKV 的整体架构，包括三层架构设计、各层职责、以及代码组织方式。

