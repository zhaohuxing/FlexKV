# 第十六章：性能优化与调优

> 本章详细介绍 FlexKV 的性能优化方法，包括性能指标分析、调优实践，以及故障排查技巧。

## 16.1 性能指标分析

### 16.1.1 缓存命中率

**定义**：从缓存中获取的 Block 数量 / 总请求的 Block 数量

**测量方法**：

```python
# FlexKV 自动统计
flexkv_stats.record_get(
    num_prompt_tokens=request.num_tokens,
    num_gpu_matched_tokens=num_computed_tokens,
    num_flexkv_matched_tokens=num_new_matched_tokens
)

# 命中率计算
hit_rate = num_flexkv_matched_tokens / num_prompt_tokens
```

**优化目标**：
- **> 80%**：优秀
- **50-80%**：良好
- **< 50%**：需要优化

### 16.1.2 传输延迟

**测量方法**：

```python
# 传输开始时间
transfer_start = time.perf_counter()

# 传输操作
transfer_engine.submit_transfer_graph(transfer_graph)

# 等待完成
completed = transfer_engine.get_completed_graphs_and_ops(timeout=10.0)

# 传输结束时间
transfer_end = time.perf_counter()

# 延迟
latency = transfer_end - transfer_start
```

**优化目标**：
- **GPU↔CPU**：< 10ms（对于 1 Block）
- **CPU↔SSD**：< 50ms（对于 1 Block）
- **CPU↔Remote**：< 500ms（取决于网络）

### 16.1.3 吞吐量

**定义**：每秒处理的请求数或 token 数

**测量方法**：

```python
# 统计一段时间内的请求数
requests_per_second = total_requests / time_elapsed
tokens_per_second = total_tokens / time_elapsed
```

## 16.2 调优实践

### 16.2.1 缓存命中率优化

#### 方法 1：增加缓存容量

```json
{
    "cache_config": {
        "num_cpu_blocks": 100000,  // 增加 CPU 缓存
        "num_ssd_blocks": 1000000  // 增加 SSD 缓存
    }
}
```

**效果**：提高缓存容量，减少淘汰

#### 方法 2：优化 Block 大小

```json
{
    "cache_config": {
        "tokens_per_block": 32  // 从 16 增加到 32
    }
}
```

**效果**：
- 减少传输次数
- 可能降低内存利用率

#### 方法 3：降低淘汰比例

```json
{
    "cache_config": {
        "evict_ratio": 0.0  // 不淘汰（如果容量充足）
    }
}
```

**效果**：保留更多缓存，提高命中率

### 16.2.2 传输延迟优化

#### 方法 1：启用 Copy Engine

```json
{
    "cache_config": {
        "use_ce_transfer_h2d": true,
        "use_ce_transfer_d2h": true,
        "transfer_sms_h2d": 16,  // 增加 SM 数量
        "transfer_sms_d2h": 16
    }
}
```

**效果**：GPU↔CPU 传输速度提升 20-30%

#### 方法 2：优化 io_uring

```json
{
    "cache_config": {
        "ssd_cache_iouring_entries": 1024  // 增加队列大小
    }
}
```

**效果**：SSD 传输吞吐量提升

#### 方法 3：批量传输

通过调整 Block 大小，增加单次传输的数据量：

```json
{
    "cache_config": {
        "tokens_per_block": 64  // 增加 Block 大小
    }
}
```

**效果**：减少传输次数，提升总体吞吐量

### 16.2.3 内存使用优化

#### 方法 1：启用 SSD 缓存

```json
{
    "cache_config": {
        "enable_ssd": true,
        "num_cpu_blocks": 10000,   // 减少 CPU 缓存
        "num_ssd_blocks": 1000000  // 增加 SSD 缓存
    }
}
```

**效果**：CPU 内存占用减少，使用更便宜的 SSD

#### 方法 2：启用 Remote 缓存

```json
{
    "cache_config": {
        "enable_remote": true,
        "num_ssd_blocks": 100000,      // 减少 SSD 缓存
        "num_remote_blocks": 10000000  // 增加 Remote 缓存
    }
}
```

**效果**：进一步减少本地存储占用

## 16.3 性能瓶颈定位

### 16.3.1 使用追踪工具

启用追踪：

```json
{
    "cache_config": {
        "enable_trace": true,
        "trace_file_path": "./flexkv_trace.log"
    }
}
```

**追踪内容**：
- GET/PUT 操作时间
- 传输操作时间
- 匹配时间

### 16.3.2 分析日志

查看 FlexKV 日志：

```python
# 查看匹配时间
logger.debug(f"Get match cost {match_cost*1000:.2f} ms")

# 查看传输时间
logger.debug(f"Transfer cost {transfer_cost*1000:.2f} ms")
```

### 16.3.3 性能分析工具

使用 Nsight Systems 分析 GPU 传输：

```bash
nsys profile --trace=cuda,nvtx python your_script.py
```

## 16.4 故障排查

### 16.4.1 缓存命中率低

**可能原因**：
1. 缓存容量不足
2. 淘汰比例过高
3. Block 大小不合适
4. 请求模式不适合缓存

**排查步骤**：
1. 检查 `num_cpu_blocks` 是否足够
2. 检查 `evict_ratio` 设置
3. 分析请求的重复性
4. 查看缓存使用情况

### 16.4.2 传输速度慢

**可能原因**：
1. 未启用 Copy Engine
2. 未使用 Pin Memory
3. SSD I/O 瓶颈
4. 网络带宽限制

**排查步骤**：
1. 检查 `use_ce_transfer_*` 配置
2. 检查 CPU 内存是否使用 Pin Memory
3. 测试 SSD 读写速度
4. 检查网络带宽

### 16.4.3 内存占用过高

**可能原因**：
1. 缓存容量设置过大
2. 未启用 SSD/Remote 缓存
3. 内存泄漏

**排查步骤**：
1. 检查配置的 Block 数量
2. 计算实际内存占用
3. 启用 SSD/Remote 缓存
4. 检查内存使用趋势

## 16.5 调优案例

### 16.5.1 案例 1：提升缓存命中率

**场景**：缓存命中率只有 30%

**调优步骤**：
1. 增加 CPU 缓存：`num_cpu_blocks: 10000 → 100000`
2. 启用 SSD 缓存：`enable_ssd: true`
3. 降低淘汰比例：`evict_ratio: 0.1 → 0.05`

**结果**：命中率提升到 75%

### 16.5.2 案例 2：优化传输延迟

**场景**：GPU↔CPU 传输延迟过高

**调优步骤**：
1. 启用 Copy Engine：`use_ce_transfer_h2d: true`
2. 增加 SM 数量：`transfer_sms_h2d: 8 → 16`
3. 增加 Block 大小：`tokens_per_block: 16 → 32`

**结果**：传输延迟降低 40%

## 16.6 本章小结

本章介绍了性能优化与调优：

1. **性能指标**：
   - 缓存命中率
   - 传输延迟
   - 吞吐量

2. **调优方法**：
   - 缓存命中率优化
   - 传输延迟优化
   - 内存使用优化

3. **瓶颈定位**：
   - 追踪工具
   - 日志分析
   - 性能分析工具

4. **故障排查**：
   - 常见问题诊断
   - 系统化排查步骤

合理的性能调优可以显著提升 FlexKV 的性能，需要根据实际场景进行针对性优化。

---

**下一章预告**：第十七章将提供实战案例，包括单机部署、分布式部署，以及性能测试的具体示例。

