# 附录 D：常见问题 FAQ

## D.1 安装与编译

### Q1: 编译失败，提示缺少依赖

**A**: 安装系统依赖：
```bash
apt install liburing-dev      # io_uring 支持
apt install libxxhash-dev     # Hash 计算
```

### Q2: CUDA 相关错误

**A**: 确保：
1. CUDA 已正确安装
2. PyTorch 支持 CUDA
3. CUDA 版本兼容

## D.2 配置问题

### Q3: tokens_per_block 必须是 2 的幂吗？

**A**: 是的，必须是 2 的幂（如 16, 32, 64），这是设计约束。

### Q4: 如何计算需要的 Block 数量？

**A**: 
```
Block 大小 = tokens_per_block × token_size_in_bytes
所需 Block 数 = 缓存容量 / Block 大小
```

### Q5: CPU/SSD/Remote 布局必须一致吗？

**A**: CPU、SSD、Remote 的布局必须一致（都是 BLOCKWISE），但 GPU 可以是 LAYERWISE。

## D.3 性能问题

### Q6: 缓存命中率低

**A**: 可能原因：
1. 缓存容量不足 → 增加 `num_cpu_blocks`
2. 淘汰比例过高 → 降低 `evict_ratio`
3. Block 大小不合适 → 调整 `tokens_per_block`

### Q7: 传输速度慢

**A**: 优化建议：
1. 启用 Copy Engine：`use_ce_transfer_h2d = true`
2. 增加 SM 数量：`transfer_sms_h2d = 16`
3. 使用 Pin Memory（自动启用）

### Q8: 内存占用过高

**A**: 优化建议：
1. 减少 `num_cpu_blocks`
2. 启用 SSD 缓存，减少 CPU 缓存
3. 启用 Remote 缓存

## D.4 集成问题

### Q9: vLLM 无法找到 FlexKV Connector

**A**: 检查：
1. Patch 是否已应用
2. 环境变量 `FLEXKV_CONFIG_PATH` 是否设置
3. vLLM 启动参数 `--kv-transfer-config` 是否正确

### Q10: GPU 块注册失败

**A**: 检查：
1. GPU KV Cache 的形状是否正确
2. 布局信息（LAYERWISE）是否正确
3. 端口是否被占用

## D.5 故障排查

### Q11: 如何查看日志？

**A**: FlexKV 使用 `flexkv_logger`，日志级别可通过环境变量设置。

### Q12: 如何追踪性能？

**A**: 
1. 启用追踪：`enable_trace = true`
2. 查看追踪文件：`trace_file_path`
3. 使用统计信息：`FlexKVStats`

---

**更多问题请参考 GitHub Issues 或提交新的 Issue。**

