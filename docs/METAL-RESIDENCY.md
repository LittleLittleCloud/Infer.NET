# Metal 空闲驻留 A/B

2026-10-22；[PR #15](https://github.com/LittleLittleCloud/Infer.NET/pull/15)，关联 #9 / #3。固定 8bit 模型与 FP32 KV，两端 Native AOT 源码均为 `b69f90e601e2e0f434bd153298a4475b3f62774d`。M5 Pro 48 GiB + RTX 4090，现有家庭局域网；Mac 插电，High Power 未重新核验。

## 结果

同一二进制，仅改变 `metal-residency=on/off`，每个配置预热一次，空闲 0/2/10/60 秒各正式三次。24 输入、自然 EOS 13 输出，全部 token 与先前诊断一致。

|空闲|开关|首步 decode 中位数（ms）|prefill 中位数（ms）|客户端整次中位数（s）|整机最大新增 swap-out|
|---:|---|---:|---:|---:|---:|
|0s|off|21.019|734.673|0.926|0 B|
|0s|on|21.594|800.701|0.991|0 B|
|2s|off|198.924|1385.449|1.753|0 B|
|2s|on|27.486|1181.659|1.376|0 B|
|10s|off|200.196|1358.970|1.732|0 B|
|10s|on|24.358|1173.629|1.372|0 B|
|60s|off|203.298|1432.194|1.803|0 B|
|60s|on|26.296|1513.007|1.710|0 B|

启用后消除了本轮稳定复现的约 0.2 秒首步等待；不能把此前内存压力下 16–84 秒停顿与本轮低压力结果直接相除，宣称数千倍提速。ON→OFF 顺序未随机化，全部测量保留。两组正式请求均无新增 swap-out。

短请求仍有固定双机状态交接成本。每次 24 token 请求回传 33,423,360 字节前段 KV/Delta/卷积状态；此次阶段日志包含等待与传输，不能当成纯 CUDA 时间。长上下文 37 tokens/s、128K prefill 1600/2000 tokens/s 与整次 200/180 秒目标均未验收。

## 实现与失败记录

- 对原有权重 buffer 创建有预算的 Metal residency set，无权重副本、无 dummy GPU 计算。已登记 36,825,666,560 字节；预算 39,664,025,600 字节，按推荐工作集/物理内存上限减 512 MiB 计算。这里只管理权重，不保证所有长上下文工作区都能驻留。
- Graph 复制矩阵描述符时共享 lease；最后一个描述符释放才取消登记。每 1000ms 续请求驻留，退出时停止线程并解除。
- 0bcbbb6 初次部署无效：托管 `Environment.SetEnvironmentVariable` 在 Unix 不更新原生 `getenv`。b69f90e 改为显式 C ABI 参数，跨 C#/Metal 测试验证实际登记与输出一致。
- 另一次未启用诊断在两端编译期间出现约 80 秒首步等待及大量整机换页，记录 `idle-latency-20261023T064533Z`；存在后台编译干扰，不纳入 A/B。
- 历史文档对部署 `Profile` 的 matrix=0 / sequential Delta 推断不准确。赋值只改托管副本；没有外部原生覆盖时，实际 native 默认 matrix=true、chunked Delta、FP32 GEMM/KV、2048 scratch；C# 与网络调用仍为 32 token。独立旧 split 脚本通过启动环境设置的参数不受该问题影响。

## 验证与复现

Mac 107 项单测（含真实 Metal 驻留开关、相同 GEMV 输出、最后释放清理）；Windows 107 项测试返回通过，其中 Mac 专用测试在 Windows 直接返回；CUDA GEMV/Delta/attention 检查通过；两端 Native AOT 构建及不可变包生成成功。原生生命周期/预算检查此前七项通过。

运行 `scripts/benchmark-idle-latency.py --help` 查看参数。先在部署配置切换驻留并重启同一版本，再在 Ready 后顺序测量，保持每台设备只有一个模型实例。探针不会重启服务，也不会输出 token 文件内容。

公开逐次结果及原始文件散列见 [residency-summary.json](../site/residency-summary.json)。原始事件、阶段日志、供电及版本记录仅本机 `artifacts/dual-host/residency-{on,off}-*`，散列不是公共下载链接。

参考：[Microsoft 托管环境变量文档](https://learn.microsoft.com/dotnet/api/system.environment.setenvironmentvariable)、[Apple residency sets](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets)。
