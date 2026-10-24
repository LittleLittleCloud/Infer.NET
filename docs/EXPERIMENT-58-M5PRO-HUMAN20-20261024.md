# 实验 #58：M5 Pro 全本地 HumanEval 20

关联 Issue #58、代码 PR #59。实际测试候选 `87c2b7dcaf4a418dcb18707e1913562717b43cdc`，基于最新 main `eaaf81f630614b4d6416bc62776c13d36331fa57`，增加已获批准的全本地路由；Windows 不参与。

20/20 Pi 任务完成，20/20 官方断言通过，无任务超时或错误。107 个模型请求全部 metal-local，输入 243663 / 输出 19104 token；worker prefill 268.2048 token/s、decode 66.4336 token/s；Pi 总耗时 1202.992 秒。原 Mac 服务已恢复并 Ready。

完整配置、客户端 TTFT、资源采样、首页趋势数据与原始证据散列正在回填本结果 PR；保留首次断言运行因相对路径导致的 Docker 挂载错误及修正后的完整断言结果。等待人工 review，不自动合并。
