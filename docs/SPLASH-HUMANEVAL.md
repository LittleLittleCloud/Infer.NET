# Splash / Pi HumanEval 参考测试

本工具只观测第三方引擎，不修改 Infer.NET 产品后端。实验计划和授权见 [Issue #61](https://github.com/LittleLittleCloud/Infer.NET/issues/61)。实验开始前固定最新 main、Splash 完整源码版本、release 二进制散列、target/draft revision 及文件散列。

复用 `benchmark-pi-humaneval.py` 的固定 cohort、seed 0、Pi 1.4.0、32768 窗口、4096 输出上限、300 秒逐题超时。模型通过 `--served-model-name qwen3.6-35b-a3b --announce-served-name` 保持现有 harness 的模型身份核查。只绑定 loopback，observer 使用独立端口 5288，Splash 使用 5289。

```sh
splash serve --model mlx-community/Qwen3.6-35B-A3B-8bit \
  --revision e06a74e6236a60c8367e1a3214e83d8b61b637b0 \
  --port 5289 --language-only --max-context 32768 --max-cache-disk 0 \
  --idle-release off --expert-memory off --kv-format int8 --state-format fp32 \
  --default-reasoning-effort none --served-model-name qwen3.6-35b-a3b \
  --announce-served-name --no-webui
python3 scripts/observe-splash-humaneval.py --output NEW_RUN/observer
python3 scripts/benchmark-pi-humaneval.py --output NEW_RUN/cohort-20 \
  --token-file LOCAL_NONEMPTY_TOKEN_FILE --endpoint http://127.0.0.1:5288 \
  --cohort cohort --seed 0 --timeout 300 --context-window 32768 --max-tokens 4096 \
  --endpoint-build SPLASH_FULL_COMMIT \
  --model-revision e06a74e6236a60c8367e1a3214e83d8b61b637b0
python3 scripts/eval-pi-humaneval-docker.py --results ABSOLUTE_NEW_RUN/cohort-20 \
  --dataset PINNED_HUMANEVAL_GZ --output NEW_RUN/functional.json
```

先核对旧 worker 空闲并保留恢复方式，再通过原托管管理器暂停，避免两个模型同时加载。若完整常驻加载失败，记录失败日志；未经本轮授权不改量化精度或启用专家流式。结束后先退出本轮 observer/Splash，再恢复原 worker，检查原版本和 Ready。

observer 向每次聊天请求加入 Splash 文档定义的独立 `cache_salt`，隔离所有请求的 KV/状态缓存，且不编辑提示词；同时固定 temperature=0、reasoning_effort=none 和 stream usage。DFlash2 保持 Splash 原生默认设置。检查每次返回 cache matched tokens 为 0；不能只根据 SSD cache 关闭就声称无前缀缓存。单独预热使用独立 observer 输出目录，不混入正式题组。

## 指标边界

`requests.jsonl` 为每个真实请求保留最终 usage、metrics、timings 及前后 `/status` 快照，且记录首个有效文本/推理/工具输出到达的客户端 TTFT。只计数的角色事件、usage、心跳不算首 token。TTFT 起点为 observer HTTP client 转发之前，排除 Pi 至 observer 派发，必须披露这一边界。日志不保存请求正文和生成文本；Pi 自身的本地原始会话仍保存在忽略目录。

Splash 1.4.0 有两套不同的指标：

- 原生 `/status.metrics` 的 `prefill_input_tokens / prefill_wall_ms`、`decode_output_tokens / decode_wall_ms` 为已完成 executor command 的累计计数。独占、串行请求前后求差，验证 completed/submitted 数量和单并发，再用 Σtokens / Σseconds 汇总。它们不包含所有客户端、模板和调度时间，也不是 Infer.NET SOP 的首末 token 间隔。
- 请求最终 `metrics.request_latency.start_to_first_token_ms` 与 `first_token_to_done_ms` 是原生请求生命周期。推测解码的第一次 emission 可包含多个 token；decode 分子必须用 `metrics.decode.tokens`，不能擅自改成 N−1。`timings.predicted_per_second` 已遵循该块输出口径。

完整保存两种原始指标、token 数、缓存、draft 接受情况和失败请求，分别命名。标准 SOP 的 N−1 首末 token 速度如果没有等价观测，写“未取得”，不根据 Pi wall time、Dashboard 最近一次请求或 HTTP token 块间隔推算。首页标准 decode 图可以为空；第三方原生速度在独立系列或选中点说明中展示。不同 KV、DFlash2、缓存、专家驻留条件不能直接与历史 Infer.NET 点作因果提速比较，也不替代 128K 正式验收。

observer 每 10 秒采集原始 vm_stat、供电及完整引擎 status，并在退出时补一份边界采样。新增 swap-out 用页数差乘实际页大小，不使用当前 swap 使用量代替；这些为整机采样，不归因于某一进程。逐题与逐请求按 UTC 请求开始时间匹配，并核查 Pi 请求数和 token 总数。缺失数据、超时、错误和未完成题目不从分母删除。官方断言采用既有固定镜像、禁网/只读/资源限制容器及错误答案负对照。
