# 首页四项趋势图与逐任务明细：Issue #44 evaluation

状态：本地 evaluation 通过；代码与结果均待用户 review，尚未合并或发布。

本实验经用户在聊天中批准。实现由 GitHub Copilot CLI 1.0.103、`gpt-6.1-sol` 完成，Codex 独立审查并运行浏览器验收。代码与结果使用从 main 独立建立的两个分支，均待用户 human review；未合并、未发布、未关闭 Issue。

## 范围与基线

- 关联 [Issue #44](https://github.com/LittleLittleCloud/Infer.NET/issues/44) 与 [代码 PR #46](https://github.com/LittleLittleCloud/Infer.NET/pull/46)。
- 网站基线 main：`58bf95018975297ba6f2f965948e2c5f4944cd95`；原提案 `630ce55` 后 main 合入 #45，只收敛模型支持，没有改变首页目标。
- 实现期间 main 新增部署记录后，代码已 rebase 至 `a5f7bc79e2d9eab308717722c52f859bb92fe2de`；最终被验收代码为 `37c93b63e6dd0eb2780d7a67989b51dbba601475`，构建保留 44 篇实验笔记。基线截图保存的是实验起始 `58bf950` 版本。
- 对照：旧首页包含进展、拓扑、目标和实验卡片；候选首页包含四项趋势图、联动实验详情及全部任务明细。旧内容保留在 `progress.html`，完整实验笔记和公开证据不删除。
- 本实验只改变展示与数据组织；不更新推理服务、不重新部署模型、不运行 GPU benchmark，不宣称推理吞吐提高。

## 历史数据核对

保留的 Pi 复测运行源码是 `e017d84e805c016d5912b5051e4c69f758056363`；网页、评测驱动和结果记录的 commit 不替代该运行版本。

| 运行 | 官方逐题通过率 | 端到端整题耗时合计 | 输入 tokens | 生成 tokens | 逐题 token 数据 |
|---|---:|---:|---:|---:|---|
| 同版本复测 | 19/20，95% | 774.288364666 s | 241162 | 19949 | 未发布，保留 null |

输入计数累计每题所有模型请求，包含重复上下文。逐题耗时四舍五入后相加与整轮原始计时可能有小差异。复测第 5 题 HumanEval/149 的工具调用错误留在固定 20 题分母中；单题独立重试不混入这轮结果。

这轮没有合格的逐请求 worker prefill/decode timing 或客户端首 token 测量，因此生产首页的 P/D 和 TTFT 明确显示“未取得”，不画零点、不从 Pi wall time 推导速度。TTFT 要求客户端请求发送到第一个有效输出 token，不能拿 worker ready 替代。既有固定长度推理实验不混入 Pi 工作负载。

证据：[复测报告](PI-HUMANEVAL-RERUN-20261023.md)、[复测摘要](../site/pi-humaneval-rerun-summary.json)。

## 验收方法与结果

数据验收核对原始摘要中的每一任务和整体统计；离线交互检查覆盖深链接、候选身份、同 SHA 复测、缺失与零、失败分母及配置隔离。独立 Chromium 验收覆盖生产数据和明确标识的合成 fixture，两者严格隔离，合成值不发布为测量结果。

首轮审查要求修正单 token 输出的 decode 语义、commit 排序、同 commit 点选区域、结果 PR 链接状态，并补齐合并前只读站点 CI；移动端坐标文字也经过浏览器检查。上述问题已修正，并加入对应回归检查。

Codex 在最终代码上独立执行：

| 检查 | 实际结果 |
|---|---|
| `python3 scripts/build-progress-site.py` | 通过：3 页、44 篇笔记、2 个真实运行 |
| `python3 scripts/test-home-trends.py --fixture-output artifacts/home-trends-fixture` | 18 项通过 |
| `node scripts/test-home-trends.js` | 15 项通过 |
| `node --check site/trends.js`、`git diff --check` | 通过 |
| Playwright 1.64.1 / Chromium 149.0.7827.55 | 18 项浏览器验收通过；无运行时错误、无资源加载错误 |

浏览器验收覆盖 1280、768、390、320px；检查四图与逐题联动、键盘操作、页面回退、深链接恢复、全部 20 题、缺失与真实零区分、官方断言分母、相同 SHA 的独立运行以及 BF16/FP32 系列隔离。合成 fixture 中 P/D/TTFT 的数值仅验证交互，不算实测。CLI 另验证 320px 下 34px 鼠标与 44px 触屏点击区域；最终数据与 JavaScript 测试保留该回归。

公开复核入口：

- [浏览器逐项结果](evaluations/issue-44/browser-results.json)、[构建和离线检查记录](evaluations/issue-44/final-checks.json)、[版本与散列清单](evaluations/issue-44/manifest.json)。
- [两分支整合检查](evaluations/issue-44/integration-results.json)：合并无冲突；结果分支单独构建为 2 页、45 篇笔记，组合后的站点构建为 3 页、45 篇笔记，18 项 Python 与 15 项 JavaScript 检查仍通过。结果记录可独立先合入，不要求代码 PR 已合并。
- [独立浏览器验收脚本](evaluations/issue-44/browser-acceptance.cjs)。在代码 PR checkout 中构建生产页和合成 fixture，分别用本机 HTTP 服务提供；设置 `PLAYWRIGHT_MODULE`（本机 Playwright 包绝对路径）、`CHROMIUM_PATH`、`PRODUCTION_URL`、`FIXTURE_URL`、`EVAL_OUTPUT`（新的本地 artifacts 目录）后运行 `node <本脚本路径>`。脚本不下载浏览器、不连接推理服务。
- [旧首页基线截图](evaluations/issue-44/baseline-main-desktop.png)、[最终桌面首页](evaluations/issue-44/desktop-overview.png)、[逐任务详情](evaluations/issue-44/task-detail.png)、[320px 首页](evaluations/issue-44/mobile-overview.png)。

CLI 会话、完整原始输出、使用量元数据、首轮检查与审查反馈仅保留在开发机忽略的 `artifacts/issue-44/` 中，公开 manifest 保存文件名和 SHA-256；不是公开下载链接。使用量记录确认实际模型为 `gpt-6.1-sol`。未公开 token、凭据或私有提示词。

## 采用建议与限制

建议采用该首页实现：已满足四图、联动详情、逐任务明细、可追溯数据和缺测处理的验收要求。通过率只有固定 20 题小样本含义，不是全量 164 题 HumanEval 榜单结果；本实验不补做缺失的推理测量，也不把界面验收当作 128K 正式性能验收。

目前只有一个被测运行 commit、两次 Pi 运行，因此任务通过率只有同一 commit 的两个独立点；三项速度/延迟图为空。这是证据现状，不是渲染故障。后续结果 PR 可按代码 PR 的数据契约加入新运行，保留实际 commit 和配置；不把未合并候选升级为 main。

结果 PR 按 SOP 经 human review 后保存；代码 PR 由用户 review 后决定是否合入。两份 PR 都不使用自动关闭实验 Issue 的关键字，待合并决定、首页状态与证据链接核对后再手动关闭 Issue。
