#!/usr/bin/env python3
"""Build the Infer.NET progress site using only the Python standard library."""
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from home_trends import homepage, load_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/pages'
REPO = 'https://github.com/LittleLittleCloud/Infer.NET'


def esc(value):
    return html.escape(str(value), quote=True)


def source(path, label='完整记录'):
    assert (ROOT / path).is_file(), path
    return f'<a href="{REPO}/blob/{esc(revision)}/{esc(path)}">{esc(label)} ↗</a>'


def table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>' + ''.join(
        f'<th scope="col">{esc(x)}</th>' for x in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def page(title, body, active):
    return f'''<!doctype html>
<html lang="{'en' if active == 'trends' else 'zh-CN'}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · Infer.NET</title><meta name="description" content="Infer.NET 项目进展、可复现实验笔记与 8bit 双机性能目标。">
<link rel="stylesheet" href="style.css"></head><body>
<a class="skip" href="#main">跳至内容</a><header><a class="brand" href="index.html">Infer.NET<span>INFERENCE RESEARCH LOG</span></a>
<nav aria-label="主导航"><a {'aria-current="page"' if active == 'trends' else ''} href="index.html">Commit trends</a>
<a {'aria-current="page"' if active == 'progress' else ''} href="progress.html">项目进展</a>
<a {'aria-current="page"' if active == 'notes' else ''} href="experiments.html">实验笔记</a><a href="{REPO}/issues">Issues ↗</a><a href="{REPO}">GitHub ↗</a></nav></header>
<main id="main">{body}</main><footer><span>内容整理于 2026-10-23 · 实测范围与缺项见逐次记录</span>
<span>页面版本：{esc(revision[:12])} · 历史测量版本见二进制散列</span></footer></body></html>'''


revision = os.environ.get('GITHUB_SHA')
if not revision:
    result = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=ROOT, text=True, capture_output=True)
    revision = result.stdout.strip() if result.returncode == 0 else 'main'
baseline = json.loads((ROOT / 'site/baseline-8bit.json').read_text())
notes = json.loads((ROOT / 'site/experiments.json').read_text())
assert len({n['id'] for n in notes}) == len(notes)
measurements = baseline['measurements']
assert {(m['inputTokens'], m['outputTokens']) for m in measurements} == {(8192,128),(32768,128),(131072,128),(131072,4096)}
for m in measurements:
    assert m['prefillSeconds'] > 0 and m['trials'] > 0
    assert abs(m['prefillTokensPerSecond'] * m['prefillSeconds'] - m['inputTokens']) < .001
    assert len(m['rawReportSha256']) == 64

target_text = (ROOT / 'docs/QWEN36-8BIT-TARGETS.md').read_text()
target_rows = [line.strip('|').split('|') for line in target_text.splitlines()
               if line.startswith('|128K 输入')]
assert len(target_rows) == 4, 'Target table changed: review site mapping before publishing.'

body = '''<section class="hero"><p class="eyebrow">研究进展 / 2026.10.23</p><h1>从单机基线，<br>走向双机推理。</h1>
<p class="lead">项目仅保留 Qwen3.6-35B-A3B，默认使用 8bit 权重；同模型 5bit 检查点继续支持。<a href="https://github.com/LittleLittleCloud/Infer.NET/issues/45">模型清理 #45</a>。自写 .NET / Metal 推理已完成 Qwen3.6-35B-A3B 8bit 单机验证。真实双机8K/32K输入与4096输出已验证：4090前20层、Mac后20层与完整状态接管。FP32 KV 诊断通过；当前 Q8 KV 目标和 128K 性能仍未验收。</p>
<div class="tags"><span>8bit 单机基线完成</span><span class="pending">双机8K/32K已验证</span><span>无前缀缓存</span></div></section>
<section class="stats" aria-label="关键实测"><article><p>128K + 128 · 单机 prefill</p><strong>964.0 <small>token/s</small></strong><span>135.966 秒 · 3 次正式测量</span></article>
<article><p>128K + 4096 · 单机 decode</p><strong>38.95 <small>token/s</small></strong><span>固定输出 · 1 次正式测量</span></article>
<article><p>128K · 双机最终 prefill 目标</p><strong>2000 <small>token/s</small></strong><span>尚未达标 · 不是实测成绩</span></article></section>
<section><div class="section-head"><h2>当前进展</h2><a href="''' + REPO + '''/issues/1">总目标 #1 ↗</a></div>
<div class="notice"><strong>CUDA GEMM / 专家：计算略快，双机未提速</strong><p>8bit / FP32 KV，同一Mac进程上切换4090 packed与fast：固定18,453输入的前段计算约快1%，两轮A–B长档完整prefill合并中位27.578/27.559秒，基本持平。固定20题HumanEval的worker prefill为561.6/532.3 token/s，decode为66.1/64.3 token/s；请求组合不同，不能把聚合降幅全归因于内核。fast保留为候选，Q8 KV与128K目标未验收。<a href="experiments.html#cuda-gemm-experts-20261023">实验记录 →</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/issues/38">Issue #38</a></p></div>
<div class="notice"><strong>Pi 固定 20 题复测：19/20，工具调用错误可复现</strong><p>同一 e017d84 推理部署与同一评测脚本复测：19 题完成并通过官方逐题断言；HumanEval/149 在工具调用解析时报错，单题重试也失败。整轮 774.3 秒含提前失败题，不能当作比首次 808.9 秒更快。<a href="experiments.html#pi-humaneval-rerun-20">复测记录 →</a> · <a href="pi-humaneval-rerun-summary.json">逐题 JSON</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/issues/32">工具调用 Issue #32</a></p></div>
<div class="notice"><strong>首次测量：固定 20 题全部通过</strong><p>coordinator、Mac serve 与 Windows agent 均运行 e017d84；Pi 1.4.0 串行完成固定 HumanEval 20 题，整题 wall time 合计 808.9 秒、中位 34.5 秒、P90 61.6 秒。20 份完整代码均通过对应题目断言。这是有工具的 Pi 端到端小样本基线，不是 164 题榜单成绩或 GPU 吞吐上限。<a href="experiments.html#pi-humaneval-main-20">实验记录 →</a> · <a href="pi-humaneval-main-summary.json">逐题 JSON</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/issues/40">Issue #40</a></p></div>
<div class="notice"><strong>BF16 QKV补偿：局部更准，整层仍未通过</strong><p>同版4090诊断中，三项BF16高低位乘积让token16的QKV相对FP32误差从9.92e-4降至8.01e-6；首层相对独立FP32参考仍约0.00704，20层0项通过。开关默认关闭、未部署；目标BF16 KV、独立状态/logits和128K验收仍待完成。<a href="experiments.html#bf16-highlow-qkv">新A/B →</a> · <a href="experiments.html#bf16-independent-oracle">独立参考 →</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/31">诊断PR #31</a></p></div>
<div class="notice"><strong>唯一模型与双机版本已统一：58bf950</strong><p>仅保留 Qwen3.6-35B-A3B，默认 8bit。真实部署15项、双机断连重连和输出一致性通过；三平台CI通过。Windows仍以前台agent运行，SCM与128K性能未验收。<a href="experiments.html#single-model-deployment">更新记录 →</a> · <a href="single-model-deployment-summary.json">验收摘要 JSON</a></p></div>
<div class="notice"><strong>CUDA权重复用：8K prefill 14.32→11.35秒</strong><p>同包对照，32K输入66.93→52.91秒；32K＋4096输出整次143.48秒，decode45.48token/s。全部输出一致；仍为8bit/FP32诊断，BF16/128K未通过。<a href="experiments.html#cuda-dense-reuse">完整A/B →</a> · <a href="dense-reuse-summary.json">逐次JSON</a></p></div>
<div class="notice"><strong>8K 双机 prefill：14.35 秒，累计约 3.66 倍</strong><p>KV 分段交接再减少 14.79%；完整请求 16.62 秒，decode 约 56 token/s。84-token 数值对照、真实取消恢复通过。仍为 8bit / FP32 诊断；128K / BF16 目标未通过。<a href="experiments.html#kv-streaming">A/B 与数值检查 →</a> · <a href="kv-streaming-summary.json">逐次 JSON</a></p></div>
<div class="notice"><strong>8K 输入 + 4096 输出：55.05 token/s</strong><p>末尾 1024 个 token 仍为 53.90 token/s，4 轮输出一致，零新增 swap-out。三次正式请求中位 91.15 秒（上一版 2760e44）；32K 新内核已测：35.10→45.43 token/s（+29.45%），4096输出整次156.77秒；BF16/128K仍未验收。<a href="experiments.html#decode-attention-fp32">新内核对照 →</a><a href="experiments.html#sustained-decode">持续解码记录 →</a> · <a href="long-output-summary.json">逐次 JSON</a></p></div>
<div class="notice"><strong>8K 双机 prefill：52.50s → 18.64s</strong><p>CUDA 按层批量执行，端到端请求由 54.78s 降至 20.93s；decode 约 56 token/s，8 次输出 token 一致。8bit / FP32 KV，128K 与 BF16 正式目标仍未通过。<a href="experiments.html#cuda-chunk">完整对照 →</a> · <a href="cuda-chunk-summary.json">逐次结果 JSON</a></p></div>
<div class="notice"><strong>你好首字：820ms → 91ms</strong><p>短输入在驻留 Mac 完成 prefill，避免固定双机交接；六组输入共 48 次输出 token 一致。长输入仍走 4090→Mac。CUDA 合并读回反而慢约 4%，已保留较快的原默认。<a href="experiments.html#short-prefill">路由与失败实验 →</a> · <a href="short-prefill-summary.json">逐次结果 JSON</a></p></div>
<div class="notice"><strong>空闲首步 decode：约 200ms → 25ms</strong><p>同版本开启/关闭 Metal 权重驻留，2/10/60 秒空闲各三次。24 次正式短请求输出一致、无新增整机 swap-out；高内存压力与 128K 长请求仍待验收。<a href="experiments.html#metal-residency">A/B 记录 →</a> · <a href="residency-summary.json">逐次结果 JSON</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/15">PR #15</a></p></div>
<div class="notice"><strong>鼠标 Dashboard 与 Auto 输出已接入</strong><p>TUI 默认 5200，Dashboard / Playground 分页，鼠标选 worker、滚轮、真实状态与最近性能。部署取消旧 2K context / 256 output 上限；本轮实测 1962 + 300、3012 + 32 tokens，262K 整模仍未验收。双端各 106 单测、59 受控检查，真实 API/TUI/上下文 45 项及断连恢复通过。<a href="experiments.html#tui-dashboard">验收记录 →</a> · <a href="dashboard-acceptance-summary.json">摘要 JSON</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/14">PR #14</a></p></div>
<ol class="milestones"><li><span class="status done">已完成</span><div><h3>Mac 8bit 正确性与容量基线</h3><p>128K 输入、4096 固定输出跑通；短数值参考与长请求后的 RPC 检查通过。</p></div></li>
<li><span class="status done">已固定</span><div><h3>性能目标 v2</h3><p>最终 2000 t/s、长请求 180 秒、decode 至少 37 t/s；阶段线 1600 t/s / 200 秒。</p></div></li>
<li><span class="status done">基线完成</span><div><h3><a href="''' + REPO + '''/issues/2">Windows 与 CUDA 环境 #2</a></h3><p>工具链、68 项单测、两组 25 项 RPC 和 SM89 数值检查通过。Mac Wi-Fi ↔ Windows 千兆网口已实测：128K 前 20 层状态，4090 → Mac 中位 24.66 秒，反向 41.20 秒；40 次传输校验全部通过。合成 GPU 状态传输不代表模型状态恢复。<a href="experiments.html#kv-sync-lan">实验记录 →</a></p></div></li>
<li><span class="status pending">诊断通过</span><div><h3><a href="''' + REPO + '''/issues/3">20/20 流水线与状态交接 #3</a></h3><p>FP32 KV：8K/128 A/B 与 8K/4096 持续输出通过；分段状态齐备后接管 decode。Q8 KV 与 128K 正式目标未通过；有线 8K CUDA 分块对照见 <a href="experiments.html#cuda-midchunk-wired">最新实验</a>，大块 CUDA 候选 <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/30">PR #30</a> 在现有链路上退化，未纳入主线。<a href="experiments.html#split-prefill-mvp">流水线记录 →</a> · <a href="split-prefill-summary.json">数值摘要 JSON</a></p></div></li>
<li><span class="status pending">候选已测</span><div><h3><a href="https://github.com/LittleLittleCloud/Infer.NET/issues/9">常驻服务与部署 #9</a></h3><p>真实模型 15 项验收通过：连续请求、SSE、取消恢复、EOS 与限流；双向 TLS，8bit/FP32 KV，已扩大上下文并实测8K长输出。Mac由LaunchAgent托管，Windows仍为SSH控制的前台进程，SCM待管理员安装；首请求优化见上方实测。<a href="experiments.html#persistent-deployment">完整记录 →</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/10">PR #10</a></p></div></li>
<li><span class="status done">已接入</span><div><h3><a href="https://github.com/LittleLittleCloud/Infer.NET/issues/11">Chat API 与 TUI #11</a></h3><p>Microsoft.Extensions.AI.IChatClient、真实流式、多轮历史、取消恢复；三格式官方 SDK 互通。8bit/FP32 KV；上下文与Auto输出已扩展，实际验证长度见最新实验。<a href="experiments.html#chat-playground">验收记录 →</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/12">PR #12</a></p></div></li>
<li><span class="status done">已接入</span><div><h3><a href="https://github.com/LittleLittleCloud/Infer.NET/issues/13">鼠标 Dashboard / 上下文 #13</a></h3><p>真实 worker 状态、性能和 Playground；Auto 输出与按需 KV 扩容。<a href="experiments.html#tui-dashboard">实验记录 →</a></p></div></li>
<li><span class="status pending">未验收</span><div><h3><a href="''' + REPO + '''/issues/4">128K 双机正式验收 #4</a></h3><p>两档输出都需通过；含通信与状态导入，不能只测 GPU 算子。</p></div></li></ol></section>
<section id="device-topology" aria-labelledby="topology-heading"><div class="section-head"><h2 id="topology-heading">当前设备拓扑</h2><a href="experiments.html#kv-sync-lan">传输实测 →</a></div>
<figure class="device-topology"><a href="assets/device-topology-pixel.png" aria-label="查看设备拓扑原图"><img src="assets/device-topology-pixel.png" width="1942" height="809" loading="lazy" decoding="async" alt="M5 Pro，48 GiB 统一内存，Metal，经 Wi-Fi 连接家庭路由器；Windows RTX 4090，24 GiB 显存，CUDA，经 1 Gbps 有线网络连接路由器。GPU 缓冲同步实测：4090 到 Mac 为 55.77 MB/s，Mac 到 4090 为 33.37 MB/s。"></a>
<figcaption class="note">2026-10-22 · 128K 前 20 层等容量合成状态，三次正式测量的中位数；图中速度为 GPU 缓冲传输吞吐，不是模型推理速度。</figcaption></figure></section>
<section><div class="section-head"><h2>8bit 单机实测</h2>''' + source('docs/QWEN36-8BIT.md') + '''</div>
<p>M5 Pro 20 核 GPU / 48 GiB，AC / High Power。模型已加载，每次清空请求状态；BF16 GEMM 输入及 KV，FP32 状态与累加。</p>'''
body += table(['输入 / 输出 tokens', 'prefill 秒', 'prefill token/s', 'decode token/s', '分块', '正式次数'], [
    [f"{m['inputTokens']:,} / {m['outputTokens']:,}", f"{m['prefillSeconds']:.3f}", f"{m['prefillTokensPerSecond']:.1f}", f"{m['decodeTokensPerSecond']:.2f}", m['prefillChunkTokens'], m['trials']] for m in measurements
])
body += '''<p class="note">短输出各档为三次均值；长输出只有一次正式测量。固定输出忽略 EOS，用于吞吐测试，不等于自然生成质量。历史 worker prefill 与双机端到端指标的计时范围不同。</p>
<div class="notice"><strong>内存限制仍然存在</strong><p>128K + 4096 必须使用较小工作区。该轮正式请求整机新增 swap-out 约 260 MiB，首次完整预热约 9.47 GiB。不能宣称“48 GiB 无 swap”。</p></div></section>
<section><div class="section-head"><h2>固定验收目标</h2>''' + source('docs/QWEN36-8BIT-TARGETS.md', '规范 v2') + '</div>'
body += table(['指标', '阶段验收', '最终目标'], target_rows)
body += '''<p class="note">目标设备：M5 Pro 48 GiB + RTX 4090 24 GiB。128K = 131072 tokens，无 prefix cache；每档完整预热一次、正式至少三次。只有正确性、资源、稳定性与性能全部通过才算完成。</p>
<p>128K 的 20/20 prefill 仍只有理想估算：短输出约 68 秒，长输出约 82 秒。网络已有独立基线，真实流水线已有8K/4096 FP32 KV测量，Q8/128K仍未验收，不能把估算当作已达到目标。</p></section>
<section><div class="section-head"><h2>实验笔记</h2><a href="experiments.html">查看完整笔记 →</a></div><div class="cards">'''
for n in notes:
    body += f'<a class="card" href="experiments.html#{esc(n["id"])}"><span class="eyebrow">{esc(n["kind"])}</span><h3>{esc(n["title"])}</h3><p>{esc(n["conclusion"])}</p><span>阅读记录 →</span></a>'
body += '''</div></section><section><h2>证据与复现</h2><p>历史实验发生在首次 Git 提交之前，通过二进制与报告 SHA-256 定位。页面 commit 是发布版本，不是历史测量版本。</p>
<p><a href="baseline-8bit.json">下载 8bit 测量摘要 JSON</a> · <a href="experiments.json">下载实验笔记 JSON</a></p>
<p class="note">公开摘要不含原始提示词和完整 RPC 日志；这些原始文件仍保存在本地，摘要保留其文件名与散列。当前链接不意味着原始日志已公开。</p></section>'''
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'progress.html').write_text(page('项目进展', body, 'progress'))
trends = load_data()
(OUT / 'index.html').write_text(page('Commit trends', homepage(trends), 'trends'))
(OUT / 'trends-data.json').write_text(json.dumps(trends, ensure_ascii=False, indent=2, allow_nan=False))
shutil.copyfile(ROOT / 'site/trends.js', OUT / 'trends.js')

body = '''<section class="hero compact"><p class="eyebrow">EXPERIMENT NOTEBOOK</p><h1>记录每一次尝试。</h1><p class="lead">保留收益、退化、失败与边界。5bit 历史实验和当前 8bit 基线分别记录；理论分析与实际测量分别标明。</p>
<p class="note">整理日期：2026-10-23。仅明确有日期的实验标出测量日期；其余按已有记录整理，不推定执行时间。新增有线 CUDA 计算块 A/B，保留端到端退化、分段 KV 交接与 4096-token 持续生成记录；BF16/128K 正式验收仍未完成。</p></section><nav class="notebook-nav" aria-label="实验目录">'''
body += ''.join(f'<a href="#{esc(n["id"])}">{esc(n["title"])}</a>' for n in notes) + '</nav>'
for n in notes:
    body += f'<article class="experiment" id="{esc(n["id"])}"><p class="eyebrow">{esc(n["kind"])}</p><h2>{esc(n["title"])}</h2>'
    for key, label in [('question','问题与假设'),('setup','实验条件'),('change','本轮改动'),('result','观察结果'),('validation','验证与证据范围'),('conclusion','结论'),('next','后续方向')]:
        body += f'<h3>{label}</h3><p>{esc(n[key])}</p>'
    if n['commands']:
        body += '<details><summary>复现入口与前提</summary><p>先按完整报告准备固定模型、输入与数值参考；GPU 测试串行运行，保留已有服务，使用新的运行标识。汇总命令需要历史原始产物；本页面不提供这些大文件。</p><pre><code>' + esc('\n'.join(n['commands'])) + '</code></pre></details>'
    body += '<p>' + source(n['source']) + f' · <a href="{REPO}/issues/1">关联总目标 #1</a></p>'
    if n.get('evidence'):
        public = re.match(r'^site/([A-Za-z0-9._-]+\.json)', n['evidence'])
        if public:
            body += f'<p class="note"><a href="{esc(public.group(1))}">公开摘要 JSON</a>；原始日志与提示词仍保留本地。</p>'
        else:
            body += f'<p class="note">本地证据索引：<code>{esc(n["evidence"])}</code>（原始文件未公开）</p>'
    body += '</article>'
(OUT / 'experiments.html').write_text(page('实验笔记', body, 'notes'))
for name in ['single-model-deployment-summary.json', 'baseline-8bit.json', 'kv-sync-summary.json', 'split-prefill-summary.json', 'experiments.json', 'pi-humaneval-main-summary.json', 'pi-humaneval-rerun-summary.json', 'chat-acceptance-summary.json', 'dashboard-acceptance-summary.json', 'idle-latency-summary.json', 'residency-summary.json', 'short-prefill-summary.json', 'cuda-chunk-summary.json', 'suffix-batching-summary.json', 'send-overlap-summary.json', 'kv-streaming-summary.json', 'long-output-summary.json', 'long-context-summary.json', 'decode-attention-summary.json', 'receive-overlap-summary.json', 'prefix-diagnostics.json', 'dense-reuse-summary.json', 'metal-scratch-summary.json', 'prefill-stage-summary.json', 'cuda-midchunk-summary.json', 'bf16-fp32-control-summary.json', 'bf16-fp32kv-control-summary.json', 'bf16-independent-oracle-summary.json', 'bf16-highlow-qkv-summary.json', 'bf16-all-projections-summary.json', 'cuda-query-reuse-summary.json', 'cuda-score-reuse-summary.json', 'style.css']:
    shutil.copyfile(ROOT / 'site' / name, OUT / name)
shutil.copytree(ROOT / 'site/assets', OUT / 'assets', dirs_exist_ok=True)
(OUT / '.nojekyll').write_text('')
print(f'Built {OUT}: 3 pages, {len(notes)} notes, {len(trends["runs"])} trend runs; revision {revision}')
