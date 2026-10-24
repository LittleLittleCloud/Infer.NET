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


def source(path, label='Full record'):
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
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · Infer.NET</title><meta name="description" content="Infer.NET project progress, reproducible experiment notes, and 8bit dual-host performance targets.">
<link rel="stylesheet" href="style.css"></head><body>
<a class="skip" href="#main">Skip to content</a><header><a class="brand" href="index.html">Infer.NET<span>INFERENCE RESEARCH LOG</span></a>
<nav aria-label="Main navigation"><a {'aria-current="page"' if active == 'trends' else ''} href="index.html">Commit trends</a>
<a {'aria-current="page"' if active == 'progress' else ''} href="progress.html">Project progress</a>
<a {'aria-current="page"' if active == 'notes' else ''} href="experiments.html">Experiment notes</a><a href="{REPO}/issues">Issues ↗</a><a href="{REPO}">GitHub ↗</a></nav></header>
<main id="main">{body}</main><footer><span>Content compiled on 2026-10-23 · See individual records for measured scope and missing evidence</span>
<span>Page version: {esc(revision[:12])} · Historical measurement versions are identified by binary hashes</span></footer></body></html>'''


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
target_lines = [line for line in target_text.splitlines() if line.startswith('|')]
target_rows = [line.strip('|').split('|') for line in target_lines[2:6]]
assert len(target_rows) == 4 and all(len(row) == 3 and '128K' in row[0] for row in target_rows), 'Target table changed: review site mapping before publishing.'

body = '''<section class="hero"><p class="eyebrow">Research Progress / 2026.10.23</p><h1>From a single-host baseline<br>to dual-host inference. </h1>
<p class="lead">Qwen3.6-35B-A3B is the only supported model. Weights default to 8bit; 5bit checkpoints of the same model remain supported. <a href="https://github.com/LittleLittleCloud/Infer.NET/issues/45">Model cleanup #45</a>. The native .NET / Metal runtime has passed Qwen3.6-35B-A3B 8bit single-host validation. Dual-host tests cover 8K/32K inputs and 4096-token outputs: the RTX 4090 runs the first 20 layers, while the Mac runs the final 20 layers and takes over decoding after complete state transfer. FP32 KV diagnostics passed; Q8 KV and 128K performance acceptance remain incomplete.</p>
<div class="tags"><span>8bit Single-machine baseline completed</span><span class="pending"> Dual-machine 8K/32K verified</span><span> No prefix cache</span></div></section>
<section class="stats" aria-label="Key measurements"><article><p>128K + 128 · Single-machine prefill</p><strong>964.0 <small>token/s</small></strong><span>135.966 seconds · 3 formal measurements</span></article>
<article><p>128K + 4096 · Single-machine decode</p><strong>38.95 <small>token/s</small></strong><span> Fixed output · 1 measured run</span></article>
<article><p>128K · Dual-machine final prefill target </p><strong>2000 <small> token/s </small></strong><span> not yet met · Not actual measured results </span></article></section>
<section><div class="section-head"><h2> current progress </h2><a href="''' + REPO + '''/issues/1">total target #1 ↗</a></div>
<div class="notice"><strong> Pi fixed 20 question retest: 19/20, tool call error reproducible </strong><p>e017d84 inference deployment, 19 questions completed and passed official per-question assertion; HumanEval/149 error during tool call parsing, single-question retry also failed. Whole round 774.3 seconds is Pi end-to-end time, cannot be used as worker prefill/decode speed. <a href="experiments.html#pi-humaneval-rerun-20"> retest record →</a> · <a href="pi-humaneval-rerun-summary.json"> per-question JSON</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/issues/32"> tool call Issue #32</a></p></div>
<div class="notice"><strong>BF16 QKV compensation: local more accurate, whole layer still not passed </strong><p> same version 4090 diagnosis in progress, three items BF16 high-low bit product let token16's QKV relative to FP32 error reduced from 9.92e-4 to 8.01e-6; first layer relatively independent FP32 reference still about 0.00704, 20 layer 0 items passed. Switch default off, not deployed; target BF16 KV, independent state/logits and 128K acceptance still pending. <a href="experiments.html#bf16-highlow-qkv"> new A/B →</a> · <a href="experiments.html#bf16-independent-oracle"> independent reference →</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/31"> diagnosis PR #31</a></p></div>
<div class="notice"><strong> unique model and dual-machine version unified: 58bf950</strong><p> only retain Qwen3.6-35B-A3B, default 8bit. Real deployment 15 items, dual-machine disconnect reconnect and output consistency passed; three platforms CI passed. Windows still runs as foreground agent, SCM and 128K performance not accepted. <a href="experiments.html#single-model-deployment"> update record →</a> · <a href="single-model-deployment-summary.json"> acceptance summary JSON</a></p></div>
<div class="notice"><strong> CUDA weight reuse: 8K prefill 14.32→11.35 seconds </strong><p> same package comparison, 32K input 66.93→52.91 seconds; 32K＋4096 output whole 143.48 seconds, decode45.48token/s. All outputs consistent; still 8bit/FP32 diagnosis, BF16/128K not passed. <a href="experiments.html#cuda-dense-reuse"> complete A/B →</a> · <a href="dense-reuse-summary.json"> per-time JSON</a></p></div>
<div class="notice"><strong>8K dual-machine prefill: 14.35 seconds, cumulative about 3.66 times </strong><p> KV segmented handover then reduce 14.79%; complete request 16.62 seconds, decode about 56 token/s. 84-token numerical comparison, real cancel recovery passed. Still 8bit / FP32 diagnosis; 128K / BF16 target not passed. <a href="experiments.html#kv-streaming"> A/B and numerical check →</a> · <a href="kv-streaming-summary.json"> per-time JSON</a></p></div>
<div class="notice"><strong>8K input + 4096 output: 55.05 token/s </strong><p> end 1024 tokens still 53.90 token/s, 4 round output consistent, zero new swap-out. Three formal requests median 91.15 seconds (previous version 2760e44); 32K new kernel tested: 35.10→45.43 token/s (+29.45%), 4096 output whole 156.77 seconds; BF16/128K still not accepted. <a href="experiments.html#decode-attention-fp32"> new kernel comparison →</a><a href="experiments.html#sustained-decode"> continuous decode record →</a> · <a href="long-output-summary.json"> per-time JSON</a></p></div>
<div class="notice"><strong>8K Dual-machine prefill: 52.50s → 18.64s</strong><p> CUDA executes layer-by-layer in batches; end-to-end requests reduced from 54.78s to 20.93s; decode achieves approx. 56 tokens/s, with 8 output tokens consistent. 8bit / FP32 KV, 128K and BF16 formal targets not yet passed. <a href="experiments.html#cuda-chunk"> Full comparison → </a> · <a href="cuda-chunk-summary.json"> Step-by-step results JSON </a></p></div>
<div class="notice"><strong> Chinese greeting, first token: 820ms → 91ms</strong><p> Short inputs complete prefill on resident Mac, avoiding fixed dual-machine handover; output token sequences matched across 48 requests spanning six inputs. Long inputs still route via 4090→Mac. CUDA merged read-back is approx. 4% slower, so the faster original default is retained. <a href="experiments.html#short-prefill"> Routing and failure experiments → </a> · <a href="short-prefill-summary.json"> Step-by-step results JSON </a></p></div>
<div class="notice"><strong> Idle first-step decode: approx. 200ms → 25ms</strong><p> Same version with/without Metal weight residency, 2/10/60 seconds idle three times each. 24 formal short requests output consistently, no new whole-machine swap-out; high memory pressure and 128K long requests still await acceptance. <a href="experiments.html#metal-residency"> A/B records → </a> · <a href="residency-summary.json"> Step-by-step results JSON </a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/15"> PR #15</a></p></div>
<div class="notice"><strong> Mouse Dashboard and Auto output integrated into </strong><p> TUI default 5200, Dashboard / Playground pagination, mouse selects worker, scroll wheel, real status, and recent performance. Deployment removed old 2K context / 256 output limits; this round tested 1962 + 300, 3012 + 32 tokens, 262K full model still not accepted. Both ends: 106 unit tests, 59 controlled checks, real API/TUI/context 45 items and disconnection recovery passed. <a href="experiments.html#tui-dashboard"> Acceptance records → </a> · <a href="dashboard-acceptance-summary.json"> Summary JSON </a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/14"> PR #14</a></p></div>
<ol class="milestones"><li><span class="status done"> Completed </span><div><h3> Mac 8bit Correctness and Capacity Baseline </h3><p>128K Input, 4096 Fixed output run-through; short numerical reference and RPC checks after long requests passed. </p></div></li>
<li><span class="status done"> Fixed </span><div><h3> Performance Targets v2</h3><p> Final 2000 t/s, long requests 180 seconds, decode at least 37 t/s; phase line 1600 t/s / 200 seconds. </p></div></li>
<li><span class="status done"> Baseline completed </span><div><h3><a href="''' + REPO + '''/issues/2">Windows and CUDA Environment #2</a></h3><p> Toolchain, 68 unit tests, two sets of 25 RPC and SM89 numerical checks passed. Mac Wi-Fi ↔ Windows Gigabit port tested: 128K before 20 layer status, 4090 → Mac median 24.66 seconds, reverse 41.20 seconds; 40 transmission verifications all passed. Synthetic GPU state transfer does not represent model state recovery. <a href="experiments.html#kv-sync-lan"> Experiment records → </a></p></div></li>
<li><span class="status pending"> Diagnosis passed </span><div><h3><a href="''' + REPO + '''/issues/3">20/20 Pipeline and State Handover #3</a></h3><p>FP32 KV: 8K/128 A/B and 8K/4096 continuous output passed; take over decode after segmented state is complete. Q8 KV and 128K formal targets not passed; wired 8K CUDA block comparison see <a href="experiments.html#cuda-midchunk-wired"> latest experiments </a>, large block CUDA candidate <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/30"> PR #30</a> degraded on existing link, not included in mainline. <a href="experiments.html#split-prefill-mvp"> Pipeline records → </a> · <a href="split-prefill-summary.json"> Numerical summary JSON </a></p></div></li>
<li><span class="status pending"> Candidates tested </span><div><h3><a href="https://github.com/LittleLittleCloud/Infer.NET/issues/9"> Resident services and deployment #9</a></h3><p> Real model 15 Items passed acceptance: continuous requests, SSE, cancellation recovery, EOS, and rate limiting; bidirectional TLS, 8bit/FP32 KV, context expanded and long outputs tested 8K. Mac hosted by LaunchAgent, Windows still a foreground process controlled via SSH, SCM pending admin installation; first-request optimization see above test results. <a href="experiments.html#persistent-deployment"> Full record →</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/10">PR #10</a></p></div></li>
<li><span class="status done"> Integrated </span><div><h3><a href="https://github.com/LittleLittleCloud/Infer.NET/issues/11"> Chat API and TUI #11</a></h3><p> Microsoft.<wbr>Extensions.<wbr>AI.<wbr>IChatClient, real streaming, multi-turn history, cancellation recovery; interoperable with three formats of official SDKs. 8bit/FP32 KV; context and Auto outputs expanded, actual verified lengths see latest experiments. <a href="experiments.html#chat-playground"> Acceptance record →</a> · <a href="https://github.com/LittleLittleCloud/Infer.NET/pull/12">PR #12</a></p></div></li>
<li><span class="status done"> Integrated </span><div><h3><a href="https://github.com/LittleLittleCloud/Infer.NET/issues/13"> Mouse Dashboard / Context #13</a></h3><p> Real worker status, performance, and Playground; Auto outputs and on-demand KV expansion. <a href="experiments.html#tui-dashboard"> Experiment record →</a></p></div></li>
<li><span class="status pending"> Not accepted </span><div><h3> <a href="''' + REPO + '''/issues/4">128K Dual-machine formal acceptance #4</a></h3><p> Both output tiers must pass; includes communication and state import, cannot just test GPU operators. </p></div></li></ol></section>
<section id="device-topology" aria-labelledby="topology-heading"><div class="section-head"><h2 id="topology-heading"> Current device topology </h2><a href="experiments.html#kv-sync-lan"> Transmission test results →</a></div>
<figure class="device-topology"><a href="assets/device-topology-pixel.png" aria-label="View the original device topology image"><img src="assets/device-topology-pixel.png" width="1942" height="809" loading="lazy" decoding="async" alt="M5 Pro with 48 GiB unified memory and Metal, connected to the home router over Wi-Fi; Windows RTX 4090 with 24 GiB VRAM and CUDA, connected over 1 Gbps Ethernet. Measured GPU buffer synchronization: 55.77 MB/s from RTX 4090 to Mac and 33.37 MB/s from Mac to RTX 4090."></a>
<figcaption class="note">2026-10-22 · 128K First 20 layers and other capacity synthesis states, median of three formal measurements; the speed in the chart is GPU buffer transfer throughput, not model inference speed. </figcaption></figure></section>
<section><div class="section-head"><h2>8bit Single-machine test results </h2>''' + source('docs/QWEN36-8BIT.md') + '''</div>
<p>M5 Pro 20 core GPU / 48 GiB, AC / High Power. Model loaded, clearing request state each time; BF16 GEMM inputs and KV, FP32 states and accumulations. </p>'''
body += table(['Input / Output tokens', 'Prefill seconds', 'prefill token/s', 'decode token/s', 'Chunks', 'Formal attempts'], [
    [f"{m['inputTokens']:,} / {m['outputTokens']:,}", f"{m['prefillSeconds']:.3f}", f"{m['prefillTokensPerSecond']:.1f}", f"{m['decodeTokensPerSecond']:.2f}", m['prefillChunkTokens'], m['trials']] for m in measurements
])
body += '''<p class="note"> Short outputs at each tier are averages of three; long outputs have only one formal measurement. Fixed outputs ignore EOS, used for throughput testing, not equal to natural generation quality. Historical worker prefill and dual-machine end-to-end metrics have different timing ranges. </p>
<div class="notice"><strong> Memory limits still exist </strong><p>128K + 4096 Must use smaller workspaces. This round of formal requests added about 260 MiB of swap-out to the whole machine, first full warm-up about 9.47 GiB. Cannot claim "48 GiB no swap". </p></div></section>
<section><div class="section-head"><h2> Fixed acceptance targets </h2>''' + source('docs/QWEN36-8BIT-TARGETS.md', 'Specification v2') + '</div>'
body += table(['Metrics', 'Stage acceptance', 'Final goal'], target_rows)
body += '''<p class="note"> Target device: M5 Pro 48 GiB + RTX 4090 24 GiB. 128K = 131072 tokens, no prefix cache; each tier fully warmed up once, formally at least three times. Only when correctness, resources, stability, and performance all pass is it considered complete. </p>
The 20/20 prefill of <p>128K remains only ideally estimated: short outputs take about 68 seconds, long outputs about 82 seconds. The network already has an independent baseline, and the real pipeline already has 8K/4096 FP32 KV measurements. Q8/128K has not yet been accepted; estimates cannot be treated as having reached the target. </p></section>
<section><div class="section-head"><h2> experiment notes </h2><a href="experiments.html"> View full notes → </a></div><div class="cards">'''
for n in notes:
    body += f'<a class="card" href="experiments.html#{esc(n["id"])}"><span class="eyebrow">{esc(n["kind"])}</span><h3>{esc(n["title"])}</h3><p>{esc(n["conclusion"])}</p><span> Reading log → </span></a>'
body += '''</div></section><section><h2> Evidence and reproduction </h2><p> Historical experiments occurred before the first Git commit, located via binaries and reports SHA-256. The page commit is a release version, not a historical measurement version. </p>
<p><a href="baseline-8bit.json"> Download 8bit measurement summary JSON </a> · <a href="experiments.json"> Download experiment notes JSON </a></p>
<p class="note"> public summary does not contain original prompts and complete RPC logs; these original files are still stored locally, summary retains their filenames and hashes. Current link does not imply original logs are public. </p></section>'''
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'progress.html').write_text(page('Project progress', body, 'progress'))
trends = load_data()
(OUT / 'index.html').write_text(page('Commit trends', homepage(trends), 'trends'))
(OUT / 'trends-data.json').write_text(json.dumps(trends, ensure_ascii=False, indent=2, allow_nan=False))
shutil.copyfile(ROOT / 'site/trends.js', OUT / 'trends.js')

body = '''<section class="hero compact"><p class="eyebrow"> EXPERIMENT NOTEBOOK </p><h1> Record every attempt. </h1><p class="lead"> Record gains, regressions, failures, and limits. Historical 5bit experiments and the current 8bit baseline are kept separate, with theoretical estimates clearly distinguished from measurements. </p>
<p class="note"> Organized date: 2026-10-23. Only experiments with explicit dates have their measurement dates marked; others are organized according to existing records without assuming execution time. New wired CUDA compute blocks A/B added, retaining end-to-end regression, segmented KV handover, and 4096-token continuous generation records; BF16/128K formal acceptance is still incomplete. </p></section><nav class="notebook-nav" aria-label="Experiment directory">'''
body += ''.join(f'<a href="#{esc(n["id"])}">{esc(n["title"])}</a>' for n in notes) + '</nav>'
for n in notes:
    body += f'<article class="experiment" id="{esc(n["id"])}"><p class="eyebrow">{esc(n["kind"])}</p><h2>{esc(n["title"])}</h2>'
    for key, label in [('question','Issues and hypotheses'),('setup','Experimental conditions'),('change','Changes in this round'),('result','Observations'),('validation','Scope of verification and evidence'),('conclusion','Conclusion'),('next','Future directions')]:
        body += f'<h3>{label}</h3><p>{esc(n[key])}</p>'
    if n['commands']:
        body += '<details><summary> Reproduction entry and prerequisites </summary><p> First, prepare fixed models, inputs, and numerical references according to the complete report; GPU tests run serially, retaining existing services and using new run identifiers. The summary command requires historical original artifacts; this page does not provide these large files. </p><pre><code>' + esc('\n'.join(n['commands'])) + '</code></pre></details>'
    body += '<p>' + source(n['source']) + f' · <a href="{REPO}/issues/1"> Associated overall goal #1</a></p>'
    if n.get('evidence'):
        public = re.match(r'^site/([A-Za-z0-9._-]+\.json)', n['evidence'])
        if public:
            body += f'<p class="note"><a href="{esc(public.group(1))}"> Public summary JSON </a>; original logs and prompts are still kept locally. </p>'
        else:
            body += f'<p class="note"> Local evidence index: <code>{esc(n["evidence"])}</code> (original files not public) </p>'
    body += '</article>'
(OUT / 'experiments.html').write_text(page('Experiment notes', body, 'notes'))
for name in ['single-model-deployment-summary.json', 'baseline-8bit.json', 'kv-sync-summary.json', 'split-prefill-summary.json', 'experiments.json', 'pi-humaneval-rerun-summary.json', 'chat-acceptance-summary.json', 'dashboard-acceptance-summary.json', 'idle-latency-summary.json', 'residency-summary.json', 'short-prefill-summary.json', 'cuda-chunk-summary.json', 'suffix-batching-summary.json', 'send-overlap-summary.json', 'kv-streaming-summary.json', 'long-output-summary.json', 'long-context-summary.json', 'decode-attention-summary.json', 'receive-overlap-summary.json', 'prefix-diagnostics.json', 'dense-reuse-summary.json', 'metal-scratch-summary.json', 'prefill-stage-summary.json', 'cuda-midchunk-summary.json', 'bf16-fp32-control-summary.json', 'bf16-fp32kv-control-summary.json', 'bf16-independent-oracle-summary.json', 'bf16-highlow-qkv-summary.json', 'bf16-all-projections-summary.json', 'cuda-query-reuse-summary.json', 'cuda-score-reuse-summary.json', 'style.css']:
    shutil.copyfile(ROOT / 'site' / name, OUT / name)
shutil.copytree(ROOT / 'site/assets', OUT / 'assets', dirs_exist_ok=True)
(OUT / '.nojekyll').write_text('')
print(f'Built {OUT}: 3 pages, {len(notes)} notes, {len(trends["runs"])} trend runs; revision {revision}')
