"""Normalized homepage data, historical adapters and strict stdlib validation."""
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"^[0-9a-f]{40}$")
ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
STATUSES = {"passed", "failed", "error", "timeout", "incomplete", "unverified"}
CONFIG_FIELDS = (
    "model", "modelRevision", "weights", "kv", "cache", "chunking", "topology",
    "network", "power", "workload", "datasetRevision", "datasetSha256", "plannedTasks", "timing"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, integer=False):
    require(value is None or (
        type(value) in (int, float) and math.isfinite(value) and value >= 0
        and (not integer or type(value) is int)
    ), f"{label}: expected null or finite nonnegative {'integer' if integer else 'number'}")


def public_url(value):
    require(isinstance(value, str) and value, "Evidence URL required")
    parts = urlsplit(value)
    require(
        (parts.scheme == "https" and parts.hostname == "github.com"
         and not parts.username and not parts.password)
        or (not parts.scheme and not parts.netloc and not value.startswith("/")
            and ".." not in parts.path.split("/") and not "\\" in value),
        f"Unsafe/nonpublic evidence URL: {value}"
    )


def total(tasks, field):
    values = [t[field] for t in tasks if t[field] is not None]
    return {"value": sum(values) if values else None, "samples": len(values),
            "total": len(tasks), "complete": len(values) == len(tasks)}


def aggregate_requests(requests, request_count):
    """Aggregate paired observations, never means of request throughput."""
    number(request_count, "requestCount", integer=True)
    if requests is None:
        requests = []
    require(isinstance(requests, list), "requests must be null or a list")
    require(request_count is not None and len(requests) <= request_count,
            "Request samples exceed total request count")
    ids = set()
    pairs = {"prefill": [], "decode": []}
    ttfts = []
    failures = 0
    unknown_failures = 0
    for row in requests:
        require(isinstance(row["id"], str) and row["id"], "Request ID required")
        require(row["id"] not in ids, "Duplicate request ID")
        ids.add(row["id"])
        require(row["outcome"] in {"completed", "failed", "timeout", "incomplete"},
                "Invalid request outcome")
        require(row.get("firstTokenReceived") is None or type(row["firstTokenReceived"]) is bool,
                "First-token receipt must be null or boolean")
        for field in ("inputTokens", "outputTokens", "prefillSeconds", "decodeSeconds", "clientTtftMs"):
            number(row[field], field, integer=field.endswith("Tokens"))
        for metric, token_field, seconds_field in (
            ("prefill", "inputTokens", "prefillSeconds"),
            ("decode", "outputTokens", "decodeSeconds")
        ):
            tokens, seconds = row[token_field], row[seconds_field]
            if metric == "decode" and tokens is not None and tokens < 2:
                continue
            if seconds is not None:
                require(seconds > 0, f"{seconds_field} must be positive when measured")
            if tokens is not None and seconds is not None:
                pairs[metric].append((tokens if metric == "prefill" else tokens - 1, seconds))
        if row["clientTtftMs"] is not None:
            require(row.get("ttftBoundary") == "client-send-to-first-valid-output-token",
                    "TTFT requires actual client first valid output-token timing")
            require(row.get("firstTokenReceived") is not False, "TTFT conflicts with no first token")
            ttfts.append(row["clientTtftMs"])
        elif row["outcome"] != "completed":
            if row.get("firstTokenReceived") is False:
                failures += 1
            elif row.get("firstTokenReceived") is None:
                unknown_failures += 1
    metrics = {}
    for name, observations in pairs.items():
        metrics[name] = {
            "value": sum(n for n, _ in observations) / sum(s for _, s in observations)
            if observations else None,
            "samples": len(observations), "total": request_count
        }
    metrics["ttft"] = {
        "value": statistics.median(ttfts) if ttfts else None,
        "samples": len(ttfts), "total": request_count,
        "failedWithoutToken": failures,
        "unknownFirstTokenFailures": unknown_failures,
        "unobservedRequests": request_count - len(requests)
    }
    return metrics


def validate(data):
    require(data["schemaVersion"] == 1, "Unsupported trend schema")
    timeline = {}
    for commit in data["commitTimeline"]:
        require(SHA.fullmatch(commit["sha"]) and commit["sha"] not in timeline,
                "Commit timeline requires unique full SHAs")
        require(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", commit["committedAt"]),
                "Commit timestamp must be UTC ISO-8601")
        datetime.strptime(commit["committedAt"], "%Y-%m-%dT%H:%M:%SZ")
        public_url(commit["evidence"])
        timeline[commit["sha"]] = commit
    configs = {c["id"]: c for c in data["configurations"]}
    require(len(configs) == len(data["configurations"]), "Duplicate configuration ID")
    signatures = set()
    for config in configs.values():
        require(ID.fullmatch(config["id"]), "Invalid configuration ID")
        require(all(config.get(f) for f in CONFIG_FIELDS), "Incomplete comparison configuration")
        require(SHA.fullmatch(config["modelRevision"]) and SHA.fullmatch(config["datasetRevision"]),
                "Configuration revisions must be full SHAs")
        require(re.fullmatch(r"[0-9a-f]{64}", config["datasetSha256"]), "Invalid dataset SHA-256")
        require(config["timing"].get("ttft") == "client-send-to-first-valid-output-token",
                "Invalid TTFT boundary")
        require(all(config["timing"].get(k) for k in ("wall", "prefill", "decode")),
                "Missing timing scopes")
        planned = config["plannedTasks"]
        require(isinstance(planned, list) and len(planned) == len(set(planned)), "Invalid planned cohort")
        signature = json.dumps({f: config[f] for f in CONFIG_FIELDS}, sort_keys=True)
        require(signature not in signatures, "Duplicate comparable configuration")
        signatures.add(signature)
    runs = {}
    for run in data["runs"]:
        require(ID.fullmatch(run["id"]) and ID.fullmatch(run["experimentId"]), "Invalid run/experiment ID")
        require(run["id"] not in runs, "Duplicate run ID")
        runs[run["id"]] = run
        require(run["configurationId"] in configs, "Unknown configuration")
        config = configs[run["configurationId"]]
        require(SHA.fullmatch(run["measuredCommit"]), "Measured runtime SHA required")
        require(run["measuredCommit"] in timeline, "Measured SHA absent from commit timeline")
        require(run["state"] in {"adopted", "pending-review", "regressed", "failed", "not-adopted"},
                "Unknown review/result state")
        provenance = run["provenance"]
        require(provenance["kind"] in {"main", "candidate"}, "Explicit code provenance required")
        require(provenance.get("evidence"), "Code provenance evidence required")
        if provenance["kind"] == "main":
            require(SHA.fullmatch(provenance.get("mainCommit", "")), "Explicit main/merge SHA required")
        else:
            require(provenance.get("mainCommit") is None, "Candidate cannot claim main")
        public_url(provenance["evidence"])
        require(run["runtime"] and all(SHA.fullmatch(v["commit"]) for v in run["runtime"].values()),
                "Actual runtime component identities required")
        require(all(v["commit"] == run["measuredCommit"] for v in run["runtime"].values()),
                "Mixed runtime SHAs must use a separate explicit configuration/adapter")
        for digest in run["binarySha256"].values():
            require(re.fullmatch(r"[0-9a-f]{64}", digest), "Invalid artifact SHA-256")
        require(run["title"] and run["date"] and run["change"] and run["result"], "Run description required")
        for link in run["links"]:
            require(link["label"], "Link label required")
            require(link.get("role") in {"issue", "code-pr", "results-pr", "harness-pr",
                                     "notebook", "evidence", "report"}, "Explicit link role required")
            public_url(link["url"])
        tasks = run["tasks"]
        require([t["taskId"] for t in tasks] == config["plannedTasks"],
                "Every planned task must appear once, in cohort order")
        for task in tasks:
            require(type(task["attempt"]) is int and task["attempt"] > 0, "Positive attempt identity required")
            require(task["status"] in STATUSES, "Unknown task status")
            for f in ("wallSeconds", "inputTokens", "outputTokens", "modelRequests"):
                number(task[f], f, integer=f != "wallSeconds")
            require(task["source"], "Task source mapping required")
            public_url(task["source"])
            require(task["officialValidation"] in {"passed", "failed", "not-run", "unverified"},
                    "Unknown official validation")
            require((task["status"] == "passed") == (task["officialValidation"] == "passed"),
                    "Pass requires official task assertions, not completion/syntax")
            require(task["officialValidation"] != "not-run" or task["status"] in
                    {"error", "timeout", "incomplete"}, "Unexecuted checks need an explicit execution failure")
            require(task["officialValidation"] != "failed" or task["status"] == "failed",
                    "Official assertion failure needs failed task status")
        verified = all(t["officialValidation"] != "unverified" for t in tasks)
        passed = sum(t["officialValidation"] == "passed" for t in tasks)
        denominator = len(config["plannedTasks"])
        run["metrics"] = aggregate_requests(run["requests"], run["requestCount"])
        task_requests = total(tasks, "modelRequests")
        if task_requests["complete"]:
            require(task_requests["value"] == run["requestCount"], "Task/run request totals differ")
        for request in run["requests"] or []:
            require(any(t["taskId"] == request["taskId"] and t["attempt"] == request["attempt"]
                        for t in tasks), "Request must map to a planned task attempt")
            public_url(request["source"])
        for task in tasks:
            observations = [r for r in run["requests"] or []
                            if r["taskId"] == task["taskId"] and r["attempt"] == task["attempt"]]
            count = task["modelRequests"]
            if count is not None:
                require(len(observations) <= count, "Too many request samples for task")
                if len(observations) == count:
                    for field in ("inputTokens", "outputTokens"):
                        if task[field] is not None and all(r[field] is not None for r in observations):
                            require(sum(r[field] for r in observations) == task[field],
                                    f"Task {field} differs from cumulative model requests")
        run["metrics"]["pass"] = {
            "value": 100 * passed / denominator if verified else None,
            "passed": passed, "total": denominator, "verified": verified,
            "samples": sum(t["officialValidation"] in {"passed", "failed"} for t in tasks)
        }
        run["totals"] = {field: total(tasks, field)
                         for field in ("wallSeconds", "inputTokens", "outputTokens", "modelRequests")}
        for field, reported in run.get("reportedTotals", {}).items():
            require(field in run["totals"], "Unknown reported total")
            number(reported, f"reported {field}", integer=field != "wallSeconds")
            observed = run["totals"][field]
            if reported is not None and observed["complete"]:
                tolerance = .00051 * len(tasks) if field == "wallSeconds" else 0
                require(abs(observed["value"] - reported) <= tolerance,
                        f"{run['id']}: task {field} total differs from source")
            elif reported is not None and observed["value"] is not None:
                require(observed["value"] <= reported + (.00051 * len(tasks) if field == "wallSeconds" else 0),
                        f"{run['id']}: known partial {field} exceeds reported total")
    for run in runs.values():
        base = run["baselineRunId"]
        if base is not None:
            require(base in runs and base != run["id"], "Invalid baseline reference")
            require(runs[base]["configurationId"] == run["configurationId"], "Incompatible baseline series")
    for config in configs.values():
        baseline = config["latestBaselineRun"]
        if baseline is not None:
            require(baseline in runs, "Unknown latest baseline")
            run = runs[baseline]
            require(run["configurationId"] == config["id"] and run["provenance"]["kind"] == "main"
                    and run["metrics"]["pass"]["verified"], "Baseline must be verified main in this series")
    require(set(timeline) == {r["measuredCommit"] for r in runs.values()},
            "Commit timeline must exactly cover measured SHAs")
    return data


def import_pi(spec, root):
    path = root / spec["source"]
    raw = json.loads(path.read_text())
    main = "sourceCommit" in raw
    rows = raw["results"]["rows"] if main else raw["rows"]
    sha = raw["sourceCommit"] if main else raw["runtimeSourceCommit"]
    run = copy.deepcopy({k: v for k, v in spec.items() if k not in {"source", "report", "binarySource"}})
    binaries = raw.get("binarySha256", {})
    if spec.get("binarySource"):
        prior = json.loads((root / spec["binarySource"]).read_text())
        require(prior["sourceCommit"] == sha, "Inherited binaries must match measured runtime")
        binaries = prior["binarySha256"]
    run.update({
        "date": raw["date"], "measuredCommit": sha,
        "runtime": {name: {"commit": sha} for name in ("coordinator", "macServe", "windowsAgent")},
        "binarySha256": binaries,
        "binaryEvidence": spec.get("binarySource", spec["source"]),
        "harnessCommit": raw["harnessCommit"], "harnessSha256": raw["harnessSha256"],
        "conditions": {"topology": raw["topology"], "run": raw["run"]},
        "requestCount": raw["results"]["requestCount"], "requests": None,
        "reportedTotals": {
            "wallSeconds": raw["results"]["wallSeconds"]["total"],
            "inputTokens": raw["results"]["tokens"]["input"],
            "outputTokens": raw["results"]["tokens"]["output"],
            "modelRequests": raw["results"]["requestCount"]
        },
        "tasks": [], "source": spec["source"],
        "rawEvidence": raw.get("rawLocalOnlySha256", raw.get("rawEvidence")),
    })
    for i, row in enumerate(rows):
        check = row.get("functional") if main else row.get("officialCheck")
        official = ("passed" if check == "passed" else "not-run"
                    if check == "missing_solution" else "failed" if check == "failed" else "unverified")
        run["tasks"].append({
            "taskId": row["taskId"], "attempt": 1, "wallSeconds": row.get("wallSeconds"),
            "inputTokens": row.get("inputTokens"), "outputTokens": row.get("outputTokens"),
            "modelRequests": row.get("modelRequests"),
            "status": "passed" if official == "passed" else "error"
            if check == "missing_solution" else "failed" if official == "failed" else "unverified",
            "officialValidation": official,
            "source": f"{path.name}#/{'results/rows' if main else 'rows'}/{i}"
        })
    official_count = raw["results"].get("officialAssertionsSubsetPassed") if main else raw["results"].get("officialTaskChecksPassed")
    require(sum(t["officialValidation"] == "passed" for t in run["tasks"]) == official_count,
            "Official pass count differs from source")
    require(len(rows) == raw["results"]["count"], "Planned denominator differs from source")
    run["links"] += [
        {"role": "notebook", "label": "Experiment notebook", "url": f"experiments.html#{spec['experimentId']}"},
        {"role": "evidence", "label": "Published evidence JSON", "url": path.name},
        {"role": "report", "label": "Full report / raw-log hashes (logs local only)",
         "url": f"https://github.com/LittleLittleCloud/Infer.NET/blob/main/{spec['report']}"}
    ]
    return run


def load_data(root=ROOT):
    manifest = json.loads((root / "site/trends.json").read_text())
    data = {"schemaVersion": manifest["schemaVersion"], "commitTimeline": manifest["commitTimeline"],
            "configurations": manifest["configurations"],
            "runs": [import_pi(spec, root) for spec in manifest["imports"]] + manifest["runs"]}
    configs = {c["id"]: c for c in data["configurations"]}
    for spec, run in zip(manifest["imports"], data["runs"]):
        raw = json.loads((root / spec["source"]).read_text())
        config = configs[run["configurationId"]]
        require(raw["model"] == config["model"] and raw["modelRevision"] == config["modelRevision"]
                and raw["datasetRevision"] == config["datasetRevision"]
                and raw["datasetSha256"] == config["datasetSha256"]
                and raw["kv"].lower() == config["kv"].lower(),
                "Imported model, dataset or precision differs from series")
        require(raw.get("weights") == config["weights"] or
                (raw.get("weightBits") == 8 and config["weights"] == "affine 8bit / group64"),
                "Imported weight precision differs from series")
        require(raw["results"]["tokens"]["cacheRead"] == 0 and raw["results"]["tokens"]["cacheWrite"] == 0,
                "Pi adapter requires documented uncached runs")
        require(raw["run"]["piVersion"] == "1.4.0" and raw["run"]["timeoutSeconds"] == 300
                and raw["run"]["contextWindow"] == 32768 and raw["run"]["maxTokens"] == 4096
                and raw["run"]["freshWorkspacePerTask"] is True,
                "Pi adapter requires the documented fixed workload")
        require([t["taskId"] for t in run["tasks"]] == config["plannedTasks"],
                "Imported tasks differ from planned cohort")
        if "taskOrder" in raw["run"]:
            require(raw["run"]["taskOrder"] == config["plannedTasks"], "Source task order differs")
        for path in (spec["source"], spec["report"], run["provenance"]["evidence"]):
            require((root / path).is_file(), f"Missing source evidence: {path}")
    return validate(data)


def homepage(data):
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    charts = "".join(
        f'<section class="trend-card" id="{anchor}" aria-labelledby="{key}-heading">'
        f'<div class="plot-head"><h2 id="{key}-heading">{label}</h2>'
        f'<span id="{key}-value" class="plot-value"></span></div>'
        f'<div class="plot" id="{key}-chart"></div><p class="note" id="{key}-coverage"></p></section>'
        for key, label, anchor in (
            ("prefill", "Prefill · token/s", "prefill-trend"),
            ("decode", "Decode · token/s", "decode-trend"),
            ("ttft", "TTFT · ms · request median ↓", "ttft-trend"),
            ("pass", "Task pass rate · %", "task-pass-trend")
        )
    )
    return f'''<div class="trends-home">
<h1 class="trends-title">Commit trends</h1>
<div class="trend-legend"><span>● main</span><span class="candidate-key">◆ candidate · dashed branch</span>
<span>◎ latest verified main baseline</span><span>□ selected · Tab + Enter/Space to select</span></div>
<div class="run-controls"><label>Configuration filter <select id="configuration"></select></label>
<label>Run · all configurations <select id="run"></select></label></div>
<p id="selection-notice" role="status"></p>
<div class="trend-grid">{charts}</div>
<p class="note">Measured commit axes; repeated runs at the same SHA are offset for visibility, not new commits.
All runs appear by default. Lines connect only comparable configurations. No Pi wall-time-derived throughput.
Missing metrics are marked 未测, not zero. The issue #54 main and BF16/2048 runs did not record client first-token timestamps, so their TTFT is unavailable.</p>
<section id="run-detail" aria-labelledby="run-title"><div class="selected-head"><h2 id="run-title"></h2>
<span id="run-state"></span><a id="permalink">Link to this run</a></div>
<p id="run-identity"></p><div class="run-summary"><div><h3>Change</h3><p id="run-change"></p></div>
<div><h3>Results</h3><p id="run-result"></p></div></div>
<details><summary>Runtime, configuration, coverage and provenance</summary><dl id="run-metadata"></dl></details>
<nav id="run-links" aria-label="Selected run evidence"></nav>
<h3>All planned tasks</h3><p class="note">Pi subprocess start → exit, not GPU time.
Tokens accumulate every model request, including repeated contexts; input tokens are prefill workload, not measured prefill speed.
Retries are separate attempts and never replace the planned cohort.</p>
<div class="table-scroll"><table id="tasks"><caption>Selected run · all planned tasks and attempt 1</caption>
<thead><tr><th scope="col">Task / attempt</th><th scope="col">End-to-end s</th><th scope="col">Input / prefill tokens</th>
<th scope="col">Generated tokens</th><th scope="col">Status / official check</th><th scope="col">Source</th></tr></thead>
<tbody></tbody><tfoot></tfoot></table></div><p id="reported-totals" class="note"></p></section>
<noscript><p>Interactive charts require local JavaScript. No network libraries are used.
<a href="trends-data.json">Normalized runs and all task rows</a> ·
<a href="experiments.html">Complete experiment notebook and evidence</a>.</p></noscript>
<p class="note"><a href="trends-data.json">Normalized data</a> ·
<a href="https://github.com/LittleLittleCloud/Infer.NET/blob/main/docs/HOME-TRENDS-DATA.md">Data contract</a></p>
</div><script id="trend-data" type="application/json">{payload}</script>
<script src="trends.js" defer></script>'''
