"""Shared, CPU-testable experiment #103 evidence and safety primitives."""
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

MAIN = "4f51f241e87999c8cc3f79d733cd95ade9c711cd"
ORIGINAL = "cd6f7da401fadf044dac13e87b415d27bb5eadfb"
REVISION = "e06a74e6236a60c8367e1a3214e83d8b61b637b0"
TOOLS_SHA = "349400b92b9de5efb88d73ce8cf5931f94f9e7b7"
TOOLS = {
    "benchmark-pi-humaneval.py": "594f60571fabc5d05d7aa849cfd25ad668a3ee8fed7c5f278b6a7ea47b9d9c22",
    "summarize-pi-humaneval-performance.py": "ee47653bedd02c1770eb6a1091266c13dd73f0236ec5dc69be667277c0169df6",
    "eval-pi-humaneval-docker.py": "89b8cc5f63819146d986490fd725a2cf79da240d8b6a2e343e42aab720baddef",
}
ADAPTED_REFERENCES = {
    "scripts/observe-dflash-e2e.py": "ca15a4fbfee2a6a5e85d904c902b5eb1a8778eb16341159da0f527bb9fc3f77b",
    "artifacts/dual-host/issue98-dflash8-e2e/control/run-human.py": "b6553a79009d0fd819308fd73e1973dff8494b16cff9b8d9bb2afcff0c26ef69",
    "artifacts/dual-host/issue98-dflash8-e2e/control/run-e2e.py": "d7b428bfc0af292eb1d97aab365552e1247b16f6ce2f530736b6f08029b0f7ce",
    "artifacts/dual-host/issue98-dflash8-e2e/control/resource_admission.py": "7ea52aff64943013c4a84ddeb6f9e30b752c97fe719d83d0ff47d8a7b1a1fd0d",
}
DATASET = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
IMAGE = "python@sha256:5fb4bc53dd0180955e31e5245b648bbb24a4ac4ab03a1de81513642325220bcd"
PILOT = [156, 94, 72, 80, 149]
COHORT = PILOT + [2, 35, 32, 146, 137, 97, 154, 6, 161, 115, 131, 162, 50, 47, 148]
SHAPES = [(21, 128), (8192, 128), (32768, 128), (8192, 4096), (32768, 4096)]
FIXTURES = {
    21: ("73b47cdbb8a53ae09c1e86cec80ac6c8f52c2364ae7f8c40f2f26e1165694961",
         "957872369108c3bb32e3e1486e8ea4935da3e7ea55eb9514f498ae24f177137f"),
    8192: ("defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d",
           "555093831eb803da2167f605c055896173cac47a71c83e3f7b4861aae0249168"),
    32768: ("4f2dfa83eb5f48eda2c8106a775d42d346d9dc72fa210c179c4db285b486f7f8",
            "96c49b4fbf11b98c1abc279a9b07821b92196075a3b106198c56755c7b932309"),
}
METAL_LIMIT = int(38.5 * 2**30)
SWAP_LIMIT = 512 * 2**20
PACKED_BYTES = 540344320
POWER_POLICIES = {
    "stable-140w": {"adapterWatts": 140, "initialMinSoc": None, "minSoc": None, "maxSocDropPoints": 1},
    "battery-assisted-35w": {"adapterWatts": 35, "initialMinSoc": 99, "minSoc": 80, "maxSocDropPoints": 20},
}
BATTERY_ASSISTED_WARNING = "Battery-assisted 35W is exploratory, not stable-140W acceptance."


class GateError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise GateError(message)


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def instant(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def append(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")


def rows(path, partial=False):
    if not Path(path).exists():
        return []
    text = Path(path).read_text()
    lines = text.splitlines()
    if partial and text and not text.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def new_output(path, repo):
    path = Path(path).absolute()
    artifacts = (repo / "artifacts").resolve()
    require(artifacts in path.resolve().parents, "Evidence must stay under this checkout's ignored artifacts/")
    ignored = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=repo).returncode == 0
    require(ignored, "Evidence path is not Git-ignored")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(mode=0o700, exist_ok=False)
    os.chmod(path, 0o700)
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def command(args, timeout=30):
    return subprocess.check_output([str(x) for x in args], text=True, stderr=subprocess.PIPE, timeout=timeout)


def http(url, token, body=None, timeout=5):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-DeepInfer-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as ex:
        return ex.code, json.loads(ex.read() or "{}")


def free_ports(ports):
    require(len(set(ports)) == len(ports), "Duplicate experiment ports")
    for port in ports:
        require(1024 <= port <= 65535, "Invalid experiment port")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))


def identity(pid):
    require(isinstance(pid, int) and pid > 1, "Invalid process PID")
    data = {}
    for key in ("comm", "lstart", "args"):
        data[key] = command(["ps", "-ww", "-p", pid, "-o", key + "="]).strip()
    require(all(data.values()), "Process identity unavailable")
    return {"pid": pid, **data}


def same_identity(expected, current):
    return all(expected[k] == current[k] for k in ("pid", "comm", "lstart", "args"))


def verify_original_identity(current, exe, health):
    require(current["pid"] == health["pid"], "Health/process PID mismatch")
    require(current["comm"] == str(exe), "Original process executable mismatch")
    require(current["args"] == str(exe) or current["args"].startswith(str(exe) + " "),
            "Original command mismatch")
    require("split-serve" in current["args"], "Original PID is not split-serve")
    require(health.get("build", "").split("+")[-1] == ORIGINAL, "Original build mismatch")


def wait_pid_gone(expected, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = subprocess.run(["ps", "-p", str(expected["pid"]), "-o", "lstart="],
                               text=True, capture_output=True)
        if probe.returncode or probe.stdout.strip() != expected["lstart"]:
            return
        time.sleep(.5)
    raise GateError("Identified original PID did not exit after bootout")


def stop_owned(process, expected, probe=identity, timeout=45):
    if process.poll() is not None:
        return
    require(same_identity(expected, probe(process.pid)), "Child identity changed; refusing signal")
    first = signal.SIGINT if any(str(x).endswith("benchmark-pi-humaneval.py") for x in process.args) else signal.SIGTERM
    process.send_signal(first)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        require(same_identity(expected, probe(process.pid)), "Child identity changed before escalation")
        process.kill()
        process.wait(timeout=10)


class Children:
    def __init__(self, output):
        self.output, self.items, self.logs = Path(output), [], []

    def start(self, args, log, **kwargs):
        handle = Path(log).open("x")
        self.logs.append(handle)
        process = subprocess.Popen([str(x) for x in args], stdout=handle, stderr=subprocess.STDOUT, **kwargs)
        try:
            deadline, previous, stable_samples = time.monotonic() + 2, None, 0
            while True:
                # Do not poll/reap during sampling: a launcher can exec, but
                # this unreaped Popen child's PID cannot have been reused.
                record = identity(process.pid)
                require(record["pid"] == process.pid and
                        (previous is None or record["lstart"] == previous["lstart"]),
                        "Child PID/start identity changed during startup")
                require(time.monotonic() < deadline, "Child identity did not stabilize")
                stable_samples = stable_samples + 1 if previous and same_identity(previous, record) else 1
                if stable_samples == 3:
                    break
                previous = record
                time.sleep(.05)
        except (GateError, subprocess.SubprocessError):
            if process.poll() is None:
                # The Popen child has not been reaped; its PID cannot have
                # been reused. Do not lose ownership on an identity-probe error.
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise
            append(self.output / "children.jsonl", {"startedUtc": stamp(), "pid": process.pid,
                                                   "earlyExitCode": process.returncode})
            return process
        self.items.append((process, record))
        append(self.output / "children.jsonl", {"startedUtc": stamp(), **record})
        return process

    def cleanup(self):
        failures = []
        for process, record in reversed(self.items):
            try:
                stop_owned(process, record)
            except (OSError, GateError, subprocess.SubprocessError) as ex:
                failures.append(type(ex).__name__)
        self.items = [(p, r) for p, r in self.items if p.poll() is None]
        for handle in self.logs:
            handle.close()
        self.logs = []
        require(not failures, "Owned-child cleanup incomplete: " + ",".join(failures))


def vm_snapshot():
    raw = command(["vm_stat"])
    page = int(re.search(r"page size of (\d+)", raw)[1])

    def size(key):
        return int(re.search(re.escape(key) + r":\s+(\d+)", raw)[1]) * page

    return {"utc": stamp(), "swapOutBytes": size("Swapouts"),
            "availableBytes": sum(size(k) for k in ("Pages free", "Pages inactive", "Pages speculative")),
            "wiredBytes": size("Pages wired down"), "compressorBytes": size("Pages occupied by compressor"),
            "raw": raw}


def model_processes():
    result = []
    for line in command(["ps", "-ww", "-axo", "pid=,comm=,args="]).splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid, exe, args = parts
        name = Path(exe).name
        if (name == "deepinfer" and any(k in args for k in ("split-serve", "verify-qwen36", "demo-qwen36")) or
            name in ("splash", "lmhead-driver", "backend-benchmark", "decode-profile", "target-block-checks") or
            name == "dotnet" and any(k in args for k in ("testhost.dll", "TargetBlockBench", "Dflash"))):
            result.append({"pid": int(pid), "name": name})
    return result


def wait_quiescent(path, timeout=180):
    stable = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = vm_snapshot()
        row["modelProcesses"] = model_processes()
        eligible = (not row["modelProcesses"] and row["availableBytes"] >= 35 * 2**30 and
                    row["wiredBytes"] <= 6 * 2**30 and row["compressorBytes"] <= 4 * 2**30)
        row["eligible"] = eligible
        append(path, row)
        stable = stable + 1 if eligible else 0
        if stable == 3:
            return
        time.sleep(1)
    raise GateError("Model/resource quiescence admission failed")


def power_snapshot():
    batt = command(["pmset", "-g", "batt"])
    profile = json.loads(command(["system_profiler", "SPPowerDataType", "-json"], timeout=45))["SPPowerDataType"]
    battery = next(v for v in profile if v.get("_name") == "spbattery_information")["sppower_battery_charge_info"]
    adapter = next(v for v in profile if v.get("_name") == "sppower_ac_charger_information")
    return {"utc": stamp(), "onAC": "AC Power" in batt,
            "adapterWatts": int(adapter.get("sppower_ac_charger_watts", 0)),
            "adapterIdentity": {k: v for k, v in adapter.items() if any(term in k for term in (
                "serial", "manufacturer", "family", "identifier", "charger_id", "charger_name"))},
            "soc": int(battery["sppower_battery_state_of_charge"]), "batteryRaw": batt,
            "charging": battery.get("sppower_battery_is_charging"), "thermal": command(["pmset", "-g", "therm"]),
            "custom": command(["pmset", "-g", "custom"])}


def thermal_gate(raw):
    for name, value in re.findall(
            r"\b(CPU_Speed_Limit|(?:CPU_)?Scheduler_Limit)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", raw, re.IGNORECASE):
        require(float(value) >= 100, f"Thermal/power limit reported: {name}={value}")
    for value in re.findall(
            r"\bThermal(?:[ _]+Warning(?:[ _]+Level)?|[ _]+Level)\s*[:=]\s*([^\r\n,;}]+)",
            raw, re.IGNORECASE):
        value = value.strip()
        require(value.lower() in ("0", "0.0", "none", "normal", "nominal", "no warning"),
                "Thermal warning reported: " + value)


def power_gate(current, initial=None, policy="stable-140w"):
    thresholds = POWER_POLICIES[policy]
    initial = current if initial is None else initial
    require(current["onAC"] and current["adapterWatts"] == thresholds["adapterWatts"],
            f"Require verified {thresholds['adapterWatts']}W AC supply throughout")
    require(current.get("adapterIdentity") == initial.get("adapterIdentity"), "Power adapter identity changed")
    if thresholds["initialMinSoc"] is not None:
        require(initial["soc"] >= thresholds["initialMinSoc"], "Initial battery SOC must be at least 99 percent")
    if thresholds["minSoc"] is not None:
        require(current["soc"] >= thresholds["minSoc"], "Battery SOC must remain at least 80 percent")
    require(current["soc"] >= initial["soc"] - thresholds["maxSocDropPoints"],
            f"Battery declined by more than {thresholds['maxSocDropPoints']} percentage points")
    thermal_gate(current.get("thermal", ""))


def high_power_evidence(value, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    require(value.get("energyModeControls") == ["High Power", "High Power"], "Fresh UI High Power evidence required")
    require(value.get("verifiedBy") == "parent" and value.get("source"), "Require parent UI verification provenance")
    age = (now - instant(value["utc"])).total_seconds()
    require(0 <= age <= 1800, "High Power evidence must be from the last 30 minutes at admission")


def metric_from_observation(row, build):
    result = row.get("workerResult") or {}
    timing = result.get("timings") or {}
    return {"requestId": result.get("requestId"), "completedAt": row.get("workerCompletedUtc", row["completedUtc"]), "build": build,
            "executionPath": result.get("executionPath"), "kv": "fp32",
            "inputTokens": row.get("inputTokens"), "outputTokens": row.get("outputTokens"),
            "decodeIntervals": timing.get("decodeTokens"), "prefillMs": timing.get("prefillMs"),
            "decodeMs": timing.get("decodeMs")}


def metric_problems(m):
    problems = []
    n, output = m.get("inputTokens"), m.get("outputTokens")
    if type(n) is not int or n <= 0 or type(output) is not int or output < 0:
        problems.append("missing/invalid tokens")
    if type(output) is int and (type(m.get("decodeIntervals")) is not int or
                               m["decodeIntervals"] != max(0, output - 1)):
        problems.append("N-1 mismatch")
    for key in ("prefillMs", "decodeMs"):
        value = m.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            problems.append("missing/invalid " + key)
        elif key == "prefillMs" and value <= 0 or key == "decodeMs" and (m.get("decodeIntervals") or 0) > 0 and value <= 0:
            problems.append("nonpositive " + key)
    if not m.get("requestId"):
        problems.append("missing request identity")
    if m.get("executionPath") != "metal-local":
        problems.append("not metal-local")
    return problems


def aggregate(metrics, complete=True):
    complete = complete and bool(metrics) and not any(metric_problems(m) for m in metrics)
    if not complete:
        return {"complete": False, "inputTokens": None, "outputTokens": None, "decodeIntervals": None,
                "prefillMs": None, "decodeMs": None, "prefillTokensPerSecond": None, "decodeTokensPerSecond": None}
    n = sum(m["inputTokens"] for m in metrics)
    o = sum(m["outputTokens"] for m in metrics)
    intervals = sum(m["decodeIntervals"] for m in metrics)
    prefill = sum(m["prefillMs"] for m in metrics)
    decode = sum(m["decodeMs"] for m in metrics)
    return {"complete": True, "inputTokens": n, "outputTokens": o, "decodeIntervals": intervals,
            "prefillMs": prefill, "decodeMs": decode, "prefillTokensPerSecond": n / (prefill / 1000),
            "decodeTokensPerSecond": intervals / (decode / 1000) if decode > 0 else None}


def reconcile(tasks, observations, build):
    selected, issues, task_results = [], [], []
    assigned = set()
    for task in tasks:
        clocks = [r for r in observations if instant(task["start_utc"]) <=
                  instant(r.get("workerCompletedUtc", r["completedUtc"])) <= instant(task["end_utc"])]
        expected = task["pi"]["requests"]
        if len(clocks) != len(expected):
            issues.append(task["task_id"] + ": coverage mismatch")
        for row, pi in zip(clocks, expected):
            m = metric_from_observation(row, build)
            issues.extend(task["task_id"] + ": " + p for p in metric_problems(m))
            if row.get("outcome") != "completed" or pi.get("error"):
                issues.append(task["task_id"] + ": request failed")
            if (m["inputTokens"], m["outputTokens"]) != (pi["usage"].get("input"), pi["usage"].get("output")):
                issues.append(task["task_id"] + ": Pi/worker token mismatch")
            if row["id"] in assigned:
                issues.append("multiply assigned observer request")
            assigned.add(row["id"])
            selected.append(m)
        declared = task["pi"].get("usage") or {}
        for field in ("input", "output"):
            total = sum((p.get("usage") or {}).get(field, 0) or 0 for p in expected)
            if field in declared and declared[field] != total:
                issues.append(task["task_id"] + ": Pi cumulative token mismatch")
        if not task.get("completed") or task.get("timeout") or task.get("error"):
            issues.append(task["task_id"] + ": incomplete task retained")
        task_results.append({"taskId": task["task_id"], "expected": len(expected), "observed": len(clocks)})
    if len({m["requestId"] for m in selected}) != len(selected):
        issues.append("duplicate worker request ID")
    return {"problems": issues, "tasks": task_results, "metrics": aggregate(selected, not issues)}
