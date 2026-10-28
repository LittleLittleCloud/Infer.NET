#!/usr/bin/env python3
"""Bounded #103 controller. Default: read-only preflight; never builds packages."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from issue103_common import (
    ADAPTED_REFERENCES, Children, COHORT, DATASET, FIXTURES, GateError, IMAGE, MAIN, METAL_LIMIT,
    PACKED_BYTES, PILOT, REVISION, SHAPES, SWAP_LIMIT, TOOLS, TOOLS_SHA, append, command,
    free_ports, high_power_evidence, http, identity, instant, metric_from_observation,
    metric_problems, model_processes, new_output, power_gate, power_snapshot, read_json,
    reconcile, require, rows, same_identity, save, sha, stamp, verify_original_identity,
    vm_snapshot, wait_pid_gone, wait_quiescent,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE84 = Path("/Users/littlecloud/.codex/worktrees/experiment-84-prefill-delivery/Infer.NET")
REFERENCE98 = Path("/Users/littlecloud/.codex/worktrees/experiment-98-dflash8-e2e/Infer.NET")
DATA_ROOT = Path("/Users/littlecloud/repos/Infer.NET/artifacts")
MODEL = "qwen3.6-35b-a3b"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package(exe, expected):
    exe = exe.resolve()
    root = exe.parent
    manifest = read_json(root / "manifest.json")
    require(manifest["source"] == expected and manifest["rid"] == "osx-arm64", "Package source/RID mismatch")
    require(manifest["modelRevision"] == REVISION and manifest["weightBits"] == 8 and manifest["kv"] == "fp32",
            "Package model/precision mismatch")
    require(all(name in manifest["files"] for name in ("deepinfer", "libdeepinfer_metal.dylib")),
            "Package executable/native hashes missing")
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        require(root in path.parents and path.is_file() and sha(path) == digest, "Immutable package hash mismatch")
    version = command([exe, "version"]).strip()
    require(expected in version, "Actual executable version mismatch")
    return {"source": expected, "version": version, "manifestSha256": sha(root / "manifest.json"),
            "executableSha256": sha(exe), "nativeSha256": sha(root / "libdeepinfer_metal.dylib")}


def environment(arm):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("DEEPINFER_", "DYLD_", "DOTNET_"))}
    env.update({"DEEPINFER_METAL_KV": "fp32", "DEEPINFER_METAL_PREFILL_CHUNK": "32",
                "DEEPINFER_METAL_PREFILL_GEMM": "fp32", "DEEPINFER_METAL_PREFILL_MATRIX": "0",
                "DEEPINFER_METAL_PREFILL_DELTA": "sequential", "DOTNET_GCHeapHardLimit": "0x40000000"})
    if arm != "main":
        env["DEEPINFER_METAL_LMHEAD"] = "tile4" if arm == "on" else "off"
    return env


def selector_evidence(log, arm):
    entries = [line for line in log.splitlines() if line.startswith("LMHEAD ")]
    if arm == "main":
        require(not entries, "Actual main must not contain candidate LMHEAD metadata")
        return {"requested": "absent", "effective": "original-main", "extraPayloadBytes": 0,
                "evidence": "Verified immutable main source/executable/native; no candidate selector exists"}
    require(len(entries) == 1, "Require exactly one native LMHEAD selector record")
    values = dict(re.findall(r"(\w+)=([^\s]+)", entries[0]))
    selected = "tile4" if arm == "on" else "off"
    require(values.get("requested") == selected and values.get("effective") == selected, "Effective LM head selection mismatch")
    require(int(values["extraPayloadBytes"]) == (PACKED_BYTES if arm == "on" else 0), "LM head payload mismatch")
    require(int(values["scratchBytes"]) == 0 and values.get("batchedPrefill") == "original", "LM head scope mismatch")
    require(int(values["extraAllocatedBytes"]) >= int(values["extraPayloadBytes"]), "LM head allocation accounting")
    if arm == "off":
        require(int(values["extraAllocatedBytes"]) == 0 and int(values["transientExtraPeakBytes"]) == 0, "Off allocation not zero")
    else:
        require(values.get("layout") == "tile4-block256-lane32-octet8-v1" and
                [values.get(k) for k in ("N", "K", "group", "bits")] == ["248320", "2048", "64", "8"],
                "LM head layout/geometry mismatch")
        require(int(values["transientExtraPeakBytes"]) == int(values["extraAllocatedBytes"]), "Unexpected repack allocation")
    return values


def native_profile(log):
    lines = [line for line in log.splitlines() if line.startswith("METAL_NATIVE_PROFILE ")]
    require(len(lines) == 1, "Missing/duplicate effective native profile")
    text = lines[0]
    for field in ("KvElementBytes = 4", "Bf16Gemm = False", "Matrix = False",
                  "ChunkedDelta = False", "ScratchTokens = 128"):
        require(field in text, "Effective native FP32/chunk profile mismatch")
    require("RESIDENCY weightsBytes=" in log, "Native residency-on evidence missing")
    return text


def official_controls(module, dataset, out):
    stage = out / "official-controls"
    stage.mkdir()
    task = dataset["HumanEval/156"]
    wrong = f"def {task['entry_point']}(*args, **kwargs): return None\n" + task["test"] + f"\ncheck({task['entry_point']})\n"
    positive = module.run_case("assert True\n", stage)
    negative = module.run_case(wrong, stage)
    save(out / "official-controls.json", {"utc": stamp(), "positive": positive, "negative": negative, "image": IMAGE})
    require(positive["status"] == "passed" and positive["exit_code"] == 0, "Docker positive execution control failed")
    require(negative["status"] == "failed" and negative["exit_code"] == 1 and
            "AssertionError" in negative["stderr"] and "/work/case.py" in negative["stderr"],
            "Negative must be an executed Python assertion, not a Docker launch failure")


class Controller:
    def __init__(self, args, out):
        self.args, self.out = args, out
        self.children = Children(out)
        self.worker = None
        self.stop = threading.Event()
        self.monitor_error = None
        self.initial_power = None
        self.attempted_bootout = False
        self.restored = False
        self.references = {}
        self.checked = set()
        self.observation_phases = {}
        self.observation_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.active_out = None
        self.worker_out = None
        self.power_last = 0
        self.observer_port, self.clone_port, self.inner_port, self.data_port, self.worker_port, self.registry_port = range(args.base_port, args.base_port + 6)
        self.endpoint = f"http://127.0.0.1:{self.observer_port}"
        self.private = args.config_dir.resolve()
        self.coordinator = read_json(self.private / "coordinator.json")
        self.original_config = read_json(self.private / "worker.json")
        self.token_file = Path(self.coordinator["token-file"])
        self.token = self.token_file.read_text().strip()
        require(len(self.token) >= 32, "Invalid private authentication")
        require(Path(self.original_config["token-file"]).read_text().strip() == self.token, "Configuration token mismatch")
        self.plist = args.plist.expanduser().resolve()
        definition = plistlib.loads(self.plist.read_bytes())
        require(definition["Label"] == "com.infernet.metal", "Wrong original LaunchAgent")
        self.original_exe = Path(definition["ProgramArguments"][0]).resolve()
        require("--config" in definition["ProgramArguments"] and
                str(self.private / "worker.json") in definition["ProgramArguments"], "Original plist config mismatch")
        self.domain = f"gui/{os.getuid()}"
        self.label = self.domain + "/com.infernet.metal"
        self.tracked = {"plist": self.plist, "config": self.private / "worker.json", "binary": self.original_exe,
                        "native": self.original_exe.parent / "libdeepinfer_metal.dylib",
                        "coordinatorConfig": self.private / "coordinator.json"}
        for key in ("token-file", "tls-cert", "tls-password-file"):
            if key in self.original_config:
                self.tracked[key] = Path(self.original_config[key])
        self.original_hashes = {k: sha(v) for k, v in self.tracked.items()}
        self.files = {21: args.fixtures / "21.txt",
                      8192: DATA_ROOT / "baselines/qwen36-context-fixed-data/8192-4096-prompt.txt",
                      32768: DATA_ROOT / "baselines/qwen36-context-large-fixed-data/32768-8192-prompt.txt"}
        self.packages = {}

    def health(self, original=False):
        url = self.original_config["listen"] if original else f"http://127.0.0.1:{self.worker_port}"
        code, h = http(url + "/health", self.token)
        require(code == 200, "Health unavailable")
        return h

    def preflight(self):
        require(re.fullmatch("[0-9a-f]{40}", self.args.candidate_sha) is not None, "Require full candidate SHA")
        require(self.args.candidate_sha != MAIN, "Candidate cannot be relabeled main")
        self.packages["main"] = package(self.args.main_package, MAIN)
        self.packages["candidate"] = package(self.args.candidate_package, self.args.candidate_sha)
        source = self.args.candidate_source.resolve()
        head = command(["git", "-C", source, "rev-parse", "HEAD"]).strip()
        dirty = command(["git", "-C", source, "status", "--porcelain"]).strip()
        require(head == self.args.candidate_sha, "Candidate source checkout differs from package")
        require(command(["git", "-C", source, "merge-base", MAIN, head]).strip() == MAIN, "Candidate is not main-derived")
        if self.args.execute:
            require(source == ROOT, "Execute the controller from the frozen candidate source checkout")
            require(not dirty, "Freeze source before execution; never build dirty packages")
        require(command(["git", "-C", self.args.tools, "rev-parse", "HEAD"]).strip() == TOOLS_SHA, "Independent tools checkout changed")
        for name, digest in TOOLS.items():
            require(sha(self.args.tools / "scripts" / name) == digest, "Independent harness hash changed")
        require(sha(self.args.dataset) == DATASET, "Dataset hash mismatch")
        self.dataset = {r["task_id"]: r for r in map(json.loads, gzip.decompress(self.args.dataset.read_bytes()).splitlines())}
        require(len(self.dataset) == 164, "Dataset task coverage")
        require(command([self.args.pi, "--version"]).strip() == "1.4.0", "Pi version mismatch")
        command(["docker", "image", "inspect", IMAGE], timeout=15)
        fixture_evidence = {}
        for count, path in self.files.items():
            manifest = read_json(self.args.fixtures / str(count) / "fixture.json")
            require(manifest == {"inputTokens": count, "textSha256": FIXTURES[count][0], "tokenIdsSha256": FIXTURES[count][1]},
                    "Pinned fixture manifest changed")
            require(sha(path) == FIXTURES[count][0], "Fixed prompt hash mismatch")
            fixture_evidence[count] = {**manifest, "manifestSha256": sha(self.args.fixtures / str(count) / "fixture.json")}
        parent_record = read_json(self.args.original_preflight)
        for key in ("plist", "config", "binary", "native"):
            require(parent_record["hashes"][key] == self.original_hashes[key], "Original recovery identity differs from parent record")
        h = self.health(original=True)
        current = identity(h["pid"])
        verify_original_identity(current, self.original_exe, h)
        launch = command(["launchctl", "print", self.label])
        require(re.search(r"\bpid = " + str(h["pid"]) + r"\b", launch), "LaunchAgent PID/health mismatch")
        processes = model_processes()
        require(all(p["pid"] == h["pid"] for p in processes), "Another model process is present")
        for key in ("build", "pid"):
            require(h[key] == parent_record["health"][key], "Original service identity changed since parent preflight")
        self.original_health, self.original_identity = h, current
        free_ports(list(range(self.args.base_port, self.args.base_port + 6)))
        # Model hashes come from the already-pinned main source; read no weight contents into logs.
        pinned = command(["git", "show", MAIN + ":src/DeepInfer.Host/PinnedSplitModel.cs"], timeout=30)
        model_hashes = dict(re.findall(r'\["([^"]+)"\] = "([0-9a-f]{64})"', pinned))
        require(len(model_hashes) == 11, "Pinned model checksum coverage changed")
        model_dir = Path(self.original_config["weights"])
        for name, digest in model_hashes.items():
            require(sha(model_dir / name) == digest, "Pinned model file checksum mismatch")
        save(self.out / "model-hashes.json", {"revision": REVISION, "files": model_hashes})
        if self.args.native_driver:
            require(self.args.native_driver_source == self.args.candidate_sha and
                    sha(self.args.native_driver) == self.args.native_driver_sha256, "Native gate driver provenance mismatch")
        observed_power = power_snapshot()
        supply_ok = observed_power["onAC"] and observed_power["adapterWatts"] == 140
        ui_ok = False
        if self.args.high_power_evidence:
            try:
                high_power_evidence(read_json(self.args.high_power_evidence))
                ui_ok = True
            except GateError:
                if self.args.execute:
                    raise
        report = {"utc": stamp(), "mode": "execute-admission" if self.args.execute else "non-mutating-preflight",
                  "phase": self.args.phase, "packages": self.packages, "candidateSourceClean": not dirty,
                  "sourceSha": head, "fixtures": fixture_evidence, "toolsCommit": TOOLS_SHA, "toolHashes": TOOLS,
                  "piVersion": "1.4.0", "piEntrySha256": sha(self.args.pi.resolve()), "datasetSha256": DATASET,
                  "originalHealth": h, "originalHashes": self.original_hashes,
                  "power": observed_power, "supply140W": supply_ok, "freshHighPowerEvidence": ui_ok,
                  "blockers": ([] if supply_ok else ["140W AC supply not verified"]) +
                              ([] if ui_ok else ["fresh parent High Power evidence missing"]) +
                              ([] if not dirty else ["candidate checkout not frozen"]),
                  "limitations": ["Streaming API exposes no full logits or internal states. Independent native operator plus exact deterministic full-model checks only.",
                                  "Input token-ID hashes are pinned historical manifests; API validates actual input counts, not full input-ID hashes.",
                                  "No physical GPU residency high-water query; device allocation observations are not physical-residency proof."]}
        save(self.out / "preflight.json", report)
        self.evidence_identity = {
            "packages": self.packages, "sourceSha": head, "toolsCommit": TOOLS_SHA, "toolHashes": TOOLS,
            "datasetSha256": DATASET, "fixtures": fixture_evidence,
            "harnessSha256": {p.name: sha(p) for p in (ROOT / "scripts").glob("*lmhead*.py")} |
                             {"issue103_common.py": sha(ROOT / "scripts/issue103_common.py")},
            "nativeDriverSha256": self.args.native_driver_sha256,
            "adaptedReferenceSha256": ADAPTED_REFERENCES,
        }
        save(self.out / "plan.json", {
            "schema": "issue103-controller-v1", "createdUtc": stamp(), "main": MAIN, "candidate": self.args.candidate_sha,
            "phase": self.args.phase, "fixedShapes": SHAPES, "fixedOrders": ["main,on", "on,main", "main,on", "on,main", "main,on"],
            "warmupsPerArmShape": 1, "formalPerArmShape": 3, "pilotOrder": ["main", "off", "on"],
            "cohortOrder": ["on", "off", "main"], "pilotIds": PILOT, "cohortIds": COHORT,
            "piTimeoutSeconds": 300, "fixedRequestTimeoutSeconds": 3600, "sameObserverHops": 2,
            "armEnvironments": {arm: {k: v for k, v in environment(arm).items() if k.startswith(("DEEPINFER_", "DOTNET_"))}
                                for arm in ("main", "off", "on")},
            "identity": self.evidence_identity,
        })
        if self.args.execute:
            require(supply_ok and ui_ok, "Power admission not satisfied")
            self.initial_power = observed_power
            require(self.args.exclusive_gpu and self.args.clients_paused, "Parent-exclusive GPU scheduling and paused clients required")
            require(self.args.phase != "preflight", "Choose an explicit execution phase")
            require(self.args.native_driver is not None, "Frozen native correctness driver required")
            require(self.args.high_power_evidence is not None, "High Power evidence required")
        return report

    def idle_original(self):
        for _ in range(3):
            h = self.health(original=True)
            verify_original_identity(identity(h["pid"]), self.original_exe, h)
            require(h["pid"] == self.original_identity["pid"] and h["state"] not in ("Busy", "Loading", "Warming", "Faulted"),
                    "Original worker not idle")
            code, cap = http(self.original_config["listen"] + "/v1/capabilities", self.token)
            telemetry = cap.get("telemetry") or {}
            require(code == 200 and telemetry.get("busy") is False and telemetry.get("queueDepth") == 0,
                    "Original active/queued requests not proven idle")
            require(h["completedRequests"] == self.original_health["completedRequests"], "Original request count changed")
            append(self.out / "original-idle.jsonl", {"utc": stamp(), "health": h, "telemetry": telemetry})
            time.sleep(1)

    def unchanged_original(self):
        require({k: sha(p) for k, p in self.tracked.items()} == self.original_hashes, "Original files changed; recovery identity failure")

    def pause(self):
        self.idle_original()
        self.unchanged_original()
        free_ports(list(range(self.args.base_port, self.args.base_port + 6)))
        require(all(p["pid"] == self.original_identity["pid"] for p in model_processes()),
                "Another model appeared before pausing the original")
        for key, exe, expected in (("main", self.args.main_package, MAIN),
                                   ("candidate", self.args.candidate_package, self.args.candidate_sha)):
            require(package(exe, expected) == self.packages[key], "Package changed before bootout")
        high_power_evidence(read_json(self.args.high_power_evidence))
        self.check_power()
        require(same_identity(self.original_identity, identity(self.original_identity["pid"])), "Original identity changed before bootout")
        save(self.out / "recovery-plan.json", {
            "utc": stamp(), "label": self.label, "domain": self.domain, "plist": str(self.plist),
            "identity": self.original_identity, "hashes": self.original_hashes,
            "tracked": {k: str(v) for k, v in self.tracked.items()}, "originalHealth": self.original_health,
            "bootstrap": ["launchctl", "bootstrap", self.domain, str(self.plist)],
            "restoreOriginalDefinitionOnly": True, "coordinatorAndWindowsUntouched": True,
            "readinessMustBeVerified": True, "preexistingReadyFalse": not self.original_health["ready"],
        })
        self.attempted_bootout = True
        append(self.out / "lifecycle.jsonl", {"utc": stamp(), "action": "bootout-intent", "pid": self.original_identity["pid"]})
        subprocess.run(["launchctl", "bootout", self.label], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        wait_pid_gone(self.original_identity)
        wait_quiescent(self.out / "pause-admission.jsonl")

    def check_power(self):
        value = power_snapshot()
        append(self.out / "power.jsonl", value)
        power_gate(value, self.initial_power)

    def monitor(self):
        try:
            while not self.stop.wait(.5):
                row = vm_snapshot()
                with self.state_lock:
                    if self.worker:
                        require(self.worker.poll() is None, "Worker exited/crashed during evaluation")
                        row["workerPid"] = self.worker.pid
                        row["rssKiB"] = command(["ps", "-p", self.worker.pid, "-o", "rss="]).strip()
                append(self.out / "resources.jsonl", row)
                if time.monotonic() - self.power_last >= 15:
                    self.check_power()
                    append(self.out / "memory-pressure.jsonl",
                           {"utc": stamp(), "raw": command(["memory_pressure", "-Q"])})
                    owned = {p.pid for p, _ in self.children.items if p.poll() is None}
                    require(all(p["pid"] in owned for p in model_processes()),
                            "Unowned model process appeared during exclusive phase")
                    if self.active_out:
                        code, status = http(self.endpoint + "/observer/health", self.token)
                        require(code == 200 and status.get("ready") and not status.get("fatal"),
                                "Observer registration/resource coverage lost")
                    self.power_last = time.monotonic()
                active_out = self.active_out
                worker_out = self.worker_out
                if worker_out and (worker_out / "worker.log").exists():
                    log = (worker_out / "worker.log").read_text()
                    peaks = [int(x) for x in re.findall(
                        r"(?:sampledPeakBytes|submissionPeakBytes|metalAllocatedBytes|currentMetalBytes|projectedMetalBytes)=(\d+)", log)]
                    if peaks:
                        require(max(peaks) <= METAL_LIMIT, "Sampled Metal allocation exceeds 38.5 GiB")
                if active_out:
                    require(not (active_out / "observer/fatal.jsonl").exists(), "Observer resource coverage failed")
                    starts = rows(active_out / "observer/request-starts.jsonl", partial=True)
                    completed = {r["id"] for r in rows(active_out / "observer/requests.jsonl", partial=True)}
                    for started in starts:
                        if started["id"] not in completed:
                            key = (str(active_out), started["id"])
                            if self.observation_phases.get(key) != "warmup" and instant(row["utc"]) >= instant(started["vmBefore"]["utc"]):
                                delta = row["swapOutBytes"] - started["vmBefore"]["swapOutBytes"]
                                require(0 <= delta <= SWAP_LIMIT, "Live steady-request swap-out budget failure")
                    self.check_observations()
        except Exception as ex:
            self.monitor_error = ex
            append(self.out / "monitor-failure.jsonl", {"utc": stamp(), "type": type(ex).__name__,
                                                      "reason": str(ex) if isinstance(ex, GateError) else None})
            # Interrupt only controller-owned identities, so blocked HTTP reads
            # unblock promptly on an admission/resource failure.
            for process, record in list(self.children.items):
                if process.poll() is None:
                    try:
                        require(same_identity(record, identity(process.pid)), "Child identity changed on monitor failure")
                        process.send_signal(signal.SIGINT if any(str(x).endswith("benchmark-pi-humaneval.py") for x in process.args)
                                            else signal.SIGTERM)
                    except (OSError, GateError, subprocess.SubprocessError) as cleanup_error:
                        append(self.out / "monitor-cleanup-errors.jsonl", {"utc": stamp(), "pid": process.pid,
                                                                         "errorType": type(cleanup_error).__name__})

    def healthy(self):
        if self.monitor_error:
            raise self.monitor_error
        if self.worker:
            require(self.worker.poll() is None, "Worker crashed")

    def check_observations(self):
        active_out = self.active_out
        if not active_out:
            return
        with self.observation_lock:
            for row in rows(active_out / "observer/requests.jsonl", partial=True):
                key = (str(active_out), row["id"])
                if key in self.checked:
                    continue
                warmup = self.observation_phases.get(key) == "warmup"
                require(row["newSwapOutBytes"] >= 0 and (warmup or row["newSwapOutBytes"] <= SWAP_LIMIT),
                        "Whole-system steady-request swap-out exceeds 512 MiB or counter reset")
                timing = (row.get("workerResult") or {}).get("timings") or {}
                if row.get("outcome") == "completed" or timing:
                    value = timing.get("metalAllocatedBytes")
                    require(isinstance(value, int) and 0 < value <= METAL_LIMIT, "Missing/excessive Metal allocation")
                append(self.out / "request-resource-gates.jsonl", {
                    "armDirectory": active_out.name, "id": row["id"], "excludedWarmup": warmup,
                    "newSwapOutBytes": row["newSwapOutBytes"],
                    "metalAllocatedBytes": timing.get("metalAllocatedBytes"), "outcome": row["outcome"]})
                self.checked.add(key)

    def wait(self, predicate, timeout, message):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.healthy()
            if predicate():
                return
            time.sleep(.25)
        raise GateError(message)

    def start_arm(self, arm, out):
        self.active_out = None
        wait_quiescent(out / "admission.jsonl")
        self.check_power()
        exe = self.args.main_package if arm == "main" else self.args.candidate_package
        expected = MAIN if arm == "main" else self.args.candidate_sha
        require(package(exe, expected) == self.packages["main" if arm == "main" else "candidate"], "Package changed after admission")
        config_dir = out / "private"
        config_dir.mkdir(mode=0o700)
        registry = f"http://127.0.0.1:{self.registry_port}"
        coor = dict(self.coordinator, listen=registry)
        coor.pop("log-file", None)
        save(config_dir / "coordinator.json", coor)
        save(config_dir / "clone.json", dict(coor, listen=f"http://127.0.0.1:{self.clone_port}"))
        conf = dict(self.original_config)
        # Fail rather than silently retaining unreviewed experiment options.
        require(not any(k.startswith(("dflash", "mtp", "speculat")) for k in conf), "Original config contains out-of-scope speculation options")
        conf.pop("log-file", None)
        conf.pop("windows-service", None)
        conf["allow-remote"] = False
        conf.update({"listen": f"http://127.0.0.1:{self.worker_port}",
                     "data-listen": f"tcp://127.0.0.1:{self.data_port}", "coordinator": registry,
                     "id": "issue103-" + arm, "lock-file": str(out / "model.lock"),
                     "short-prefill-tokens": "262144", "decode-attention": "fused-fp32",
                     "deadline-seconds": "1800", "metal-scratch-tokens": "128", "suffix-chunk-tokens": "128",
                     "metal-residency": "on", "residency-refresh-ms": "1000", "kv": "fp32", "queue-size": "0"})
        save(config_dir / "worker.json", conf)
        env = environment(arm)
        before = vm_snapshot()
        self.children.start([exe, "coordinator", "--config", config_dir / "coordinator.json"],
                            out / "registry.log", env=env)
        self.children.start([exe, "coordinator", "--config", config_dir / "clone.json"],
                            out / "clone.log", env=env)
        self.worker_out = out
        self.worker = self.children.start([exe, "split-serve", "--config", config_dir / "worker.json"],
                                          out / "worker.log", env=env)

        def ready():
            try:
                h = self.health()
                return h.get("ready") and h["pid"] == self.worker.pid
            except (OSError, ValueError):
                return False

        self.wait(ready, 900, "Worker load/warm readiness timeout")
        h = self.health()
        require(h["build"].split("+")[-1] == expected and h["revision"] == REVISION and h["kv"] == "fp32" and h["weightBits"] == 8,
                "Actual runtime model/build/precision mismatch")
        require(h.get("maxContextTokens", 0) >= max(n + m for n, m in SHAPES) and
                h.get("maxOutputTokens", 0) >= 4096, "Runtime capacity does not cover approved long fixed requests")
        log = (out / "worker.log").read_text()
        selector = selector_evidence(log, arm)
        profile = native_profile(log)
        # Fused decode is an explicit managed->native ABI selector, not getenv.
        require(conf["decode-attention"] == "fused-fp32", "Decode attention selector changed")
        save(out / "runtime.json", {"arm": arm, "health": h, "package": self.packages["main" if arm == "main" else "candidate"],
                                   "environment": {k: v for k, v in env.items() if k.startswith(("DEEPINFER_", "DOTNET_"))},
                                   "selector": selector, "nativeProfile": profile, "beforeLoad": before,
                                   "afterLoadWarmup": vm_snapshot(), "managedChunk": 32, "nativeScratch": 128,
                                   "decodeAttention": "fused-fp32", "routingThreshold": 262144,
                                   "modelHashEvidence": "../model-hashes.json"})
        observer = self.children.start([
            sys.executable, ROOT / "scripts/observe-lmhead-e2e.py", "--config-dir", config_dir,
            "--outer-port", self.observer_port, "--clone-port", self.clone_port, "--inner-port", self.inner_port,
            "--output", out / "observer", "--request-timeout", "3600"], out / "observer-control.log", env=env)

        def observer_ready():
            require(observer.poll() is None, "Observer crashed")
            try:
                code, status = http(self.endpoint + "/observer/health", self.token)
                if code != 200 or not status.get("ready") or status.get("fatal"):
                    return False
                code, registry = http(self.endpoint + "/v1/workers", self.token)
                return code == 200 and len(registry["workers"]) == 1 and registry["workers"][0].get("healthy")
            except (OSError, ValueError):
                return False

        self.wait(observer_ready, 60, "Two-hop observer readiness timeout")
        self.active_out = out
        return h

    def finish_arm(self):
        self.check_observations()
        self.healthy()
        self.active_out = None
        self.worker_out = None
        with self.state_lock:
            self.worker = None
        self.children.cleanup()
        self.check_power()

    def observer_complete(self, previous, timeout=20):
        path = self.active_out / "observer/requests.jsonl"
        self.wait(lambda: len(rows(path, partial=True)) == previous + 1, timeout, "Observer terminal coverage timeout")
        observed = rows(path)[-1]
        self.check_observations()
        return observed

    def trial(self, arm, n, m, trial, out, *, natural=False, cancel=False, label=None):
        self.healthy()
        self.check_power()
        previous = len(rows(self.active_out / "observer/requests.jsonl"))
        with self.observation_lock:
            key = (str(self.active_out), f"observer-{previous + 1:05d}")
            self.observation_phases[key] = "warmup" if trial == 0 else "steady"
            append(self.active_out / "observation-phases.jsonl",
                   {"id": key[1], "phase": self.observation_phases[key], "intentionalCancellation": cancel})
        body = {"prompt": self.files[n].read_text(), "maxTokens": m, "model": MODEL, "backend": "metal",
                "benchmarkIgnoreEos": not natural, "profile": True}
        if natural:
            body["prompt"] = "只回答一个字：好。不要解释。"
        req = urllib.request.Request(self.endpoint + "/v1/infer/stream", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-DeepInfer-Token": self.token})
        start, started = time.monotonic(), stamp()
        result, error, receipts, token_ids = None, None, [], []
        try:
            with urllib.request.urlopen(req, timeout=3600) as response:
                for line in response:
                    self.healthy()
                    require(time.monotonic() - start <= 3600, "Fixed request total deadline")
                    if not line.startswith(b"data:"):
                        continue
                    event = json.loads(line[5:])
                    if type(event.get("tokenId")) is int and event["tokenId"] >= 0:
                        receipts.append((time.monotonic() - start) * 1000)
                        token_ids.append(event["tokenId"])
                        if cancel:
                            break
                    if event.get("result"):
                        result = event["result"]
                    if event.get("error"):
                        error = True
        except Exception as ex:
            error = type(ex).__name__
        record = {"arm": arm, "inputTokens": None if natural else n, "outputLimit": m,
                  "trial": trial, "excludedWarmup": trial == 0, "naturalEos": natural,
                  "intentionalCancellation": cancel, "startedUtc": started, "endedUtc": stamp(),
                  "clientWallSeconds": time.monotonic() - start,
                  "clientTtftMs": receipts[0] if receipts else None, "nativeTokenReceiptMs": receipts,
                  "deliveredTokenIds": token_ids, "result": result, "error": error}
        # Remove decoded text/pieces; raw token evidence stays private.
        if result is not None:
            result = dict(result)
            result.pop("text", None)
            result.pop("tokens", None)
            record["result"] = result
        name = label or f"trial-{trial}"
        save(out / (name + ".json"), record)
        observed = self.observer_complete(previous)
        save(out / (name + "-observation.json"), observed)
        self.check_power()
        require(error is None, "Fixed/gate request error retained")
        if cancel:
            require(len(receipts) == 1 and result is None, "Cancellation must occur after first real token")
            self.wait(lambda: self.health().get("state") != "Busy", 60, "Worker did not recover from cancellation")
            return record
        require(result is not None and observed["outcome"] == "completed", "Missing terminal worker result")
        require(result["executionPath"] == "metal-local" and result["tokenIds"] == token_ids, "Generated/delivered token reconciliation failed")
        if natural:
            require(result["finishReason"] == "eos" and len(token_ids) < m, "Natural EOS did not terminate before bound")
        else:
            require(result["promptTokens"] == n and len(token_ids) == m and result["finishReason"] == "length",
                    "Fixed input/output count or finish reason mismatch")
        metric = metric_from_observation(observed, self.health()["build"])
        require(not metric_problems(metric), "Invalid/missing worker clocks")
        require(observed["outputTokens"] == len(token_ids), "Observer token coverage mismatch")
        key = ("natural", m) if natural else (n, m)
        if key in self.references:
            require(token_ids == self.references[key], "First exact generated-sequence divergence; stop and retain")
        else:
            self.references[key] = token_ids
        return record

    def run_native(self):
        out = self.out / "native"
        out.mkdir()
        process = self.children.start([self.args.native_driver, "--gpu"], out / "gpu.log", env=environment("on"))
        self.wait(lambda: process.poll() is not None, 900, "Native operator correctness timeout")
        save(out / "status.json", {"exitCode": process.returncode, "sha256": sha(self.args.native_driver),
                                   "source": self.args.native_driver_source})
        require(process.returncode == 0, "Native operator correctness gate failed")
        self.children.cleanup()
        wait_quiescent(out / "post-gpu-admission.jsonl")

    def gates(self):
        self.run_native()
        for arm in ("main", "off", "on"):
            out = self.out / ("gates-" + arm)
            out.mkdir()
            self.start_arm(arm, out)
            self.trial(arm, 21, 128, 0, out, label="excluded-warmup")
            self.trial(arm, 21, 128, 1, out, label="reset-reference")
            self.trial(arm, 8192, 128, 1, out, label="8k-off-sequence")
            self.trial(arm, 21, 256, 1, out, natural=True, label="natural-eos")
            self.trial(arm, 21, 4096, 1, out, cancel=True, label="cancel")
            self.trial(arm, 21, 128, 1, out, label="post-cancel-reset")
            self.finish_arm()
        save(self.out / "gates.json", {"passed": True, "utc": stamp(), "packages": self.packages,
                                      "identity": self.evidence_identity,
                                      "nativeOperatorIndependent": True, "fullModelExactSequences": True,
                                      "resetNaturalEosCancellation": True, "fullCheckpointLogits": None,
                                      "internalStates": None,
                                      "limitation": "No full-logit/state endpoint in immutable packages; not claimed."})

    def fixed(self):
        for index, (n, m) in enumerate(SHAPES):
            for arm in (("main", "on") if index % 2 == 0 else ("on", "main")):
                out = self.out / f"fixed-{n}-{m}-{arm}"
                out.mkdir()
                self.start_arm(arm, out)
                for trial in range(4):
                    self.trial(arm, n, m, trial, out)
                self.finish_arm()

    def run_command(self, args, path, timeout):
        process = self.children.start(args, path)
        self.wait(lambda: process.poll() is not None, timeout, "Harness subprocess deadline")
        return process.returncode

    def human(self, cohort):
        for arm in (("main", "off", "on") if cohort == "pilot" else ("on", "off", "main")):
            out = self.out / f"{cohort}-{arm}"
            out.mkdir()
            h = self.start_arm(arm, out)
            self.trial(arm, 21, 128, 0, out, label="excluded-warmup")
            official = load_module("issue103_official", self.args.tools / "scripts/eval-pi-humaneval-docker.py")
            official_controls(official, self.dataset, out)
            selected = PILOT if cohort == "pilot" else COHORT
            code = self.run_command([
                sys.executable, self.args.tools / "scripts/benchmark-pi-humaneval.py",
                "--output", out / "pi", "--token-file", self.token_file, "--endpoint", self.endpoint,
                "--pi", self.args.pi, "--cohort", cohort, "--ids", ",".join("HumanEval/" + str(i) for i in selected),
                "--seed", "0", "--timeout", "300", "--dataset", self.args.dataset,
                "--context-window", "32768", "--max-tokens", "4096",
                "--endpoint-build", h["build"], "--model-revision", REVISION],
                out / "pi-driver.log", len(selected) * 300 + 120)
            save(out / "pi-exit.json", {"exitCode": code, "tasksRetained": True})
            tasks = rows(out / "pi/tasks.jsonl")
            require([t["task_id"] for t in tasks] == ["HumanEval/" + str(i) for i in selected], "Task cohort coverage mismatch")
            # Await the outer request's resource bracket, including a cancelled
            # timeout, without demanding success-shaped clocks for failed tasks.
            self.wait(lambda: self.health().get("state") != "Busy", 60, "Pi cancellation did not drain worker")
            time.sleep(1)
            observations = rows(out / "observer/requests.jsonl")
            clocks = [metric_from_observation(r, h["build"]) for r in observations if r.get("workerResult")]
            with (out / "service-events-metrics.log").open("x") as f:
                for metric in clocks:
                    f.write("INFERENCE_METRIC " + json.dumps(metric, allow_nan=False) + "\n")
            result = reconcile(tasks, observations, h["build"])
            save(out / "reconciliation.json", result)
            # The pinned summarizer is run only when its required fields exist;
            # missing clocks are explicitly null in our independent wrapper.
            safe = all(not metric_problems(m) for m in clocks)
            if safe:
                summary_code = self.run_command([
                    sys.executable, self.args.tools / "scripts/summarize-pi-humaneval-performance.py",
                    "--run", out / "pi", "--serve-log", out / "service-events-metrics.log",
                    "--mac-commit", MAIN if arm == "main" else self.args.candidate_sha,
                    "--change", "#103 two-hop ServiceEvent observer; not native terminal log",
                    "--output", out / "standard-performance.json"], out / "standard-summarizer.log", 60)
                require(summary_code == 0, "Pinned standard summarizer execution failed")
            else:
                save(out / "standard-performance-unavailable.json", {"reason": "Missing/error durations; no fabricated zeros"})
            official_code = self.run_command([
                sys.executable, self.args.tools / "scripts/eval-pi-humaneval-docker.py",
                "--results", out / "pi", "--dataset", self.args.dataset,
                "--output", out / "official.json"], out / "official.log", len(selected) * 45 + 60)
            save(out / "official-exit.json", {"exitCode": official_code})
            require(official_code == 0, "Official evaluator infrastructure failure")
            validation = read_json(out / "official.json")
            require(validation["count"] == len(selected) and validation["negative_control"] == "failed",
                    "Official assertion coverage missing")
            require(not any(r.get("exit_code") in (125, 126, 127) or
                            (r.get("status") == "failed" and r.get("exit_code") != 1)
                            for r in validation["rows"]), "Docker launch/runtime failure is not model quality")
            self.finish_arm()
            # Slow or assertion-failed pilots do not cancel formal20. Invalid
            # clocks/tokens or infrastructure/resource failures are different.
            wrong = [p for p in result["problems"] if any(k in p for k in (
                "Pi/worker token mismatch", "N-1 mismatch", "not metal-local", "duplicate worker", "multiply assigned"))]
            require(not wrong, "Pi/worker correctness mismatch; no whole-cohort claim")

    def dependency_evidence(self):
        if self.args.phase in ("gates", "all"):
            return
        require(self.args.gates_evidence is not None, "Fixed/pilot/cohort require reviewed prior gates evidence")
        gates = read_json(self.args.gates_evidence)
        require(gates.get("passed") is True and gates.get("identity") == self.evidence_identity,
                "Prior gates package/source/harness/input identity mismatch")
        save(self.out / "prior-gates.json", {"sha256": sha(self.args.gates_evidence), "gates": gates})
        if self.args.phase == "cohort":
            require(self.args.pilot_evidence is not None, "Formal20 requires retained pilot evidence")
            pilot = read_json(self.args.pilot_evidence)
            require(pilot.get("completed") is True and pilot.get("identity") == self.evidence_identity and "pilot" in pilot.get("phases", []),
                    "Pilot completion/package evidence mismatch (speed is not a gate)")
            save(self.out / "prior-pilot.json", {"sha256": sha(self.args.pilot_evidence)})

    def restore(self):
        record = {"utc": stamp(), "attemptedBootout": self.attempted_bootout, "hashesUnchanged": False,
                  "definitionRestored": False, "healthReady": False, "realRequestVerified": False,
                  "originalReady": self.original_health.get("ready"), "errors": []}
        cleanup_ok = True
        self.stop.set()
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=240)
            if self.monitor_thread.is_alive():
                record["errors"].append("monitorShutdownTimeout")
        try:
            self.children.cleanup()
        except Exception as ex:
            cleanup_ok = False
            record["errors"].append("childCleanup:" + type(ex).__name__)
        self.worker, self.active_out, self.worker_out = None, None, None
        if self.attempted_bootout:
            try:
                self.unchanged_original()
                record["hashesUnchanged"] = True
                require(cleanup_ok, "Do not reload original alongside an uncleaned owned model")
                exists = subprocess.run(["launchctl", "print", self.label], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, timeout=15).returncode == 0
                if exists:
                    current = self.health(original=True)
                    verify_original_identity(identity(current["pid"]), self.original_exe, current)
                    require(same_identity(self.original_identity, identity(current["pid"])),
                            "Unexpected service identity after bootout attempt")
                else:
                    wait_quiescent(self.out / "restore-admission.jsonl")
                    subprocess.run(["launchctl", "bootstrap", self.domain, str(self.plist)], check=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                record["definitionRestored"] = True
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    try:
                        h = self.health(original=True)
                        verify_original_identity(identity(h["pid"]), self.original_exe, h)
                        record["health"] = h
                        status, ready = http(self.original_config["listen"] + "/ready", self.token)
                        if h.get("ready") and status == 200 and ready.get("ready"):
                            launch = command(["launchctl", "print", self.label])
                            require(re.search(r"\bpid = " + str(h["pid"]) + r"\b", launch),
                                    "Restored LaunchAgent/health PID mismatch")
                            record["identity"] = identity(h["pid"])
                            record["healthReady"] = True
                            break
                    except (OSError, GateError, ValueError, subprocess.SubprocessError) as ex:
                        record["lastProbeError"] = type(ex).__name__
                    time.sleep(1)
                if record["healthReady"]:
                    # Original routing/configuration, never overrides Windows or coordinator.
                    body = {"prompt": self.files[21].read_text(), "maxTokens": 1, "model": MODEL, "backend": "metal"}
                    code, response = http(self.original_config["listen"] + "/v1/infer", self.token, body, timeout=120)
                    record["realRequestVerified"] = code == 200 and bool(response.get("tokenIds"))
                    record["realRequestStatus"] = code
                else:
                    record["limitation"] = "Unchanged definition restored, but real readiness not obtained; original ready=false may predate experiment. No successful request restoration claimed."
                self.unchanged_original()
            except Exception as ex:
                record["errors"].append(type(ex).__name__)
        save(self.out / "restoration.json", record)
        self.restored = record["definitionRestored"] and record["healthReady"] and record["realRequestVerified"] and not record["errors"]
        return record

    def execute(self):
        self.dependency_evidence()
        completed = []
        previous_handlers = {}

        def interrupted(signum, _frame):
            raise InterruptedError(f"signal-{signum}")

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        try:
            self.pause()
            self.monitor_thread = threading.Thread(target=self.monitor)
            self.monitor_thread.start()
            phases = ("gates", "fixed", "pilot", "cohort") if self.args.phase == "all" else (self.args.phase,)
            for phase in phases:
                self.healthy()
                append(self.out / "phases.jsonl", {"utc": stamp(), "phase": phase, "status": "starting"})
                if phase == "gates":
                    self.gates()
                elif phase == "fixed":
                    self.fixed()
                else:
                    self.human(phase)
                completed.append(phase)
                append(self.out / "phases.jsonl", {"utc": stamp(), "phase": phase, "status": "completed"})
            save(self.out / "completed.json", {"utc": stamp(), "completed": True, "phases": completed,
                                              "packages": self.packages, "identity": self.evidence_identity,
                                              "performanceAdoptionClaim": False})
        except BaseException as ex:
            save(self.out / "failure.json", {"utc": stamp(), "type": type(ex).__name__, "completedPhases": completed,
                                            "reason": str(ex) if isinstance(ex, (GateError, InterruptedError)) else None})
            raise
        finally:
            for signum in previous_handlers:
                signal.signal(signum, signal.SIG_IGN)
            try:
                self.restore()
            finally:
                for signum, previous in previous_handlers.items():
                    signal.signal(signum, previous)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "gates", "fixed", "pilot", "cohort", "all"), default="preflight")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-package", type=Path, required=True, help="Frozen Native AOT deepinfer executable")
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--main-package", type=Path, default=Path("/Users/littlecloud/.codex/worktrees/experiment-92-main/Infer.NET/artifacts/deploy/releases/issue95-main/deepinfer"))
    parser.add_argument("--tools", type=Path, default=REFERENCE84)
    parser.add_argument("--fixtures", type=Path, default=REFERENCE98 / "artifacts/dual-host/issue98-dflash8-e2e/fixtures")
    parser.add_argument("--dataset", type=Path, default=DATA_ROOT / "main-eval-20261023/artifacts/HumanEval.jsonl.gz")
    parser.add_argument("--pi", type=Path, default=Path.home() / ".local/bin/pi")
    parser.add_argument("--config-dir", type=Path, default=DATA_ROOT / "deploy/private/mac")
    parser.add_argument("--plist", type=Path, default=Path.home() / "Library/LaunchAgents/com.infernet.metal.plist")
    parser.add_argument("--original-preflight", type=Path, default=ROOT / "artifacts/dual-host/issue103-lmhead-affine8/control/original-service-preflight.json")
    parser.add_argument("--high-power-evidence", type=Path)
    parser.add_argument("--native-driver", type=Path)
    parser.add_argument("--native-driver-sha256")
    parser.add_argument("--native-driver-source")
    parser.add_argument("--gates-evidence", type=Path)
    parser.add_argument("--pilot-evidence", type=Path)
    parser.add_argument("--base-port", type=int, default=5380)
    parser.add_argument("--execute", action="store_true", help="Explicit service/GPU mutation authorization for this invocation")
    parser.add_argument("--exclusive-gpu", action="store_true")
    parser.add_argument("--clients-paused", action="store_true")
    return parser.parse_args()


def main():
    os.umask(0o077)
    args = arguments()
    out = new_output(args.output, ROOT)
    try:
        controller = Controller(args, out)
        report = controller.preflight()
        if args.execute:
            controller.execute()
            print(json.dumps({"completed": True, "restorationVerified": controller.restored, "evidence": str(out)}))
            return 0 if controller.restored else 3
        print(json.dumps({"mode": "non-mutating-preflight", "blockers": report["blockers"], "evidence": str(out)}))
        return 0
    except BaseException as ex:
        if not (out / "failure.json").exists():
            save(out / "failure.json", {"utc": stamp(), "type": type(ex).__name__,
                                        "reason": str(ex) if isinstance(ex, GateError) else None})
        print(json.dumps({"completed": False, "errorType": type(ex).__name__, "evidence": str(out)}))
        return 2
    finally:
        hashes = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        save(out / "evidence-sha256.json", hashes)


if __name__ == "__main__":
    raise SystemExit(main())
