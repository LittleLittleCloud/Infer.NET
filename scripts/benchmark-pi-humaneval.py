#!/usr/bin/env python3
"""Measure real Pi coding-agent wall time, not HumanEval correctness."""

import argparse
import ast
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from urllib.parse import urlsplit


DATASET_REVISION = "b9453e8f1865df0a0057a1383767caa400acfbcf"
DATASET_URL = (
    f"https://raw.githubusercontent.com/openai/human-eval/{DATASET_REVISION}"
    "/data/HumanEval.jsonl.gz"
)
DATASET_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
PROVIDER = "infernet"
MODEL = "qwen3.6-35b-a3b"
TOKEN_KEYS = ("input", "output", "cacheRead", "cacheWrite", "totalTokens")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_dataset(path=None):
    data = path.read_bytes() if path else urllib.request.urlopen(
        DATASET_URL, timeout=30
    ).read()
    if sha256(data) != DATASET_SHA256:
        raise ValueError("HumanEval compressed dataset SHA-256 mismatch")
    rows = [json.loads(line) for line in gzip.decompress(data).split(b"\n") if line]
    if len(rows) != 164 or {r["task_id"] for r in rows} != {
        f"HumanEval/{i}" for i in range(164)
    }:
        raise ValueError("Unexpected HumanEval task IDs")
    # Never retain tests or canonical solutions in the harness's task records.
    return {r["task_id"]: r["prompt"] for r in rows}


def select_ids(tasks, cohort, seed, ids=None):
    if ids is not None:
        selected = ids.split(",")
        if len(set(selected)) != len(selected) or any(t not in tasks for t in selected):
            raise ValueError("Task IDs must be unique official HumanEval IDs")
        return selected
    # Hash ranking is stable across Python versions; pilot is a prefix of cohort.
    ranked = sorted(tasks, key=lambda t: (sha256(f"{seed}:{t}".encode()), t))
    return ranked[:{"pilot": 5, "cohort": 20, "full": 164}[cohort]]


def make_prompt(prompt):
    return (
        "Implement the Python function specified below. Use your standard tools "
        "to write the complete implementation, including required imports, to "
        "solution.py in the current directory. Do not merely print the code. "
        "Do not fetch reference solutions or external tests. You may inspect "
        "and syntax-check your own file. Finish when the file is ready.\n\n" + prompt
    )


def redact(text, token):
    if token:
        text = text.replace(token, "[REDACTED]")
    return re.sub(r"(?i)(bearer\s+)[^\s\"'\\]+", r"\1[REDACTED]", text)


def scrub(value, token):
    if isinstance(value, str):
        return redact(value, token)
    if isinstance(value, dict):
        return {redact(k, token): scrub(v, token) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, token) for v in value]
    return value


def parse_events(records):
    """Count authoritative assistant completions once, never streaming snapshots."""
    requests, turns, tools = [], [], []
    message_start = turn_start = None
    malformed = 0
    ended = settled = aborted = False
    for record in records:
        event = record.get("event")
        if not isinstance(event, dict):
            malformed += 1
            continue
        kind, at = event.get("type"), record["seconds"]
        if kind == "turn_start":
            turn_start = at
        elif kind == "message_start" and event.get("message", {}).get("role") == "assistant":
            message_start = at
        elif kind == "message_end" and event.get("message", {}).get("role") == "assistant":
            message = event["message"]
            requests.append({
                "observed_seconds": None if message_start is None else at - message_start,
                "duration_ms": message.get("durationMs"),
                "usage": message.get("usage", {}),
                "stop_reason": message.get("stopReason"),
                "error": message.get("errorMessage"),
                "provider": message.get("provider"),
                "model": message.get("model"),
            })
            message_start = None
        elif kind == "turn_end":
            turns.append(None if turn_start is None else at - turn_start)
            turn_start = None
        elif kind in ("tool_execution_start", "tool_execution_end"):
            tools.append({"seconds": at, **event})
        elif kind == "agent_end":
            ended = not event.get("willRetry", False)
        elif kind == "agent_settled":
            settled, aborted = True, event.get("aborted", False)
    usage = {key: sum(r["usage"].get(key, 0) or 0 for r in requests) for key in TOKEN_KEYS}
    final = requests[-1]["stop_reason"] if requests else None
    mismatch = any(
        r["provider"] not in (None, PROVIDER) or r["model"] not in (None, MODEL)
        for r in requests
    )
    return {
        "requests": requests, "turn_seconds": turns, "tool_calls": tools,
        "usage": usage, "malformed_records": malformed, "finish_status": final,
        "stream_complete": (settled or ended) and not aborted,
        "model_mismatch": mismatch,
        "agent_ok": (settled or ended) and not aborted and not mismatch
        and not malformed and final == "stop",
    }


def summarize(rows):
    durations = [r["wall_seconds"] for r in rows]
    request_times = [
        q["observed_seconds"] for r in rows for q in r["pi"]["requests"]
        if q["observed_seconds"] is not None
    ]
    return {
        "count": len(rows),
        "completion_count": sum(r["completed"] for r in rows),
        "syntax_ok_count": sum(r["syntax_status"] == "ok" for r in rows),
        "timeout_count": sum(r["timeout"] for r in rows),
        "error_count": sum(r["error"] is not None for r in rows),
        "wall_seconds": {
            "median": statistics.median(durations) if durations else None,
            "p90": sorted(durations)[math.ceil(.9 * len(durations)) - 1] if durations else None,
            "total": sum(durations),
        },
        "tokens": {k: sum(r["pi"]["usage"][k] for r in rows) for k in TOKEN_KEYS},
        "request_count": sum(len(r["pi"]["requests"]) for r in rows),
        "observed_request_seconds": {
            "count": len(request_times), "total": sum(request_times),
            "median": statistics.median(request_times) if request_times else None,
        },
        "correctness_evaluated": False,
    }


class WindowsJob:
    """Own only this capture's processes, including children outliving their parent."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class Limits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", Limits),
                ("IoInfo", ctypes.c_ulonglong * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class Accounting(ctypes.Structure):
            _fields_ = [
                ("Times", ctypes.c_longlong * 4),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        self.ctypes, self.Accounting = ctypes, Accounting
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, arguments, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int,
                                        ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int,
                                          ctypes.c_void_p, wintypes.DWORD,
                                          ctypes.c_void_p], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ):
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self.api.CreateJobObjectW(None, None)
        self.check(self.handle)
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            self.check(self.api.SetInformationJobObject(
                self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ))
        except BaseException:
            self.close()
            raise

    def check(self, result):
        if not result:
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def assign(self, process):
        self.check(self.api.AssignProcessToJobObject(self.handle, int(process._handle)))

    def terminate(self):
        self.check(self.api.TerminateJobObject(self.handle, 1))
        deadline = time.monotonic() + 3
        while True:
            accounting = self.Accounting()
            self.check(self.api.QueryInformationJobObject(
                self.handle, 1, self.ctypes.byref(accounting),
                self.ctypes.sizeof(accounting), None,
            ))
            if not accounting.ActiveProcesses:
                return
            if time.monotonic() >= deadline:
                raise OSError("Windows job cleanup did not finish")
            time.sleep(.01)

    def close(self):
        self.check(self.api.CloseHandle(self.handle))


def kill_tree(process, job=None):
    if job is not None:
        job.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def pi_command(executable, prompt):
    prefix = [executable] if isinstance(executable, str) else executable
    return [
        *prefix, "--provider", PROVIDER, "--model", MODEL,
        "--mode", "json", "--no-session", "--tools", "read,write,bash,edit",
        "--no-extensions", "--no-mcp", "--no-skills", "--no-prompt-templates",
        "--no-themes", "--no-context-files", "--no-approve",
        "--thinking", "off", "-p", prompt,
    ]


def capture(command, cwd, env, timeout, token, event_path):
    records = []
    timed_out, error, returncode = False, None, None
    start_wall, start = utc_now(), time.monotonic()
    process = job = None
    readers = []
    with event_path.open("w", encoding="utf-8") as log:
        try:
            if os.name == "nt":
                executable = command[0]
                if os.path.dirname(executable):
                    executable = os.path.abspath(os.path.join(cwd, executable))
                executable = shutil.which(executable)
                if executable is None:
                    raise FileNotFoundError("Capture executable not found")
                job = WindowsJob()
                # The gate prevents Pi spawning children before job assignment.
                command = [
                    sys.executable, "-I", "-c",
                    "import subprocess,sys; "
                    "gate=sys.stdin.buffer.read(1); "
                    "sys.exit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL) "
                    "if gate == b'1' else 1)",
                    executable, *command[1:],
                ]
            process = subprocess.Popen(
                command, cwd=cwd, env=env,
                stdin=subprocess.PIPE if job is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            if job is not None:
                job.assign(process)
                process.stdin.write(b"1")
                process.stdin.close()
            messages = queue.Queue()

            def read_pipe(pipe, stream):
                try:
                    for line in iter(pipe.readline, b""):
                        messages.put((stream, time.monotonic() - start, line))
                finally:
                    pipe.close()
                    messages.put((stream, None, None))

            for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                reader = threading.Thread(target=read_pipe, args=(pipe, name), daemon=True)
                readers.append(reader)
                reader.start()
            closed = 0
            while closed < 2:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stream, seconds, raw = messages.get(timeout=min(remaining, .1))
                except queue.Empty:
                    continue
                if raw is None:
                    closed += 1
                    continue
                text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if stream == "stderr":
                    record = {"seconds": seconds, "stderr": redact(text, token)}
                else:
                    try:
                        event = scrub(json.loads(text), token)
                    except ValueError:
                        event = None
                    record = {"seconds": seconds, "event": event}
                    if event is None:
                        record["invalid_json"] = redact(text, token)
                    records.append(record)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
            returncode = process.wait(timeout=max(.001, timeout - (time.monotonic() - start)))
        except subprocess.TimeoutExpired:
            timed_out, error = True, "task_timeout"
        except OSError as ex:
            error = type(ex).__name__
        finally:
            try:
                if process is not None:
                    try:
                        kill_tree(process, job)
                    finally:
                        # Also reap the gated launcher if job assignment failed.
                        if process.poll() is None:
                            process.kill()
                        returncode = process.wait(timeout=3)
                        if process.stdin is not None:
                            process.stdin.close()
                        if not readers:
                            process.stdout.close()
                            process.stderr.close()
            except (OSError, subprocess.TimeoutExpired) as ex:
                error = f"cleanup_failed:{type(ex).__name__}"
            finally:
                if job is not None:
                    job.close()
    end, end_wall = time.monotonic(), utc_now()
    return {
        "start_utc": start_wall, "end_utc": end_wall,
        "start_monotonic": start, "end_monotonic": end, "wall_seconds": end - start,
        "timeout": timed_out, "error": error, "returncode": returncode,
        "pi": parse_events(records),
    }


def run_task(args, task_id, prompt, executable, version, token):
    task_dir = args.output / task_id.replace("/", "-")
    task_dir.mkdir()
    with tempfile.TemporaryDirectory(prefix="pi-humaneval-") as temporary:
        root = Path(temporary)
        workspace, agent = root / "workspace", root / "agent"
        workspace.mkdir()
        agent.mkdir()
        model_config = {"providers": {PROVIDER: {
            "baseUrl": args.endpoint.rstrip("/") + "/v1",
            "api": "openai-completions", "apiKey": "$INFERNET_BENCH_TOKEN",
            "authHeader": True,
            "models": [{
                "id": MODEL, "reasoning": False, "input": ["text"],
                "contextWindow": args.context_window, "maxTokens": args.max_tokens,
                "compat": {
                    "maxTokensField": "max_tokens",
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "supportsStore": False,
                },
            }],
        }}}
        (agent / "models.json").write_text(json.dumps(model_config), encoding="utf-8")
        (agent / "settings.json").write_text(json.dumps({
            "compaction": {"enabled": False}, "retry": {"enabled": False},
        }), encoding="utf-8")
        env = {
            k: v for k, v in os.environ.items()
            if k in ("PATH", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL")
        }
        env.update({
            "HOME": str(root), "USERPROFILE": str(root), "TMPDIR": str(root),
            "TMP": str(root), "TEMP": str(root),
            "PI_CODING_AGENT_DIR": str(agent), "INFERNET_BENCH_TOKEN": token,
        })
        full_prompt = make_prompt(prompt)
        command = pi_command(executable, full_prompt)
        result = capture(command, workspace, env, args.timeout, token, task_dir / "events.jsonl")
        result.update({
            "task_id": task_id, "prompt_sha256": sha256(full_prompt.encode()),
            "dataset_prompt_sha256": sha256(prompt.encode()),
            "command": command, "pi_version": version, "provider": PROVIDER, "model": MODEL,
            "endpoint_build": args.endpoint_build, "model_revision": args.model_revision,
            "solution_sha256": None, "syntax_status": "missing",
        })
        solution = workspace / "solution.py"
        if solution.is_file() and not solution.is_symlink():
            data = solution.read_bytes()
            # Preserve the actual artifact hash, but never persist credentials it may echo.
            result["solution_sha256"] = sha256(data)
            safe_data = redact(data.decode("utf-8", errors="replace"), token).encode()
            (task_dir / "solution.py").write_bytes(safe_data)
            result["stored_solution_sha256"] = sha256(safe_data)
            try:
                ast.parse(data, filename="solution.py")
                result["syntax_status"] = "ok"
            except (SyntaxError, ValueError):
                result["syntax_status"] = "invalid"
        result["completed"] = (
            result["returncode"] == 0 and result["error"] is None
            and result["pi"]["agent_ok"] and result["solution_sha256"] is not None
        )
        if result["error"] is None and not result["completed"]:
            result["error"] = "incomplete_or_failed_pi_run"
        return scrub(result, token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; must not exist")
    parser.add_argument("--token-file", type=Path, required=True, help="Read privately; never logged")
    parser.add_argument("--endpoint", default="http://127.0.0.1:5200", help="Local server origin, no /v1")
    parser.add_argument("--pi", default="pi", help="Pi executable or absolute path to Pi's cli.js")
    parser.add_argument("--cohort", choices=("pilot", "cohort", "full"), default="pilot")
    parser.add_argument("--seed", type=int, default=0, help="Task selection only, not model sampling")
    parser.add_argument("--ids", help="Explicit comma-separated HumanEval/N IDs, overriding cohort")
    parser.add_argument("--timeout", type=float, default=600, help="Seconds per task")
    parser.add_argument("--dataset", type=Path, help="Offline copy of the pinned .jsonl.gz")
    parser.add_argument("--context-window", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--endpoint-build", help="Verified server build ID, otherwise null")
    parser.add_argument("--model-revision", help="Verified model revision, otherwise null")
    args = parser.parse_args()
    origin = urlsplit(args.endpoint)
    if (origin.scheme != "http" or origin.hostname not in ("127.0.0.1", "localhost", "::1")
            or origin.username or origin.password or origin.query or origin.fragment
            or origin.path not in ("", "/")):
        parser.error("--endpoint must be a credential-free HTTP loopback origin")
    if (not math.isfinite(args.timeout) or args.timeout <= 0
            or not 0 < args.max_tokens < args.context_window):
        parser.error("Require positive finite timeout and 0 < max-tokens < context-window")
    try:
        token = args.token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Empty token")
        tasks = load_dataset(args.dataset)
        ids = select_ids(tasks, args.cohort, args.seed, args.ids)
        if Path(args.pi).suffix == ".js" and Path(args.pi).is_file():
            node = shutil.which("node")
            executable = [node, str(Path(args.pi).resolve())] if node else None
        else:
            resolved = shutil.which(args.pi)
            executable = [resolved] if resolved else None
        if not executable or Path(executable[0]).suffix.lower() in (".cmd", ".bat"):
            raise ValueError("Pi executable not found")
        version = redact(subprocess.check_output(
            [*executable, "--version"], stderr=subprocess.STDOUT, timeout=15
        ).decode("utf-8", errors="replace").strip(), token)
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=False)
    except Exception as ex:
        # Do not include exception text: subprocess/network errors can contain credentials.
        parser.exit(2, f"Benchmark setup failed ({type(ex).__name__}); no model tasks run.\n")
    manifest = {
        "dataset_url": DATASET_URL, "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA256, "ids": ids, "seed": args.seed,
        "selection": "explicit" if args.ids else args.cohort,
        "endpoint": args.endpoint, "endpoint_build": args.endpoint_build,
        "model_revision": args.model_revision, "pi_version": version,
        "provider": PROVIDER, "model": MODEL, "timeout_seconds": args.timeout,
        "context_window": args.context_window, "max_tokens": args.max_tokens,
        "harness_sha256": sha256(Path(__file__).read_bytes()), "created_utc": utc_now(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(scrub(manifest, token), indent=2), encoding="utf-8"
    )
    rows = []
    with (args.output / "tasks.jsonl").open("w", encoding="utf-8") as output:
        for task_id in ids:
            start_wall, start = utc_now(), time.monotonic()
            try:
                row = run_task(args, task_id, tasks[task_id], executable, version, token)
            except Exception as ex:
                end = time.monotonic()
                prompt = make_prompt(tasks[task_id])
                row = {
                    "task_id": task_id, "start_utc": start_wall, "end_utc": utc_now(),
                    "start_monotonic": start, "end_monotonic": end,
                    "wall_seconds": end - start,
                    "prompt_sha256": sha256(prompt.encode()),
                    "dataset_prompt_sha256": sha256(tasks[task_id].encode()),
                    "command": pi_command(executable, prompt),
                    "pi_version": version, "provider": PROVIDER, "model": MODEL,
                    "endpoint_build": args.endpoint_build, "model_revision": args.model_revision,
                    "completed": False, "timeout": False,
                    "returncode": None, "solution_sha256": None,
                    "syntax_status": "unavailable", "error": type(ex).__name__,
                    "pi": parse_events([]),
                }
                row = scrub(row, token)
            rows.append(row)
            output.write(json.dumps(row) + "\n")
            output.flush()
            (args.output / "summary.json").write_text(
                json.dumps(summarize(rows), indent=2), encoding="utf-8"
            )
            print(f"{task_id}: completed={row['completed']} timeout={row['timeout']}")
    print(json.dumps(summarize(rows), indent=2))
    return 0 if all(r["completed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
