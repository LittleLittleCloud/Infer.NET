#!/usr/bin/env python3
"""Run pinned HumanEval checks against Pi solution files in an offline container."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import uuid

DATASET_SHA = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
IMAGE = "python@sha256:5fb4bc53dd0180955e31e5245b648bbb24a4ac4ab03a1de81513642325220bcd"


def run_case(code, staging):
    with tempfile.TemporaryDirectory(dir=staging) as dirname:
        case_dir = Path(dirname)
        os.chmod(case_dir, 0o755)
        case = case_dir / "case.py"
        case.write_text(code, encoding="utf-8")
        os.chmod(case, 0o644)
        container_name = "infernet-humaneval-" + uuid.uuid4().hex
        command = [
            "docker", "run", "--rm", "--name", container_name, "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "64", "--memory", "256m", "--cpus", "1",
            "--user", "65534:65534", "--ulimit", "nofile=64:64",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--mount", f"type=bind,src={case_dir},dst=/work,readonly",
            "--workdir", "/work", IMAGE, "python", "-I", "-B", "/work/case.py",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            return {"status": "passed" if result.returncode == 0 else "failed",
                    "exit_code": result.returncode, "stderr": result.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "exit_code": None, "stderr": "30-second container timeout"}
        finally:
            subprocess.run(["docker", "rm", "-f", container_name],
                           capture_output=True, timeout=10, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, choices=(5, 20), default=20,
                        help="Fixed pilot or formal cohort size; default 20 preserves formal evaluation")
    args = parser.parse_args()
    assert hashlib.sha256(args.dataset.read_bytes()).hexdigest() == DATASET_SHA
    tasks = {item["task_id"]: item for item in
             (json.loads(line) for line in gzip.decompress(args.dataset.read_bytes()).splitlines())}
    manifest = json.loads((args.results / "manifest.json").read_text())
    assert manifest["dataset_sha256"] == DATASET_SHA
    assert len(manifest["ids"]) == args.expected_count and len(set(manifest["ids"])) == args.expected_count
    staging = args.results / "functional-staging"
    staging.mkdir(exist_ok=True)
    negative = tasks[manifest["ids"][0]]
    negative_code = (
        f"def {negative['entry_point']}(*args, **kwargs): return None\n"
        + negative["test"] + f"\ncheck({negative['entry_point']})\n"
    )
    negative_result = run_case(negative_code, staging)
    assert negative_result["status"] == "failed", negative_result
    rows = []
    for task_id in manifest["ids"]:
        item = tasks[task_id]
        solution_path = args.results / task_id.replace("/", "-") / "solution.py"
        if not solution_path.is_file():
            rows.append({"task_id": task_id, "status": "missing_solution"})
            continue
        solution = solution_path.read_text(encoding="utf-8")
        code = solution + "\n\n" + item["test"] + f"\ncheck({item['entry_point']})\n"
        result = run_case(code, staging)
        rows.append({"task_id": task_id, **result})
        print(task_id, result["status"], flush=True)
    output = {"method": "Official HumanEval test field against complete solution.py files; one offline restricted Docker container per task",
              "dataset_sha256": DATASET_SHA, "image": IMAGE,
              "negative_control": negative_result["status"], "count": len(rows),
              "passed": sum(row["status"] == "passed" for row in rows), "rows": rows}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print("passed", output["passed"], "of", len(rows), flush=True)


if __name__ == "__main__":
    main()
