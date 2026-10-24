#!/usr/bin/env python3
"""Offline acceptance/data tests; optional explicitly synthetic browser fixture."""
import argparse
import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from home_trends import ROOT, aggregate_requests, homepage, import_pi, load_data, total, validate

FIXTURES = ROOT / "tests/home-trends/fixtures.json"


def fixture_data():
    fixture = json.loads(FIXTURES.read_text())
    data = load_data()
    original = data["runs"][0]
    data["runs"] = []
    data["commitTimeline"] = [
        {"sha": sha, "committedAt": f"2026-01-{i + 1:02d}T00:00:00Z",
         "evidence": "fixture-input.json"}
        for i, sha in enumerate(dict.fromkeys(s["commit"] for s in fixture["scenarios"]))
    ]
    config = data["configurations"][0]
    config["label"] = "SYNTHETIC ACCEPTANCE FIXTURE · NOT PRODUCTION"
    config["latestBaselineRun"] = "fixture-repeat"
    for scenario in fixture["scenarios"]:
        run = copy.deepcopy(original)
        run.update({
            "id": scenario["id"], "experimentId": "synthetic-tests",
            "title": "SYNTHETIC · " + scenario["id"], "state": scenario["state"],
            "change": "Synthetic fixture, not a measurement or production claim.",
            "result": "Exercises UI interactions only; never publish as historical evidence.",
            "measuredCommit": scenario["commit"],
            "provenance": {"kind": scenario["kind"],
                           "evidence": "https://github.com/LittleLittleCloud/Infer.NET/issues/44"},
            "baselineRunId": None if scenario["id"] == "fixture-main" else "fixture-main",
            "reportedTotals": {}, "requests": copy.deepcopy(fixture["requests"]),
            "requestCount": 4, "binarySha256": {}, "binaryEvidence": None
        })
        if scenario["kind"] == "main":
            run["provenance"]["mainCommit"] = scenario["commit"]
        for component in run["runtime"].values():
            component["commit"] = scenario["commit"]
        for task in run["tasks"]:
            task.update({"modelRequests": None, "inputTokens": None, "outputTokens": None,
                         "source": "fixture-input.json", "officialValidation": "not-run",
                         "status": "incomplete"})
        run["tasks"][0].update({"inputTokens": 1000, "outputTokens": 42,
                                "status": "passed", "officialValidation": "passed"})
        run["tasks"][1].update({"inputTokens": 0, "outputTokens": 1,
                                "status": "passed", "officialValidation": "passed"})
        run["tasks"][4]["status"] = "timeout"
        if scenario["mode"] == "regressed":
            for request in run["requests"]:
                if request["prefillSeconds"] is not None:
                    request["prefillSeconds"] *= 2
                    request["decodeSeconds"] *= 2
        if scenario["mode"] in {"missing", "unverified"}:
            run["requests"] = None
            run["tasks"][0]["status"] = "error"
            run["tasks"][0]["officialValidation"] = "not-run"
        if scenario["mode"] == "unverified":
            run["tasks"][0]["status"] = "unverified"
            run["tasks"][0]["officialValidation"] = "unverified"
        if scenario["mode"] == "zero":
            run["requests"] = [copy.deepcopy(fixture["requests"][3])]
            run["requests"][0]["clientTtftMs"] = 0
        data["runs"].append(run)
    other = copy.deepcopy(config)
    other.update({"id": "fixture-incompatible-bf16", "label": "SYNTHETIC · incompatible BF16 KV",
                  "kv": "BF16", "latestBaselineRun": None})
    data["configurations"].append(other)
    other_run = copy.deepcopy(data["runs"][1])
    other_run.update({"id": "fixture-bf16", "configurationId": other["id"], "baselineRunId": None})
    data["runs"].append(other_run)
    return validate(data)


class TrendTests(unittest.TestCase):
    def setUp(self):
        self.data = load_data()

    def test_production_only_real_history_and_same_measured_sha(self):
        self.assertEqual(len(self.data["runs"]), 2)
        for run in self.data["runs"]:
            self.assertEqual(run["measuredCommit"], "e017d84e805c016d5912b5051e4c69f758056363")
            self.assertNotEqual(run["measuredCommit"], run["harnessCommit"])
            self.assertEqual(len(run["tasks"]), 20)
            for key in ("prefill", "decode", "ttft"):
                self.assertIsNone(run["metrics"][key]["value"])
                self.assertEqual(run["metrics"][key]["samples"], 0)
            self.assertEqual(len(run["binarySha256"]), 4)
        self.assertNotEqual(self.data["runs"][0]["id"], self.data["runs"][1]["id"])

    def test_source_rows_and_totals_exactly_mapped(self):
        manifest = json.loads((ROOT / "site/trends.json").read_text())
        for spec, run in zip(manifest["imports"], self.data["runs"]):
            source = json.loads((ROOT / spec["source"]).read_text())
            rows = source["results"].get("rows", source.get("rows"))
            for index, (task, row) in enumerate(zip(run["tasks"], rows)):
                for field in ("taskId", "wallSeconds", "inputTokens", "outputTokens", "modelRequests"):
                    self.assertEqual(task[field], row.get(field))
                self.assertTrue(task["source"].endswith("/" + str(index)))
            self.assertAlmostEqual(run["totals"]["wallSeconds"]["value"],
                                   source["results"]["wallSeconds"]["total"], delta=.011)
        first, rerun = self.data["runs"]
        self.assertEqual(first["totals"]["inputTokens"]["value"], 222744)
        self.assertEqual(first["totals"]["outputTokens"]["value"], 19212)
        self.assertIsNone(rerun["totals"]["inputTokens"]["value"])
        self.assertFalse(rerun["totals"]["inputTokens"]["complete"])
        self.assertEqual(rerun["reportedTotals"]["inputTokens"], 241162)

    def test_official_denominator_includes_failure(self):
        first, rerun = self.data["runs"]
        self.assertEqual(first["metrics"]["pass"]["value"], 100)
        self.assertEqual(rerun["metrics"]["pass"]["value"], 95)
        self.assertEqual(rerun["metrics"]["pass"]["total"], 20)
        self.assertEqual(rerun["metrics"]["pass"]["samples"], 19)
        failed = rerun["tasks"][4]
        self.assertEqual((failed["taskId"], failed["status"]), ("HumanEval/149", "error"))
        rerun["tasks"].pop()
        with self.assertRaisesRegex(ValueError, "Every planned task"):
            validate(self.data)

    def test_missing_assertions_are_unverified_not_completion(self):
        run = self.data["runs"][0]
        run["tasks"][0].update({"status": "unverified", "officialValidation": "unverified"})
        self.data["configurations"][0]["latestBaselineRun"] = None
        validate(self.data)
        self.assertIsNone(run["metrics"]["pass"]["value"])
        run["tasks"][0]["status"] = "passed"
        with self.assertRaisesRegex(ValueError, "official task assertions"):
            validate(self.data)

    def test_absence_zero_and_partial_fields(self):
        tasks = self.data["runs"][0]["tasks"][:3]
        tasks[0]["inputTokens"] = None
        tasks[1]["inputTokens"] = 0
        tasks[2]["inputTokens"] = 10
        self.assertEqual(total(tasks, "inputTokens"),
                         {"value": 10, "samples": 2, "total": 3, "complete": False})
        tasks[2]["inputTokens"] = None
        self.assertEqual(total(tasks, "inputTokens")["value"], 0)
        tasks[1]["inputTokens"] = None
        self.assertIsNone(total(tasks, "inputTokens")["value"])
        for field, invalid in (("inputTokens", -1), ("outputTokens", 1.5),
                               ("wallSeconds", float("nan")), ("inputTokens", True)):
            data = load_data()
            data["runs"][0]["tasks"][0][field] = invalid
            with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                validate(data)

    def test_paired_aggregation_not_average_of_rates(self):
        requests = json.loads(FIXTURES.read_text())["requests"]
        metrics = aggregate_requests(requests, 5)
        self.assertAlmostEqual(metrics["prefill"]["value"], 1000 / 6)
        self.assertAlmostEqual(metrics["decode"]["value"], 40 / 5)
        self.assertEqual(metrics["decode"]["samples"], 2)
        self.assertEqual(metrics["prefill"]["samples"], 3)
        self.assertEqual(metrics["prefill"]["total"], 5)
        self.assertEqual(metrics["ttft"]["value"], 120)
        self.assertEqual(metrics["ttft"]["failedWithoutToken"], 1)
        self.assertEqual(metrics["ttft"]["unobservedRequests"], 1)
        zero = aggregate_requests([requests[3]], 1)
        self.assertEqual(zero["prefill"]["value"], 0)
        self.assertIsNone(zero["decode"]["value"])
        partial = copy.deepcopy(requests[:1])
        partial[0]["inputTokens"] = None
        self.assertIsNone(aggregate_requests(partial, 1)["prefill"]["value"])
        self.assertEqual(aggregate_requests(partial, 1)["decode"]["value"], 5)

    def test_no_decode_intervals_for_zero_or_one_output_token(self):
        request = json.loads(FIXTURES.read_text())["requests"][0]
        for output in (0, 1):
            for seconds in (None, 0, 2):
                row = dict(request, outputTokens=output, decodeSeconds=seconds, inputTokens=0, clientTtftMs=0)
                metric = aggregate_requests([row], 1)
                self.assertIsNone(metric["decode"]["value"])
                self.assertEqual(metric["decode"]["samples"], 0)
                self.assertEqual(metric["decode"]["total"], 1)
                self.assertEqual(metric["prefill"]["value"], 0)
                self.assertEqual(metric["ttft"]["value"], 0)
                mixed = aggregate_requests([row, dict(request, id="other")], 2)["decode"]
                self.assertEqual(mixed["value"], 5)
                self.assertEqual(mixed["samples"], 1)

    def test_commit_timeline_requires_valid_unique_full_coverage(self):
        stamp = subprocess.check_output(
            ["git", "show", "-s", "--format=%ct", self.data["commitTimeline"][0]["sha"]],
            cwd=ROOT, text=True).strip()
        from datetime import datetime, timezone
        self.assertEqual(self.data["commitTimeline"][0]["committedAt"],
                         datetime.fromtimestamp(int(stamp), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        for timeline in ([], self.data["commitTimeline"] * 2,
                         [dict(self.data["commitTimeline"][0], committedAt="yesterday")]):
            data = copy.deepcopy(self.data)
            data["commitTimeline"] = timeline
            with self.subTest(timeline=timeline), self.assertRaises(ValueError):
                validate(data)

    def test_invalid_normalized_observations(self):
        request = json.loads(FIXTURES.read_text())["requests"][0]
        for key, value in (("prefillSeconds", 0), ("decodeSeconds", 0), ("decodeSeconds", -1),
                           ("clientTtftMs", float("inf")), ("inputTokens", -1),
                           ("outputTokens", 1.5), ("ttftBoundary", "worker-ready"),
                           ("ttftBoundary", "total-response-time"), ("firstTokenReceived", False),
                           ("firstTokenReceived", "false")):
            row = dict(request, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                aggregate_requests([row], 1)
        with self.assertRaisesRegex(ValueError, "exceed"):
            aggregate_requests([request], 0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_requests([request, request], 2)
        unknown = dict(request, outcome="failed", clientTtftMs=None)
        metric = aggregate_requests([unknown], 1)["ttft"]
        self.assertEqual(metric["failedWithoutToken"], 0)
        self.assertEqual(metric["unknownFirstTokenFailures"], 1)

    def test_request_attempt_mapping_and_cumulative_tokens(self):
        data = fixture_data()
        run = data["runs"][0]
        run["tasks"][0]["modelRequests"] = 2
        validate(data)
        run["tasks"][0]["inputTokens"] = 999
        with self.assertRaisesRegex(ValueError, "cumulative model requests"):
            validate(data)
        run["tasks"][0]["inputTokens"] = 1000
        run["requests"][0]["attempt"] = 2
        with self.assertRaisesRegex(ValueError, "planned task attempt"):
            validate(data)

    def test_partial_source_totals_cannot_hide_contradictions(self):
        data = fixture_data()
        data["runs"][0]["reportedTotals"] = {"inputTokens": 999}
        with self.assertRaisesRegex(ValueError, "partial inputTokens exceeds"):
            validate(data)

    def test_candidate_never_promoted_by_result_link(self):
        data = fixture_data()
        candidate = data["runs"][1]
        candidate["links"].append({"role": "results-pr", "label": "Results merged into main",
                                  "url": "https://github.com/LittleLittleCloud/Infer.NET/pull/42"})
        validate(data)
        self.assertEqual(candidate["provenance"]["kind"], "candidate")
        data["configurations"][0]["latestBaselineRun"] = candidate["id"]
        with self.assertRaisesRegex(ValueError, "verified main"):
            validate(data)

    def test_configuration_separation_and_incompatible_baseline(self):
        data = fixture_data()
        other = data["runs"][-1]
        other["baselineRunId"] = data["runs"][0]["id"]
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            validate(data)

    def test_invalid_source_totals_and_official_claim(self):
        self.data["runs"][0]["tasks"][0]["inputTokens"] += 1
        with self.assertRaisesRegex(ValueError, "differs from source"):
            validate(self.data)
        manifest = json.loads((ROOT / "site/trends.json").read_text())
        spec = manifest["imports"][0]
        self.assertEqual(import_pi(spec, ROOT)["measuredCommit"], self.data["runs"][0]["measuredCommit"])

    def test_links_require_explicit_roles(self):
        del self.data["runs"][0]["links"][0]["role"]
        with self.assertRaisesRegex(ValueError, "Explicit link role"):
            validate(self.data)

    def test_pull_request_site_acceptance_is_read_only(self):
        workflow = (ROOT / ".github/workflows/site-acceptance.yml").read_text()
        self.assertIn("pull_request:", workflow)
        self.assertIn("contents: read", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("fetch-depth: 0", workflow)
        for command in ("python3 scripts/build-progress-site.py",
                        "python3 scripts/test-home-trends.py", "node scripts/test-home-trends.js"):
            self.assertIn(command, workflow)
        for action in ("pages: write", "id-token: write", "deploy-pages", "upload-pages-artifact"):
            self.assertNotIn(action, workflow)

    def test_fixture_and_payload_are_isolated(self):
        data = fixture_data()
        self.assertEqual(len(data["runs"]), 8)
        self.assertNotIn("fixture-main", homepage(load_data()))
        self.assertNotIn("demo-05", homepage(load_data()))
        data["runs"][0]["change"] = "</script><script>alert(1)</script>"
        payload = homepage(data)
        self.assertNotIn("</script><script>alert", payload)
        self.assertIn("\\u003c/script>", payload)

    def test_build_and_local_links(self):
        subprocess.run(["python3", "scripts/build-progress-site.py"], cwd=ROOT, check=True,
                       capture_output=True, text=True)
        out = ROOT / "artifacts/pages"
        class Links(HTMLParser):
            def handle_starttag(self, tag, attrs):
                for key, value in attrs:
                    if key in {"href", "src"} and value and not value.startswith(("https://", "#")):
                        self.links.append(value.split("#")[0])
        for name in ("index.html", "progress.html", "experiments.html"):
            parser = Links()
            parser.links = []
            text = (out / name).read_text()
            parser.feed(text)
            for link in parser.links:
                self.assertTrue((out / link).is_file(), (name, link))
        index = (out / "index.html").read_text()
        self.assertNotIn('class="notice"', index)
        self.assertNotIn('id="device-topology"', index)
        self.assertIn('id="device-topology"', (out / "progress.html").read_text())
        self.assertEqual(index.count('class="trend-card"'), 4)
        for run in json.loads((out / "trends-data.json").read_text())["runs"]:
            for task in run["tasks"]:
                filename, pointer = task["source"].split("#/")
                source = json.loads((out / filename).read_text())
                for part in pointer.split("/"):
                    source = source[int(part)] if isinstance(source, list) else source[part]
                self.assertEqual(task["taskId"], source["taskId"])
            for link in run["links"]:
                if not link["url"].startswith("https://"):
                    self.assertTrue((out / link["url"].split("#")[0]).is_file())


def write_fixture(output):
    output.mkdir(parents=True, exist_ok=True)
    data = fixture_data()
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>SYNTHETIC trend acceptance fixture</title><link rel="stylesheet" href="style.css">'
        '<body><header>SYNTHETIC TEST DATA ONLY · NOT PRODUCTION</header><main>' +
        homepage(data) + "</main></body></html>")
    (output / "trends-data.json").write_text(json.dumps(data, indent=2))
    (output / "fixture-input.json").write_text(FIXTURES.read_text())
    for name in ("style.css", "trends.js"):
        shutil.copyfile(ROOT / "site" / name, output / name)
    print(f"Synthetic-only browser fixture: {output / 'index.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-output", type=Path)
    args, rest = parser.parse_known_args()
    if args.fixture_output:
        require_output = args.fixture_output.resolve()
        if (not require_output.is_relative_to(ROOT / "artifacts")
                or require_output == ROOT / "artifacts"
                or require_output.is_relative_to(ROOT / "artifacts/pages")):
            parser.error("Fixture output must be a separate artifacts/ subdirectory, never artifacts/pages")
        write_fixture(require_output)
    unittest.main(argv=["test-home-trends.py"] + rest)
