#!/usr/bin/env python3
"""Offline acceptance/data tests; optional explicitly synthetic browser fixture."""
import argparse
import copy
from html.parser import HTMLParser
import json
import re
from pathlib import Path
import shutil
import subprocess
import unittest

from home_trends import ROOT, aggregate_requests, homepage, import_pi, load_data, total, validate, sota_summary

FIXTURES = ROOT / "tests/home-trends/fixtures.json"


def historical_runs(data):
    by_id = {run["id"]: run for run in data["runs"]}
    return by_id["main-cd6f7da-humaneval20-20261023"], by_id["pi-main-rerun-20261023"]


def fixture_data():
    fixture = json.loads(FIXTURES.read_text())
    data = load_data()
    original, _ = historical_runs(data)
    config = copy.deepcopy(next(c for c in data["configurations"]
                                if c["id"] == original["configurationId"]))
    data["configurations"] = [config]
    data["runs"] = []
    data["commitTimeline"] = [
        {"sha": sha, "committedAt": f"2026-01-{i + 1:02d}T00:00:00Z",
         "evidence": "fixture-input.json"}
        for i, sha in enumerate(dict.fromkeys(s["commit"] for s in fixture["scenarios"]))
    ]
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
            "requestCount": 4, "binarySha256": {}, "binaryEvidence": None,
            "links": []
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
        self.first, self.rerun = historical_runs(self.data)

    def test_sota_requires_official_passes_complete_coverage_and_separate_scopes(self):
        entries = {(e['family'], e['metric']): e for e in sota_summary(self.data)}
        self.assertGreaterEqual(entries['infernet', 'prefill']['value'], 531.19940338122)
        self.assertAlmostEqual(next(r for r in self.data['runs'] if r['id'] == 'issue58-m5pro-local-human20-20261024')['metrics']['decode']['value'], 66.4336423808985)
        self.assertGreaterEqual(entries['infernet', 'decode']['value'], 66.4336423808985)
        self.assertAlmostEqual(entries['external', 'prefill']['value'], 3066.6108426327232)
        self.assertAlmostEqual(entries['external', 'decode']['value'], 148.91366915373305)
        for entry in entries.values():
            self.assertTrue(entry['run']['metrics']['pass']['verified'])
        # A faster record without official checks cannot become verified SOTA.
        candidate = next(r for r in self.data['runs'] if r['id'] == 'issue54-main99e-human20-20261024')
        candidate['metrics']['prefill']['value'] = 1e9
        self.assertEqual(sota_summary(self.data)[0]['run']['id'], entries['infernet', 'prefill']['run']['id'])
        # Missing native observations never become zero or an inferred speed.
        external = next(r for r in self.data['runs'] if 'splash' in r['runtime'])
        external['requests'][0].pop('nativeCommandDelta')
        self.assertIsNone(sota_summary(self.data)[2]['value'])
        self.assertIn('P / D SOTA', homepage(self.data))
        self.assertIn('No speedup ratio is computed across these measurement scopes', homepage(self.data))

    def test_production_only_real_history_and_measured_worker_speed(self):
        self.assertEqual(self.first["measuredCommit"], "cd6f7da401fadf044dac13e87b415d27bb5eadfb")
        self.assertEqual(self.rerun["measuredCommit"], "e017d84e805c016d5912b5051e4c69f758056363")
        self.assertGreater(self.first["metrics"]["prefill"]["value"], 500)
        self.assertGreater(self.first["metrics"]["decode"]["value"], 60)
        for key in ("prefill", "decode", "ttft"):
            self.assertIsNone(self.rerun["metrics"][key]["value"])
            self.assertEqual(self.rerun["metrics"][key]["samples"], 0)
        for run in (self.first, self.rerun):
            self.assertEqual(len(run["tasks"]), 20)
            self.assertEqual(len(run["binarySha256"]), 4)
        self.assertNotEqual(self.first["id"], self.rerun["id"])

    def test_source_rows_and_totals_exactly_mapped(self):
        manifest = json.loads((ROOT / "site/trends.json").read_text())
        specs = {spec["id"]: spec for spec in manifest["imports"]}
        for run in (self.rerun,):
            spec = specs[run["id"]]
            source = json.loads((ROOT / spec["source"]).read_text())
            rows = source["results"].get("rows", source.get("rows"))
            self.assertEqual(len(run["tasks"]), len(rows))
            for index, (task, row) in enumerate(zip(run["tasks"], rows)):
                for field in ("taskId", "wallSeconds", "inputTokens", "outputTokens", "modelRequests"):
                    self.assertEqual(task[field], row.get(field))
                self.assertTrue(task["source"].endswith("/" + str(index)))
            self.assertAlmostEqual(run["totals"]["wallSeconds"]["value"],
                                   source["results"]["wallSeconds"]["total"], delta=.011)
        first, rerun = self.first, self.rerun
        self.assertEqual(first["totals"]["inputTokens"]["value"], 211383)
        self.assertEqual(first["totals"]["outputTokens"]["value"], 19892)
        self.assertIsNone(rerun["totals"]["inputTokens"]["value"])
        self.assertFalse(rerun["totals"]["inputTokens"]["complete"])
        self.assertEqual(rerun["reportedTotals"]["inputTokens"], 241162)

    def test_official_denominator_includes_failure(self):
        first, rerun = self.first, self.rerun
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
        run = self.first
        run["tasks"][0].update({"status": "unverified", "officialValidation": "unverified"})
        for config in self.data["configurations"]:
            if config["latestBaselineRun"] == run["id"]:
                config["latestBaselineRun"] = None
        validate(self.data)
        self.assertIsNone(run["metrics"]["pass"]["value"])
        run["tasks"][0]["status"] = "passed"
        with self.assertRaisesRegex(ValueError, "official task assertions"):
            validate(self.data)

    def test_absence_zero_and_partial_fields(self):
        tasks = self.first["tasks"][:3]
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
            historical_runs(data)[0]["tasks"][0][field] = invalid
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
        historical_commit = next(commit for commit in self.data["commitTimeline"]
                                 if commit["sha"] == self.first["measuredCommit"])
        stamp = subprocess.check_output(
            ["git", "show", "-s", "--format=%ct", historical_commit["sha"]],
            cwd=ROOT, text=True).strip()
        from datetime import datetime, timezone
        self.assertEqual(historical_commit["committedAt"],
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
        self.first["tasks"][0]["inputTokens"] += 1
        with self.assertRaisesRegex(ValueError, "differs from cumulative model requests"):
            validate(self.data)
        manifest = json.loads((ROOT / "site/trends.json").read_text())
        spec = next(spec for spec in manifest["imports"] if spec["id"] == self.rerun["id"])
        self.assertEqual(import_pi(spec, ROOT)["measuredCommit"], self.rerun["measuredCommit"])

    def test_links_require_explicit_roles(self):
        del self.first["links"][0]["role"]
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

    def test_issue54_retained_missing_ttft_and_separate_observed_retake(self):
        by_id = {run["id"]: run for run in self.data["runs"]}
        old = by_id["issue54-bf16-2048-human20-20261024"]
        retake = by_id["issue54-bf16-2048-ttft-human20-20261024"]
        self.assertIsNone(old["metrics"]["ttft"]["value"])
        self.assertEqual(old["metrics"]["ttft"]["samples"], 0)
        self.assertEqual(old["requestCount"], 116)
        self.assertEqual(retake["measuredCommit"], old["measuredCommit"])
        self.assertNotEqual(retake["configurationId"], old["configurationId"])
        self.assertEqual(retake["requestCount"], 121)
        self.assertEqual(retake["metrics"]["ttft"]["samples"], 121)
        self.assertAlmostEqual(retake["metrics"]["ttft"]["value"], 7998.56475, places=4)
        self.assertAlmostEqual(retake["metrics"]["prefill"]["value"], 413.43, delta=.01)
        self.assertAlmostEqual(retake["metrics"]["decode"]["value"], 66.42, delta=.01)
        self.assertIsNone(retake["metrics"]["pass"]["value"])

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
        self.assertIn('issue54-bf16-2048-human20-20261024', index)
        self.assertIn('Configuration filter <select id="configuration"', index)
        self.assertIn('All runs appear by default', index)
        self.assertIn('a separate BF16/2048 retake supplies the observed TTFT point', index)
        self.assertNotIn('id="device-topology"', index)
        self.assertIn('id="device-topology"', (out / "progress.html").read_text())
        self.assertEqual(index.count('class="trend-card"'), 4)
        for run in (historical_runs(json.loads((out / "trends-data.json").read_text()))[1],):
            for task in run["tasks"]:
                filename, pointer = task["source"].split("#/")
                source = json.loads((out / filename).read_text())
                for part in pointer.split("/"):
                    source = source[int(part)] if isinstance(source, list) else source[part]
                self.assertEqual(task["taskId"], source["taskId"])
            for link in run["links"]:
                if not link["url"].startswith("https://"):
                    self.assertTrue((out / link["url"].split("#")[0]).is_file())


    def test_sota_cards_expose_derived_configuration_and_adjacent_qualification(self):
        html = homepage(self.data)
        cards = re.findall(r'<article class="sota-card.*?</article>', html, re.S)
        self.assertEqual(len(cards), 4)
        names = re.findall(r'data-sota-evidence href="[^"]+" aria-label="([^"]+)"', html)
        self.assertEqual(len(names), 4)
        self.assertEqual(len(set(names)), 4, "each SOTA link needs a distinct accessible name")
        for entry, card in zip(sota_summary(self.data), cards):
            config = entry["config"]
            self.assertIn(f"KV: {config['kv']}", card)
            self.assertIn(config["topology"], card)
            self.assertIn("sota-chips", card)
            self.assertIn("no cross-scope ratio is computed", card)
            self.assertIn("DFlash2: used" if "DFlash2" in config["label"] + config["chunking"]
                          else "DFlash2: not in configuration", card)
            self.assertIn("Clock: Native command" if entry["family"] == "external" else "Clock: Worker", card)
            self.assertNotIn("<details", card, "configuration chips must be visible without expansion")
        self.assertNotRegex(re.sub(r"<[^>]+>", " ", html), r"\d+(\.\d+)?\s*[x×] (faster|speedup)")
        self.assertIn('id="run-title" tabindex="-1"', html)
        for key in ("prefill", "decode", "ttft", "pass"):
            self.assertIn(f'id="{key}-scroll-hint" hidden', html)
        for key in ("prefill", "decode"):
            self.assertIn(f'id="{key}-clock-note" hidden', html)

    def test_external_qualifier_depends_on_worker_metric_presence(self):
        self.assertIn("is unavailable for this run", homepage(self.data))
        external = next(r for r in self.data["runs"] if "splash" in r["runtime"])
        external["metrics"]["prefill"]["value"] = 1234.5
        html = homepage(self.data)
        cards = re.findall(r'<article class="sota-card external.*?</article>', html, re.S)
        self.assertEqual(len(cards), 2)
        self.assertNotIn("unavailable", cards[0])
        self.assertIn("reported separately", cards[0])
        self.assertIn("unavailable", cards[1])

    def test_external_card_describes_only_its_missing_worker_metric(self):
        data = copy.deepcopy(self.data)
        external = next(run for run in data["runs"] if "splash" in run["runtime"])
        external["metrics"]["decode"]["value"] = 1.0
        html = homepage(data)
        self.assertIn("Standard worker-clock prefill is unavailable", html)
        self.assertNotIn("Standard worker-clock decode is unavailable", html)
        self.assertIn("Any worker-clock value for this run is reported separately", html)

    def test_css_contrast_and_minimum_text_size(self):
        css = (ROOT / "site/style.css").read_text()
        def channel(value):
            value /= 255
            return value / 12.92 if value <= .03928 else ((value + .055) / 1.055) ** 2.4
        def luminance(color):
            r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
            return .2126 * channel(r) + .7152 * channel(g) + .0722 * channel(b)
        def contrast(a, b):
            high, low = sorted((luminance(a), luminance(b)), reverse=True)
            return (high + .05) / (low + .05)
        variables = dict(re.findall(r"--([a-z]+):(#[0-9a-f]{6})", css))
        self.assertGreaterEqual(contrast(variables["gold"], variables["paper"]), 4.5)
        self.assertGreaterEqual(contrast(variables["gold"], "#ffffff"), 4.5)
        self.assertGreaterEqual(contrast(variables["muted"], variables["paper"]), 4.5)
        self.assertGreaterEqual(contrast(variables["accent"], variables["paper"]), 4.5)
        small = [int(v) for v in re.findall(r"font-size:(\d+)px", css)]
        self.assertGreaterEqual(min(small), 12, "no text below 12px")

    def test_notebook_groups_filter_fallback_and_anchors(self):
        subprocess.run(["python3", "scripts/build-progress-site.py"], cwd=ROOT, check=True,
                       capture_output=True, text=True)
        out = ROOT / "artifacts/pages"
        notes = json.loads((ROOT / "site/experiments.json").read_text())
        html = (out / "experiments.html").read_text()
        article_ids = re.findall(r'<article class="experiment" id="([^"]+)"', html)
        self.assertEqual(article_ids, [n["id"] for n in notes])
        self.assertEqual(len(article_ids), len(notes))
        linked = re.findall(r'<li data-note-link="([^"]+)"><a href="#([^"]+)">', html)
        self.assertEqual(sorted(a for a, _ in linked), sorted(article_ids))
        self.assertTrue(all(a == b for a, b in linked))
        # Without JavaScript the tools stay hidden and no note or link carries hidden.
        self.assertIn('id="notebook-tools" hidden', html)
        self.assertEqual(html.count("<article"), html.count("</article>"))
        self.assertNotRegex(html, r'<(article|li)[^>]*\shidden')
        self.assertIn('<script src="notebook.js" defer>', html)
        self.assertTrue((out / "notebook.js").is_file())
        dated = re.findall(r'<h2 id="notes-date-([^"]+)">[^<]* · (\d+) notes</h2>', html)
        self.assertEqual(sum(int(c) for _, c in dated), len(notes))
        self.assertEqual([d for d, _ in dated], sorted((d for d, _ in dated if d != "none"), reverse=True) +
                         (["none"] if any(d == "none" for d, _ in dated) else []))
        for note in notes:
            match = re.match(r"(\d{4}-\d\d-\d\d)\b", note["kind"])
            self.assertIn(note["id"], dict(linked))
            section = re.search(rf'<h2 id="notes-date-{match.group(1) if match else "none"}">.*?</ul>', html, re.S).group(0)
            self.assertIn(f'data-note-link="{note["id"]}"', section)
        # Retained failures remain visible and no outcome badge is inferred from prose.
        self.assertIn("first-public-ci", article_ids)
        self.assertNotRegex(html, r'class="[^"]*(badge|outcome)')

    def test_progress_preserves_evidence_and_separates_status_classes(self):
        subprocess.run(["python3", "scripts/build-progress-site.py"], cwd=ROOT, check=True,
                       capture_output=True, text=True)
        html = (ROOT / "artifacts/pages/progress.html").read_text()
        for token in ("e017d84", "58bf950", "2760e44", "9.92e-4", "8.01e-6", "0.00704", "774.3 seconds",
                      "HumanEval/149", "issues/32", "pull/31", "pull/30", "pull/15", "pull/14", "pull/10",
                      "0 of 20 layer checks passed", "262K full model is still not accepted",
                      "Windows still ran as a foreground agent", "cannot be claimed"):
            self.assertIn(token, html)
        self.assertEqual(html.count('kicker">Snapshot finding<'), 3)
        self.assertEqual(html.count('kicker">Historical result<'), 7)
        self.assertIn("Recorded snapshot of 2026-10-23", html)
        self.assertIn('href="index.html">commit trends page', html)
        self.assertIn("Three BF16 high/low product terms for QKV", html)
        self.assertNotIn("diagnosis is in progress", html)
        self.assertIn("output token sequences matching across 8 requests", html)
        self.assertNotIn("8 output tokens", html)
        self.assertIn("Unmeasured target", html)
        self.assertIn("Unmeasured estimate", html)
        self.assertNotIn("<details", html.split('id="milestones"')[1].split('id="device-topology"')[0],
                         "no accordion in this phase")
        self.assertEqual(html.count("<p><p"), 0)


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
