#!/usr/bin/env node
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {execFileSync} = require("node:child_process");
const root = path.resolve(__dirname, "..");
const ui = require("../site/trends.js");
const {production, fixtures} = JSON.parse(execFileSync("python3", ["-c",
  "import importlib.util,json; s=importlib.util.spec_from_file_location('fixture','scripts/test-home-trends.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps({'production': m.load_data(), 'fixtures': m.fixture_data()}))"
], {cwd: root, env: {...process.env, PYTHONPATH: path.join(root, "scripts")}, encoding: "utf8"}));
const historical = {...production, runs: ["pi-main-rerun-20261023"].map(id => {
  const run = production.runs.find(run => run.id === id);
  assert.ok(run, "historical run " + id);
  return run;
})};
const historicalConfigId = historical.runs[0].configurationId;
let checks = 0;
function test(name, run) {
  run();
  checks++;
  console.log("ok " + checks + " - " + name);
}
test("defaults to latest explicitly verified main baseline, not highest pass rate", () => {
  assert.ok(production.configurations[0].latestBaselineRun);
  assert.equal(ui.resolveSelection(production, "").run.id, production.configurations[0].latestBaselineRun);
  assert.equal(ui.resolveSelection(fixtures, "").run.id, "fixture-repeat");
});
test("stable experiment/run deep link resolves both and rejects mismatched IDs", () => {
  for (const run of fixtures.runs) {
    assert.equal(ui.resolveSelection(fixtures, ui.runHash(run)).run.id, run.id);
  }
  assert.equal(ui.resolveSelection(production, "#experiment=pi-humaneval-rerun-20").run.id, "pi-main-rerun-20261023");
  for (const run of historical.runs) {
    assert.equal(ui.resolveSelection(production, ui.runHash(run)).run.id, run.id);
  }
  const mismatch = ui.resolveSelection(fixtures, "#experiment=wrong&run=fixture-candidate");
  assert.ok(mismatch.error.includes("Unknown"));
  assert.equal(mismatch.run.id, "fixture-repeat");
  assert.equal(ui.resolveSelection(fixtures, "#prefill-trend").error, "");
});
test("all runs are visible by default and optional filter narrows the view", () => {
  const other = fixtures.runs.at(-1);
  assert.equal(ui.resolveSelection(fixtures, ui.runHash(other)).run.configurationId, other.configurationId);
  assert.equal(ui.chartModel(fixtures, "all", "prefill").points.length, fixtures.runs.length);
  const points = ui.chartModel(fixtures, other.configurationId, "prefill").points;
  assert.deepEqual(points.map(p => p.run.id), ["fixture-bf16"]);
});
test("repeated runtime SHA has one commit tick but individually selectable runs", () => {
  const repeated = {...fixtures, runs: fixtures.runs.filter(r =>
    r.id === "fixture-main" || r.id === "fixture-repeat")};
  const chart = ui.chartModel(repeated, repeated.configurations[0].id, "pass");
  assert.equal(chart.commits.length, 1);
  assert.equal(chart.points.length, 2);
  assert.equal(chart.counts[0], 2);
  assert.notEqual(chart.points[0].x, chart.points[1].x);
  assert.deepEqual(chart.points.map(p => p.value), [10, 10]);
});
test("missing and real zero remain distinct, failure runs remain selectable", () => {
  assert.equal(ui.format(null), "Unavailable");
  assert.equal(ui.format(0), "0");
  const chart = ui.chartModel(fixtures, fixtures.configurations[0].id, "prefill");
  assert.equal(chart.points.find(p => p.run.id === "fixture-zero").value, 0);
  assert.equal(chart.points.find(p => p.run.id === "fixture-failure").value, null);
  assert.equal(ui.resolveSelection(fixtures, "#run=fixture-failure").run.id, "fixture-failure");
  assert.equal(ui.resolveSelection(fixtures, "#run=fixture-unverified").run.metrics.pass.value, null);
});
test("result publication never changes candidate identity", () => {
  const run = ui.resolveSelection(fixtures, "#run=fixture-candidate").run;
  assert.equal(run.provenance.kind, "candidate");
  assert.equal(run.state, "pending-review");
  assert.notEqual(run.id, fixtures.configurations[0].latestBaselineRun);
});
test("commit timeline and main lines ignore shuffled run and timeline input order", () => {
  const shuffled = {...fixtures, runs: [...fixtures.runs].reverse(),
    commitTimeline: [...fixtures.commitTimeline].reverse()};
  const expected = ui.chartModel(fixtures, fixtures.configurations[0].id, "pass");
  const actual = ui.chartModel(shuffled, fixtures.configurations[0].id, "pass");
  assert.deepEqual(actual, expected);
  assert.equal(actual.points.at(-1).run.id, "fixture-unverified");
  assert.deepEqual(actual.points.slice(0, 2).map(p => p.run.id), ["fixture-main", "fixture-repeat"]);
});
test("same-value repeated runs retain nonoverlapping fine/coarse targets even in large cohorts", () => {
  for (const target of [34, 44]) {
    const many = {...historical, runs: Array.from({length: 12}, (_, i) => ({
      ...historical.runs[0], id: "repeat-" + String(i).padStart(2, "0")
    }))};
    const chart = ui.chartModel(many, historicalConfigId, "pass", 280, target);
    assert.ok(chart.width > 280);
    chart.points.forEach((point, i) => {
      assert.ok(point.x - target / 2 >= 0);
      assert.ok(point.x + target / 2 <= chart.width);
      if (i) assert.ok(point.x - chart.points[i - 1].x >= target + 8);
    });
  }
});

// Minimal DOM adapter exercises real render/event code offline; not a browser/layout substitute.
class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.style = {};
    this.listeners = {};
    this.clientWidth = 500;
    this.className = "";
    this.textContent = "";
    this.value = "";
    this.classList = {add: name => { this.className += " " + name; }};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  addEventListener(event, handler) { this.listeners[event] = handler; }
  click() { this.listeners.click(); }
  querySelector(tag) { return this.children.find(child => child.tagName === tag); }
}
function render(data, hash = "") {
  const elements = {};
  const ids = ["trend-data", "configuration", "run", "selection-notice", "run-title", "run-state",
    "run-identity", "run-change", "run-result", "permalink", "run-metadata", "run-links", "tasks",
    "reported-totals"];
  for (const metric of ["prefill", "decode", "ttft", "pass"]) {
    ids.push(metric + "-chart", metric + "-value", metric + "-coverage");
  }
  ids.forEach(id => { elements[id] = new Element("div"); });
  elements["trend-data"].textContent = JSON.stringify(data);
  elements.tasks.append(new Element("tbody"), new Element("tfoot"));
  const listeners = {}, history = [];
  const location = {hash};
  const document = {
    readyState: "complete",
    getElementById: id => {
      assert.ok(elements[id], "known element " + id);
      return elements[id];
    },
    createElement: tag => new Element(tag),
    createElementNS: (ns, tag) => new Element(tag),
    querySelectorAll: selector => {
      assert.equal(selector, ".trend-point");
      return ["prefill", "decode", "ttft", "pass"].flatMap(key =>
        elements[key + "-chart"].children.filter(child => child.tagName === "button"));
    }
  };
  const window = {document, location, addEventListener: (event, handler) => {listeners[event] = handler;},
    history: {pushState: (state, title, hash) => {location.hash = hash; history.push(hash);}}};
  vm.runInNewContext(fs.readFileSync(path.join(root, "site/trends.js"), "utf8"),
    {window, URLSearchParams, Intl, console});
  return {elements, listeners, window, history,
    points: key => elements[key + "-chart"].children.filter(child => child.tagName === "button")};
}
test("production DOM renders missing metrics as selectable unmeasured runs", () => {
  const page = render(historical, "#run=pi-main-rerun-20261023");
  assert.equal(page.elements.configuration.value, "all");
  assert.equal(page.points("prefill").length, 1);
  assert.equal(page.points("ttft").length, 1);
  assert.ok(page.points("ttft")[0].className.includes("missing"));
  assert.ok(page.elements["ttft-chart"].children[0].children.some(child => child.textContent === "Not measured"));
  assert.equal(page.points("pass").length, 1);
  assert.equal(page.elements["prefill-value"].textContent, "Unavailable");
  assert.equal(page.elements["pass-value"].textContent, "95% · 19/20");
  assert.equal(page.elements.tasks.querySelector("tbody").children.length, 20);
  const inputTotal = page.elements.tasks.querySelector("tfoot").children[0].children[2].textContent;
  assert.ok(inputTotal.includes("Unavailable"));
  assert.ok(inputTotal.includes("0/20"));
});
test("point clicks update every metric, all task rows and pressed state without recreating focus target", () => {
  const page = render(fixtures);
  const point = page.points("prefill").find(p => p.dataset.run === "fixture-candidate");
  point.click();
  assert.equal(page.elements.run.value, "fixture-candidate");
  assert.equal(page.elements["ttft-value"].textContent, "120 ms");
  assert.equal(page.elements["pass-value"].textContent, "10% · 2/20");
  assert.ok(page.elements["run-state"].textContent.startsWith("candidate"));
  assert.equal(page.elements.tasks.querySelector("tbody").children.length, 20);
  assert.ok(page.history.at(-1).includes("run=fixture-candidate"));
  assert.ok(page.points("prefill").includes(point));
  for (const key of ["prefill", "decode", "ttft", "pass"]) {
    assert.equal(page.points(key).find(p => p.dataset.run === "fixture-candidate").getAttribute("aria-pressed"), "true");
    assert.ok(page.points(key).every(p => p.getAttribute("aria-label").includes("measured")));
  }
});
test("main/candidate/baseline have distinct shapes and only explicit baseline branches", () => {
  const page = render(fixtures);
  const points = page.points("prefill");
  const baseline = points.find(p => p.dataset.run === "fixture-repeat");
  const candidate = points.find(p => p.dataset.run === "fixture-candidate");
  assert.ok(baseline.className.includes("main"));
  assert.ok(baseline.className.includes("latest-baseline"));
  assert.ok(candidate.className.includes("candidate"));
  assert.equal(candidate.type, "button");
  assert.equal(candidate.getAttribute("aria-controls"), "run-detail");
  const svg = page.elements["prefill-chart"].children[0];
  const branches = svg.children.filter(p => p.attributes.class === "candidate-line");
  assert.equal(branches.length, 3);
  assert.equal(new Set(branches.map(line => line.attributes.x1)).size, 1);
  assert.equal(svg.children.filter(p => p.attributes.class === "main-line").length, 1);
});
test("higher plot position always means better measured performance", () => {
  const page = render(fixtures);
  const top = (key, id) => Number.parseFloat(page.points(key).find(p => p.dataset.run === id).style.top);
  assert.ok(top("prefill", "fixture-main") < top("prefill", "fixture-regressed"));
  assert.ok(top("decode", "fixture-main") < top("decode", "fixture-regressed"));
  assert.ok(top("pass", "fixture-main") < top("pass", "fixture-failure"));
  assert.ok(top("ttft", "fixture-zero") < top("ttft", "fixture-main"));
});
test("partial task totals and merge-not-remeasured identity are explicit", () => {
  const modified = JSON.parse(JSON.stringify(fixtures));
  modified.runs[0].provenance.mainCommit = "1234567890123456789012345678901234567890";
  const page = render(modified, "#run=fixture-main");
  const metadata = page.elements["run-metadata"].children.map(n => n.textContent).join(" ");
  assert.ok(metadata.includes("merge revision NOT remeasured"));
  assert.ok(metadata.includes(modified.runs[0].measuredCommit));
  const total = page.elements.tasks.querySelector("tfoot").children[0].children[2].textContent;
  assert.ok(total.includes("1,000"));
  assert.ok(total.includes("2/20"));
  assert.ok(total.includes("PARTIAL"));
});
test("results absence uses explicit link role, never the display label", () => {
  const data = JSON.parse(JSON.stringify(fixtures));
  data.runs[0].links.push({role: "results-pr", label: "Evaluation record",
    url: "https://github.com/LittleLittleCloud/Infer.NET/pull/42"});
  const present = render(data, "#run=fixture-main").elements["run-links"].children;
  assert.ok(present.some(link => link.textContent === "Evaluation record"));
  assert.ok(!present.some(link => link.textContent === "Separate results PR: not recorded"));
  data.runs[0].links.at(-1).role = "harness-pr";
  data.runs[0].links.at(-1).label = "Results PR mentioned by harness";
  const absent = render(data, "#run=fixture-main").elements["run-links"].children;
  assert.ok(absent.some(link => link.textContent === "Separate results PR: not recorded"));
});
test("run controls select failures without speed points, real zero, and browser Back", () => {
  const page = render(fixtures);
  page.elements.run.value = "fixture-failure";
  page.elements.run.listeners.change();
  assert.equal(page.elements["prefill-value"].textContent, "Unavailable");
  assert.ok(page.elements["run-state"].textContent.includes("failed"));
  page.elements.run.value = "fixture-zero";
  page.elements.run.listeners.change();
  assert.equal(page.elements["prefill-value"].textContent, "0 token/s");
  assert.equal(page.elements["ttft-value"].textContent, "0 ms");
  page.window.location.hash = "#run=fixture-main&experiment=synthetic-tests";
  page.listeners.popstate();
  assert.equal(page.elements.run.value, "fixture-main");
  assert.ok(page.elements["run-state"].textContent.startsWith("main"));
});
test("configuration filter narrows, then deep link restores all runs if needed", () => {
  const page = render(fixtures);
  assert.equal(page.elements.configuration.value, "all");
  assert.equal(page.points("prefill").length, fixtures.runs.length);
  page.elements.configuration.value = "fixture-incompatible-bf16";
  page.elements.configuration.listeners.change();
  assert.deepEqual(page.points("prefill").map(p => p.dataset.run), ["fixture-bf16"]);
  page.window.location.hash = "#run=fixture-candidate";
  page.listeners.hashchange();
  assert.equal(page.elements.configuration.value, "all");
  assert.equal(page.points("prefill").length, fixtures.runs.length);
});
test("default homepage exposes the BF16/2048 run across configurations", () => {
  const page = render(production);
  const candidateId = "issue54-bf16-2048-human20-20261024";
  assert.equal(page.elements.configuration.value, "all");
  assert.deepEqual(new Set(page.points("prefill").map(point => point.dataset.run)),
    new Set(production.runs.map(run => run.id)));
  const prefillSvg = page.elements["prefill-chart"].children[0];
  assert.equal(prefillSvg.children.filter(child =>
    ["main-line", "candidate-line"].includes(child.attributes.class)).length, 0,
  "different configurations must not be connected as one trend");
  assert.ok(page.points("ttft").find(point => point.dataset.run === candidateId).className.includes("missing"));
  assert.ok(page.elements.run.children.some(option => option.value === candidateId));
  page.elements.run.value = candidateId;
  page.elements.run.listeners.change();
  assert.equal(page.elements.configuration.value, "all");
  assert.equal(page.points("prefill").length, production.runs.length);
  assert.equal(page.elements["prefill-value"].textContent, "417.363 token/s");
  assert.equal(page.elements["ttft-value"].textContent, "Unavailable");
  assert.equal(page.elements["ttft-coverage"].textContent, "0/116 client first-token samples");
});
console.log(`${checks} homepage selection/render acceptance checks passed (offline DOM, no browser layout).`);
