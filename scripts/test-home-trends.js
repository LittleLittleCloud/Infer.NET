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
    this.hidden = false;
    this.focused = false;
    this.scrolled = 0;
  }
  focus(options) { this.focused = true; this.focusOptions = options; }
  scrollIntoView(options) { this.scrolled++; this.scrollOptions = options; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  addEventListener(event, handler) { this.listeners[event] = handler; }
  click(event = {button: 0}) { this.listeners.click({defaultPrevented: false, ...event, preventDefault() { this.defaultPrevented = true; event.preventDefault?.(); }}); }
  querySelector(tag) { return this.children.find(child => child.tagName === tag); }
}
function render(data, hash = "") {
  const elements = {};
  const ids = ["trend-data", "configuration", "run", "selection-notice", "run-title", "run-state",
    "run-identity", "run-change", "run-result", "permalink", "run-metadata", "run-links", "tasks",
    "reported-totals", "run-detail"];
  for (const metric of ["prefill", "decode", "ttft", "pass"]) {
    ids.push(metric + "-chart", metric + "-value", metric + "-coverage", metric + "-scroll-hint");
  }
  ids.push("prefill-clock-note", "decode-clock-note");
  ids.forEach(id => { elements[id] = new Element("div"); });
  elements["trend-data"].textContent = JSON.stringify(data);
  elements.tasks.append(new Element("tbody"), new Element("tfoot"));
  const listeners = {}, history = [];
  const sotaLinks = [];
  for (const run of data.runs.slice(0, 2)) {
    const link = new Element("a");
    link.setAttribute("href", ui.runHash(run));
    sotaLinks.push(link);
  }
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
      if (selector === "a[data-sota-evidence]") return sotaLinks;
      assert.equal(selector, ".trend-point");
      return ["prefill", "decode", "ttft", "pass"].flatMap(key =>
        elements[key + "-chart"].children.filter(child => child.tagName === "button"));
    }
  };
  const window = {document, location, addEventListener: (event, handler) => {listeners[event] = handler;},
    history: {pushState: (state, title, hash) => {location.hash = hash; history.push(hash);}}};
  vm.runInNewContext(fs.readFileSync(path.join(root, "site/trends.js"), "utf8"),
    {window, URLSearchParams, Intl, console});
  return {elements, listeners, window, history, sotaLinks,
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
test("axis ticks are round, short and stay inside the left margin", () => {
  for (const [max, key] of [[12224.66, "ttft"], [4074.887, "ttft"], [531.2, "prefill"], [66.43, "decode"], [0.37, "prefill"], [0, "ttft"]]) {
    const axis = ui.axisModel(max, key);
    assert.equal(axis.ticks.length, 5);
    assert.ok(axis.max >= max, "axis must contain the maximum value");
    assert.equal(axis.max, axis.ticks.at(-1));
    for (const tick of axis.ticks) {
      const label = ui.tickLabel(tick, axis.decimals);
      assert.ok(label.length * 7 + 8 <= ui.plotLeft, `${label} fits the left margin`);
    }
  }
  assert.deepEqual(ui.axisModel(12224.66, "ttft").ticks, [0, 4000, 8000, 12000, 16000]);
  assert.deepEqual(ui.axisModel(100, "pass").ticks, [0, 25, 50, 75, 100]);
  assert.equal(ui.tickLabel(2.5, ui.axisModel(8, "decode").decimals), "2.5");
});
test("missing labels are one per commit group and cannot overlap", () => {
  const data = {...production};
  const all = ui.chartModel(data, "all", "ttft");
  const labels = ui.missingLabels(all);
  const missingPoints = all.points.filter(p => p.value === null).length;
  assert.equal(labels.reduce((sum, label) => sum + label.count, 0), missingPoints);
  assert.ok(labels.length <= all.ticks.length);
  labels.forEach((label, i) => {
    assert.ok(label.text.length * 6.2 <= 90, "label fits the minimum commit slot");
    if (i) assert.ok(label.x - labels[i - 1].x >= 90, "neighbouring labels are separated");
  });
  const many = {...fixtures, runs: fixtures.runs.filter(r => r.id === "fixture-failure" || r.id === "fixture-main")};
  const grouped = ui.missingLabels(ui.chartModel(many, "all", "prefill"));
  assert.equal(grouped.length, 1);
  assert.equal(grouped[0].text, "Not measured");
});
test("chart regions are named, keyboard focusable and announce horizontal scrolling", () => {
  const page = render(production);
  for (const key of ["prefill", "decode", "ttft", "pass"]) {
    const container = page.elements[key + "-chart"];
    assert.equal(container.getAttribute("role"), "region");
    assert.equal(container.getAttribute("tabindex"), "0");
    assert.ok(container.getAttribute("aria-label").includes(`${production.runs.length} runs`));
    const svg = container.children[0];
    const wide = Number.parseFloat(svg.style.width) > container.clientWidth + 1;
    assert.equal(page.elements[key + "-scroll-hint"].hidden, !wide);
    if (wide) {
      assert.ok(container.getAttribute("aria-label").includes("scrolls horizontally"));
      assert.ok(page.elements[key + "-scroll-hint"].textContent.includes(`all ${production.runs.length} runs`));
    }
    assert.equal(page.points(key).length, production.runs.length, "no run point is dropped");
  }
  assert.ok(Number.parseFloat(page.elements["prefill-chart"].children[0].style.width) > 500, "production chart overflows 500px");
  assert.equal(page.elements["prefill-scroll-hint"].hidden, false);
});
test("native-command runs explain unavailable worker-clock charts without substituting values", () => {
  const external = production.runs.find(run => ui.usesNativeClock(run));
  assert.ok(external, "production has a native-command run");
  const page = render(production, ui.runHash(external));
  for (const key of ["prefill", "decode"]) {
    const note = page.elements[key + "-clock-note"];
    assert.equal(note.hidden, false);
    assert.ok(note.textContent.includes("unavailable"));
    assert.ok(note.textContent.includes("no cross-scope ratio"));
    assert.equal(page.elements[key + "-value"].textContent, "Unavailable");
    assert.ok(!/\d/.test(note.textContent), "explanatory note contains no substituted number");
  }
  const worker = production.runs.find(run => !ui.usesNativeClock(run));
  page.window.location.hash = ui.runHash(worker);
  page.listeners.hashchange();
  assert.equal(page.elements["prefill-clock-note"].hidden, true);
});
test("SOTA evidence activation selects, scrolls and focuses details; ordinary selection does not", () => {
  const page = render(production);
  assert.equal(page.elements["run-detail"].scrolled, 0, "page load never scrolls");
  page.points("prefill")[1].click();
  assert.equal(page.elements["run-detail"].scrolled, 0, "chart selection never scrolls");
  assert.equal(page.elements["run-title"].focused, false);
  const target = production.runs[1];
  const link = page.sotaLinks[1];
  const historyBefore = page.history.length;
  let prevented = false;
  link.click({button: 0, preventDefault() { prevented = true; }});
  assert.ok(prevented);
  assert.equal(page.elements.run.value, target.id);
  assert.equal(page.elements["run-detail"].scrolled, 1);
  assert.equal(page.elements["run-title"].focused, true);
  assert.equal(page.elements["run-title"].focusOptions.preventScroll, true);
  assert.ok(page.history.length >= historyBefore);
  // Repeating the same activation still focuses details even though the hash is unchanged.
  link.click({button: 0});
  assert.equal(page.elements["run-detail"].scrolled, 2);
  // Back/forward navigation is handled by selection only and does not scroll.
  page.window.location.hash = ui.runHash(production.runs[0]);
  page.listeners.popstate();
  assert.equal(page.elements.run.value, production.runs[0].id);
  assert.equal(page.elements["run-detail"].scrolled, 2);
  // Modified clicks keep native browser behaviour.
  link.click({button: 0, ctrlKey: true});
  assert.equal(page.elements["run-detail"].scrolled, 2);
});
test("vertical lanes keep points, missing captions, ticks and axis caption disjoint inside the SVG", () => {
  const L = ui.layout, half = L.hitHalf, glyph = 12;
  assert.ok(L.pointBottom + half <= L.missingY - half, "value points end before the missing-point lane");
  assert.ok(L.missingY + half <= L.captionY - glyph, "missing hit targets end before captions");
  assert.ok(L.captionY + 4 < L.tickY - glyph, "captions end before commit ticks");
  assert.ok(L.tickY + 4 < L.axisY - glyph, "ticks end before the axis caption");
  assert.ok(L.axisY + 6 <= L.height, "axis caption is not clipped");
  assert.ok(L.pointTop - half >= 0, "highest point target is not clipped");
  const page = render(production);
  for (const key of ["prefill", "decode", "ttft", "pass"]) {
    const svg = page.elements[key + "-chart"].children[0];
    assert.ok(svg.attributes.viewBox.endsWith(" " + L.height));
    const captions = svg.children.filter(c => c.attributes.class === "missing-label");
    assert.ok(captions.length > 0);
    captions.forEach(c => assert.equal(Number(c.attributes.y), L.captionY));
    page.points(key).forEach(button => {
      const top = Number.parseFloat(button.style.top);
      const missing = button.className.includes("missing");
      assert.ok(top - half >= 0 && top + half <= L.height);
      if (missing) assert.equal(top, L.missingY);
      else assert.ok(top + half <= L.missingY - half, "value point target clears the missing lane");
    });
  }
});
test("worker metric is called unavailable only when native counters exist and the metric is null", () => {
  const external = production.runs.find(run => ui.usesNativeClock(run));
  assert.equal(ui.workerMetricUnavailable(external, "prefill"), true);
  const both = JSON.parse(JSON.stringify(external));
  both.id = "copy-with-both-clocks";
  both.metrics.prefill.value = 1234.5;
  assert.equal(ui.workerMetricUnavailable(both, "prefill"), false);
  assert.equal(ui.workerMetricUnavailable(both, "decode"), true);
  const data = {...production, runs: [both]};
  const page = render(data, ui.runHash(both));
  assert.equal(page.elements["prefill-clock-note"].hidden, true);
  assert.equal(page.elements["prefill-clock-note"].textContent, "");
  assert.equal(page.elements["prefill-value"].textContent, "1,234.5 token/s");
  assert.equal(page.elements["decode-clock-note"].hidden, false);
  assert.equal(page.elements["decode-value"].textContent, "Unavailable");
});
const notebook = require("../site/notebook.js");
test("notebook matching requires every term, ignores case, and never filters on an empty query", () => {
  const text = notebook.normalize("CUDA  QK reuse: values consistent\nFailed experiment");
  assert.ok(notebook.matches(text, ""));
  assert.ok(notebook.matches(text, "cuda FAILED"));
  assert.ok(!notebook.matches(text, "cuda passed"));
  assert.equal(notebook.countText(3, 48, ""), "Showing all 48 notes.");
  assert.ok(notebook.countText(3, 48, " cuda ").includes("3 of 48"));
});
function notebookPage(hash = "") {
  const mk = (extra = {}) => ({hidden: false, dataset: {}, listeners: {}, textContent: "", children: [],
    addEventListener(event, handler) { this.listeners[event] = handler; }, focus() { this.focused = true; },
    scrollIntoView() { this.scrolled = (this.scrolled || 0) + 1; }, querySelectorAll() { return this.entries || []; }, ...extra});
  const notes = [["a-first", "CUDA reuse failed regression"], ["b-second", "Metal residency passed"], ["c-third", "Dual host cuda passed"]];
  const articles = notes.map(([id, text]) => mk({id, textContent: text}));
  const entries = notes.map(([id]) => mk({dataset: {noteLink: id}}));
  const group = mk({entries});
  const elements = {"notebook-tools": mk({hidden: true}), "notebook-filter": mk({value: ""}),
    "notebook-reset": mk({hidden: true}), "notebook-count": mk(), "notebook-empty": mk({hidden: true})};
  const listeners = {}, location = {hash};
  const document = {readyState: "complete", getElementById: id => elements[id],
    querySelectorAll: selector => selector === "article.experiment" ? articles :
      selector === "[data-note-link]" ? entries : selector === ".notebook-group" ? [group] : assert.fail(selector)};
  vm.runInNewContext(fs.readFileSync(path.join(root, "site/notebook.js"), "utf8"),
    {window: {document, location, addEventListener: (event, handler) => {listeners[event] = handler;}}, console});
  return {elements, articles, entries, group, listeners, location};
}
test("notebook filter keeps notes and failures visible by default, filters links and notes together, and resets", () => {
  const page = notebookPage();
  assert.equal(page.elements["notebook-tools"].hidden, false);
  assert.ok(page.articles.every(a => !a.hidden) && page.entries.every(e => !e.hidden));
  assert.equal(page.elements["notebook-count"].textContent, "Showing all 3 notes.");
  page.elements["notebook-filter"].value = "cuda";
  page.elements["notebook-filter"].listeners.input();
  assert.deepEqual(page.articles.map(a => a.hidden), [false, true, false]);
  assert.deepEqual(page.entries.map(e => e.hidden), [false, true, false]);
  assert.equal(page.elements["notebook-reset"].hidden, false);
  page.elements["notebook-filter"].value = "nothing-matches";
  page.elements["notebook-filter"].listeners.input();
  assert.equal(page.elements["notebook-empty"].hidden, false);
  assert.equal(page.group.hidden, true);
  page.elements["notebook-reset"].listeners.click();
  assert.ok(page.articles.every(a => !a.hidden) && !page.group.hidden);
  assert.equal(page.elements["notebook-filter"].focused, true);
  assert.equal(page.elements["notebook-reset"].hidden, true);
});
test("deep links reveal a note hidden by the active filter", () => {
  const page = notebookPage();
  page.elements["notebook-filter"].value = "cuda";
  page.elements["notebook-filter"].listeners.input();
  assert.equal(page.articles[1].hidden, true);
  page.location.hash = "#b-second";
  page.listeners.hashchange();
  assert.equal(page.elements["notebook-filter"].value, "");
  assert.ok(page.articles.every(a => !a.hidden));
  assert.equal(page.articles[1].scrolled, 1);
  assert.ok(page.elements["notebook-count"].textContent.startsWith("Filter cleared"));
  for (const hash of ["#%", "#%E0%A4%A", "#not-a-note", "#", ""]) {
    const safe = notebookPage();
    safe.elements["notebook-filter"].value = "cuda";
    safe.elements["notebook-filter"].listeners.input();
    safe.location.hash = hash;
    safe.listeners.hashchange();
    assert.equal(safe.elements["notebook-filter"].value, "cuda", "unrelated fragment keeps the filter: " + hash);
    assert.deepEqual(safe.articles.map(a => a.hidden), [false, true, false]);
    safe.elements["notebook-filter"].value = "metal";
    safe.elements["notebook-filter"].listeners.input();
    assert.deepEqual(safe.articles.map(a => a.hidden), [true, false, true], "filtering still works after: " + hash);
  }
  assert.doesNotThrow(() => notebookPage("#%"));
  const initial = notebookPage("#c-third");
  assert.ok(initial.articles.every(a => !a.hidden));
});
console.log(`${checks} homepage selection/render acceptance checks passed (offline DOM, no browser layout).`);
