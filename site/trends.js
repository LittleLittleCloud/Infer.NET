/* Local, dependency-free rendering; pure selection helpers are also exercised by Node. */
(function (scope) {
  "use strict";
  const specs = {
    prefill: {unit: "token/s", label: "Prefill"},
    decode: {unit: "token/s", label: "Decode"},
    ttft: {unit: "ms", label: "TTFT request median"},
    pass: {unit: "%", label: "Official task pass rate"}
  };
  const missing = "未取得";
  const format = value => value === null ? missing : new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 3
  }).format(value);
  const runHash = run => "#experiment=" + encodeURIComponent(run.experimentId) +
    "&run=" + encodeURIComponent(run.id);
  function defaultRun(data, configurationId) {
    const config = data.configurations.find(c => c.id === configurationId) || data.configurations[0];
    if (!config) return undefined;
    return data.runs.find(r => r.id === config.latestBaselineRun) ||
      data.runs.find(r => r.configurationId === config.id);
  }
  function resolveSelection(data, hash, configurationId) {
    const params = new URLSearchParams(hash.replace(/^#/, ""));
    const runId = params.get("run"), experimentId = params.get("experiment");
    const run = data.runs.find(r => (!runId || r.id === runId) &&
      (!experimentId || r.experimentId === experimentId));
    if (run && (runId || experimentId)) return {run, error: ""};
    const requested = runId || experimentId;
    return {run: defaultRun(data, configurationId),
      error: requested ? "Unknown experiment/run link. Showing the configured default run." : ""};
  }
  function seriesRuns(data, configId = "all") {
    const timeline = [...data.commitTimeline].sort((a, b) =>
      a.committedAt.localeCompare(b.committedAt) || a.sha.localeCompare(b.sha));
    const rank = new Map(timeline.map((commit, i) => [commit.sha, i]));
    return data.runs.filter(r => configId === "all" || r.configurationId === configId).sort((a, b) =>
      rank.get(a.measuredCommit) - rank.get(b.measuredCommit) ||
      a.date.localeCompare(b.date) || a.id.localeCompare(b.id));
  }
  function chartModel(data, configId = "all", key, availableWidth = 640, targetSize = 34) {
    const runs = seriesRuns(data, configId);
    const commits = [...new Set(runs.map(r => r.measuredCommit))];
    const counts = commits.map(sha => runs.filter(r => r.measuredCommit === sha).length);
    const spacing = targetSize + 8;
    const slots = counts.map(count => Math.max(90, count * spacing));
    const minimum = 90 + slots.reduce((sum, slot) => sum + slot, 0);
    const width = Math.max(availableWidth, minimum);
    const extra = (width - minimum) / Math.max(1, commits.length);
    let left = 60;
    const ticks = commits.map((sha, i) => {
      const slot = slots[i] + extra;
      const tick = {sha, x: left + slot / 2};
      left += slot;
      return tick;
    });
    const points = runs.map(run => {
      const index = commits.indexOf(run.measuredCommit);
      const siblings = runs.filter(r => r.measuredCommit === run.measuredCommit);
      const offset = (siblings.indexOf(run) - (siblings.length - 1) / 2) * spacing;
      return {run, x: ticks[index].x + offset, value: run.metrics[key].value};
    });
    return {commits, counts, points, ticks, width};
  }
  const api = {format, runHash, defaultRun, resolveSelection, seriesRuns, chartModel};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (!scope.document) return;
  const document = scope.document;
  function start() {
    const embedded = document.getElementById("trend-data");
    if (!embedded) return;
    const data = JSON.parse(embedded.textContent);
    const byId = id => document.getElementById(id);
    const node = (tag, text, className) => {
      const el = document.createElement(tag);
      if (text !== undefined) el.textContent = text;
      if (className) el.className = className;
      return el;
    };
    const svgNode = (tag, attributes, text) => {
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, String(value)));
      if (text !== undefined) el.textContent = text;
      return el;
    };
    const configControl = byId("configuration"), runControl = byId("run");
    const initial = resolveSelection(data, scope.location.hash);
    let selected = initial.run;
    let filterId = "all";
    if (!selected) {
      byId("selection-notice").textContent = "No measured runs available. See the experiment notebook.";
      return;
    }
    const allOption = node("option", "All runs");
    allOption.value = "all";
    configControl.append(allOption);
    data.configurations.forEach(config => {
      const option = node("option", config.label);
      option.value = config.id;
      configControl.append(option);
    });
    function config() {
      return data.configurations.find(c => c.id === selected.configurationId);
    }
    function state(run) {
      const baseline = data.configurations.find(c => c.id === run.configurationId).latestBaselineRun === run.id;
      return run.provenance.kind + " · " + run.state + (baseline ? " · latest verified main baseline" : "");
    }
    function metricText(run, key) {
      const metric = run.metrics[key];
      if (key === "pass") return metric.verified ?
        `${format(metric.value)}% · ${metric.passed}/${metric.total}` :
        `Unverified · ${metric.passed} confirmed / ${metric.total} planned`;
      return metric.value === null ? missing : `${format(metric.value)} ${specs[key].unit}`;
    }
    function coverage(run, key) {
      const m = run.metrics[key];
      if (key === "pass") return `${m.samples}/${m.total} official checks executed; failures, timeouts and incomplete tasks remain in the planned denominator.`;
      if (key === "ttft") return `${m.samples}/${m.total} client first-token samples; ${m.failedWithoutToken} observed failures without first token; ${m.unknownFirstTokenFailures} observed failures with unknown first-token receipt; ${m.unobservedRequests} requests without instrumentation.`;
      return `${m.samples}/${m.total} paired request measurements` +
        (m.value === null ? "; no worker timings acquired." : "; aggregate of matched token/time pairs.");
    }
    function drawChart(key) {
      const container = byId(key + "-chart");
      container.replaceChildren();
      const targetSize = scope.matchMedia?.("(pointer: coarse)").matches ? 44 : 34;
      const model = chartModel(data, filterId, key,
        Math.max(240, container.clientWidth), targetSize);
      const valid = model.points.filter(p => p.value !== null);
      const width = model.width;
      const x = value => value;
      const max = key === "pass" ? 100 : Math.max(1, ...valid.map(p => p.value)) * 1.12;
      // Keep better results higher on every chart: throughput/pass rise, TTFT falls.
      const y = value => key === "ttft" ? 40 + value / max * 166 : 206 - value / max * 166;
      const svg = svgNode("svg", {viewBox: `0 0 ${width} 300`, role: "img",
        "aria-label": `${specs[key].label}; measured commit horizontal axis; ${specs[key].unit} vertical axis`});
      svg.style.width = width + "px";
      for (let i = 0; i < 4; i++) {
        const value = max * i / 3, yy = y(value);
        svg.append(svgNode("line", {x1: 60, x2: width - 30, y1: yy, y2: yy, class: "grid-line"}));
        if (valid.length) svg.append(svgNode("text", {x: 54, y: yy + 4, "text-anchor": "end"}, format(value)));
      }
      const visibleTicks = model.ticks.filter((tick, i) => i === 0 || i === model.ticks.length - 1 ||
        i % Math.max(1, Math.ceil(model.ticks.length / (width < 400 ? 3 : 5))) === 0);
      visibleTicks.forEach(tick => {
        const text = svgNode("text", {x: x(tick.x), y: 266, "text-anchor": "middle"}, tick.sha.slice(0, 7));
        text.append(svgNode("title", {}, tick.sha));
        svg.append(text);
      });
      svg.append(svgNode("text", {x: width / 2, y: 290, "text-anchor": "middle"}, "Measured commit →"));
      // Connect only measurements from the same comparable configuration.
      const previousByConfig = new Map();
      model.points.forEach(point => {
        if (point.run.provenance.kind !== "main") return;
        const configId = point.run.configurationId;
        const previous = previousByConfig.get(configId);
        if (point.value !== null && previous) svg.append(svgNode("line", {
          x1: x(previous.x), y1: y(previous.value), x2: x(point.x), y2: y(point.value), class: "main-line"
        }));
        previousByConfig.set(configId, point.value === null ? null : point);
      });
      valid.filter(p => p.run.provenance.kind === "candidate").forEach(point => {
        const base = valid.find(p => p.run.id === point.run.baselineRunId);
        if (base) svg.append(svgNode("line", {
          x1: x(base.x), y1: y(base.value), x2: x(point.x), y2: y(point.value), class: "candidate-line"
        }));
      });
      model.points.filter(point => point.value === null).forEach(point =>
        svg.append(svgNode("text", {x: x(point.x), y: 244, "text-anchor": "middle",
          class: "missing-label"}, "未测")));
      container.append(svg);
      model.points.forEach(point => {
        const button = node("button", undefined, "trend-point " + point.run.provenance.kind +
          (point.value === null ? " missing" : ""));
        button.type = "button";
        button.dataset.run = point.run.id;
        button.style.left = x(point.x) + "px";
        button.style.top = (point.value === null ? 220 : y(point.value)) + "px";
        button.setAttribute("aria-label", `${point.run.title}; ${point.run.id}; measured ${point.run.measuredCommit}; ${metricText(point.run, key)}; ${state(point.run)}`);
        button.setAttribute("aria-controls", "run-detail");
        button.title = button.getAttribute("aria-label");
        if (point.run.id === data.configurations.find(c => c.id === point.run.configurationId).latestBaselineRun)
          button.classList.add("latest-baseline");
        button.addEventListener("click", () => select(point.run, true));
        container.append(button);
      });
    }
    function updateControls() {
      configControl.value = filterId;
      runControl.replaceChildren();
      data.runs.forEach(run => {
        const option = node("option", `${run.title} · ${run.measuredCommit.slice(0, 7)} · ${metricText(run, "prefill")}`);
        option.value = run.id;
        runControl.append(option);
      });
      runControl.value = selected.id;
    }
    function details() {
      const c = config();
      byId("run-title").textContent = selected.title;
      byId("run-state").textContent = state(selected);
      byId("run-identity").textContent = `${selected.date} · ${selected.experimentId} / ${selected.id}`;
      byId("run-change").textContent = selected.change;
      byId("run-result").textContent = selected.result;
      byId("permalink").href = runHash(selected);
      const metadata = byId("run-metadata");
      metadata.replaceChildren();
      const base = data.runs.find(r => r.id === selected.baselineRunId);
      const entries = [
        ["Measured runtime SHA (not harness/site SHA)", selected.measuredCommit],
        ["Baseline run / full SHA", base ? `${base.id} / ${base.measuredCommit}` : "No prior baseline"],
        ["Candidate SHA", selected.provenance.kind === "candidate" ? selected.measuredCommit : "Not a candidate"],
        ["Main provenance", selected.provenance.kind === "main" ?
          `${selected.provenance.mainCommit}; ${selected.provenance.mainCommit === selected.measuredCommit ?
            "measured main revision" : "mapped merge revision NOT remeasured"}` : "Candidate code not adopted; results publication does not imply adoption"],
        ["Provenance evidence", selected.provenance.evidence],
        ["Runtime components", JSON.stringify(selected.runtime)],
        ["Published run conditions / warmup", JSON.stringify(selected.conditions || {})],
        ["Artifact SHA-256", Object.keys(selected.binarySha256).length ? JSON.stringify(selected.binarySha256) : missing],
        ["Artifact identity evidence", selected.binaryEvidence || missing],
        ["Harness SHA / script hash (not runtime)", `${selected.harnessCommit || missing} / ${selected.harnessSha256 || missing}`],
        ...["model", "modelRevision", "weights", "kv", "cache", "chunking", "topology", "network",
          "power", "workload", "datasetRevision", "datasetSha256"].map(k => [k, c[k]]),
        ["Planned cohort", c.plannedTasks.join(", ")],
        ["Timing boundaries", JSON.stringify(c.timing)],
        ["Correctness", metricText(selected, "pass")],
        ...Object.keys(specs).map(k => [specs[k].label + " coverage", coverage(selected, k)]),
        ["Raw evidence (local only unless explicitly published)", JSON.stringify(selected.rawEvidence)]
      ];
      entries.forEach(([term, value]) => metadata.append(node("dt", term), node("dd", value)));
      const links = byId("run-links");
      links.replaceChildren();
      selected.links.forEach(link => {
        const a = node("a", link.label);
        a.href = link.url;
        links.append(a);
      });
      const commitLink = node("a", "Measured runtime commit");
      commitLink.href = "https://github.com/LittleLittleCloud/Infer.NET/commit/" + selected.measuredCommit;
      links.append(commitLink);
      if (!selected.links.some(link => link.role === "results-pr")) {
        links.append(node("span", "Separate results PR: not recorded", "note"));
      }
      const body = byId("tasks").querySelector("tbody");
      body.replaceChildren();
      selected.tasks.forEach(task => {
        const tr = node("tr");
        const heading = node("th", `${task.taskId} / ${task.attempt}`);
        heading.scope = "row";
        tr.append(heading);
        [format(task.wallSeconds), format(task.inputTokens),
          format(task.outputTokens), `${task.status} / ${task.officialValidation}`].forEach(value => {
          tr.append(node("td", value));
        });
        const cell = node("td"), a = node("a", "JSON row");
        a.href = task.source;
        a.title = task.source;
        cell.append(a);
        tr.append(cell);
        body.append(tr);
      });
      const foot = byId("tasks").querySelector("tfoot"), row = node("tr"), heading = node("th", "Totals");
      heading.scope = "row";
      foot.replaceChildren(row);
      row.append(heading);
      ["wallSeconds", "inputTokens", "outputTokens"].forEach(field => {
        const t = selected.totals[field];
        row.append(node("td", `${format(t.value)} · ${t.samples}/${t.total} rows` +
          (t.complete ? "" : " · PARTIAL / not a complete total")));
      });
      row.append(node("td", metricText(selected, "pass")), node("td", "All planned tasks retained"));
      const reported = selected.reportedTotals || {};
      byId("reported-totals").textContent = "Source-reported run totals (not a substitute for missing per-task data): " +
        `wall ${format(reported.wallSeconds ?? null)} s; input ${format(reported.inputTokens ?? null)}; ` +
        `generated ${format(reported.outputTokens ?? null)}; requests ${format(reported.modelRequests ?? null)}. ` +
        "Published per-task wall seconds are rounded; totals may differ slightly.";
      Object.keys(specs).forEach(key => {
        byId(key + "-value").textContent = metricText(selected, key);
        const metric = selected.metrics[key];
        byId(key + "-coverage").textContent = key === "pass" ?
          `${metric.samples}/${metric.total} official checks · fixed cohort` :
          `${metric.samples}/${metric.total} ${key === "ttft" ? "client first-token" : "paired request"} samples`;
      });
      document.querySelectorAll(".trend-point").forEach(button => {
        button.setAttribute("aria-pressed", String(button.dataset.run === selected.id));
      });
      runControl.value = selected.id;
    }
    function select(run, push, error = "") {
      const changedFilter = filterId !== "all" && run.configurationId !== filterId;
      if (changedFilter) filterId = "all";
      selected = run;
      byId("selection-notice").textContent = error || `Selected ${run.id} · ${state(run)}`;
      if (changedFilter) {
        updateControls();
        Object.keys(specs).forEach(drawChart);
      }
      details();
      if (push && scope.location.hash !== runHash(run)) scope.history.pushState(null, "", runHash(run));
    }
    configControl.addEventListener("change", () => {
      filterId = configControl.value;
      Object.keys(specs).forEach(drawChart);
      if (filterId !== "all" && selected.configurationId !== filterId)
        select(defaultRun(data, filterId), true);
      else details();
    });
    runControl.addEventListener("change", () => select(data.runs.find(r => r.id === runControl.value), true));
    const locationChanged = () => {
      const resolved = resolveSelection(data, scope.location.hash, selected.configurationId);
      select(resolved.run, false, resolved.error);
    };
    scope.addEventListener("hashchange", locationChanged);
    scope.addEventListener("popstate", locationChanged);
    updateControls();
    Object.keys(specs).forEach(drawChart);
    select(selected, false, initial.error);
    if (scope.ResizeObserver) {
      let previousWidth = byId("prefill-chart").clientWidth;
      new scope.ResizeObserver(() => {
        const width = byId("prefill-chart").clientWidth;
        if (width !== previousWidth) {
          previousWidth = width;
          Object.keys(specs).forEach(drawChart);
          details();
        }
      }).observe(byId("prefill-chart"));
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})(typeof window !== "undefined" ? window : globalThis);
