// Conversation UI and evaluation charts for jobs.kasra.page. No framework, no build step.
// All text from the API or results.json goes into the page through textContent.

const $ = (selector) => document.querySelector(selector);
const SVG = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

const money = (usd) => `$${Math.round(usd / 1000)}k`;
const pct = (x) => `${Math.round(x * 100)}%`;

// --- conversation ---------------------------------------------------------

const FIELD_NAMES = {
  seniority: "level",
  category: "job family",
  country: "location",
  employment_type: "contract",
  salary_min_usd: "salary",
};
const MAX_CONTEXT_TURNS = 4;
const STAGES = ["Understanding your request", "Finding matching jobs", "Reading the best matches"];

// What the page sends back with a follow-up: each earlier message, the filters
// it was understood as, and the jobs picked for it.
let turns = [];

function filterChips(filters, requireSalary) {
  const chips = [];
  for (const level of filters.seniority) chips.push(level);
  for (const category of filters.categories) chips.push(category);
  if (filters.countries.length > 6) chips.push(`open to ${filters.countries.length} countries`);
  else for (const country of filters.countries) chips.push(`open to ${country}`);
  for (const type of filters.employment_types) chips.push(type);
  if (filters.salary_min_usd) chips.push(`at least ${money(filters.salary_min_usd)} a year`);
  if (requireSalary) chips.push("salary stated");
  return chips;
}

function whereText(job) {
  if (job.worldwide) return "Worldwide";
  const shown = job.countries.slice(0, 3).join(", ");
  return job.countries.length > 3 ? `${shown} +${job.countries.length - 3}` : shown;
}

function salary(job) {
  if (job.salary_status !== "stated") return null;
  return job.salary_min_usd === job.salary_max_usd
    ? `${money(job.salary_min_usd)} a year`
    : `${money(job.salary_min_usd)}–${money(job.salary_max_usd)} a year`;
}

function factText(field, job) {
  switch (field) {
    case "seniority": return job.seniority.join(" / ");
    case "category": return job.categories.join(" / ") || "job family not stated";
    case "country": return whereText(job);
    case "employment_type": return job.employment_type || "contract not stated";
    case "salary_min_usd": return salary(job) || "salary not stated";
  }
}

function facts(job) {
  const row = el("div", { class: "facts" });
  for (const [field, status] of Object.entries(job.meets)) {
    row.append(el("span", {
      class: `fact ${status}`,
      title: `${FIELD_NAMES[field]}: ${status === "yes" ? "matches" : "not stated in the posting"}`,
    }, factText(field, job)));
  }
  for (const field of ["seniority", "country", "employment_type", "salary_min_usd"]) {
    if (field in job.meets) continue;
    const text = field === "salary_min_usd" ? salary(job) : factText(field, job);
    if (text && !text.endsWith("not stated")) row.append(el("span", { class: "fact" }, text));
  }
  return row;
}

function jobTitle(job, tag = "h3") {
  const title = el(tag);
  title.append(el("a", { href: job.id, target: "_blank", rel: "noopener" }, job.title));
  return title;
}

function posted(job) {
  return new Date(job.published_at).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}

function pickCard(job) {
  const card = el("li", { class: "job" });
  card.append(jobTitle(job));
  card.append(el("p", { class: "meta" }, `${job.company} · posted ${posted(job)}`));
  card.append(facts(job));
  if (job.reason) card.append(el("p", { class: "reason" }, job.reason));
  card.append(el("p", { class: "excerpt" }, job.excerpt));
  const source = el("p", { class: "source" });
  source.append(el("a", { href: job.id, target: "_blank", rel: "noopener" }, "View and apply on Himalayas ↗"));
  card.append(source);
  return card;
}

function moreRow(job) {
  const row = el("li", { class: "more-job" });
  row.append(jobTitle(job, "h4"));
  row.append(el("p", { class: "meta" }, `${job.company} · posted ${posted(job)}`));
  row.append(facts(job));
  return row;
}

function renderAnswer(block, data) {
  block.replaceChildren();
  const understood = el("div", { class: "understood" });
  understood.append(el("span", { class: "label" }, "Understood as"));
  const chips = filterChips(data.filters, data.require_salary);
  if (chips.length === 0) understood.append(el("span", { class: "chip" }, "no filters, ranked by meaning"));
  for (const chip of chips) understood.append(el("span", { class: "chip" }, chip));
  block.append(understood);

  block.append(el("p", { class: "summary" }, data.summary));
  const maybe = data.counts.maybe ? ` · ${data.counts.maybe.toLocaleString()} more might (detail not in the posting)` : "";
  const ms = Object.values(data.timing).reduce((a, b) => a + b, 0);
  block.append(el("p", { class: "muted" },
    `${data.counts.match.toLocaleString()} of ${data.counts.total.toLocaleString()} jobs meet everything${maybe} · ${(ms / 1000).toFixed(1)} s`));

  const picks = el("ol", { class: "picks" });
  picks.append(...data.picks.map(pickCard));
  block.append(picks);

  if (data.more.length) {
    const more = el("details", { class: "more" });
    more.append(el("summary", {}, `${data.more.length} more matches`));
    const list = el("ol");
    list.append(...data.more.map(moreRow));
    more.append(list);
    block.append(more);
  }
}

function contextParam() {
  const json = JSON.stringify(turns.slice(-MAX_CONTEXT_TURNS));
  const bytes = new TextEncoder().encode(json);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function setComposing(busy) {
  const button = $("#composer button[type=submit]");
  button.disabled = busy;
  $("#q").disabled = busy;
  button.textContent = busy ? "Working…" : turns.length ? "Send" : "Search";
}

async function send(message) {
  const thread = $("#thread");
  const turn = el("article", { class: "turn" });
  turn.append(el("p", { class: "user-message" }, message));
  const block = el("div", { class: "answer" });
  const stage = el("p", { class: "muted working" }, STAGES[0]);
  block.append(stage);
  turn.append(block);
  thread.append(turn);
  turn.scrollIntoView({ behavior: "smooth", block: "start" });

  let step = 0;
  const timer = setInterval(() => { step = Math.min(step + 1, STAGES.length - 1); stage.textContent = STAGES[step]; }, 1800);
  setComposing(true);
  try {
    const url = turns.length
      ? `/api/followup?q=${encodeURIComponent(message)}&ctx=${contextParam()}`
      : `/api/search?q=${encodeURIComponent(message)}`;
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || "Search failed.");
    renderAnswer(block, data);
    if (turns.length === 0) history.replaceState(null, "", `?q=${encodeURIComponent(message)}`);
    turns.push({ q: message, filters: data.filters, picks: data.picks.map((p) => p.id) });
    $("#examples").hidden = true;
    $("#new-search").hidden = false;
    $("#q").placeholder = "Refine it: only full-time, what about Canada, only ones that list pay…";
  } catch (exc) {
    block.replaceChildren(el("p", { class: "error" },
      exc.message === "Failed to fetch" ? "Search is unavailable right now." : exc.message));
  } finally {
    clearInterval(timer);
    setComposing(false);
    $("#q").value = "";
    $("#q").focus();
  }
}

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const message = $("#q").value.trim();
  if (message.length >= 3) send(message);
});

$("#new-search").addEventListener("click", () => {
  turns = [];
  $("#thread").replaceChildren();
  $("#examples").hidden = false;
  $("#new-search").hidden = true;
  $("#q").placeholder = "e.g. senior data jobs open to people in Germany, at least $90k";
  setComposing(false);
  history.replaceState(null, "", window.location.pathname);
  $("#q").focus();
});

for (const button of document.querySelectorAll("#examples button")) {
  button.addEventListener("click", () => send(button.dataset.q));
}

const initial = new URLSearchParams(window.location.search).get("q");
if (initial) send(initial);

// --- charts ---------------------------------------------------------------

const tooltip = $("#tooltip");

function showTooltip(event, value, label) {
  tooltip.replaceChildren(el("strong", {}, value), document.createTextNode(label));
  tooltip.hidden = false;
  const rect = event.target.getBoundingClientRect();
  const x = event.clientX ?? rect.right;
  const y = event.clientY ?? rect.top;
  tooltip.style.left = `${Math.min(x + 12, window.innerWidth - tooltip.offsetWidth - 8)}px`;
  tooltip.style.top = `${y - tooltip.offsetHeight - 10}px`;
}

function hideTooltip() { tooltip.hidden = true; }

// A bar with a 4px rounded data end and a square baseline end.
function barPath(x, y, width, height, r = 4) {
  const w = Math.max(width, 0.5);
  const radius = Math.min(r, w, height / 2);
  return `M${x},${y} H${x + w - radius} Q${x + w},${y} ${x + w},${y + radius} ` +
         `V${y + height - radius} Q${x + w},${y + height} ${x + w - radius},${y + height} H${x} Z`;
}

function axis(svg, left, right, top, bottom) {
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const x = left + t * (right - left);
    svg.append(svgEl("line", { class: t === 0 ? "baseline" : "grid", x1: x, x2: x, y1: top, y2: bottom }));
    svg.append(svgEl("text", { class: "tick", x, y: bottom + 16, "text-anchor": t === 0 ? "start" : t === 1 ? "end" : "middle" }, pct(t)));
  }
}

function bar(svg, { x, y, width, height, color, value, label }) {
  const path = svgEl("path", { class: "bar", d: barPath(x, y, width, height), fill: color, tabindex: 0,
    "aria-label": `${label}: ${pct(value)}` });
  path.addEventListener("pointermove", (e) => showTooltip(e, value.toFixed(3), label));
  path.addEventListener("focus", (e) => showTooltip(e, value.toFixed(3), label));
  path.addEventListener("pointerleave", hideTooltip);
  path.addEventListener("blur", hideTooltip);
  svg.append(path);
  svg.append(svgEl("text", { class: "value", x: x + width + 6, y: y + height / 2 + 4 }, pct(value)));
}

function systemsChart(systems) {
  const W = 680, left = 0, right = W - 48, row = 46, barH = 18;
  const H = systems.length * row + 24;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Precision at 10 by search system" });
  axis(svg, left, right, 0, H - 24);
  systems.forEach((s, i) => {
    const y = i * row;
    svg.append(svgEl("text", { class: "cat", x: left, y: y + 16, "font-weight": s.name === "filtered" ? 600 : 400 }, s.label));
    bar(svg, { x: left, y: y + 22, width: s["precision@10"] * (right - left), height: barH,
      color: "var(--series-1)", value: s["precision@10"], label: s.label });
  });
  return svg;
}

const FIELD_SERIES = [
  ["embeddings", "Meaning search", "var(--series-1)"],
  ["textonly", "LLM, posting text only", "var(--series-2)"],
  ["filtered", "LLM filters (this demo)", "var(--series-3)"],
];

function fieldsChart(rows) {
  const W = 680, left = 0, right = W - 48, barH = 12, gap = 2;
  const group = 24 + FIELD_SERIES.length * (barH + gap) + 14;
  const H = rows.length * group + 24;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Precision at 10 by field and system" });
  axis(svg, left, right, 0, H - 24);
  rows.forEach((r, i) => {
    const y0 = i * group;
    svg.append(svgEl("text", { class: "cat", x: left, y: y0 + 16 }, r.field));
    FIELD_SERIES.forEach(([key, name, color], j) => {
      bar(svg, { x: left, y: y0 + 24 + j * (barH + gap), width: r[key] * (right - left), height: barH,
        color, value: r[key], label: `${r.field} · ${name}` });
    });
  });
  return svg;
}

function table(target, headers, rows) {
  const head = el("tr");
  headers.forEach((h) => head.append(el("th", {}, h)));
  target.replaceChildren(head);
  for (const row of rows) {
    const tr = el("tr");
    row.forEach((cell, i) => tr.append(el("td", i ? { class: "num" } : {}, cell)));
    target.append(tr);
  }
}

async function charts() {
  const results = await (await fetch("results.json")).json();
  $("#query-count").textContent = results.queries.toLocaleString();
  $("#chart-systems").replaceChildren(systemsChart(results.systems));
  table($("#table-systems"), ["System", "Precision@10", "nDCG@10", "Recall@100"],
    results.systems.map((s) => [s.label, s["precision@10"].toFixed(3), s["ndcg@10"].toFixed(3), s["recall@100"].toFixed(3)]));

  $("#legend-fields").replaceChildren(...FIELD_SERIES.map(([, name, color]) => {
    const item = el("span");
    const swatch = el("i");
    swatch.style.background = color;
    item.append(swatch, document.createTextNode(name));
    return item;
  }));
  $("#chart-fields").replaceChildren(fieldsChart(results.by_field));
  table($("#table-fields"), ["Search asks for", ...FIELD_SERIES.map(([, name]) => name)],
    results.by_field.map((r) => [r.field, ...FIELD_SERIES.map(([key]) => r[key].toFixed(3))]));
}

charts();
