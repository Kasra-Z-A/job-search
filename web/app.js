// Search UI and evaluation charts for jobs.kasra.page. No framework, no build step.
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

// --- search ---------------------------------------------------------------

const FIELD_NAMES = {
  seniority: "level",
  category: "job family",
  country: "location",
  employment_type: "contract",
  salary_min_usd: "salary",
};

function filterChips(filters) {
  const chips = [];
  for (const level of filters.seniority) chips.push(level);
  for (const category of filters.categories) chips.push(category);
  for (const country of filters.countries) chips.push(`open to ${country}`);
  for (const type of filters.employment_types) chips.push(type);
  if (filters.salary_min_usd) chips.push(`at least ${money(filters.salary_min_usd)} a year`);
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

function renderJob(job) {
  const item = el("li", { class: "job" });
  const title = el("h3");
  title.append(el("a", { href: job.id, target: "_blank", rel: "noopener" }, job.title));
  item.append(title);
  const posted = new Date(job.published_at).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
  item.append(el("p", { class: "meta" }, `${job.company} · posted ${posted}`));

  const facts = el("div", { class: "facts" });
  for (const [field, status] of Object.entries(job.meets)) {
    facts.append(el("span", { class: `fact ${status}`, title: `${FIELD_NAMES[field]}: ${status === "yes" ? "matches" : "not stated in the posting"}` },
      factText(field, job)));
  }
  for (const field of ["seniority", "country", "employment_type", "salary_min_usd"]) {
    if (field in job.meets) continue;
    const text = field === "salary_min_usd" ? salary(job) : factText(field, job);
    if (text && !text.endsWith("not stated")) facts.append(el("span", { class: "fact" }, text));
  }
  item.append(facts);
  item.append(el("p", { class: "excerpt" }, job.excerpt));
  return item;
}

function renderResults(data) {
  const understood = $("#understood");
  understood.replaceChildren(el("span", { class: "label" }, "Understood as"));
  const chips = filterChips(data.filters);
  if (chips.length === 0) understood.append(el("span", { class: "chip" }, "no filters, ranked by meaning"));
  for (const chip of chips) understood.append(el("span", { class: "chip" }, chip));

  const seconds = (data.timing.understand_ms + data.timing.embed_ms + data.timing.rank_ms) / 1000;
  const maybe = data.counts.maybe ? ` · ${data.counts.maybe.toLocaleString()} more might (detail not in the posting)` : "";
  $("#summary").textContent =
    `${data.counts.match.toLocaleString()} of ${data.counts.total.toLocaleString()} jobs meet everything${maybe} · ${seconds.toFixed(1)} s`;

  const list = $("#list");
  list.replaceChildren(...data.results.map(renderJob));
  if (data.results.length === 0) list.append(el("li", { class: "muted" }, "No job meets all of that. Try loosening one part."));
  $("#results").hidden = false;
}

async function search(query) {
  const button = $("#search button");
  const error = $("#error");
  error.hidden = true;
  button.disabled = true;
  button.textContent = "Searching…";
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || "Search failed.");
    renderResults(data);
    history.replaceState(null, "", `?q=${encodeURIComponent(query)}`);
  } catch (exc) {
    error.textContent = exc.message === "Failed to fetch" ? "Search is unavailable right now." : exc.message;
    error.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Search";
  }
}

$("#search").addEventListener("submit", (event) => {
  event.preventDefault();
  const query = $("#q").value.trim();
  if (query.length >= 3) search(query);
});

for (const button of document.querySelectorAll(".examples button")) {
  button.addEventListener("click", () => {
    $("#q").value = button.dataset.q;
    search(button.dataset.q);
  });
}

const initial = new URLSearchParams(window.location.search).get("q");
if (initial) {
  $("#q").value = initial;
  search(initial);
}

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
