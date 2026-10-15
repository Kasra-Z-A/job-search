// Charts for "How it works", drawn from results.json. Single hue: the demo in
// rust, the other systems in a neutral, so colour marks the story and labels
// carry identity. All data goes into the page through textContent.

const $ = (selector) => document.querySelector(selector);
const SVG = "http://www.w3.org/2000/svg";
const pct = (x) => `${Math.round(x * 100)}%`;

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

const tooltip = $("#tooltip");

function hover(node, value, label) {
  const show = (event) => {
    tooltip.replaceChildren(el("strong", {}, value), document.createTextNode(label));
    tooltip.hidden = false;
    const rect = node.getBoundingClientRect();
    const x = event.clientX ?? rect.right;
    const y = event.clientY ?? rect.top;
    tooltip.style.left = `${Math.min(x + 12, window.innerWidth - tooltip.offsetWidth - 8)}px`;
    tooltip.style.top = `${y - tooltip.offsetHeight - 10}px`;
  };
  const hide = () => { tooltip.hidden = true; };
  node.addEventListener("pointermove", show);
  node.addEventListener("focus", show);
  node.addEventListener("pointerleave", hide);
  node.addEventListener("blur", hide);
}

// A bar with a 4px rounded data end and a square baseline end.
function barPath(x, y, width, height, r = 4) {
  const w = Math.max(width, 0.5);
  const radius = Math.min(r, w, height / 2);
  return `M${x},${y} H${x + w - radius} Q${x + w},${y} ${x + w},${y + radius} ` +
         `V${y + height - radius} Q${x + w},${y + height} ${x + w - radius},${y + height} H${x} Z`;
}

function systemsChart(systems) {
  const W = 760, right = W - 56, row = 50, barH = 20;
  const H = systems.length * row + 26;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Precision at 10 by search system" });
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const x = t * right;
    svg.append(svgEl("line", { class: "grid", x1: x, x2: x, y1: 0, y2: H - 26 }));
    svg.append(svgEl("text", { class: "tick", x, y: H - 8, "text-anchor": t === 0 ? "start" : t === 1 ? "end" : "middle" }, pct(t)));
  }
  systems.forEach((s, i) => {
    const demo = s.name === "filtered";
    const y = i * row;
    const value = s["precision@10"];
    svg.append(svgEl("text", { class: `cat${demo ? " cat--demo" : ""}`, x: 0, y: y + 16 }, s.label));
    const bar = svgEl("path", { class: `bar${demo ? " bar--demo" : ""}`, d: barPath(0, y + 23, value * right, barH),
      tabindex: 0, "aria-label": `${s.label}: ${pct(value)}` });
    hover(bar, value.toFixed(3), s.label);
    svg.append(bar);
    svg.append(svgEl("text", { class: "value", x: value * right + 8, y: y + 38 }, pct(value)));
  });
  return svg;
}

const FIELD_COLUMNS = [
  ["embeddings", "Meaning search"],
  ["textonly", "LLM, posting text only"],
  ["filtered", "This assistant"],
];

function miniBar(value, demo, label) {
  const svg = svgEl("svg", { viewBox: "0 0 100 14", preserveAspectRatio: "none", "aria-hidden": "true" });
  svg.append(svgEl("rect", { class: "bar", x: 0, y: 0, width: 100, height: 14, rx: 3, opacity: 0.25 }));
  const bar = svgEl("path", { class: `bar${demo ? " bar--demo" : ""}`, d: barPath(0, 0, value * 100, 14, 3) });
  svg.append(bar);
  hover(svg, value.toFixed(3), label);
  return svg;
}

function fieldsChart(target, rows) {
  target.replaceChildren(el("span"));
  for (const [key, name] of FIELD_COLUMNS) {
    target.append(el("span", { class: `label head${key === "filtered" ? " head--demo" : ""}` }, name));
  }
  for (const row of rows) {
    target.append(el("span", { class: "field" }, row.field));
    for (const [key, name] of FIELD_COLUMNS) {
      const cell = el("span", { class: "cell" });
      cell.append(miniBar(row[key], key === "filtered", `${row.field} · ${name}`));
      cell.append(el("span", { class: "num" }, pct(row[key])));
      target.append(cell);
    }
  }
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

async function main() {
  const results = await (await fetch("results.json")).json();
  const demo = results.systems.find((s) => s.name === "filtered");
  $("#m-precision").textContent = pct(demo["precision@10"]);
  $("#m-queries").textContent = results.queries.toLocaleString("en-US");

  $("#chart-systems").replaceChildren(systemsChart(results.systems));
  table($("#table-systems"), ["System", "Precision@10", "nDCG@10", "Recall@100"],
    results.systems.map((s) => [s.label, s["precision@10"].toFixed(3), s["ndcg@10"].toFixed(3), s["recall@100"].toFixed(3)]));

  fieldsChart($("#chart-fields"), results.by_field);
  table($("#table-fields"), ["Search asks for", ...FIELD_COLUMNS.map(([, name]) => name)],
    results.by_field.map((r) => [r.field, ...FIELD_COLUMNS.map(([key]) => r[key].toFixed(3))]));
}

main();
