// Chart for "How it works", drawn from results.json: nDCG@10 from the LLM
// judge. The demo in rust, the other systems in a neutral, so colour marks the
// story and labels carry identity. All data goes into the page through textContent.

const $ = (selector) => document.querySelector(selector);
const SVG = "http://www.w3.org/2000/svg";
const score = (x) => x.toFixed(2);

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

// Drawn at the container's width, so labels stay at their set size.
function systemsChart(systems, width) {
  const W = Math.max(width, 320), right = W - 56, row = 50, barH = 20;
  const H = systems.length * row + 26;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "nDCG at 10 by search system" });
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const x = t * right;
    svg.append(svgEl("line", { class: "grid", x1: x, x2: x, y1: 0, y2: H - 26 }));
    svg.append(svgEl("text", { class: "tick", x, y: H - 8, "text-anchor": t === 0 ? "start" : t === 1 ? "end" : "middle" }, score(t)));
  }
  systems.forEach((s, i) => {
    const y = i * row;
    const demo = s.name === "filtered";
    const value = s["ndcg@10"];
    svg.append(svgEl("text", { class: `cat${demo ? " cat--demo" : ""}`, x: 0, y: y + 16 }, s.label));
    // Above the end of the demo's bar; dropped in main() if it would overlap the label.
    if (demo) {
      svg.append(svgEl("text", { class: "ours", x: value * right, y: y + 16, "text-anchor": "end" },
        "Current implementation"));
    }
    const bar = svgEl("path", { class: `bar${demo ? " bar--demo" : ""}`, d: barPath(0, y + 23, value * right, barH),
      tabindex: 0, "aria-label": `${s.label}: ${score(value)}` });
    hover(bar, `${Math.round(s["good@10"] * 100)}% good matches`, s.label);
    svg.append(bar);
    svg.append(svgEl("text", { class: "value", x: value * right + 8, y: y + 38 }, score(value)));
  });
  return svg;
}

async function main() {
  const results = await (await fetch("results.json")).json();
  const systemsTarget = $("#chart-systems");
  systemsTarget.replaceChildren(systemsChart(results.systems, systemsTarget.clientWidth));
  // Text widths are only known once drawn; the legend still says which system it is.
  const ours = systemsTarget.querySelector(".ours"), cat = systemsTarget.querySelector(".cat--demo");
  if (ours && cat && ours.getBBox().x < cat.getBBox().x + cat.getBBox().width + 16) ours.remove();
}

main();
