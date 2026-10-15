// The job search chat. Plain JS, no build step. Everything from the API goes
// into the page through textContent, never as HTML.

const $ = (selector) => document.querySelector(selector);

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

const money = (usd) => `$${Math.round(usd / 1000)}k`;
const STAGES = ["Reading your message", "Finding matching jobs", "Picking the best fits"];
const MAX_CONTEXT_TURNS = 4;

// Sent back with every follow-up: each earlier message, the filters it was
// understood as, and the jobs picked for it.
let turns = [];
let busy = false;

// --- formatting ------------------------------------------------------------

function understood(filters, requireSalary) {
  const chips = [...filters.seniority, ...filters.categories];
  if (filters.countries.length > 5) chips.push(`${filters.countries.length} countries`);
  else chips.push(...filters.countries);
  chips.push(...filters.employment_types);
  if (filters.salary_min_usd) chips.push(`${money(filters.salary_min_usd)}+ a year`);
  if (requireSalary) chips.push("pay stated");
  return chips;
}

function where(job) {
  if (job.worldwide) return "Worldwide";
  const shown = job.countries.slice(0, 2).join(", ");
  return job.countries.length > 2 ? `${shown} +${job.countries.length - 2}` : shown;
}

function pay(job) {
  if (job.salary_status !== "stated") return "pay not stated";
  return job.salary_min_usd === job.salary_max_usd
    ? `${money(job.salary_min_usd)} a year`
    : `${money(job.salary_min_usd)}–${money(job.salary_max_usd)} a year`;
}

const CHECK_TEXT = {
  seniority: (job) => job.seniority.join(" / "),
  category: (job) => job.categories.join(" / ") || "job family not stated",
  country: where,
  employment_type: (job) => job.employment_type || "contract not stated",
  salary_min_usd: pay,
};

function posted(job) {
  return new Date(job.published_at).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}

// --- rendering ---------------------------------------------------------------

function jobItem(job, n) {
  const item = el("li", { class: "jobitem" });
  item.append(el("span", { class: "label jobitem__n" }, String(n).padStart(2, "0")));
  const body = el("div");
  const title = el("h3", { class: "jobitem__title" });
  title.append(el("a", { href: job.id, target: "_blank", rel: "noopener" }, job.title));
  body.append(title);
  body.append(el("p", { class: "jobitem__meta" }, [job.company, where(job), pay(job), `posted ${posted(job)}`].join(" · ")));
  if (job.reason) body.append(el("p", { class: "jobitem__reason" }, job.reason));
  const checks = Object.entries(job.meets);
  if (checks.length) {
    const row = el("div", { class: "jobitem__checks" });
    for (const [field, status] of checks) {
      row.append(el("span", { class: `tag check--${status}`,
        title: status === "yes" ? "Matches what you asked for" : "Not stated in the posting" }, CHECK_TEXT[field](job)));
    }
    body.append(row);
  }
  item.append(body);
  return item;
}

function botShell(working) {
  const msg = el("li", { class: `msg-bot${working ? " msg-bot--working" : ""}` });
  const who = el("div", { class: "msg-bot__who" });
  who.append(el("span", { class: "dot", "aria-hidden": "true" }));
  const status = el("span", { class: "label" }, working ? STAGES[0] : "Job search");
  who.append(status);
  msg.append(who);
  return { msg, status };
}

function renderReply(msg, data) {
  msg.classList.remove("msg-bot--working");
  msg.querySelector(".msg-bot__who .label").textContent = "Job search";
  msg.append(el("p", { class: "msg-bot__text" }, data.summary));

  const chips = understood(data.filters, data.require_salary);
  const row = el("div", { class: "understood" });
  row.append(el("span", { class: "label" }, "Understood as"));
  if (chips.length === 0) row.append(el("span", { class: "tag" }, "no filters"));
  for (const chip of chips) row.append(el("span", { class: "tag" }, chip));
  msg.append(row);

  if (data.picks.length) {
    const list = el("ol", { class: "jobs" });
    data.picks.forEach((job, i) => list.append(jobItem(job, i + 1)));
    msg.append(list);
  }

  if (data.more.length) {
    const toggle = el("button", { type: "button", class: "more-toggle label", "aria-expanded": "false" },
      `Show ${data.more.length} more matches`);
    const more = el("ol", { class: "jobs jobs--more", hidden: "" });
    data.more.forEach((job, i) => more.append(jobItem(job, data.picks.length + i + 1)));
    toggle.addEventListener("click", () => {
      const open = more.hidden;
      more.hidden = !open;
      toggle.setAttribute("aria-expanded", String(open));
      toggle.textContent = open ? "Hide more matches" : `Show ${data.more.length} more matches`;
    });
    msg.append(toggle, more);
  }

  const seconds = Object.values(data.timing).reduce((a, b) => a + b, 0) / 1000;
  const maybe = data.counts.maybe ? ` · ${data.counts.maybe.toLocaleString()} might` : "";
  msg.append(el("p", { class: "label msg-bot__foot" },
    `${data.counts.match.toLocaleString()} of ${data.counts.total.toLocaleString()} jobs meet everything${maybe} · ${seconds.toFixed(1)} s`));
}

function renderError(msg, text) {
  msg.classList.remove("msg-bot--working");
  msg.querySelector(".msg-bot__who .label").textContent = "Job search";
  msg.append(el("p", { class: "msg-bot__text msg-bot__error" }, text));
}

// --- conversation ------------------------------------------------------------

function contextParam() {
  const bytes = new TextEncoder().encode(JSON.stringify(turns.slice(-MAX_CONTEXT_TURNS)));
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function scrollToEnd() {
  const scroller = $("#scroller");
  scroller.scrollTop = scroller.scrollHeight;
}

function updateComposer() {
  const input = $("#q");
  $("#composer .send").disabled = busy || input.value.trim().length < 3;
  input.placeholder = turns.length
    ? "Refine it: only full-time, Canada instead…"
    : "Describe the job you're looking for…";
}

function autosize() {
  const input = $("#q");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight + 2, 144)}px`;
}

async function send(message) {
  if (busy) return;
  busy = true;
  $("#intro").hidden = true;
  $("#new-chat").hidden = false;

  const thread = $("#thread");
  thread.append(el("li", { class: "msg-user" }, message));
  const { msg, status } = botShell(true);
  thread.append(msg);
  $("#q").value = "";
  autosize();
  updateComposer();
  scrollToEnd();

  let stage = 0;
  const timer = setInterval(() => {
    stage = Math.min(stage + 1, STAGES.length - 1);
    status.textContent = STAGES[stage];
  }, 1600);

  try {
    const url = turns.length
      ? `/api/followup?q=${encodeURIComponent(message)}&ctx=${contextParam()}`
      : `/api/search?q=${encodeURIComponent(message)}`;
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || "Something went wrong.");
    renderReply(msg, data);
    if (turns.length === 0) history.replaceState(null, "", `?q=${encodeURIComponent(message)}`);
    turns.push({ q: message, filters: data.filters, picks: data.picks.map((p) => p.id) });
    msg.scrollIntoView({ block: "start" });
  } catch (exc) {
    renderError(msg, exc.message === "Failed to fetch" ? "Search is unavailable right now. Try again in a moment." : exc.message);
    scrollToEnd();
  } finally {
    clearInterval(timer);
    busy = false;
    updateComposer();
    $("#q").focus();
  }
}

function newChat() {
  if (busy) return;
  turns = [];
  $("#thread").replaceChildren();
  $("#intro").hidden = false;
  $("#new-chat").hidden = true;
  history.replaceState(null, "", window.location.pathname);
  updateComposer();
  $("#q").focus();
}

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const message = $("#q").value.trim();
  if (message.length >= 3) send(message);
});

$("#q").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});

$("#q").addEventListener("input", () => {
  autosize();
  updateComposer();
});

$("#new-chat").addEventListener("click", newChat);

for (const button of document.querySelectorAll(".suggestions button")) {
  button.addEventListener("click", () => send(button.dataset.q));
}

updateComposer();
const initial = new URLSearchParams(window.location.search).get("q");
if (initial) send(initial.slice(0, 200));
else $("#q").focus();
