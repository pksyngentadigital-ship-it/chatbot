// ── Suggested prompt cards ──
document.querySelectorAll(".prompt-card").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById("chat-input").value = btn.textContent;
    document.getElementById("chat-form").requestSubmit();
  });
});

// ── Badge label (mirrors the Streamlit build's badge text, minimal style) ──
// Built as a node with textContent rather than an HTML string: badge text is
// assembled server-side from data-derived labels, so it should never be
// parsed as markup.
function appendBadge(container, badgeText) {
  if (!badgeText) return;
  const span = document.createElement("span");
  span.className = "badge";
  span.textContent = badgeText;
  container.appendChild(span);
}

function scrollToBottom() {
  const area = document.getElementById("scroll-area");
  if (area) area.scrollTo({ top: area.scrollHeight, behavior: "smooth" });
}

function collapseWelcome() {
  const welcome = document.getElementById("welcome");
  if (welcome) welcome.classList.add("collapsed");
}

// ── Sidebar: mobile toggle + New chat ──
const sidebar = document.getElementById("sidebar");
const sidebarScrim = document.getElementById("sidebar-scrim");
const sidebarToggle = document.getElementById("sidebar-toggle");

function closeSidebar() {
  sidebar.classList.remove("open");
  sidebarScrim.classList.remove("open");
}

if (sidebarToggle) {
  sidebarToggle.addEventListener("click", () => {
    sidebar.classList.add("open");
    sidebarScrim.classList.add("open");
  });
}
if (sidebarScrim) {
  sidebarScrim.addEventListener("click", closeSidebar);
}

document.getElementById("new-chat-btn").addEventListener("click", () => {
  document.getElementById("chat-log").innerHTML = "";
  const welcome = document.getElementById("welcome");
  if (welcome) welcome.classList.remove("collapsed");
  closeSidebar();
  chatInputFocus();
  // Fire-and-forget: clears server-side history + follow-up context for
  // this session. Nothing in the UI depends on the response.
  fetch("/chat/clear", { method: "POST" }).catch(() => {});
});

function chatInputFocus() {
  const el = document.getElementById("chat-input");
  if (el) el.focus();
}

function addUserMessage(text) {
  const log = document.getElementById("chat-log");
  const div = document.createElement("div");
  div.className = "msg user";
  const body = document.createElement("div");
  body.className = "msg-body";
  body.textContent = text;
  div.appendChild(body);
  log.appendChild(div);
  scrollToBottom();
  return div;
}

function addAssistantMessage() {
  const log = document.getElementById("chat-log");
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  const spinner = document.createElement("div");
  spinner.className = "spinner-row";
  spinner.innerHTML = '<span class="spinner"></span> Thinking...';
  wrap.appendChild(spinner);
  log.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

// Assistant text is model output, and the model is instructed to quote the
// ingested spreadsheet verbatim — so it is untrusted input, not trusted
// markup. Everything goes through DOMPurify before it reaches innerHTML.
function renderMarkdown(text) {
  let html;
  try {
    html = marked.parse(text);
  } catch (e) {
    html = String(text).replace(/\n/g, "<br>");
  }
  try {
    return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
  } catch (e) {
    // If the sanitizer is unavailable for any reason, degrade to plain
    // text rather than rendering unsanitized HTML.
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
}

function setMarkdown(el, text) {
  el.innerHTML = renderMarkdown(text);
}

function renderChart(container, chart) {
  if (!chart || !chart.labels || !chart.labels.length) return;
  const chartWrap = document.createElement("div");
  chartWrap.className = "chart-wrap";
  const canvas = document.createElement("canvas");
  chartWrap.appendChild(canvas);
  container.appendChild(chartWrap);

  const styles = getComputedStyle(document.documentElement);
  const accent = styles.getPropertyValue("--accent").trim() || "#3f6b3f";
  const gridColor = styles.getPropertyValue("--border").trim() || "#e5e5e5";
  const textColor = styles.getPropertyValue("--text-muted").trim() || "#666";

  new Chart(canvas, {
    type: chart.type === "line" ? "line" : "bar",
    data: {
      labels: chart.labels,
      datasets: [{
        label: chart.title || "Value",
        data: chart.values,
        backgroundColor: accent,
        borderColor: accent,
        borderRadius: chart.type === "line" ? 0 : 6,
        fill: false,
        tension: 0.3,
        pointRadius: chart.type === "line" ? 2 : 0,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: gridColor }, ticks: { color: textColor, font: { size: 11 } } },
        x: { grid: { display: false }, ticks: { color: textColor, font: { size: 11 } } },
      },
    },
  });
}

function renderDownloads(container, downloadId) {
  if (!downloadId) return;
  const wrap = document.createElement("div");
  wrap.className = "downloads";
  const items = [
    ["csv", "CSV"],
    ["excel", "Excel"],
    ["pptx", "PowerPoint"],
  ];
  items.forEach(([kind, label]) => {
    const a = document.createElement("a");
    a.className = "dl-btn";
    a.href = `/download/${downloadId}/${kind}`;
    a.textContent = `↓ ${label}`;
    wrap.appendChild(a);
  });
  container.appendChild(wrap);
}

function renderSuggestions(container, suggestions) {
  if (!Array.isArray(suggestions) || suggestions.length === 0) return;
  const wrap = document.createElement("div");
  wrap.className = "followups";
  suggestions.forEach((text) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "followup-chip";
    btn.textContent = text;
    btn.addEventListener("click", () => sendQuery(text));
    wrap.appendChild(btn);
  });
  container.appendChild(wrap);
}

// Only one turn may be in flight. Two concurrent streams interleave their
// writes into the same server-side history and follow-up context, which
// silently corrupts the conversation.
let inFlight = null;

function setBusy(busy) {
  const stopBtn = document.getElementById("stop-btn");
  if (stopBtn) stopBtn.hidden = !busy;
  if (sendBtn) sendBtn.disabled = busy || chatInput.value.trim().length === 0;
}

function stopStreaming() {
  if (inFlight) {
    inFlight.close();
    inFlight = null;
  }
  setBusy(false);
}

function sendQuery(query) {
  if (inFlight) return;  // a turn is already streaming

  collapseWelcome();
  addUserMessage(query);
  const assistantWrap = addAssistantMessage();

  let bodyText = "";
  let headerText = "";
  let repaintQueued = false;

  const es = new EventSource(`/chat?q=${encodeURIComponent(query)}`);
  inFlight = es;
  setBusy(true);

  function finish() {
    if (inFlight === es) inFlight = null;
    es.close();
    setBusy(false);
  }

  es.addEventListener("start", (e) => {
    const data = JSON.parse(e.data);
    headerText = data.header || "";
    assistantWrap.innerHTML = "";
    appendBadge(assistantWrap, data.badge);
    const body = document.createElement("div");
    body.className = "msg-body";
    body.innerHTML = '<span class="cursor-blink"></span>';
    assistantWrap.appendChild(body);
  });

  es.addEventListener("token", (e) => {
    bodyText += JSON.parse(e.data).token;
    // Re-parsing the whole markdown string on every token is O(n^2) and
    // visibly janks long answers — coalesce to one repaint per frame.
    if (repaintQueued) return;
    repaintQueued = true;
    requestAnimationFrame(() => {
      repaintQueued = false;
      const body = assistantWrap.querySelector(".msg-body");
      if (body) {
        setMarkdown(body, headerText + bodyText);
        body.insertAdjacentHTML("beforeend", '<span class="cursor-blink"></span>');
      }
      scrollToBottom();
    });
  });

  es.addEventListener("final", (e) => {
    const data = JSON.parse(e.data);
    assistantWrap.innerHTML = "";
    appendBadge(assistantWrap, data.badge);

    const body = document.createElement("div");
    body.className = "msg-body";
    setMarkdown(body, data.kind === "normal" ? (data.header || "") + data.reply : data.reply);
    assistantWrap.appendChild(body);

    renderChart(assistantWrap, data.chart);
    renderDownloads(assistantWrap, data.download_id);
    renderSuggestions(assistantWrap, data.suggestions);

    scrollToBottom();
    finish();
  });

  es.addEventListener("error", (e) => {
    let message = "Something went wrong talking to the server.";
    try {
      message = JSON.parse(e.data).message || message;
    } catch (_) { /* connection-level error, no JSON payload */ }

    // Append below whatever already streamed rather than replacing it —
    // discarding a partial answer the user was mid-way through reading is
    // worse than showing it with a note attached.
    const note = document.createElement("div");
    note.className = "msg-error";
    note.textContent = message;
    const cursor = assistantWrap.querySelector(".cursor-blink");
    if (cursor) cursor.remove();
    assistantWrap.appendChild(note);
    scrollToBottom();
    finish();
  });
}

// ── Composer: auto-resize + enable/disable send + Enter-to-send ──
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

function autoResize() {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + "px";
}

chatInput.addEventListener("input", () => {
  autoResize();
  sendBtn.disabled = !!inFlight || chatInput.value.trim().length === 0;
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("chat-form").requestSubmit();
  }
});

chatInput.setAttribute("maxlength", "500");

document.getElementById("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (inFlight) return;
  const query = chatInput.value.trim();
  if (!query) return;
  chatInput.value = "";
  autoResize();
  sendQuery(query);
});

const stopBtnEl = document.getElementById("stop-btn");
if (stopBtnEl) stopBtnEl.addEventListener("click", stopStreaming);

// ── Restore server-side chat history on page load, so a refresh doesn't
// lose the conversation. Reuses the same render helpers as live turns. ──
(function hydrateHistory() {
  let history = [];
  try {
    // Read from a JSON data island rather than an inline assignment — in
    // application/json the parser only looks for "</script", so stored
    // text cannot confuse the tokenizer and break the page.
    const el = document.getElementById("init-history");
    if (el) history = JSON.parse(el.textContent || "[]");
  } catch (e) {
    history = [];
  }
  if (!Array.isArray(history) || history.length === 0) return;

  collapseWelcome();
  const log = document.getElementById("chat-log");

  history.forEach((entry) => {
    if (entry.role === "user") {
      const div = document.createElement("div");
      div.className = "msg user";
      const body = document.createElement("div");
      body.className = "msg-body";
      body.textContent = entry.content;
      div.appendChild(body);
      log.appendChild(div);
      return;
    }

    const wrap = document.createElement("div");
    wrap.className = "msg assistant";
    appendBadge(wrap, entry.badge);
    const body = document.createElement("div");
    body.className = "msg-body";
    setMarkdown(body, entry.content || "");
    wrap.appendChild(body);
    renderChart(wrap, entry.chart);
    renderDownloads(wrap, entry.download_id);
    renderSuggestions(wrap, entry.suggestions);
    log.appendChild(wrap);
  });

  scrollToBottom();
})();
