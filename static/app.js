// ── Suggested prompt cards ──
document.querySelectorAll(".prompt-card").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById("chat-input").value = btn.textContent;
    document.getElementById("chat-form").requestSubmit();
  });
});

// ── Badge label (mirrors the Streamlit build's badge text, minimal style) ──
function renderBadge(badgeText) {
  if (!badgeText) return "";
  return `<span class="badge">${badgeText}</span>`;
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

function renderMarkdown(text) {
  try {
    return marked.parse(text);
  } catch (e) {
    return text.replace(/\n/g, "<br>");
  }
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

function sendQuery(query) {
  collapseWelcome();
  addUserMessage(query);
  const assistantWrap = addAssistantMessage();

  let bodyHtml = "";
  let headerText = "";

  const es = new EventSource(`/chat?q=${encodeURIComponent(query)}`);

  es.addEventListener("start", (e) => {
    const data = JSON.parse(e.data);
    headerText = data.header || "";
    assistantWrap.innerHTML = "";
    if (data.badge) assistantWrap.insertAdjacentHTML("beforeend", renderBadge(data.badge));
    const body = document.createElement("div");
    body.className = "msg-body";
    body.innerHTML = '<span class="cursor-blink"></span>';
    assistantWrap.appendChild(body);
  });

  es.addEventListener("token", (e) => {
    const data = JSON.parse(e.data);
    bodyHtml += data.token;
    const body = assistantWrap.querySelector(".msg-body");
    if (body) body.innerHTML = renderMarkdown(headerText + bodyHtml) + '<span class="cursor-blink"></span>';
    scrollToBottom();
  });

  es.addEventListener("final", (e) => {
    const data = JSON.parse(e.data);
    assistantWrap.innerHTML = "";

    if (data.badge) assistantWrap.insertAdjacentHTML("beforeend", renderBadge(data.badge));

    const body = document.createElement("div");
    body.className = "msg-body";
    const text = data.kind === "normal" ? (data.header || "") + data.reply : data.reply;
    body.innerHTML = renderMarkdown(text);
    assistantWrap.appendChild(body);

    renderChart(assistantWrap, data.chart);
    renderDownloads(assistantWrap, data.download_id);

    scrollToBottom();
    es.close();
  });

  es.addEventListener("error", (e) => {
    let message = "Something went wrong talking to the server.";
    try {
      message = JSON.parse(e.data).message || message;
    } catch (_) { /* connection-level error, no JSON payload */ }
    assistantWrap.innerHTML = `<div class="msg-body">${message}</div>`;
    es.close();
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
  sendBtn.disabled = chatInput.value.trim().length === 0;
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("chat-form").requestSubmit();
  }
});

document.getElementById("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;
  chatInput.value = "";
  autoResize();
  sendBtn.disabled = true;
  sendQuery(query);
});

// ── Restore server-side chat history on page load, so a refresh doesn't
// lose the conversation. Reuses the same render helpers as live turns. ──
(function hydrateHistory() {
  const history = window.__initialHistory;
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
    if (entry.badge) wrap.insertAdjacentHTML("beforeend", renderBadge(entry.badge));
    const body = document.createElement("div");
    body.className = "msg-body";
    body.innerHTML = renderMarkdown(entry.content || "");
    wrap.appendChild(body);
    renderChart(wrap, entry.chart);
    renderDownloads(wrap, entry.download_id);
    log.appendChild(wrap);
  });

  scrollToBottom();
})();
