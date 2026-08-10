// ── Suggested prompt tabs ──
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

document.querySelectorAll(".prompt-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById("chat-input").value = btn.textContent;
    document.getElementById("chat-form").requestSubmit();
  });
});

// ── Badge → CSS class mapping (mirrors the Streamlit build's badge_html) ──
function badgeClass(badgeText) {
  if (!badgeText) return "badge-sentiment";
  if (badgeText.startsWith("🔀")) return "badge-comparison";
  if (badgeText.startsWith("🏷️")) return "badge-product";
  if (badgeText.startsWith("🐛")) return "badge-complaint";
  if (badgeText.startsWith("🌻")) return "badge-positive";
  if (badgeText.startsWith("💡")) return "badge-suggestion";
  if (badgeText.startsWith("📊") || badgeText.startsWith("📈")) return "badge-ranking";
  return "badge-sentiment";
}

function addUserMessage(text) {
  const log = document.getElementById("chat-log");
  const div = document.createElement("div");
  div.className = "msg user";
  div.textContent = text;
  log.appendChild(div);
  log.scrollIntoView({ behavior: "smooth", block: "end" });
  return div;
}

function addAssistantMessage() {
  const log = document.getElementById("chat-log");
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  const spinner = document.createElement("div");
  spinner.className = "spinner-row";
  spinner.innerHTML = '<span class="spinner"></span> Searching and aggregating matching historical data records...';
  wrap.appendChild(spinner);
  log.appendChild(wrap);
  log.scrollIntoView({ behavior: "smooth", block: "end" });
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

  new Chart(canvas, {
    type: chart.type === "line" ? "line" : "bar",
    data: {
      labels: chart.labels,
      datasets: [{
        label: chart.title || "Value",
        data: chart.values,
        backgroundColor: "#2e7d32",
        borderColor: "#2e7d32",
        fill: false,
        tension: 0.25,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }, title: { display: !!chart.title, text: chart.title } },
      scales: { y: { beginAtZero: true } },
    },
  });
}

function renderDownloads(container, downloadId) {
  if (!downloadId) return;
  const wrap = document.createElement("div");
  wrap.className = "downloads";
  const items = [
    ["csv", "⬇️ Chart data (CSV)"],
    ["excel", "⬇️ Excel"],
    ["pptx", "⬇️ PowerPoint"],
  ];
  items.forEach(([kind, label]) => {
    const a = document.createElement("a");
    a.className = "dl-btn";
    a.href = `/download/${downloadId}/${kind}`;
    a.textContent = label;
    wrap.appendChild(a);
  });
  container.appendChild(wrap);
}

function sendQuery(query) {
  addUserMessage(query);
  const assistantWrap = addAssistantMessage();

  let bodyHtml = "";
  let badgeText = null;
  let headerText = "";

  const es = new EventSource(`/chat?q=${encodeURIComponent(query)}`);

  es.addEventListener("start", (e) => {
    const data = JSON.parse(e.data);
    badgeText = data.badge;
    headerText = data.header || "";
    assistantWrap.innerHTML = "";
    if (badgeText) {
      const badge = document.createElement("span");
      badge.className = `intent-badge ${badgeClass(badgeText)}`;
      badge.textContent = badgeText;
      assistantWrap.appendChild(badge);
    }
    const body = document.createElement("div");
    body.className = "msg-body";
    assistantWrap.appendChild(body);
  });

  es.addEventListener("token", (e) => {
    const data = JSON.parse(e.data);
    bodyHtml += data.token;
    const body = assistantWrap.querySelector(".msg-body");
    if (body) body.innerHTML = renderMarkdown(headerText + bodyHtml);
    assistantWrap.scrollIntoView({ behavior: "smooth", block: "end" });
  });

  es.addEventListener("final", (e) => {
    const data = JSON.parse(e.data);
    assistantWrap.innerHTML = "";

    const badge = data.badge;
    if (badge) {
      const badgeEl = document.createElement("span");
      badgeEl.className = `intent-badge ${badgeClass(badge)}`;
      badgeEl.textContent = badge;
      assistantWrap.appendChild(badgeEl);
    }

    const body = document.createElement("div");
    body.className = "msg-body";
    const text = data.kind === "normal" ? (data.header || "") + data.reply : data.reply;
    body.innerHTML = renderMarkdown(text);
    assistantWrap.appendChild(body);

    renderChart(assistantWrap, data.chart);
    renderDownloads(assistantWrap, data.download_id);

    assistantWrap.scrollIntoView({ behavior: "smooth", block: "end" });
    es.close();
  });

  es.addEventListener("error", (e) => {
    let message = "Something went wrong talking to the server.";
    try {
      message = JSON.parse(e.data).message || message;
    } catch (_) { /* connection-level error, no JSON payload */ }
    assistantWrap.innerHTML = `<div class="msg-body">⚠️ ${message}</div>`;
    es.close();
  });
}

document.getElementById("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const query = input.value.trim();
  if (!query) return;
  input.value = "";
  document.getElementById("suggestions-box").open = false;
  sendQuery(query);
});
