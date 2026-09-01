// Admin panel: ingestion upload + result reporting.
// Lives in its own file rather than inline so the CSP can stay at
// script-src 'self' with no 'unsafe-inline'.

const ingestForm = document.getElementById("ingest-form");

if (ingestForm) {
  ingestForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("file");
    const purgeInput = document.getElementById("purge-first");
    const statusEl = document.getElementById("ingest-status");
    const reportEl = document.getElementById("ingest-report");
    if (!fileInput.files.length) return;

    if (purgeInput && purgeInput.checked) {
      const ok = window.confirm(
        "This will permanently delete ALL existing records from the index before " +
        "loading this file.\n\nThere is no undo. Continue?"
      );
      if (!ok) return;
    }

    reportEl.textContent = "";
    statusEl.textContent = "Processing — this can take a minute...";
    statusEl.className = "status-line";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    if (purgeInput && purgeInput.checked) formData.append("purge_first", "true");

    try {
      const res = await fetch("/admin/ingest", { method: "POST", body: formData });
      const data = await res.json();
      if (res.ok && data.success) {
        statusEl.textContent = `Done — ingested ${data.total_records} records.`;
        statusEl.className = "status-line success-text";
        renderReport(reportEl, data);
      } else {
        statusEl.textContent = data.error || "Ingestion failed.";
        statusEl.className = "status-line error-text";
      }
    } catch (err) {
      statusEl.textContent = err.message;
      statusEl.className = "status-line error-text";
    }
  });
}

// Anything the parser could not use is surfaced here. Previously these
// were silent `continue` paths, so a workbook could half-ingest while the
// UI still reported success.
function renderReport(container, data) {
  container.textContent = "";
  const skipped = data.skipped || [];
  if (!skipped.length) return;

  const h = document.createElement("p");
  h.className = "admin-status";
  h.textContent = `${skipped.length} item(s) were not ingested:`;
  container.appendChild(h);

  const ul = document.createElement("ul");
  ul.className = "feedback-log-list";
  skipped.forEach((s) => {
    const li = document.createElement("li");
    const sheet = document.createElement("span");
    sheet.className = "feedback-log-time";
    sheet.textContent = `${s.sheet} · ${s.reason}`;
    li.appendChild(sheet);
    li.appendChild(document.createTextNode(" — " + (s.detail || "")));
    ul.appendChild(li);
  });
  container.appendChild(ul);
}
