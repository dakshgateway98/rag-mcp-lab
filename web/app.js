const catalogEl = document.getElementById("catalog");
const runsEl = document.getElementById("runs");
const mcpEl = document.getElementById("mcp-copy");

document.querySelectorAll("nav button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

async function loadCatalog() {
  const data = await fetch("/api/techniques").then((r) => r.json());
  catalogEl.innerHTML = data.techniques
    .map(
      (t) => `<article class="card">
        <h3>${t.name}</h3>
        <p class="meta"><span class="chip">${t.family}</span> ${t.cost}</p>
        <p>${t.when}</p>
        <p class="meta">Weakness: ${t.weakness}</p>
        <p class="meta">${t.pipeline}</p>
      </article>`
    )
    .join("");
}

async function compare(query) {
  runsEl.innerHTML = "<p class='meta'>Running retrievers…</p>";
  const data = await fetch("/api/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  }).then((r) => r.json());
  runsEl.innerHTML = data.runs
    .map(
      (run) => `<article class="run">
        <h3>${run.name}</h3>
        <p class="meta">${run.doc_ids.join(", ") || "no hits"}</p>
        ${(run.previews || [])
          .map((p) => `<p><strong>${p.id}</strong> (${p.score}) — ${p.snippet}</p>`)
          .join("")}
      </article>`
    )
    .join("");
}

async function loadMcp() {
  const data = await fetch("/api/mcp").then((r) => r.json());
  mcpEl.innerHTML = `
    <h2>${data.protocol}</h2>
    <p><strong>Host</strong> — ${data.roles.host}</p>
    <p><strong>Client</strong> — ${data.roles.client}</p>
    <p><strong>Server</strong> — ${data.roles.server}</p>
    <h3>Tools in this lab</h3>
    <ul>${data.primitives.tools.map((t) => `<li><code>${t.name}</code> — ${t.use}</li>`).join("")}</ul>
    <h3>Resources</h3>
    <ul>${data.primitives.resources.map((t) => `<li><code>${t.uri}</code> — ${t.use}</li>`).join("")}</ul>
    <h3>RAG vs MCP</h3>
    <ul>${data.vs_rag.map((line) => `<li>${line}</li>`).join("")}</ul>
    <pre>${data.run}</pre>
  `;
}

document.getElementById("compare-form").addEventListener("submit", (e) => {
  e.preventDefault();
  compare(document.getElementById("query").value);
});

loadCatalog();
compare(document.getElementById("query").value);
loadMcp();
