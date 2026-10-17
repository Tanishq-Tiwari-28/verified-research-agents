const API = "http://localhost:8000";
const $ = (id) => document.getElementById(id);
const ORDER = ["Retriever", "Synthesizer", "Fact-Checker"];

const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function renderAnswer(text, sources) {
  const byId = Object.fromEntries(sources.map((s) => [s.id, s]));
  let html = esc(text).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\n/g, "<br>");
  return html.replace(/\[((?:S\d+(?:,\s*)?)+)\]/g, (_, ids) =>
    ids.split(/,\s*/).map((id) => byId[id]
      ? `<a class="cite" href="${esc(byId[id].url)}" target="_blank" rel="noreferrer" title="${esc(byId[id].title)}">${id}</a>`
      : id).join(""));
}

function setActive(agent) {
  const idx = ORDER.indexOf(agent);
  document.querySelectorAll("#pipeline li").forEach((li) => {
    const i = ORDER.indexOf(li.dataset.agent);
    li.classList.toggle("active", i === idx);
    li.classList.toggle("done", i < idx);
  });
}

function addTrace(e) {
  const li = document.createElement("li");
  li.innerHTML = `<b>${esc(e.agent)}</b> ${esc(e.detail)} <span>(${e.t}s)</span>`;
  $("trace").appendChild(li);
  setActive(e.agent);
}

function showFinal(r) {
  $("answer-card").hidden = false;
  const badge = $("badge");
  if (r.verified === true) { badge.className = "badge ok"; badge.textContent = `Verified${r.revisions ? ` after ${r.revisions} revision(s)` : ""}`; }
  else { badge.className = "badge warn"; badge.textContent = "Not fully verified. Check the sources."; }
  $("answer").innerHTML = renderAnswer(r.answer, r.sources);
  $("sources").innerHTML = r.sources.map((s) =>
    `<li><a href="${esc(s.url)}" target="_blank" rel="noreferrer">${s.id} ${esc(s.title)}</a>${s.date ? ` (${esc(s.date)})` : ""}</li>`).join("");
  document.querySelectorAll("#pipeline li").forEach((li) => { li.classList.remove("active"); li.classList.add("done"); });
}

async function ask() {
  const question = $("q").value.trim();
  if (!question) return;
  $("go").disabled = true;
  $("error").hidden = true; $("answer-card").hidden = true;
  $("trace").innerHTML = ""; $("trace-wrap").hidden = false; $("pipeline").hidden = false;
  setActive("Retriever");
  try {
    const res = await fetch(`${API}/ask_stream`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }),
    });
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line) continue;
        const msg = JSON.parse(line);
        if (msg.event === "trace") addTrace(msg);
        else if (msg.event === "final") showFinal(msg.result);
        else if (msg.event === "error") throw new Error(msg.message);
      }
    }
  } catch (err) {
    $("error").hidden = false;
    $("error").textContent = `Request failed: ${err.message}. Is the backend running on ${API}?`;
  } finally {
    $("go").disabled = false;
  }
}

$("go").addEventListener("click", ask);
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask(); });
