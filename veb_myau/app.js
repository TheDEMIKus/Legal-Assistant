let dialogHistory = [];
let allDocs = [];
let mascotPinned = false;
let stepInterval = null;

document.addEventListener("DOMContentLoaded", () => {
  loadStats();
  loadDocs();
});

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById("stat-docs").textContent = data.docs_count;
    document.getElementById("stat-chunks").textContent = data.chunks_count;
    updateTokenStats(data.llm_stats);
  } catch (err) {
    console.error("Ошибка загрузки статистики:", err);
  }
}

function updateTokenStats(llmStats) {
  if (!llmStats) return;
  const totalTokens = (llmStats.prompt_tokens || 0) + (llmStats.completion_tokens || 0);
  const calls = (llmStats.chat || 0) + (llmStats.rerank || 0);
  document.getElementById("stat-tokens").textContent = `${calls} запр. / ${totalTokens} ток.`;
}

async function loadDocs() {
  const container = document.getElementById("docs-list");
  try {
    const res = await fetch("/api/docs");
    if (!res.ok) throw new Error("Не удалось получить список документов");
    allDocs = await res.json();
    renderDocs(allDocs);
  } catch (err) {
    container.innerHTML = `<div class="side-hint">Ошибка загрузки базы: ${escapeHtml(err.message)}</div>`;
  }
}

function renderDocs(docs) {
  const container = document.getElementById("docs-list");
  if (!docs.length) {
    container.innerHTML = `<div class="side-hint">Документы не найдены</div>`;
    return;
  }
  container.innerHTML = docs
    .map((d) => {
      const icon = d.kind === "law" ? "⚖️" : "📄";
      const validity = d.valid_until
        ? `до ${d.valid_until}`
        : d.valid_from
        ? `с ${d.valid_from}`
        : "действует";
      return `
        <div class="doc-item" onclick="openDocModal(${d.id})">
          <div class="doc-item-title">${icon} ${escapeHtml(d.title)}</div>
          <div class="doc-item-meta">
            <span>Чанков: ${d.chunks_count}</span>
            <span>${escapeHtml(validity)}</span>
          </div>
        </div>
      `;
    })
    .join("");
}

function filterDocs() {
  const q = document.getElementById("doc-filter").value.toLowerCase().trim();
  if (!q) {
    renderDocs(allDocs);
    return;
  }
  renderDocs(
    allDocs.filter(
      (d) => d.title.toLowerCase().includes(q) || d.name.toLowerCase().includes(q)
    )
  );
}

async function openDocModal(docId) {
  const modal = document.getElementById("doc-modal");
  const titleEl = document.getElementById("modal-doc-title");
  const kindEl = document.getElementById("modal-doc-kind");
  const datesEl = document.getElementById("modal-doc-dates");
  const textEl = document.getElementById("modal-doc-text");

  modal.classList.add("open");
  titleEl.textContent = "Загрузка...";
  textEl.textContent = "Читаем документ из базы знаний...";

  try {
    const res = await fetch(`/api/docs/${docId}`);
    const doc = await res.json();
    kindEl.textContent = doc.kind === "law" ? "ЗАКОНОДАТЕЛЬСТВО РФ" : "ДОКУМЕНТ КОМПАНИИ";
    titleEl.textContent = doc.title;
    const dates = [];
    if (doc.valid_from) dates.push(`Действует с: ${doc.valid_from}`);
    if (doc.valid_until) dates.push(`Утратил силу с: ${doc.valid_until}`);
    datesEl.textContent = dates.join(" · ") || `Файл: ${doc.name}`;
    textEl.textContent = doc.text;
  } catch (err) {
    textEl.textContent = "Ошибка загрузки документа: " + err.message;
  }
}

function closeDocModal(e) {
  if (e && e.target !== document.getElementById("doc-modal")) return;
  document.getElementById("doc-modal").classList.remove("open");
}

function togglePinMascot() {
  mascotPinned = !mascotPinned;
  const btn = document.getElementById("pin-mascot-btn");
  const box = document.getElementById("mascot-box");
  btn.classList.toggle("active", mascotPinned);
  btn.textContent = mascotPinned ? "Открепить" : "Закрепить";
  if (mascotPinned) {
    box.classList.add("visible");
  } else {
    box.classList.remove("visible");
  }
}

function startMascotAnimation() {
  const box = document.getElementById("mascot-box");
  box.classList.add("visible");

  const steps = ["step-1", "step-2", "step-3", "step-4"];
  let idx = 0;
  steps.forEach((id, i) => {
    document.getElementById(id).classList.toggle("active", i === 0);
  });

  clearInterval(stepInterval);
  stepInterval = setInterval(() => {
    idx = (idx + 1) % steps.length;
    steps.forEach((id, i) => {
      document.getElementById(id).classList.toggle("active", i === idx);
    });
  }, 950);
}

function stopMascotAnimation() {
  clearInterval(stepInterval);
  if (!mascotPinned) {
    document.getElementById("mascot-box").classList.remove("visible");
  }
}

function fillAndSend(text) {
  const input = document.getElementById("question-input");
  input.value = text;
  document.querySelector(".input-panel").requestSubmit();
}

function resetChat() {
  dialogHistory = [];
  document.getElementById("chat-stream").innerHTML = `
    <div class="msg bot">
      <div class="msg-meta">
        <span class="badge ok">ДИАЛОГ ОЧИЩЕН</span>
      </div>
      <div class="msg-text">Контекст предыдущих реплик сброшен. Задайте новый вопрос!</div>
    </div>
  `;
}

async function handleAsk(event) {
  event.preventDefault();
  const input = document.getElementById("question-input");
  const sendBtn = document.getElementById("send-btn");
  const customDate = document.getElementById("custom-date").value || null;
  const question = input.value.trim();
  if (!question) return;

  const stream = document.getElementById("chat-stream");

  // 1. Выводим сообщение пользователя
  const userCard = document.createElement("div");
  userCard.className = "msg user";
  userCard.textContent = question;
  stream.appendChild(userCard);

  input.value = "";
  sendBtn.disabled = true;
  startMascotAnimation();
  stream.scrollTop = stream.scrollHeight;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        history: dialogHistory,
        custom_date: customDate,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    // Сохраняем в историю диалога
    dialogHistory.push({ role: "Пользователь", text: question });
    dialogHistory.push({ role: "Ассистент", text: data.answer });

    updateTokenStats(data.llm_stats);
    renderBotAnswer(data);
  } catch (err) {
    const errCard = document.createElement("div");
    errCard.className = "msg bot";
    errCard.innerHTML = `
      <div class="msg-meta"><span class="badge error">ОШИБКА СЕРВЕРА</span></div>
      <div class="msg-text">${escapeHtml(err.message)}</div>
    `;
    stream.appendChild(errCard);
  } finally {
    stopMascotAnimation();
    sendBtn.disabled = false;
    stream.scrollTop = stream.scrollHeight;
  }
}

function renderBotAnswer(data) {
  const stream = document.getElementById("chat-stream");

  const statusMap = {
    ok: "ПРОВЕРЕНО С 1-Й ПОПЫТКИ",
    fixed: "ИСПРАВЛЕНО СО 2-Й ПОПЫТКИ",
    not_found: "НЕТ В БАЗЕ ЗНАНИЙ",
    fallback: "ОТКАЗ ВЕРИФИКАТОРА",
  };

  // Блок дословных цитат
  let quotesHtml = "";
  if (data.quotes && data.quotes.length > 0) {
    const items = data.quotes
      .map((q) => {
        const clickAttr =
          q.doc_id !== null && q.doc_id !== undefined
            ? `onclick="openDocModal(${q.doc_id})" class="quote-doc-link"`
            : "";
        return `
          <div class="quote-card">
            <div class="quote-header">
              <span ${clickAttr}>📄 ${escapeHtml(q.doc_title)} ${
          q.section ? "· " + escapeHtml(q.section) : ""
        }</span>
              <span>фрагмент #${q.doc_index}</span>
            </div>
            <div>«${escapeHtml(q.text)}»</div>
          </div>
        `;
      })
      .join("");

    quotesHtml = `
      <details class="info-drawer" open>
        <summary>📌 Дословные цитаты из базы (${data.quotes.length})</summary>
        ${items}
      </details>
    `;
  }

  // Блок ошибок 1-й попытки, если сработало автоисправление
  let problemsHtml = "";
  const diagList =
    data.first_problems && data.first_problems.length
      ? data.first_problems
      : data.problems || [];
  if (diagList.length > 0) {
    const listItems = diagList.map((p) => `<div>• ${escapeHtml(p)}</div>`).join("");
    problemsHtml = `
      <details class="info-drawer warn-drawer">
        <summary>🛠️ Диагностика верификатора (${diagList.length})</summary>
        <div class="quote-card">${listItems}</div>
      </details>
    `;
  }

  // Блок найденных фрагментов поиска (BM25 + Vector + Rerank + Related)
  let hitsHtml = "";
  if (data.hits && data.hits.length > 0) {
    const hitItems = data.hits
      .map(
        (h) => `
        <div class="quote-card">
          <div class="quote-header">
            <span class="quote-doc-link" onclick="openDocModal(${h.doc_id})">
              #${h.doc_index} [${h.source.toUpperCase()}] ${escapeHtml(h.doc_title)}
            </span>
            <span>score: ${h.score}</span>
          </div>
          <div style="color:#475569">${escapeHtml(h.preview)}</div>
        </div>
      `
      )
      .join("");

    hitsHtml = `
      <details class="info-drawer context-drawer">
        <summary>🔍 Найденный контекст &lt;context&gt; (${data.hits.length} чанков)</summary>
        ${hitItems}
      </details>
    `;
  }

  const botCard = document.createElement("div");
  botCard.className = "msg bot";
  botCard.innerHTML = `
    <div class="msg-meta">
      <span class="badge ${data.status}">${statusMap[data.status] || data.status}</span>
      <span class="meta-note">Попыток: ${data.attempts}</span>
      <span class="meta-note">⏱ ${data.elapsed_sec} с</span>
    </div>
    <div class="msg-text">${escapeHtml(data.answer)}</div>
    ${quotesHtml}
    ${problemsHtml}
    ${hitsHtml}
  `;

  stream.appendChild(botCard);
}