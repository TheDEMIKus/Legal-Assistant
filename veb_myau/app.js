let allDocs = [];
let sessions = [];
let activeSessionId = null;

document.addEventListener("DOMContentLoaded", () => {
  initDefaultDate();
  loadStats();
  loadDocs();
  createNewSession();
});

function initDefaultDate() {
  const dateInput = document.getElementById("custom-date");
  if (!dateInput) return;
  const now = new Date();
  const yyyy = now.getFullYear();
  const mm = String(now.getMonth() + 1).padStart(2, "0");
  const dd = String(now.getDate()).padStart(2, "0");
  dateInput.value = `${yyyy}-${mm}-${dd}`;
}

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
    const docsEl = document.getElementById("stat-docs");
    if (docsEl) docsEl.textContent = data.docs_count;
  } catch (err) {
    console.error("Ошибка загрузки статистики:", err);
  }
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
      const kindLabel = d.kind === "law" ? "Закон РФ" : "Договор / Акт";
      const validity = d.valid_until
        ? `до ${d.valid_until}`
        : d.valid_from
        ? `с ${d.valid_from}`
        : "действует";
      return `
        <div class="doc-item" onclick="openDocModal(${d.id})">
          <div class="doc-item-title">${icon} ${escapeHtml(d.title)}</div>
          <div class="doc-item-meta">
            <span>${kindLabel}</span>
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

/**
 * Ищет цитату в полном тексте документа (с учетом переносов строк и многоточий)
 * и оборачивает найденный фрагмент в <mark id="doc-highlight">
 */
function renderHighlightedText(fullText, quoteText) {
  if (!quoteText || !quoteText.trim()) {
    return { html: escapeHtml(fullText), found: false };
  }

  const cleanQuote = quoteText
    .trim()
    .replace(/^[«"'\s.…]+|[»"'\s.…]+$/g, "")
    .trim();

  if (!cleanQuote) {
    return { html: escapeHtml(fullText), found: false };
  }

  const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  // 1. Поиск полного текста цитаты (допускаем любые пробелы/переносы строк между словами)
  const fullPattern = cleanQuote
    .split(/\s+/)
    .filter(Boolean)
    .map(escapeRegex)
    .join("[\\s\\r\\n]+");

  let match = fullPattern ? new RegExp(fullPattern, "i").exec(fullText) : null;

  // 2. Если в цитате есть сокращения через «...», ищем по самой длинной непрерывной части
  if (!match) {
    const parts = cleanQuote
      .split(/(?:\.\.\.|…)/)
      .map((p) => p.trim())
      .filter((p) => p.length >= 12)
      .sort((a, b) => b.length - a.length);

    for (const part of parts) {
      const partPattern = part
        .split(/\s+/)
        .filter(Boolean)
        .map(escapeRegex)
        .join("[\\s\\r\\n]+");
      match = new RegExp(partPattern, "i").exec(fullText);
      if (match) break;
    }
  }

  // 3. Резервный поиск по первым 8 словам цитаты
  if (!match) {
    const firstWords = cleanQuote.split(/\s+/).filter(Boolean).slice(0, 8);
    if (firstWords.length >= 3) {
      const fallbackPattern = firstWords.map(escapeRegex).join("[\\s\\r\\n]+");
      match = new RegExp(fallbackPattern, "i").exec(fullText);
    }
  }

  if (!match) {
    return { html: escapeHtml(fullText), found: false };
  }

  const start = match.index;
  const end = start + match[0].length;

  const before = escapeHtml(fullText.slice(0, start));
  const highlighted = escapeHtml(fullText.slice(start, end));
  const after = escapeHtml(fullText.slice(end));

  return {
    html: `${before}<mark id="doc-highlight" class="doc-quote-highlight">${highlighted}</mark>${after}`,
    found: true,
  };
}

function openDocFromQuote(el) {
  const docIdAttr = el.getAttribute("data-doc-id");
  if (docIdAttr === null || docIdAttr === "") return;
  const docId = Number(docIdAttr);
  const quoteText = el.getAttribute("data-quote") || "";
  openDocModal(docId, quoteText);
}

async function openDocModal(docId, highlightQuote = "") {
  const modal = document.getElementById("doc-modal");
  const titleEl = document.getElementById("modal-doc-title");
  const kindEl = document.getElementById("modal-doc-kind");
  const datesEl = document.getElementById("modal-doc-dates");
  const textEl = document.getElementById("modal-doc-text");
  const bodyEl = document.getElementById("modal-doc-body");

  modal.classList.add("open");
  titleEl.textContent = "Загрузка...";
  textEl.textContent = "Читаем документ из базы знаний...";
  if (bodyEl) bodyEl.scrollTop = 0;

  try {
    const res = await fetch(`/api/docs/${docId}`);
    const doc = await res.json();
    kindEl.textContent = doc.kind === "law" ? "ЗАКОНОДАТЕЛЬСТВО РФ" : "ДОКУМЕНТ КОМПАНИИ";
    titleEl.textContent = doc.title;
    const dates = [];
    if (doc.valid_from) dates.push(`Действует с: ${doc.valid_from}`);
    if (doc.valid_until) dates.push(`Утратил силу с: ${doc.valid_until}`);
    datesEl.textContent = dates.join(" · ") || `Файл: ${doc.name}`;

    if (highlightQuote) {
      const { html, found } = renderHighlightedText(doc.text, highlightQuote);
      textEl.innerHTML = html;
      if (found) {
        setTimeout(() => {
          const markEl = document.getElementById("doc-highlight");
          if (markEl) {
            markEl.scrollIntoView({ behavior: "smooth", block: "center" });
          }
        }, 60);
      }
    } else {
      textEl.textContent = doc.text;
    }
  } catch (err) {
    textEl.textContent = "Ошибка загрузки документа: " + err.message;
  }
}

function closeDocModal(e) {
  if (e && e.target !== document.getElementById("doc-modal")) return;
  document.getElementById("doc-modal").classList.remove("open");
}

function startMascotAnimation() {
  const box = document.getElementById("mascot-box");
  if (box) box.classList.add("visible");
}

function stopMascotAnimation() {
  const box = document.getElementById("mascot-box");
  if (box) box.classList.remove("visible");
}

async function handleAsk(event) {
  event.preventDefault();
  const input = document.getElementById("question-input");
  const sendBtn = document.getElementById("send-btn");
  const customDate = document.getElementById("custom-date").value || null;
  const question = input.value.trim();
  if (!question) return;

  const stream = document.getElementById("chat-stream");
  const session = sessions.find((s) => s.id === activeSessionId);

  // 1. Выводим сообщение пользователя
  const userCard = document.createElement("div");
  userCard.className = "msg user";
  userCard.textContent = question;
  stream.appendChild(userCard);

  // Сохраняем в сессию
  if (session) {
    session.messages.push({ role: "user", text: question });
    if (session.title === "Новое дело") {
      session.title = question.substring(0, 34) + (question.length > 34 ? "..." : "");
    }
    renderSessionsList();
    updateActiveCaseBar();
  }

  input.value = "";
  sendBtn.disabled = true;
  startMascotAnimation();
  stream.scrollTop = stream.scrollHeight;

  try {
    const historyPayload = session ? session.history : [];

    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        history: historyPayload,
        custom_date: customDate,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    if (session) {
      session.history.push({ role: "Пользователь", text: question });
      session.history.push({ role: "Ассистент", text: data.answer });
      session.messages.push({
        role: "bot",
        text: data.answer,
        status: data.status,
        quotes: data.quotes || [],
      });
      renderSessionsList();
      updateActiveCaseBar();
    }

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

function buildBotMessageHtml(data) {
  const statusMap = {
    ok: { label: "ПРАВОВОЕ ЗАКЛЮЧЕНИЕ", cls: "ok" },
    fixed: { label: "ПРАВОВОЕ ЗАКЛЮЧЕНИЕ", cls: "ok" },
    not_found: { label: "НЕТ ДАННЫХ В БАЗЕ", cls: "not_found" },
    fallback: { label: "ТРЕБУЕТСЯ УТОЧНЕНИЕ", cls: "fallback" },
  };

  const badgeInfo = statusMap[data.status] || { label: "ОТВЕТ СФОРМИРОВАН", cls: "ok" };

  let quotesHtml = "";
  if (data.quotes && data.quotes.length > 0) {
    const items = data.quotes
      .map((q) => {
        const hasDoc = q.doc_id !== null && q.doc_id !== undefined;
        const clickAttr = hasDoc
          ? `data-doc-id="${q.doc_id}" data-quote="${escapeHtml(q.text)}" onclick="openDocFromQuote(this)" class="quote-doc-link" title="Открыть документ и перейти к цитате"`
          : "";
        return `
          <div class="quote-card">
            <div class="quote-header">
              <span ${clickAttr}>📄 ${escapeHtml(q.doc_title)} ${
          q.section ? "· " + escapeHtml(q.section) : ""
        }</span>
            </div>
            <div>«${escapeHtml(q.text)}»</div>
          </div>
        `;
      })
      .join("");

    quotesHtml = `
      <details class="info-drawer" open>
        <summary>📌 Дословные цитаты из документов (${data.quotes.length})</summary>
        ${items}
      </details>
    `;
  }

  return `
    <div class="msg-meta">
      <span class="badge ${badgeInfo.cls}">${badgeInfo.label}</span>
    </div>
    <div class="msg-text">${escapeHtml(data.text || data.answer)}</div>
    ${quotesHtml}
  `;
}

function renderBotAnswer(data) {
  const stream = document.getElementById("chat-stream");
  const botCard = document.createElement("div");
  botCard.className = "msg bot";
  botCard.innerHTML = buildBotMessageHtml(data);
  stream.appendChild(botCard);
}

function createNewSession() {
  const newSession = {
    id: Date.now().toString(),
    title: "Новое дело",
    pinned: false,
    messages: [],
    history: [],
    date: new Date().toLocaleDateString("ru-RU"),
  };

  sessions.unshift(newSession);
  activeSessionId = newSession.id;

  renderSessionsList();
  updateActiveCaseBar();

  document.getElementById("chat-stream").innerHTML = `
    <div class="msg bot">
      <div class="msg-meta"><span class="badge ok">ГОТОВ К РАБОТЕ</span></div>
      <div class="msg-text">Здравствуйте! Задайте ваш вопрос по текущему делу.</div>
    </div>
  `;
}

function togglePinSession(id, event) {
  if (event) event.stopPropagation();
  const session = sessions.find((s) => s.id === id);
  if (!session) return;
  session.pinned = !session.pinned;
  renderSessionsList();
}

function renderSessionCardHtml(s) {
  const isActive = s.id === activeSessionId ? "active" : "";
  const isPinned = s.pinned ? "pinned" : "";
  const pinTitle = s.pinned ? "Открепить дело" : "Закрепить дело";
  const previewText =
    s.messages.length > 0
      ? escapeHtml(s.messages[s.messages.length - 1].text.substring(0, 42)) + "..."
      : "Пустое дело";

  return `
    <div class="session-item ${isActive}" onclick="switchSession('${s.id}')">
      <div class="session-top">
        <div class="session-title">${escapeHtml(s.title)}</div>
        <div class="session-actions">
          <button
            type="button"
            class="session-icon-btn ${isPinned}"
            onclick="togglePinSession('${s.id}', event)"
            title="${pinTitle}"
          >📌</button>
        </div>
      </div>
      <div class="session-preview">${previewText}</div>
      <div class="session-meta">
        <span>${s.date}</span>
        <span>${s.messages.length} сообщ.</span>
      </div>
    </div>
  `;
}

function renderSessionsList() {
  const archiveContainer = document.getElementById("sessions-list");
  const pinnedContainer = document.getElementById("pinned-list");
  const pinnedCountEl = document.getElementById("pinned-count");

  const searchQ = document.getElementById("chat-search")?.value.toLowerCase().trim() || "";

  // 1. Отрисовка архива дел
  if (archiveContainer) {
    let filtered = sessions;
    if (searchQ) {
      filtered = sessions.filter(
        (s) =>
          s.title.toLowerCase().includes(searchQ) ||
          s.messages.some((m) => m.text.toLowerCase().includes(searchQ))
      );
    }

    if (filtered.length === 0) {
      archiveContainer.innerHTML = `<div class="side-hint">Дела не найдены</div>`;
    } else {
      archiveContainer.innerHTML = filtered.map(renderSessionCardHtml).join("");
    }
  }

  // 2. Отрисовка закрепленных дел
  const pinnedSessions = sessions.filter((s) => s.pinned);
  if (pinnedCountEl) {
    pinnedCountEl.textContent = pinnedSessions.length;
  }

  if (pinnedContainer) {
    if (pinnedSessions.length === 0) {
      pinnedContainer.innerHTML = `<div class="side-hint">Нет закрепленных дел. Нажмите 📌 на нужном чате в архиве.</div>`;
    } else {
      pinnedContainer.innerHTML = pinnedSessions.map(renderSessionCardHtml).join("");
    }
  }
}

function switchSession(id) {
  activeSessionId = id;
  const session = sessions.find((s) => s.id === id);
  if (!session) return;

  renderSessionsList();
  updateActiveCaseBar();

  const stream = document.getElementById("chat-stream");
  stream.innerHTML = "";

  if (session.messages.length === 0) {
    stream.innerHTML = `
      <div class="msg bot">
        <div class="msg-meta"><span class="badge ok">ГОТОВ К РАБОТЕ</span></div>
        <div class="msg-text">Здравствуйте! Задайте ваш вопрос по текущему делу.</div>
      </div>
    `;
  } else {
    session.messages.forEach((msg) => {
      const card = document.createElement("div");
      if (msg.role === "user") {
        card.className = "msg user";
        card.textContent = msg.text;
      } else {
        card.className = "msg bot";
        card.innerHTML = buildBotMessageHtml(msg);
      }
      stream.appendChild(card);
    });
  }
  stream.scrollTop = stream.scrollHeight;
}

function updateActiveCaseBar() {
  const session = sessions.find((s) => s.id === activeSessionId);
  if (session) {
    document.getElementById("active-case-title").textContent = session.title;
    document.getElementById("active-case-count").textContent = `Сообщений: ${session.messages.length}`;
  }
}

function renameActiveSession() {
  const session = sessions.find((s) => s.id === activeSessionId);
  if (!session) return;
  const newTitle = prompt("Введите новое название для дела:", session.title);
  if (newTitle && newTitle.trim()) {
    session.title = newTitle.trim();
    renderSessionsList();
    updateActiveCaseBar();
  }
}