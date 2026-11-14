"""FastAPI веб-сервер для юридического ИИ-ассистента ПСБ (ООО «Меридиан»)."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import llm
from answer import answer_question
from loader import Chunk, Document, load_knowledge_base
from search import SearchIndex
from solution import check_reranker

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "veb_myau"


class AppState:
    docs: list[Document] = []
    chunks: list[Chunk] = []
    index: SearchIndex | None = None
    startup_sec: float = 0.0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает базу знаний, строит BM25 + векторный индекс и проверяет реранкер при старте."""
    t0 = time.monotonic()
    print(f"[startup] Конфигурация: {config.describe()}", flush=True)
    config.check()

    state.docs, state.chunks = load_knowledge_base()
    check_reranker()
    state.index = SearchIndex(state.docs, state.chunks)
    state.startup_sec = round(time.monotonic() - t0, 2)

    print(
        f"[startup] База готова за {state.startup_sec} с: "
        f"документов={len(state.docs)}, чанков={len(state.chunks)}",
        flush=True,
    )
    yield


app = FastAPI(
    title="ПСБ — Юридический ИИ-ассистент",
    description="RAG-система по договорам, регламентам ООО «Меридиан» и законодательству РФ",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic-модели запросов и ответов ---


class DialogTurn(BaseModel):
    role: str = Field(..., pattern="^(Пользователь|Ассистент)$")
    text: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Текущий вопрос пользователя")
    history: list[DialogTurn] = Field(
        default_factory=list,
        description="История диалога в формате Пользователь / Ассистент",
    )
    custom_date: str | None = Field(
        default=None,
        description="Опциональная дата в формате YYYY-MM-DD для тестирования редакций",
    )


class QuoteItem(BaseModel):
    doc_index: int
    doc_id: int | None = None
    doc_title: str = ""
    section: str = ""
    text: str


class RetrievedHitItem(BaseModel):
    doc_index: int
    doc_id: int
    doc_title: str
    kind: str
    section: str
    score: float
    source: str  # "rerank" или "related"
    preview: str


class AskResponse(BaseModel):
    answer: str
    status: str  # "ok", "fixed", "not_found", "fallback"
    attempts: int
    quotes: list[QuoteItem]
    problems: list[str]
    first_problems: list[str]
    hits: list[RetrievedHitItem]
    elapsed_sec: float
    llm_stats: dict[str, int]


# --- API Эндпоинты ---


@app.post("/api/ask", response_model=AskResponse)
def ask_question_endpoint(req: AskRequest) -> AskResponse:
    """Выполняет гибридный поиск, генерацию ответа и верификацию цитат/дат/расчётов."""
    if state.index is None:
        raise HTTPException(status_code=503, detail="Поисковый индекс ещё не загружен")

    t0 = time.monotonic()
    question_clean = req.question.strip()
    if not question_clean:
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    # Формируем строку диалога в формате, который ожидают search.parse_dialog и answer.SYSTEM
    if req.history:
        turns = [f"{turn.role}: {turn.text.strip()}" for turn in req.history if turn.text.strip()]
        turns.append(f"Пользователь: {question_clean}")
        full_question = "\n".join(turns)
    else:
        full_question = question_clean

    # Определяем дату, на которую проверяются редакции документов
    target_date = date.today()
    if req.custom_date:
        try:
            target_date = date.fromisoformat(req.custom_date)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Некорректный формат custom_date (ожидается YYYY-MM-DD)"
            )

    try:
        # 1. Эмбеддинг запроса и гибридный поиск (BM25 + bge-m3 + Rerank + связанные допсоглашения)
        query_vector = state.index.embed_queries([full_question])[0]
        hits = state.index.search(full_question, query_vector)

        # 2. Генерация ответа и двухэтапная проверка кодом
        result = answer_question(full_question, hits, state.docs, today=target_date)
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=f"Ошибка LLM/Rerank API: {exc}") from exc

    # 3. Обогащаем цитаты названиями документов и разделов для карточек на фронтенде
    enriched_quotes: list[QuoteItem] = []
    for q in result.quotes:
        if not isinstance(q, dict):
            continue
        raw_idx = q.get("doc", 0)
        doc_idx = int(raw_idx) if isinstance(raw_idx, (int, str)) and str(raw_idx).isdigit() else 0
        q_text = str(q.get("text", "")).strip()

        doc_id = None
        doc_title = f"Фрагмент #{doc_idx}"
        section = ""

        if 1 <= doc_idx <= len(hits):
            chunk = hits[doc_idx - 1].chunk
            doc_obj = state.docs[chunk.doc_id]
            doc_id = doc_obj.id
            doc_title = doc_obj.title
            section = chunk.section

        enriched_quotes.append(
            QuoteItem(
                doc_index=doc_idx,
                doc_id=doc_id,
                doc_title=doc_title,
                section=section,
                text=q_text,
            )
        )

    # 4. Формируем список найденных фрагментов контекста (<context>), чтобы видеть работу поиска
    hit_items: list[RetrievedHitItem] = []
    for idx, h in enumerate(hits, 1):
        doc_obj = state.docs[h.chunk.doc_id]
        hit_items.append(
            RetrievedHitItem(
                doc_index=idx,
                doc_id=doc_obj.id,
                doc_title=doc_obj.title,
                kind=doc_obj.kind,
                section=h.chunk.section,
                score=round(float(h.score), 4),
                source=h.source,
                preview=h.chunk.body[:300] + ("…" if len(h.chunk.body) > 300 else ""),
            )
        )

    elapsed = round(time.monotonic() - t0, 2)

    return AskResponse(
        answer=result.text,
        status=result.status,
        attempts=result.attempts,
        quotes=enriched_quotes,
        problems=result.problems,
        first_problems=result.first_problems,
        hits=hit_items,
        elapsed_sec=elapsed,
        llm_stats=dict(llm.stats),
    )


@app.get("/api/stats")
def get_stats_endpoint() -> dict[str, Any]:
    """Возвращает статистику базы знаний, конфигурации моделей и счётчики токенов llm.stats."""
    return {
        "docs_count": len(state.docs),
        "chunks_count": len(state.chunks),
        "startup_sec": state.startup_sec,
        "models": {
            "llm": config.LLM_MODEL,
            "embedder": config.EMB_MODEL,
            "reranker": config.RERANKER_MODEL,
        },
        "params": {
            "temperature": config.TEMPERATURE,
            "max_tokens": config.MAX_TOKENS,
            "top_k": config.TOP_K,
            "reranker_top_n": config.RERANKER_TOP_N,
        },
        "llm_stats": dict(llm.stats),
    }


@app.get("/api/docs")
def list_documents_endpoint() -> list[dict[str, Any]]:
    """Возвращает список всех загруженных документов базы знаний с датами редакций и связями."""
    result = []
    for doc in state.docs:
        chunks_count = sum(1 for c in state.chunks if c.doc_id == doc.id)
        related_titles = [state.docs[r_id].title for r_id in doc.related if 0 <= r_id < len(state.docs)]
        result.append(
            {
                "id": doc.id,
                "name": doc.name,
                "title": doc.title,
                "kind": doc.kind,
                "doc_date": doc.doc_date.isoformat() if doc.doc_date else None,
                "valid_from": doc.valid_from.isoformat() if doc.valid_from else None,
                "valid_until": doc.valid_until.isoformat() if doc.valid_until else None,
                "chunks_count": chunks_count,
                "related": related_titles,
            }
        )
    return result


@app.get("/api/docs/{doc_id}")
def get_document_detail_endpoint(doc_id: int) -> dict[str, Any]:
    """Возвращает полный текст документа и его нарезку на чанки для просмотра в интерфейсе."""
    if not (0 <= doc_id < len(state.docs)):
        raise HTTPException(status_code=404, detail="Документ не найден")

    doc = state.docs[doc_id]
    doc_chunks = [
        {
            "id": c.id,
            "section": c.section,
            "points": c.points,
            "text": c.text,
        }
        for c in state.chunks
        if c.doc_id == doc.id
    ]
    return {
        "id": doc.id,
        "name": doc.name,
        "title": doc.title,
        "kind": doc.kind,
        "valid_from": doc.valid_from.isoformat() if doc.valid_from else None,
        "valid_until": doc.valid_until.isoformat() if doc.valid_until else None,
        "text": doc.text,
        "chunks": doc_chunks,
    }


# --- Раздача статики (HTML, CSS, JS) ---

STATIC_DIR.mkdir(exist_ok=True)
app.mount("/veb_myau", StaticFiles(directory=STATIC_DIR), name="vab_myau")


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Отдаёт главную страницу веб-интерфейса."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Файл static/index.html не найден. Положите файлы фронтенда в папку static/.",
        )
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)