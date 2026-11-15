"""Клиент к OpenAI-совместимому API: генерация, эмбеддинги, реранк."""

from __future__ import annotations

import json
import math
import random
import re
import sys
import threading
import time

import httpx
import numpy as np

import config

EMBED_BATCH = 64
RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(Exception):
    pass


class EmptyResponse(LLMError):
    """Сбой одного ответа, а не API: обрабатывается на уровне вопроса, прогон не прерывается."""


_client: httpx.Client | None = None
_client_lock = threading.Lock()
_stats_lock = threading.Lock()
stats = {
    "chat": 0,
    "embed": 0,
    "rerank": 0,
    "retries": 0,
    "truncated": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
}


def client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                timeout=httpx.Timeout(config.HTTP_TIMEOUT, connect=10.0),
                limits=httpx.Limits(max_connections=config.MAX_WORKERS * 2 + 4),
            )
        return _client


def _count(key: str, value: int = 1) -> None:
    with _stats_lock:
        stats[key] += value


def post(path: str, payload: dict, key: str) -> dict:
    if not config.BASE_URL or not key:
        raise LLMError(f"{path}: не задан адрес API или ключ")
    url = config.BASE_URL + path
    headers = {"Authorization": f"Bearer {key}"}
    last_error = ""
    for attempt in range(config.HTTP_RETRIES):
        if attempt:
            _count("retries")
        retry_after = ""
        try:
            response = client().post(url, json=payload, headers=headers)
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ) as exc:
            last_error = type(exc).__name__
        except httpx.HTTPError as exc:
            raise LLMError(f"{path}: {type(exc).__name__}: {exc}") from exc
        else:
            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError:
                    raise LLMError(
                        f"{path}: ответ не JSON: {response.text[:200]}"
                    ) from None
                if not isinstance(data, dict):
                    raise LLMError(f"{path}: неожиданный ответ: {str(data)[:200]}")
                return data
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code not in RETRY_STATUSES:
                break
            retry_after = response.headers.get("retry-after", "")
        if attempt == config.HTTP_RETRIES - 1:
            break
        if retry_after.isdigit():
            time.sleep(min(int(retry_after), 30))
        else:
            # экспоненциальная пауза со случайной добавкой, чтобы потоки не били в API одновременно
            time.sleep(min(2**attempt, 20) + random.random())
    raise LLMError(f"{path}: {last_error}")


def chat(
    messages: list[dict],
    json_mode: bool = True,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or config.MAX_TOKENS,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # DeepSeek в JSON-режиме возвращает строку из пробелов, если не смог выдать JSON
    content = ""
    for _ in range(3):
        _count("chat")
        data = post("/chat/completions", payload, config.LLM_API_KEY)
        usage = data.get("usage") or {}
        _count("prompt_tokens", usage.get("prompt_tokens") or 0)
        _count("completion_tokens", usage.get("completion_tokens") or 0)
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMError(f"chat: неожиданный ответ: {str(data)[:200]}") from None
        if not isinstance(content, str):
            raise LLMError(f"chat: content не строка: {str(content)[:200]}")
        truncated = choice.get("finish_reason") == "length"
        if content.strip() and not truncated:
            return content.strip()
        if truncated:
            # в лимит входят и скрытые рассуждения модели, поэтому обрезаться может даже короткий ответ
            _count("truncated")
            payload["max_tokens"] = min(payload["max_tokens"] * 2, config.MAX_TOKENS_CEILING)
    if content.strip():
        print("chat: ответ обрезан по max_tokens и после повторов", file=sys.stderr)
        return content.strip()
    raise EmptyResponse("модель вернула пустой ответ")


def parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def chat_json(messages: list[dict], **kwargs) -> tuple[dict | None, str]:
    """Возвращает разобранный JSON и сырой текст. Если JSON не разобрался, dict равен None."""
    raw = chat(messages, json_mode=True, **kwargs)
    return parse_json(raw), raw


def embed(texts: list[str]) -> np.ndarray:
    """Нормированные векторы: косинусная близость считается скалярным произведением."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        _count("embed")
        data = post(
            "/embeddings",
            {"model": config.EMB_MODEL, "input": batch, "encoding_format": "float"},
            config.EMB_API_KEY,
        )
        if not isinstance(data.get("data"), list):
            raise LLMError(f"embeddings: неожиданный ответ: {str(data)[:200]}")
        items = sorted(data["data"], key=lambda item: item.get("index", 0))
        if len(items) != len(batch):
            raise LLMError(
                f"embeddings: ждали {len(batch)} векторов, пришло {len(items)}"
            )
        vectors += [item["embedding"] for item in items]
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def rerank(
    query: str, documents: list[str], top_n: int | None = None
) -> list[tuple[int, float]]:
    """Индексы документов с оценками, от лучшего к худшему."""
    if not documents:
        return []
    top_n = min(top_n or config.RERANKER_TOP_N, len(documents))
    _count("rerank")
    data = post(
        "/rerank",
        {
            "model": config.RERANKER_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        },
        config.RERANKER_API_KEY,
    )
    results = data.get("results")
    if not results:
        results = data.get("data")
    if results is None:
        results = []
    if not isinstance(results, list):
        raise LLMError("rerank: поле results/data должно быть списком")
    ranked = []
    seen: set[int] = set()
    for position, item in enumerate(results):
        if not isinstance(item, dict):
            raise LLMError(f"rerank: элемент results[{position}] должен быть объектом")
        index = item.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(documents)
        ):
            raise LLMError(f"rerank: некорректный index в results[{position}]")
        if index in seen:
            raise LLMError(f"rerank: повторный index {index} в results")
        score = item.get("relevance_score")
        if score is None:
            score = item.get("score")
        if isinstance(score, bool) or score is None:
            raise LLMError(f"rerank: отсутствует числовой score в results[{position}]")
        try:
            score = float(score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise LLMError(f"rerank: некорректный score в results[{position}]") from exc
        if not math.isfinite(score):
            raise LLMError(f"rerank: нечисловой score в results[{position}]")
        ranked.append((index, score))
        seen.add(index)
    if not ranked:
        raise LLMError(f"rerank: пустой или непонятный ответ: {str(data)[:200]}")
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return ranked[:top_n]
