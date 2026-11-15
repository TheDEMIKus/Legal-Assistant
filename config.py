from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / ".env"
DOTENV_SETTINGS: dict[str, str | None] = {}

try:
    from dotenv import dotenv_values

    DOTENV_SETTINGS = dotenv_values(ENV_FILE)
    for name, value in DOTENV_SETTINGS.items():
        if value is not None and value.strip():
            os.environ[name] = value
except ImportError:
    if ENV_FILE.exists():
        print(
            "Найден .env, но python-dotenv не установлен; используются переменные окружения.",
            file=sys.stderr,
        )


def env(*names: str, default: str = "") -> str:
    """Ищет непустое значение в .env, затем в окружении процесса."""
    for source in (DOTENV_SETTINGS, os.environ):
        for name in names:
            value = (source.get(name) or "").strip()
            if value and value != name:
                return value
    return default


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(env(name))
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return max(low, min(high, value))


BASE_URL = env("LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL").rstrip("/")
LLM_API_KEY = env("LLM_API_KEY", "OPENAI_API_KEY")
EMB_API_KEY = env("EMBEDDER_API_KEY") or LLM_API_KEY
RERANKER_API_KEY = env("RERANKER_API_KEY") or LLM_API_KEY

LLM_MODEL = env(
    "LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL", default="deepseek/deepseek-v4.1-flash"
)
EMB_MODEL = env("EMB_MODEL", default="baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", default="cohere/rerank-5-fast")

DATA_DIR = Path(env("DATA_DIR", default="data"))

TEMPERATURE = env_float("TEMPERATURE", 0.1, 0.0, 2.0)
MAX_TOKENS = env_int("MAX_TOKENS", 3000, 1, 32768)
MAX_TOKENS_CEILING = env_int(
    "MAX_TOKENS_CEILING", max(12000, MAX_TOKENS), MAX_TOKENS, 65536
)
TOP_K = env_int("TOP_K", 20, 1, 500)
MAX_RERANK_CANDIDATES = env_int("MAX_RERANK_CANDIDATES", 40, 1, 500)
RERANKER_TOP_N = env_int("RERANKER_TOP_N", 5, 1, MAX_RERANK_CANDIDATES)
MAX_RELATED = env_int("MAX_RELATED", 2, 0, 50)
CHUNK_MAX = env_int("CHUNK_MAX", 1200, 100, 10000)
CHUNK_OVERLAP = env_int("CHUNK_OVERLAP", 150, 0, CHUNK_MAX - 1)
EMBED_BATCH = env_int("EMBED_BATCH", 64, 1, 1024)
MAX_ANSWER_CHARS = env_int("MAX_ANSWER_CHARS", 10000, 100, 100000)
MAX_REMARKS = env_int("MAX_REMARKS", 3, 0, 10)
ANSWER_ATTEMPTS = env_int("ANSWER_ATTEMPTS", 2, 1, 2)
CHAT_ATTEMPTS = env_int("CHAT_ATTEMPTS", 3, 1, 5)
HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 60.0, 1.0, 600.0)
HTTP_CONNECT_TIMEOUT = env_float("HTTP_CONNECT_TIMEOUT", 10.0, 1.0, 120.0)
HTTP_RETRIES = env_int("HTTP_RETRIES", 3, 1, 10)
MAX_WORKERS = env_int("MAX_WORKERS", 4, 1, 16)


def check() -> None:
    missing = [
        name
        for name, value in [
            ("LLM_BASE_URL", BASE_URL),
            ("LLM_API_KEY", LLM_API_KEY),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError("Не заданы переменные окружения: " + ", ".join(missing))


def describe() -> str:
    """Для лога. Ключ не выводится, только факт его наличия."""
    return (
        f"base_url={BASE_URL} key_set={bool(LLM_API_KEY)} "
        f"llm={LLM_MODEL} emb={EMB_MODEL} reranker={RERANKER_MODEL} "
        f"data_dir={DATA_DIR} temperature={TEMPERATURE} max_tokens={MAX_TOKENS} "
        f"workers={MAX_WORKERS} top_k={TOP_K} top_n={RERANKER_TOP_N} "
        f"rerank_pool={MAX_RERANK_CANDIDATES} related={MAX_RELATED} "
        f"chunk={CHUNK_MAX}/{CHUNK_OVERLAP} embed_batch={EMBED_BATCH} "
        f"answer_attempts={ANSWER_ATTEMPTS} chat_attempts={CHAT_ATTEMPTS} "
        f"max_remarks={MAX_REMARKS} http_retries={HTTP_RETRIES}"
    )
