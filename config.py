from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    # не перезаписывает переменные, заданные платформой
    load_dotenv(override=False)
except ImportError:
    if Path(".env").exists():
        print(
            "ВНИМАНИЕ: найден .env, но python-dotenv не установлен — .env не загружен",
            file=sys.stderr,
        )


def env(*names: str, default: str = "") -> str:
    """Первое непустое значение из списка имён переменных.

    Значение, равное имени переменной (неподставленный шаблон вида
    LLM_MODEL=LLM_MODEL), считается отсутствующим.
    """
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return default


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


BASE_URL = env("LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL").rstrip("/")
LLM_API_KEY = env("LLM_API_KEY", "OPENAI_API_KEY")
EMB_API_KEY = env("EMBEDDER_API_KEY") or LLM_API_KEY
RERANKER_API_KEY = env("RERANKER_API_KEY") or LLM_API_KEY

LLM_MODEL = env(
    "LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL", default="deepseek/deepseek-v4.1-flash"
)
EMB_MODEL = env("EMB_MODEL", default="baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", default="qwen/qwen3-reranker-8b")

DATA_DIR = Path(env("DATA_DIR", default="data"))

# Параметры генерации и поиска заданы в коде, одноимённые переменные платформы не читаются:
# их значения подобраны под стартер, а решение проверено на этих.
TEMPERATURE = 0.1
MAX_TOKENS = 3000
# при обрезанном ответе лимит удваивается до этого потолка
MAX_TOKENS_CEILING = 12000
TOP_K = 20  # кандидатов от каждого поиска, а не итоговых
RERANKER_TOP_N = 7

# проверка ответа LLM-судьёй; REVIEW=0 отключает её для сравнения прогонов
REVIEW = env("REVIEW", default="1") != "0"

MAX_WORKERS = env_int("MAX_WORKERS", 4, 1, 16)

HTTP_TIMEOUT = 60.0
HTTP_RETRIES = 3


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
        f"workers={MAX_WORKERS} top_k={TOP_K} top_n={RERANKER_TOP_N} review={REVIEW}"
    )
