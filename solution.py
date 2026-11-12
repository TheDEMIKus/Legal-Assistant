"""Точка входа: читает вопросы и базу знаний из DATA_DIR, пишет submission.csv."""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import config
import llm
from answer import answer_question
from loader import load_knowledge_base, load_questions
from search import SearchIndex

OUTPUT = Path("submission.csv")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def write_submission(questions: list[tuple[str, str]], answers: dict[str, str]) -> None:
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for qid, _ in questions:
            writer.writerow([qid, answers[qid]])


def check_reranker() -> None:
    """Реранкер впервые нужен только на первом вопросе; проверка заранее даёт понятную ошибку вместо трейсбека из потока."""
    try:
        llm.rerank("проверка", ["первый документ", "второй документ"], top_n=1)
    except llm.LLMError as exc:
        raise SystemExit(f"Реранкер {config.RERANKER_MODEL} недоступен: {exc}") from exc


def main() -> None:
    started = time.monotonic()
    log(config.describe())
    config.check()

    questions: list[tuple[str, str]] = []
    seen: set[str] = set()
    for qid, question in load_questions():
        if qid in seen:
            log(f"Повтор id {qid} во входных данных, берётся первый")
            continue
        seen.add(qid)
        questions.append((qid, question))
    docs, chunks = load_knowledge_base()
    check_reranker()
    index = SearchIndex(docs, chunks)
    vectors = index.embed_queries([q for _, q in questions])
    log(
        f"Вопросов: {len(questions)}, документов: {len(docs)}, чанков: {len(chunks)}, "
        f"подготовка {time.monotonic() - started:.0f} с"
    )

    today = date.today()
    answers: dict[str, str] = {}
    statuses: Counter[str] = Counter()
    reviews: Counter[str] = Counter()

    def process(i: int) -> tuple[str, str, str, str]:
        qid, question = questions[i]
        t0 = time.monotonic()
        result = answer_question(
            question, index.search(question, vectors[i]), docs, today
        )
        detail = "; ".join(result.problems or result.first_problems)[:200]
        log(
            f"[{qid}] {result.status} {time.monotonic() - t0:.1f} с"
            + (f" — {detail}" if detail else "")
            + (f" | судья: {result.review}" if result.review else "")
            + (f" — {' / '.join(result.remarks)}" if result.remarks else "")
        )
        return qid, result.text, result.status, result.review

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(process, i) for i in range(len(questions))]
        for future in as_completed(futures):
            try:
                qid, text, status, review = future.result()
            except Exception:
                traceback.print_exc()
                # остальные потоки не дожидаются: их запросы могут висеть до таймаута
                os._exit(1)
            answers[qid] = text
            statuses[status] += 1
            if review:
                reviews[review] += 1

    write_submission(questions, answers)
    print(f"Создан submission.csv: {len(questions)} ответов", flush=True)
    log(
        f"Готово за {time.monotonic() - started:.0f} с: {dict(statuses)}; судья: {dict(reviews)}; API: {llm.stats}"
    )


if __name__ == "__main__":
    main()
