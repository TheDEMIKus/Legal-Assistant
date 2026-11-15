"""Гибридный поиск: BM25 + векторный, реранк объединённых кандидатов, подтягивание связанных документов."""

from __future__ import annotations

import re
import sys
import threading
from dataclasses import dataclass
from itertools import zip_longest

import bm25s
import numpy as np

import config
import llm
from loader import Chunk, Document

STOPWORDS = set("""
и в во на по с со к ко о об от до за из у не ни ли а но же бы то что как это или для при
мы нас наш наша наше наши вы вас ваш он она оно они их его ее мне меня мой я ты
можно нужно надо ли есть был была было были будет быть какой какая какое какие который
так также тоже уже еще только все всё весь вся если чтобы когда где тут там этот эта эти
""".split())

ENDINGS = sorted("""
иями ями ами ого его ому ему ыми ими ией ией ость ости остью ений ениям ениями ениях ением ению ения ение
ание ания аний анием анию ова ева ует ают яют ить ать ять еть уть ешь ете ите ишь ает яет
ую юю ая яя ое ее ые ие ой ей ий ый ом ем ах ях ов ев ам ям ут ют ат ят ет ит ть ла ли ло на ны но
а я о е ы и у ю ь
""".split(), key=len, reverse=True)

ABBREVIATIONS = [
    (r"юротдел\w*", "юридический отдел"),
    (r"юрслужб\w*", "юридический отдел"),
    (r"гендиректор\w*", "генеральный директор"),
    (r"главбух\w*", "главный бухгалтер"),
    (r"допсоглашени\w*|доп\.?\s+соглашени\w*", "дополнительное соглашение"),
    (r"кадровик\w*|отдел\w* кадров", "кадровая служба"),
    (r"пдн", "персональные данные"),
    (r"млрд\.?", "миллиард"),
    (r"млн\.?", "миллион"),
    (r"тыс\.?", "тысяч"),
    (r"руб\.?", "рублей"),
]
ABBREVIATION_RES = [(re.compile(r"\b(?:" + p + r")(?!\w)"), r) for p, r in ABBREVIATIONS]

ID_RE = re.compile(r"\b[а-яa-z]{1,3}-\d+[/-]\d{4}\b")
WORD_RE = re.compile(r"[а-яa-z0-9]+(?:-[а-яa-z0-9]+)*")
TURN_RE = re.compile(r"^(Пользователь|Ассистент)\s*:\s*", re.MULTILINE)


def stem(word: str) -> str:
    if word.isdigit() or len(word) <= 3:
        return word
    for suffix in ("ся", "сь"):
        if word.endswith(suffix) and len(word) - 2 >= 4:
            word = word[:-2]
            break
    for ending in ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            return word[: -len(ending)]
    return word


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    for pattern, replacement in ABBREVIATION_RES:
        text = pattern.sub(replacement, text)
    text = re.sub(r"(?<=\d)[\s ](?=\d{3}\b)", "", text)
    tokens = ["#" + m.replace("/", "-") for m in ID_RE.findall(text)]
    for word in WORD_RE.findall(text):
        parts = word.split("-")
        if len(parts) > 1:
            # Сохраняем составное слово целиком, чтобы различать похожие названия.
            tokens.append("-".join(stem(p) for p in parts))
        tokens += [stem(p) for p in parts if p not in STOPWORDS]
    return tokens


def parse_dialog(question: str) -> list[tuple[str, str]]:
    parts = TURN_RE.split(question)
    if len(parts) < 3:
        return [("Пользователь", question.strip())]
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def search_query(question: str) -> str:
    """Для диалога: реплики пользователя, последняя повторена, чтобы весила больше."""
    turns = [text for role, text in parse_dialog(question) if role == "Пользователь"]
    if not turns:
        return question
    return " ".join(turns[:-1] + [turns[-1], turns[-1]])


def rerank_query(question: str) -> str:
    turns = [text for role, text in parse_dialog(question) if role == "Пользователь"]
    return " ".join(turns) if turns else question


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str  # "rerank", "fallback" или "related"


class SearchIndex:
    def __init__(self, docs: list[Document], chunks: list[Chunk]):
        self.docs = docs
        self.chunks = chunks
        self.bm25 = bm25s.BM25()
        self.bm25.index([tokenize(c.text) for c in chunks], show_progress=False)
        self.vectors = llm.embed([c.text for c in chunks])
        self._reranker_available = True
        self._reranker_lock = threading.Lock()

    def embed_queries(self, questions: list[str]) -> list[np.ndarray]:
        """Векторы всех запросов пачками, а не отдельным запросом на каждый вопрос."""
        return list(llm.embed([search_query(q) for q in questions]))

    def search(self, question: str, query_vector: np.ndarray) -> list[Hit]:
        bm25 = self.bm25.get_scores(tokenize(search_query(question)))
        vector = self.vectors @ query_vector

        candidates = interleave(top_indices(bm25, config.TOP_K), top_indices(vector, config.TOP_K))
        candidates = candidates[:config.MAX_RERANK_CANDIDATES]
        with self._reranker_lock:
            reranker_available = self._reranker_available
        if reranker_available:
            try:
                ranked = llm.rerank(
                    rerank_query(question),
                    [self.chunks[i].text for i in candidates],
                )
                hits = [
                    Hit(self.chunks[candidates[i]], score, "rerank")
                    for i, score in ranked
                ]
            except llm.LLMError as exc:
                with self._reranker_lock:
                    first_failure = self._reranker_available
                    self._reranker_available = False
                if first_failure:
                    print(
                        "Реранкер недоступен; отключаю его до конца прогона и "
                        f"использую BM25 + эмбеддинги: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                hits = [
                    Hit(self.chunks[i], 0.0, "fallback")
                    for i in candidates[: config.RERANKER_TOP_N]
                ]
        else:
            hits = [
                Hit(self.chunks[i], 0.0, "fallback")
                for i in candidates[: config.RERANKER_TOP_N]
            ]
        return hits + self.related_hits(hits, vector)

    def related_hits(self, hits: list[Hit], relevance: np.ndarray) -> list[Hit]:
        """Допсоглашения и письма, меняющие найденный документ, модель должна видеть вместе с ним."""
        found_docs = {h.chunk.doc_id for h in hits}
        extra: list[Hit] = []
        # Связи ищутся только от лучших результатов, чтобы не расширять контекст случайно.
        for hit in hits[:2]:
            for doc_id in self.docs[hit.chunk.doc_id].related:
                if doc_id in found_docs or len(extra) >= config.MAX_RELATED:
                    continue
                own = [c for c in self.chunks if c.doc_id == doc_id]
                best = max(own, key=lambda c: relevance[c.id])
                if relevance[best.id] <= 0:
                    continue
                extra.append(Hit(best, float(relevance[best.id]), "related"))
                found_docs.add(doc_id)
        return extra


def top_indices(scores: np.ndarray, k: int) -> list[int]:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in idx[np.argsort(-scores[idx])] if scores[i] > 0]


def interleave(*rankings: list[int]) -> list[int]:
    """1-й из первого списка, 1-й из второго, 2-й из первого… без повторов.
    При обрезке так не теряются лучшие кандидаты ни одного из поисков."""
    merged: dict[int, None] = {}
    for group in zip_longest(*rankings):
        merged.update(dict.fromkeys(i for i in group if i is not None))
    return list(merged)
