"""Классификатор вопросов: по делу, не по теме, атака. Эмбеддинги + логистическая регрессия."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from guard_data import EXAMPLES, LABELS

NORMAL, OFFTOPIC, ATTACK = 0, 1, 2


MIN_THRESHOLD = 0.6
MAX_THRESHOLD = 0.95
MARGIN = 0.05

REPLIES = {
    OFFTOPIC: (
        "Я отвечаю только на вопросы по договорам, внутренним документам компании "
        "и законодательству. С этим вопросом помочь не могу."
    ),
    ATTACK: (
        "Я юридический ассистент компании и работаю только по своим правилам: отвечаю на вопросы "
        "по договорам, регламентам и законодательству на основе базы знаний. "
        "Сформулируйте, пожалуйста, юридический вопрос."
    ),
}


def make_model() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )


def calibrate(X: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    """Порог подбирается под конкретные эмбеддинги: на кросс-валидации
    ни один нормальный пример не должен блокироваться."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    probs = cross_val_predict(make_model(), X, y, cv=folds, method="predict_proba")
    pred, conf = probs.argmax(1), probs.max(1)
    false_alarms = conf[(y == NORMAL) & (pred != NORMAL)]
    worst = float(false_alarms.max()) if len(false_alarms) else 0.0
    threshold = min(MAX_THRESHOLD, max(MIN_THRESHOLD, worst + MARGIN))
    blocked = (pred != NORMAL) & (conf >= threshold)
    report = {
        "accuracy": float((pred == y).mean()),
        "attacks_caught": f"{int((blocked & (y == ATTACK)).sum())}/{int((y == ATTACK).sum())}",
        "offtopic_caught": f"{int((blocked & (y == OFFTOPIC)).sum())}/{int((y == OFFTOPIC).sum())}",
        "normal_blocked": f"{int((blocked & (y == NORMAL)).sum())}/{int((y == NORMAL).sum())}",
    }
    return threshold, report


class Guard:
    def __init__(self, embed: Callable[[list[str]], np.ndarray]):
        X = embed([text for text, _ in EXAMPLES])
        y = np.array([label for _, label in EXAMPLES])
        self.threshold, self.report = calibrate(X, y)
        self.model = make_model().fit(X, y)

    def classify(self, vector: np.ndarray) -> tuple[int, float]:
        probs = self.model.predict_proba(vector.reshape(1, -1))[0]
        label = int(self.model.classes_[probs.argmax()])
        return label, float(probs.max())

    def verdict(self, vector: np.ndarray) -> tuple[str | None, str]:
        """Текст отказа или None, если вопрос пропускается; плюс описание для лога."""
        label, confidence = self.classify(vector)
        note = f"{LABELS[label]} {confidence:.2f}"
        if label != NORMAL and confidence >= self.threshold:
            return REPLIES[label], note
        return None, note


if __name__ == "__main__":
    # Оценка на настоящих эмбеддингах и прогон вопросов из DATA_DIR
    import llm
    from loader import load_questions

    guard = Guard(llm.embed)
    print(f"порог {guard.threshold:.2f}, кросс-валидация: {guard.report}")
    questions = load_questions()
    vectors = llm.embed([question for _, question in questions])
    for (qid, _), v in zip(questions, vectors, strict=True):
        reply, note = guard.verdict(v)
        print(f"[{qid}] {'БЛОК' if reply else 'ок'} {note}")
