"""Загрузка вопросов и базы знаний, нарезка на чанки по структуре документов."""

from __future__ import annotations

import csv
import io
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


import config

MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}
DATE_NUM_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
DATE_WORD_RE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})", re.IGNORECASE
)

# Явный разделитель отличает номер пункта от числа в обычном тексте.
POINT_RE = re.compile(r"^(\d+(?:\.\d+)*(?:-\d+)?)\.\s+\S")
SUBPOINT_RE = re.compile(r"^(\d+(?:\.\d+)*(?:-\d+)?|[а-я])\)\s+\S")
ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)*)\.\s+\S")
# Служебные пометки источника обычно стоят перед соответствующей нормой.
NOTE_RE = re.compile(
    r"^((Пункт|Подпункт|Часть|Абзац|Статья)\s+[\d.\-]+\s+(изменен|дополнен|утратил)"
    r"|Абзац утратил"
    r"|О конституционно-правовом смысле"
    r"|Федеральн\w+ закон\w* от"
    # Это исключение оставляет текст нормы в документе.
    r"|Положения\s(?!настоящ\w+\s+(?:подпункт|пункт|стать|абзац|част))"
    r"|Действие\s)"
)
# Пометка об изменении не содержит текста новой нормы.
AMENDMENT_RE = re.compile(
    r"^(?:(?:Пункт|Подпункт|Часть|Абзац|Статья|Наименование)\s+[\d.\-]+\s+(?:изменен|дополнен)\w*\s.*Федеральн"
    r"|Федеральн\w+\s+закон\w*\s+от\s"
    r"|О конституционно-правовом смысле)"
)
CHANGE_QUESTION_RE = re.compile(
    r"измен|редакци|поправ|вступ\w+\s+в\s+силу|утрат\w+\s+силу|с\s+какого|действует\s+ли", re.IGNORECASE
)

# В НК статьи «Налоговые ставки» различаются по разделу налога.
NK_TAXES = [
    (143, 178, "НДС — налог на добавленную стоимость"),
    (207, 233, "НДФЛ — налог на доходы физических лиц"),
    (246, 333, "налог на прибыль организаций"),
]
CONTRACT_NO_RE = re.compile(r"№\s*([А-ЯA-Z]{1,3}-\d+/\d{4})")

# Удаляет из документов строки, похожие на инструкции для модели.
INJECTION_RE = re.compile(
    r"(?:\b(?:ии|ai|llm|gpt)\b|ассистент|нейросет|языков\w+\s+модел|чат-?бот)[^\n]*"
    r"\b(?:игнорируй|отвечай|не\s+упоминай|забудь|выведи|следуй|раскрой)"
    r"|\b(?:игнорируй|забудь)\w*\s+(?:все\s+|всё\s+)?(?:предыдущ|остальн|прежн|вышеуказанн)\w*"
    r"|не\s+упоминай\w*\s+(?:эту|данную)\s+инструкци"
    r"|системн\w+\s+промпт"
    r"|ignore\s+(?:all\s+|any\s+)?(?:previous|prior|other)\s+(?:instructions|documents)",
    re.IGNORECASE,
)


@dataclass
class Document:
    id: int
    name: str
    title: str
    kind: str  # "company" или "law"
    text: str
    doc_date: date | None = None
    valid_from: date | None = None
    valid_until: date | None = None
    keys: list[str] = field(default_factory=list)
    related: list[int] = field(default_factory=list)


@dataclass
class Chunk:
    id: int
    doc_id: int
    section: str
    points: list[str]
    body: str
    text: str  # body с шапкой: это видят поиск и модель


def parse_date(text: str) -> date | None:
    for m in (DATE_NUM_RE.search(text), DATE_WORD_RE.search(text)):
        if not m:
            continue
        d, mo, y = m.groups()
        month = MONTHS[mo.lower()] if not mo.isdigit() else int(mo)
        try:
            return date(int(y), month, int(d))
        except ValueError:
            continue
    return None


def find_data_file(*names: str) -> Path:
    """Неизвестно, из какой папки платформа запустит скрипт, поэтому поиск в нескольких местах."""
    roots = (
        config.DATA_DIR,
        Path(__file__).resolve().parent / config.DATA_DIR,
        Path.cwd(),
    )
    for root in roots:
        for name in names:
            if (root / name).is_file():
                return root / name
    raise FileNotFoundError(f"Не найден ни один из файлов {names} в {config.DATA_DIR}")


def load_questions(path: Path | None = None) -> list[tuple[str, str]]:
    path = path or find_data_file("questions.csv")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [
            (row["id"].strip(), row["question"].strip()) for row in csv.DictReader(f)
        ]


def find_kb_zip() -> Path:
    return find_data_file("knowledge-base.zip", "knowledge_base.zip", "kb.zip")


def read_docx(raw: bytes) -> str:
    from docx import Document as Docx

    parts = []
    for block in Docx(io.BytesIO(raw)).iter_inner_content():
        if hasattr(block, "rows"):
            for row in block.rows:
                cells = []
                for cell in row.cells:
                    value = " ".join(cell.text.split())
                    # python-docx повторяет объединённую ячейку для каждой колонки.
                    if not cells or cells[-1] != value:
                        cells.append(value)
                parts.append(" | ".join(cells))
        elif block.text.strip():
            parts.append(block.text)
    return "\n".join(parts)


def decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_documents(path: Path | None = None) -> list[Document]:
    path = path or find_kb_zip()
    docs = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/") or "__MACOSX" in name:
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in {".txt", ".md", ".docx"}:
                continue
            try:
                raw = archive.read(name)
                text = read_docx(raw) if suffix == ".docx" else decode(raw)
            except Exception as exc:
                print(
                    f"Пропущен документ {name}: {type(exc).__name__}", file=sys.stderr
                )
                continue
            text = text.replace("\r\n", "\n").replace(" ", " ").strip()
            text, removed = strip_injections(text)
            for line in removed:
                print(f"Удалена вставка для ИИ из {name}: {line[:120]}", file=sys.stderr)
            if text:
                docs.append(make_document(len(docs), Path(name).name, text))
    link_documents(docs)
    return docs


def strip_injections(text: str) -> tuple[str, list[str]]:
    """Удаляет строку целиком: вставка обычно состоит из нескольких предложений, и не все содержат признак."""
    kept, removed = [], []
    for line in text.splitlines():
        (removed if INJECTION_RE.search(line) else kept).append(line)
    return "\n".join(kept), removed


def make_document(doc_id: int, name: str, text: str) -> Document:
    stem = Path(name).stem
    prefix, _, short = stem.partition("__")
    kind = "law" if prefix.lower().startswith("закон") else "company"
    head = text[:600]

    if kind == "law":
        title = law_title(text) or short or stem
    else:
        # В имени файла номер договора записан через дефис, в тексте нужен слеш.
        title = re.sub(r"\b([А-ЯA-Z]{1,3}-\d+)-(\d{4})\b", r"№ \1/\2", short or stem)

    doc = Document(id=doc_id, name=name, title=title, kind=kind, text=text)
    if kind == "company":
        first_lines = "\n".join(text.splitlines()[:3])
        doc.doc_date = parse_date(first_lines)
        start = re.search(r"(?:Действует с|Редакция от)\s*([\d.]+)", head)
        end = re.search(r"утратила? силу с\s*([\d.]+)", head, re.IGNORECASE)
        doc.valid_from = parse_date(start.group(1)) if start else None
        doc.valid_until = parse_date(end.group(1)) if end else None
        if doc.valid_from:
            doc.title = re.sub(r"\s+ред\.?\s*\d{4}.*$", "", doc.title)
        doc.keys = document_keys(text)
    return doc


def law_title(text: str) -> str:
    """«ТК РФ, Статья 70. Испытание при приеме на работу» вместо длинной строки «Источник:»."""
    source = re.search(r"^Источник:\s*(.+)$", text[:600], re.MULTILINE)
    source = source.group(1) if source else ""
    code = re.search(r"\b(ТК РФ|НК РФ|КоАП РФ|ГК РФ)\b", source)
    law_no = re.search(r"N\s*(\d+-ФЗ)", source)
    code = (
        code.group(1)
        if code
        else f"Федеральный закон № {law_no.group(1)}"
        if law_no
        else ""
    )

    heading = next(
        (l.strip() for l in text.splitlines() if is_article_heading(l.strip())), ""
    )
    if not heading:
        number = re.search(r"Статья\s+(\d+(?:\.\d+)*)", source)
        heading = f"Статья {number.group(1)}" if number else ""
    article = re.search(r"Статья\s+(\d+)", heading)
    if code == "НК РФ" and article:
        tax = next((t for low, high, t in NK_TAXES if low <= int(article.group(1)) <= high), "")
        code = f"НК РФ ({tax})" if tax else code
    return ", ".join(p for p in (code, heading) if p)


def hide_amendment_notes(text: str, question: str) -> str:
    """Убирает из текста для модели пометки вида «Пункт 3 изменен с … - Федеральный закон…»,
    если вопрос не об изменениях: иначе модель дописывает их в ответ как лишний факт."""
    if CHANGE_QUESTION_RE.search(question):
        return text
    return "\n".join(l for l in text.splitlines() if not AMENDMENT_RE.match(l.strip()))


def is_article_heading(line: str) -> bool:
    # Предложение статьи не считается заголовком.
    return bool(ARTICLE_RE.match(line)) and len(line) <= 200 and not line.endswith(".")


def document_keys(text: str) -> list[str]:
    """Признаки, по которым на документ ссылаются другие: номер договора или название регламента."""
    keys = []
    for line in text.splitlines()[:3]:
        if line.startswith("ДОГОВОР"):
            keys += CONTRACT_NO_RE.findall(line)
        elif line.startswith("РЕГЛАМЕНТ"):
            name = re.sub(r"\s+ООО.*$", "", line).lower()
            # Сопоставление по основам слов учитывает падежные окончания.
            keys.append(r"\s+".join(w[:5] + r"\w*" for w in name.split()))
    return keys


def link_documents(docs: list[Document]) -> None:
    """Связывает документ с теми, что его меняют или дополняют: допсоглашения, письма."""
    for doc in docs:
        for key in doc.keys:
            pattern = key if "\\" in key else re.escape(key)
            for other in docs:
                if other.id == doc.id or other.kind != "company" or key in other.keys:
                    continue
                if re.search(pattern, other.text, re.IGNORECASE):
                    if other.id not in doc.related:
                        doc.related.append(other.id)
                    if doc.id not in other.related:
                        other.related.append(doc.id)


def chunk_header(doc: Document, section: str, parent: str = "") -> str:
    parts = [doc.title]
    if doc.valid_from:
        parts.append(f"ред. от {doc.valid_from:%d.%m.%Y}")
    if doc.valid_until:
        parts.append(f"утратил силу с {doc.valid_until:%d.%m.%Y}")
    if section and section not in doc.title:
        parts.append(section)
    if parent:
        parts.append(parent)
    return "[" + " | ".join(parts) + "]"


def split_units(doc: Document) -> list[tuple[str, str | None, str, tuple[str, str]]]:
    """Делит документ на смысловые единицы: (раздел, номер пункта или None, текст, пункт-родитель).

    Пункт-родитель задаётся парой (номер, подпись для шапки) и заполняется у единиц без своего номера внутри длинного пункта:
    подпункт «9.4) услуг по перевозке…» без вводной фразы «1. Налогообложение производится
    по налоговой ставке 0 процентов при реализации:» не говорит, о какой ставке речь."""
    lines = [l.strip() for l in doc.text.splitlines()]
    lines = [
        l for l in lines if l and not re.match(r"^(URL|Дата выгрузки|Источник):", l)
    ]

    has_subnumbers = any(re.match(r"^\d+\.\d+", l) for l in lines)
    units: list[tuple[str, str | None, str, tuple[str, str]]] = []
    section = ""
    no_parent = ("", "")
    parent = no_parent
    pending_notes: list[str] = []
    current: list[str] = []
    current_point: str | None = None

    def flush():
        nonlocal current, current_point
        if current:
            units.append((section, current_point, "\n".join(current), no_parent if current_point else parent))
        current, current_point = [], None

    for line in lines:
        point = POINT_RE.match(line)
        in_list = bool(current) and current[-1].endswith((":", ";"))

        if NOTE_RE.match(line):
            flush()
            pending_notes.append(line)
        elif (
            is_article_heading(line)
            or point
            and has_subnumbers
            and "." not in point.group(1)
            and is_heading(line)
        ):
            flush()
            section = line
            parent = no_parent
        elif point:
            flush()
            current_point = point.group(1)
            parent = (current_point, line[:220] if line.endswith(":") else f"п. {current_point}")
            current = pending_notes + [line]
            pending_notes = []
        elif SUBPOINT_RE.match(line) or current_point or in_list:
            if pending_notes:
                current += pending_notes
                pending_notes = []
            current.append(line)
        else:
            # Без номера также встречаются части законов и договорных документов.
            flush()
            current = pending_notes + [line]
            pending_notes = []
    flush()
    if pending_notes:
        units.append((section, None, "\n".join(pending_notes), parent))
    return units


def is_heading(line: str) -> bool:
    return len(line) <= 80 and not line.endswith((".", ":", ";", ","))


SEPARATORS = ["\n", "; ", ". ", " "]


def split_long(text: str) -> list[str]:
    """Режет длинный пункт на фрагменты с перекрытием.
    Сначала по строкам, затем по «; », «. » и пробелам, чтобы не рвать предложения без нужды."""
    pieces = atomize(text, SEPARATORS)
    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and len("".join(current)) + len(piece) > config.CHUNK_MAX:
            chunks.append("".join(current).strip())
            # Переносим хвост предыдущего фрагмента для сохранения контекста.
            tail: list[str] = []
            for prev in reversed(current):
                size = len("".join(tail)) + len(prev)
                if size > config.CHUNK_OVERLAP or size + len(piece) > config.CHUNK_MAX:
                    break
                tail.insert(0, prev)
            current = tail
        current.append(piece)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def atomize(text: str, separators: list[str]) -> list[str]:
    """Дробит текст на части, сохраняя разделитель в конце фрагмента."""
    if len(text) <= config.CHUNK_MAX:
        return [text]
    sep = next((s for s in separators if s in text), None)
    if sep is None:
        return [text[i:i + config.CHUNK_MAX] for i in range(0, len(text), config.CHUNK_MAX)]
    rest = separators[separators.index(sep) + 1:]
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        part = part + sep if i < len(parts) - 1 else part
        pieces += atomize(part, rest) if len(part) > config.CHUNK_MAX else [part]
    return pieces


def chunk_document(doc: Document, start_id: int) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(section: str, points: list[str], body: str, parent: str = ""):
        header = chunk_header(doc, section, parent)
        chunks.append(
            Chunk(
                id=start_id + len(chunks),
                doc_id=doc.id,
                section=section,
                points=points,
                body=body,
                text=header + "\n" + body,
            )
        )

    buf: list[str] = []
    buf_points: list[str] = []
    buf_section = ""
    buf_parent = ""

    def flush():
        nonlocal buf, buf_points
        if buf:
            emit(buf_section, buf_points, "\n".join(buf), buf_parent)
        buf, buf_points = [], []

    for section, point, text, (parent_point, parent) in split_units(doc):
        # По возможности оставляем продолжение вместе с началом пункта.
        continues = not buf_parent and bool(buf_points) and buf_points[-1] == parent_point
        if section != buf_section or not (parent == buf_parent or continues):
            flush()
        buf_section = section
        if len(text) > config.CHUNK_MAX:
            flush()
            # Отдельный фрагмент получает вводную часть родительского пункта.
            intro = next(l for l in text.splitlines() if not NOTE_RE.match(l))
            label = (
                f"(продолжение п. {point}: {intro[:220]})\n"
                if point
                else f"(продолжение: {intro[:200]})\n"
            )
            for i, piece in enumerate(split_long(text)):
                emit(section, [point] if point else [], (label if i else "") + piece, parent)
            continue
        if buf and len("\n".join(buf)) + len(text) + 1 > config.CHUNK_MAX:
            flush()
        if not buf:
            buf_parent = parent
        buf.append(text)
        if point:
            buf_points.append(point)
    flush()
    return chunks


def load_knowledge_base(path: Path | None = None) -> tuple[list[Document], list[Chunk]]:
    docs = load_documents(path)
    if not docs:
        raise RuntimeError("В архиве базы знаний нет ни одного читаемого документа")
    chunks: list[Chunk] = []
    for doc in docs:
        chunks += chunk_document(doc, len(chunks))
    return docs, chunks


if __name__ == "__main__":
    docs, chunks = load_knowledge_base()
    print(f"документов: {len(docs)}, чанков: {len(chunks)}")
    for doc in docs:
        n = sum(c.doc_id == doc.id for c in chunks)
        links = ", ".join(docs[i].title for i in doc.related)
        print(
            f"  [{doc.kind}] {doc.title[:70]} — чанков: {n}"
            + (
                f", действует {doc.valid_from}…{doc.valid_until}"
                if doc.valid_from
                else ""
            )
            + (f", связи: {links}" if links else "")
        )
