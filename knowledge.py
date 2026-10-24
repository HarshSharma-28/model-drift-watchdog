"""Knowledge base + agent memory for Model Drift Watchdog.

* **Knowledge base** — the Markdown documents in `knowledge_base/` (concepts,
  policy, runbooks, domain notes). They are indexed with TF-IDF (scikit-learn,
  no paid API) and retrieved by the agents during every run. Agents cite the
  documents they used, and the Reporter turns the matching runbook into the
  "recommended steps" of its report (retrieval-augmented reporting).
* **Agent memory** — every run can be appended to `data/memory.jsonl`; the
  Reporter recalls similar past runs ("we have seen this before").

Retrieval never changes a decision: decisions come from `policy.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).parent
KB_DIR = ROOT / "knowledge_base"
MEMORY_PATH = ROOT / "data" / "memory.jsonl"

CATEGORIES = ["concept", "policy", "runbook", "domain"]


@dataclass
class KBDoc:
    id: str
    title: str
    category: str
    tags: list[str]
    body: str
    path: str
    steps: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """First paragraph of the body (without the heading)."""
        text = re.sub(r"^# .*\n", "", self.body.strip())
        return text.strip().split("\n\n")[0].replace("\n", " ")


def _parse(path: Path) -> KBDoc:
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = m.group(2)
    steps = []
    s = re.search(r"## Steps\n(.*?)(\n## |\Z)", body, re.S)
    if s:
        steps = [re.sub(r"^\d+\.\s*", "", ln).strip() for ln in s.group(1).splitlines() if re.match(r"^\d+\.", ln.strip())]
    return KBDoc(
        id=meta.get("id", path.stem),
        title=meta.get("title", path.stem),
        category=meta.get("category", "other"),
        tags=[t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
        body=body.strip(),
        path=str(path.relative_to(ROOT)).replace("\\", "/"),
        steps=steps,
    )


class KnowledgeBase:
    """TF-IDF retriever over the Markdown knowledge base."""

    def __init__(self, kb_dir: Path = KB_DIR):
        self.docs = [_parse(p) for p in sorted(kb_dir.rglob("*.md")) if p.name.lower() != "readme.md"]
        self.by_id = {d.id: d for d in self.docs}
        # Title and tags are repeated so they weigh more than body text.
        corpus = [f"{d.title} {d.title} {' '.join(d.tags)} {' '.join(d.tags)} {d.body}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def __len__(self) -> int:
        return len(self.docs)

    def get(self, doc_id: str) -> KBDoc:
        return self.by_id[doc_id]

    def search(self, query: str, k: int = 5, category: str | None = None) -> list[tuple[KBDoc, float]]:
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        ranked = sorted(zip(self.docs, scores), key=lambda x: x[1], reverse=True)
        if category:
            ranked = [r for r in ranked if r[0].category == category]
        return [(d, float(s)) for d, s in ranked[:k] if s > 0]

    def lookup(self, query: str, category: str | None = None) -> tuple[KBDoc, float] | None:
        hits = self.search(query, k=1, category=category)
        return hits[0] if hits else None


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()


def cite(refs: list[dict], agent: str, query: str, reason: str, category: str | None = None) -> tuple[list[dict], str | None]:
    """Retrieve the best document for `query` and add it to the run's citations.

    Returns (updated citation list, cited doc id or None). Never mutates `refs`.
    """
    refs = list(refs or [])
    hit = get_kb().lookup(query, category)
    if not hit:
        return refs, None
    doc, score = hit
    if not any(r["id"] == doc.id and r["agent"] == agent for r in refs):
        refs.append({"id": doc.id, "title": doc.title, "category": doc.category, "agent": agent,
                     "reason": reason, "query": query, "score": round(score, 3)})
    return refs, doc.id


# ---------------------------------------------------------------------------
# Agent memory (run history)
# ---------------------------------------------------------------------------

def _mem(path: Path | None) -> Path:
    return path or MEMORY_PATH


def load_memory(path: Path | None = None) -> list[dict]:
    path = _mem(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def remember(record: dict, path: Path | None = None) -> None:
    path = _mem(path)
    record = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **record}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # read-only filesystem: memory is best-effort


def recall(decision: str, drifted: list[str], k: int = 3, path: Path | None = None) -> list[dict]:
    """Past runs most similar to this one: same decision first, then overlap of drifted features."""
    def score(m: dict) -> float:
        a, b = set(m.get("drifted", [])), set(drifted)
        overlap = len(a & b) / len(a | b) if (a | b) else 1.0
        return (1.0 if m.get("decision") == decision else 0.0) + overlap
    past = load_memory(path)
    ranked = sorted(past, key=score, reverse=True)
    return [m for m in ranked if score(m) >= 1.0][:k]


def clear_memory(path: Path | None = None) -> None:
    path = _mem(path)
    if path.exists():
        path.write_text("", encoding="utf-8")
