"""Load ALL MentalRiskES data — base corpus + every IberLEF edition — as transcripts.

One pass over the whole on-disk corpus root produces a flat list of
``RiskTranscript``, lossless and namespaced by source so the repeated ids across
sources never collide:

  corpus/<Disorder>          corpusMentalRiskES/processed/<Disorder>  (bs/rbs gold)
  <year>/<task>/<split>      mentalriskes<year>/..._<task>/<split>    (edition gold)

Each subject keeps every raw gold column (``gold_labels``, namespaced by gold file)
and a single derived ``gold_risk`` in [0, 1] where the source supports one:

  corpus       collapse_risk(bs/rbs)                 (via load_subjects)
  2023 t1/2/3  gold_b regression label  [0, 1]
  2024 t1/t2   condition label          ("none" -> 0.0, else 1.0)
  2024 t3      binary suicide label     {0, 1}
  2025 t1/t2   gambling/addiction Risk  {0, 1}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.common.logging import log

from .mental_risk_loader import load_subjects
from .risk_transcript import RiskTranscript

# corpus on-disk disorder dir -> condition key
_CORPUS_CONDITION = {"Anxiety": "anxiety", "Depress": "depression", "ED": "eating_disorder"}

# (year, task) -> condition key (from the IberLEF overviews)
_EDITION_CONDITION = {
    ("2023", "task1"): "eating_disorder",
    ("2023", "task2"): "depression",
    ("2023", "task3"): "anxiety",
    ("2024", "task1"): "depression_or_anxiety",
    ("2024", "task2"): "depression_or_anxiety",
    ("2024", "task3"): "suicide",
    ("2025", "task1"): "gambling",
    ("2025", "task2"): "addiction",
}

_SPLITS = ("train", "trial", "test")


def _transcript_text(records: list[dict]) -> str:
    """Join a subject's message records into a transcript, in streaming order.

    A few raw records carry a non-string ``message`` (NaN/number); skip those.
    """
    records = sorted(records, key=lambda m: (m.get("round", 0) or 0, m.get("id_message", 0) or 0))
    return "\n".join(
        m["message"] for m in records if isinstance(m.get("message"), str) and m["message"]
    )


def _iter_subjects(data_dir: Path, limit: int | None):
    """Yield (subject_id, records) for a split, handling BOTH on-disk layouts.

    train/trial are per-subject (subject101.json / user1036.json — one file is one
    subject's whole timeline). test is the streaming early-detection layout
    (round_1.json … round_N.json — each round file holds one message per subject,
    keyed by ``nick``), so a subject's transcript is gathered across all rounds.
    """
    files = sorted(data_dir.glob("*.json"))
    round_files = [f for f in files if f.name.startswith("round")]
    if round_files:
        by_nick: dict[str, list[dict]] = {}
        for f in round_files:
            for rec in json.loads(f.read_text(encoding="utf-8")):
                nick = rec.get("nick")
                if nick:
                    by_nick.setdefault(nick, []).append(rec)
        subjects = list(by_nick.items())
    else:
        subjects = [(f.stem, json.loads(f.read_text(encoding="utf-8"))) for f in files]
    if limit is not None:
        subjects = subjects[:limit]
    yield from subjects


# Edition gold files reuse the column name "label" across variants, so give each
# a meaningful key. The a/b/c/d convention is documented in the corpus README:
# a = hard binary, b = regression [0,1], c = text category, d = subtype distribution.
_GOLD_COL_RENAME = {
    ("gold_a", "label"): "binary",
    ("gold_b", "label"): "regression",
    ("gold_c", "label"): "category",
    # 2025 "Risk" is the gambling-risk LEVEL: 0 = low, 1 = high (per the official
    # task page) — every user is at risk, so name it risk_level, not at-risk.
    ("gold_task1", "Risk"): "risk_level",
    ("gold_task2", "Type"): "type",
}


def _clean_col(stem: str, col: str) -> str:
    """A meaningful gold key for one (file, column) — no opaque file prefixes.

    Multi-column files (gold_d's subtype distribution, 2024-t2's context factors)
    already have descriptive column names, so they pass through lower-cased.
    """
    return _GOLD_COL_RENAME.get((stem, col), col.lower())


def _read_gold_dir(gold_dir: Path) -> dict[str, dict]:
    """Merge every gold_*.txt in a split into {subject_id: {clean_col -> value}}."""
    out: dict[str, dict] = {}
    for gf in sorted(gold_dir.glob("*.txt")):
        text = gf.read_text(encoding="utf-8")
        first = text.split("\n", 1)[0]
        delim = "\t" if "\t" in first else ","
        rows = list(csv.reader(text.splitlines(), delimiter=delim))
        if not rows:
            continue
        header = rows[0]
        for row in rows[1:]:
            if not row or not row[0].strip():
                continue
            sid = row[0].strip()
            cols = out.setdefault(sid, {})
            for col, val in zip(header[1:], row[1:]):
                key = _clean_col(gf.stem, col)
                try:
                    cols[key] = float(val)
                except (TypeError, ValueError):
                    cols[key] = val
    return out


def _derive_risk(year: str, task: str, raw: dict) -> float | None:
    """Single risk in [0, 1] from one subject's raw gold, by source convention."""
    def num(key: str) -> float | None:
        try:
            return float(raw[key])
        except (KeyError, TypeError, ValueError):
            return None

    if year == "2023":
        return num("regression")  # gold_b: P(suffer) in [0, 1]
    if year == "2024":
        if task == "task3":
            return num("label")  # binary suicidal ideation (0/1)
        label = raw.get("label")  # disorder class: depression / anxiety / none
        if label is None:
            return None
        return 0.0 if str(label).strip().lower() == "none" else 1.0
    if year == "2025":
        # Official 2025 def: EVERY gambling user is at risk — label 0 = LOW risk,
        # 1 = HIGH risk (not at-risk/not). So both tasks: all at risk -> 1.0. The
        # low/high LEVEL (task1) and the TYPE (task2) are the task targets, in
        # gold_labels (risk_level / type).
        return 1.0 if raw else None
    return None


def _load_corpus(inner: Path, limit: int | None) -> list[RiskTranscript]:
    """Base corpusMentalRiskES (by disorder) via the existing subject loader."""
    if not (inner / "processed").is_dir():
        return []
    out: list[RiskTranscript] = []
    for subject in load_subjects(inner, source="processed", limit=limit):
        out.append(
            RiskTranscript(
                subject_id=subject.subject_id,
                source=f"corpus/{subject.disorder.value}",
                condition=_CORPUS_CONDITION.get(subject.disorder.value, "unknown"),
                transcript=subject.transcript,
                gold_risk=subject.risk,
                gold_labels=dict(subject.labels),
            )
        )
    return out


def _load_edition(year_dir: Path, year: str, limit: int | None) -> list[RiskTranscript]:
    """Every subject in one year's task/split tree, with derived + raw gold."""
    out: list[RiskTranscript] = []
    for task_dir in sorted(year_dir.glob(f"mentalriskes{year}_task*")):
        task = task_dir.name.split("_")[-1]
        condition = _EDITION_CONDITION.get((year, task), "unknown")
        for split in _SPLITS:
            data_dir = task_dir / split / "data"
            if not data_dir.is_dir():
                continue
            gold_dir = task_dir / split / "gold"
            gold = _read_gold_dir(gold_dir) if gold_dir.is_dir() else {}
            for sid, records in _iter_subjects(data_dir, limit):
                raw = gold.get(sid, {})
                out.append(
                    RiskTranscript(
                        subject_id=sid,
                        source=f"{year}/{task}/{split}",
                        condition=condition,
                        transcript=_transcript_text(records),
                        gold_risk=_derive_risk(year, task, raw),
                        gold_labels=raw,
                    )
                )
    return out


def load_all_transcripts(root: Path | str, limit: int | None = None) -> list[RiskTranscript]:
    """Load EVERY transcript from the corpus root: base corpus + all editions.

    ``root`` is the directory holding ``corpusMentalRiskES/`` and the
    ``mentalriskes<year>/`` edition trees. ``limit`` caps subjects per leaf source
    (per disorder / per task-split) for quick runs.
    """
    root = Path(root)
    transcripts = _load_corpus(root / "corpusMentalRiskES", limit)
    log(f"[all-data] corpus: {len(transcripts)} subjects")
    for year in ("2023", "2024", "2025"):
        year_dir = root / f"mentalriskes{year}"
        if not year_dir.is_dir():
            continue
        edition = _load_edition(year_dir, year, limit)
        log(f"[all-data] {year}: {len(edition)} subjects")
        transcripts.extend(edition)
    return transcripts
