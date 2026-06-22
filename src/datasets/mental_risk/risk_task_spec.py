"""The actual MentalRiskES task each subject was annotated for, with its question.

Each source defines a concrete classification task (taken verbatim from the
official IberLEF task pages), in Spanish. ``tasks_for`` maps one transcript to
the task(s) the model should be asked, so each subject is prompted appropriately:

  base corpus / 2023      suffer vs control of a named disorder (binary), and for
                          2023 depression also the 4-class attitude.
  2024 t1 / t2 / t3       3-class disorder {depression, anxiety, none}; +context
                          multi-label; binary suicidal ideation.
  2025 t1 / t2            gambling risk LEVEL (low/high — every user is at risk);
                          4-class addiction type (every user is at risk).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema

from .risk_transcript import RiskTranscript

# Answer kinds the parser understands.
BINARY = "binary"
SINGLE_CHOICE = "single_choice"
MULTI_LABEL = "multi_label"


@dataclass
class TaskSpec(BaseSchema):
    """One classification task: its Spanish question and answer space."""

    task_id: str
    kind: str  # BINARY | SINGLE_CHOICE | MULTI_LABEL
    question: str  # Spanish question
    options: list[str] = field(default_factory=list)  # Spanish answer options
    gold_keys: list[str] = field(default_factory=list)  # gold_labels cols it scores against


def _suffer(task_id: str, question: str, gold_keys: list[str]) -> TaskSpec:
    """A binary suffer(sí)/control(no) task."""
    return TaskSpec(task_id, BINARY, question, ["sí", "no"], gold_keys)


# 3-class disorder used by both 2024 tasks (depression / anxiety / none).
def _disorder_type(gold_keys: list[str]) -> TaskSpec:
    return TaskSpec(
        "disorder_type", SINGLE_CHOICE,
        "¿Qué trastorno presenta principalmente esta persona?",
        ["depresión", "ansiedad", "ninguno"], gold_keys,
    )


# Base corpus (LREC): binary suffer/control per disorder (gold bs/rbs).
_CONDITION_SUFFER = {
    "anxiety": _suffer("anxiety_suffer", "¿Esta persona sufre ansiedad?", ["bs", "rbs"]),
    "depression": _suffer("depression_suffer", "¿Esta persona sufre depresión?", ["bs", "rbs"]),
    "eating_disorder": _suffer("ed_suffer", "¿Esta persona sufre un trastorno alimentario?", ["bs", "rbs"]),
}

# IberLEF editions: (year, task) -> the task(s) to ask, per the official pages.
_EDITION_TASKS: dict[tuple[str, str], list[TaskSpec]] = {
    # 2023: suffer/control (gold_a binary, gold_b regression); task2 adds attitude.
    ("2023", "task1"): [
        _suffer("ed_suffer", "¿Esta persona sufre un trastorno alimentario (anorexia o bulimia)?", ["binary", "regression"]),
    ],
    ("2023", "task2"): [
        _suffer("depression_suffer", "¿Esta persona sufre depresión?", ["binary", "regression"]),
        TaskSpec(
            "depression_attitude", SINGLE_CHOICE,
            "Respecto a la depresión, ¿cuál es la actitud de la persona?",
            ["a favor del trastorno", "en contra del trastorno", "otra", "no sufre (control)"],
            ["category"],
        ),
    ],
    ("2023", "task3"): [
        _suffer("disorder_suffer", "¿Esta persona sufre algún trastorno mental?", ["binary", "regression"]),
    ],
    # 2024: 3-class disorder; task2 adds the context multi-label; task3 suicide.
    ("2024", "task1"): [_disorder_type(["label"])],
    ("2024", "task2"): [
        _disorder_type(["label"]),
        TaskSpec(
            "context", MULTI_LABEL,
            "Si la persona presenta un trastorno, ¿de qué contexto(s) parece provenir? (elige todos los que apliquen)",
            ["adicción", "emergencia", "familia", "trabajo", "social", "otro"],
            ["addiction", "emergency", "family", "work", "social", "other"],
        ),
    ],
    ("2024", "task3"): [
        _suffer("suicide_suffer", "¿Esta persona presenta ideación suicida?", ["label"]),
    ],
    # 2025: gambling risk LEVEL (every user is at risk — low vs high); 4-class type.
    ("2025", "task1"): [
        TaskSpec(
            "gambling_risk_level", SINGLE_CHOICE,
            "Esta persona tiene un problema de juego. ¿Está en alto o bajo riesgo de ludopatía?",
            ["alto riesgo", "bajo riesgo"], ["risk_level"],
        ),
    ],
    ("2025", "task2"): [
        TaskSpec(
            "addiction_type", SINGLE_CHOICE,
            "Esta persona tiene un problema de juego. ¿Qué tipo de adicción presenta?",
            ["apuestas", "juego en línea", "trading", "lootboxes"], ["type"],
        ),
    ],
}


def tasks_for(transcript: RiskTranscript) -> list[TaskSpec]:
    """The classification task(s) this subject should be prompted for."""
    src = transcript.source
    if src.startswith("corpus/"):
        spec = _CONDITION_SUFFER.get(transcript.condition)
        return [spec] if spec else []
    parts = src.split("/")
    if len(parts) >= 2:
        return _EDITION_TASKS.get((parts[0], parts[1]), [])
    return []
