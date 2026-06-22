"""A subject's transcript plus gold — the model-free prompting input, any source.

This is what feeds the prompt builder: the raw timeline text and the gold for one
subject, with NO question or framing attached (those are added at response time).
It is source-agnostic — the same record represents a subject from the base
corpusMentalRiskES or from any IberLEF edition/task/split — so ``source`` keeps
provenance and ``condition`` says what risk the subject was annotated for.

``gold_risk`` is a single derived risk in [0, 1] when the source supports one;
``gold_labels`` keeps every raw gold column verbatim (lossless), so subtype /
context-factor / type annotations are never dropped.

It deliberately exposes ``transcript`` / ``risk`` / ``condition_label`` so the
prompt renderer consumes it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema

from .risk_condition import condition_label


@dataclass
class RiskTranscript(BaseSchema):
    """One subject's transcript + gold, free of any question/framing."""

    subject_id: str  # raw id within its source (subject101 / user1036)
    source: str  # provenance, e.g. "corpus/Anxiety", "2023/task2/train"
    condition: str  # what risk this subject was annotated for (see risk_condition)
    transcript: str
    gold_risk: float | None = None  # derived risk in [0, 1] where the source supports it
    gold_labels: dict = field(default_factory=dict)  # raw gold columns, lossless

    @property
    def key(self) -> str:
        """Globally-unique key (ids repeat across sources): source + id."""
        return f"{self.source}/{self.subject_id}"

    @property
    def risk(self) -> float | None:
        """Gold risk under the name the prompt renderer expects."""
        return self.gold_risk

    def condition_label(self, lang: str) -> str:
        """English/Spanish phrase naming this subject's condition."""
        return condition_label(self.condition, lang)
