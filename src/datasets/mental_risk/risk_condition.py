"""The mental-health conditions spanned by ALL MentalRiskES data, with labels.

The base corpus covers anxiety / depression / eating disorders; the IberLEF
competition editions add suicidal ideation (2024 task3), gambling (2025 task1)
and addiction (2024 task2 / 2025 task2). Each condition carries an English and
Spanish phrase so the risk question ("¿en riesgo de {condition}?") names it
correctly per subject, whatever source the transcript came from.
"""

from __future__ import annotations

# condition key -> (English phrase, Spanish phrase) for the at-risk question slot.
_CONDITION_LABELS: dict[str, tuple[str, str]] = {
    "anxiety": ("anxiety", "ansiedad"),
    "depression": ("depression", "depresión"),
    "eating_disorder": ("an eating disorder", "un trastorno alimentario"),
    "depression_or_anxiety": ("depression or anxiety", "depresión o ansiedad"),
    "suicide": ("suicidal ideation", "ideación suicida"),
    "gambling": ("a gambling disorder", "ludopatía"),
    "addiction": ("an addiction", "una adicción"),
    "unknown": ("a mental health problem", "un problema de salud mental"),
}


def condition_label(condition: str, lang: str) -> str:
    """The English/Spanish phrase naming a condition (falls back to generic)."""
    en, es = _CONDITION_LABELS.get(condition, _CONDITION_LABELS["unknown"])
    return es if lang == "es" else en
