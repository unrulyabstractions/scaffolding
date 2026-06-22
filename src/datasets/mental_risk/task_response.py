"""One model response to one subject's task, plus the dataset that collects them.

A ``TaskResponse`` is self-describing: it records WHICH task was asked (id, kind,
question, options), the exact prompt, the model's raw answer, the parsed choice(s),
and the subject's gold (derived risk + raw labels) so it can be scored later.
``TaskResponseDataset`` is the single ``responses.json`` holding every task's
responses for one model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common import BaseSchema
from src.common.file_io import ensure_dir, save_json


@dataclass
class TaskResponse(BaseSchema):
    """The model's answer to one (subject, task)."""

    subject_id: str
    source: str
    condition: str
    task_id: str
    kind: str
    scaffold: str  # which scaffold preamble was used ("none" == without scaffolding)
    question: str
    options: list[str]
    prompt_text: str
    response_text: str  # model answer (reasoning stripped)
    parsed: list[str] = field(default_factory=list)  # parsed option(s); [] if unparsed
    gold_risk: float | None = None
    gold_labels: dict = field(default_factory=dict)
    # Heavy/private: the full raw generation is excluded from identity + to_dict.
    _raw_text: str = ""


@dataclass
class TaskResponseDataset(BaseSchema):
    """Every task response produced for one (prompt dataset, model)."""

    model: str
    prompt_dataset_id: str = ""
    responses: list[TaskResponse] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        """Bare model name, dropping any org prefix."""
        return self.model.split("/")[-1]

    def save_as_json(self, path: Path | str) -> None:
        """Write the dataset to JSON, creating the parent directory."""
        path = Path(path)
        ensure_dir(path.parent)
        save_json(self.to_dict(), path)
