"""A flat dataset of subject transcripts — the model-free prompting input.

Serialization leans entirely on BaseSchema (``to_dict``/``from_dict`` reconstruct
the nested ``RiskTranscript`` list and its ``Disorder`` enum), so save is a thin
file_io wrapper and load is the inherited ``from_json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common import BaseSchema
from src.common.file_io import ensure_dir, save_json

from .risk_transcript import RiskTranscript


@dataclass
class TranscriptDataset(BaseSchema):
    """All subject transcripts resolved from a corpus, ready to be prompted."""

    dataset_id: str
    subjects: list[RiskTranscript] = field(default_factory=list)

    def save_as_json(self, path: Path | str) -> None:
        """Write the dataset to JSON, creating the parent directory."""
        path = Path(path)
        ensure_dir(path.parent)
        save_json(self.to_dict(), path)
