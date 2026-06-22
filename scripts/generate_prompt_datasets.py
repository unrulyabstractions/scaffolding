"""Build the prompt dataset from ALL MentalRiskES data: transcripts + gold only.

Run-by-path driver. Loads EVERY subject from the whole corpus root — the base
corpusMentalRiskES (by disorder) AND every IberLEF edition (2023/2024/2025, all
tasks and splits) — into one flat dataset of raw transcripts. NO questions or
framing are baked in; those are added (in Spanish, per the subject's own
condition) at response time.

Each entry carries: subject_id, source (provenance), condition (anxiety /
depression / eating_disorder / suicide / gambling / addiction / ...), the full
transcript, a derived gold_risk in [0, 1] where the source supports one, and the
raw gold columns (gold_labels, lossless). Subject ids repeat across sources, so
they are kept unique by `source` (see RiskTranscript.key).

The prompt dataset is ALWAYS the complete data — every subject from every source.
Subsampling for quick tests happens at response time (generate_response_datasets.py
--subsample), never here. Output:
  out/prompt_dataset.json

Usage:
  uv run python scripts/generate_prompt_datasets.py
  uv run python scripts/generate_prompt_datasets.py --corpus-root datasets/corpusMentalRiskES
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/scripts/x.py, parents[1] is the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common.file_io import ensure_dir  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.mental_risk import TranscriptDataset  # noqa: E402
from src.datasets.mental_risk.all_data_loader import load_all_transcripts  # noqa: E402

# The whole corpus root: holds corpusMentalRiskES/ and the mentalriskes<year>/ trees.
DEFAULT_CORPUS_ROOT = Path("datasets/corpusMentalRiskES")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for transcript-dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate the prompt dataset from ALL MentalRiskES data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
        help=f"Root holding corpusMentalRiskES/ + mentalriskes<year>/ (default: {DEFAULT_CORPUS_ROOT})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Output dir; the dataset lands at <out-dir>/prompt_dataset.json",
    )
    return parser.parse_args()


def main() -> None:
    """Load ALL transcripts and persist them (no questions)."""
    args = parse_args()
    log_header("GENERATE PROMPT DATASET (ALL MentalRiskES data, transcripts only)")

    with P("load_all_transcripts"):
        transcripts = load_all_transcripts(args.corpus_root)

    dataset = TranscriptDataset(dataset_id="", subjects=transcripts)
    dataset.dataset_id = dataset.get_id()

    log_section("prompt_dataset")
    n = len(transcripts)
    n_gold = sum(1 for t in transcripts if t.gold_risk is not None)
    log(f"  subjects:     {n}")
    log(f"  with gold:    {n_gold}")
    log(f"  by condition: {dict(Counter(t.condition for t in transcripts))}")
    log(f"  by source:    {dict(Counter(t.source.split('/')[0] for t in transcripts))}")

    path = ensure_dir(args.out_dir) / "prompt_dataset.json"
    dataset.save_as_json(path)
    log(f"[generate] wrote {path}")


if __name__ == "__main__":
    main()
