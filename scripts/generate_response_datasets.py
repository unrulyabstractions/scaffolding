"""Prompt a model for every subject's OWN MentalRiskES task into one responses file.

Reads the model-free prompt dataset (out/prompt_dataset.json — transcripts + gold)
and, for each subject, asks the task it was actually annotated for, in Spanish,
naming its own condition: binary at-risk (anxiety / depression / eating disorder /
suicide / gambling), 3-class disorder (2024 t1), multi-label risk factors
(2024 t2), or addiction type (2025 t2). The model answers in free text; we strip
the reasoning and parse the choice(s) per task kind.

Every task's responses go into ONE file — each response carries its task_id / kind
/ options / parsed answer + the subject's gold, so you slice and score by task.

Input:   <out-dir>/prompt_dataset.json
Output:  <out-dir>/<MODEL>/responses.json

With --scaffolding, every task is run WITH and WITHOUT each scaffold preamble (a
reasoning/debiasing nudge); each response records which scaffold was used.

Usage:
  uv run python scripts/generate_response_datasets.py
  uv run python scripts/generate_response_datasets.py --model Qwen/Qwen3-0.6B
  uv run python scripts/generate_response_datasets.py --test      # 1 subject per task
  uv run python scripts/generate_response_datasets.py --test \
      --scaffolding "Piensa paso a paso antes de responder."
  uv run python scripts/generate_response_datasets.py --scaffolding scaffolds.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/scripts/x.py, parents[1] is the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.mental_risk import TranscriptDataset  # noqa: E402
from src.datasets.mental_risk.risk_task_spec import tasks_for  # noqa: E402
from src.datasets.mental_risk.task_querier import run_task_responses  # noqa: E402
from src.datasets.mental_risk.task_response import TaskResponseDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for response generation."""
    parser = argparse.ArgumentParser(
        description="Prompt a model for each subject's task into one responses.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Dir holding <out-dir>/prompt_dataset.json; output lands beside it",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B", help="HF model name (default: smallest Qwen)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Max new tokens per answer — large so the chain-of-thought finishes fully",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Decoding temperature (0 = greedy)"
    )
    parser.add_argument(
        "--subsample", type=float, default=1.0, help="Fraction of transcripts (0-1)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Smoke test: one RANDOM subject per task -> <model>/test_responses.json",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for --test random pick (default: random)"
    )
    parser.add_argument(
        "--scaffolding",
        default=None,
        help="Scaffold preamble(s): inline text, or a path to a .json (list or "
        "{name: text}) / .txt file. Runs each task WITH and WITHOUT each scaffold.",
    )
    return parser.parse_args()


def load_scaffolds(value: str) -> dict[str, str]:
    """Resolve --scaffolding into {name: preamble}: inline text or a json/txt file."""
    path = pathlib.Path(value)
    if path.exists():
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            if isinstance(data, list):
                return {f"s{i + 1}": str(s) for i, s in enumerate(data)}
            return {"scaffold": str(data)}
        return {path.stem: path.read_text(encoding="utf-8").strip()}
    return {"scaffold": value}


def _one_per_task(transcripts, seed: int | None = None):
    """Pick one RANDOM subject per distinct task (covers every task once)."""
    by_task: dict[str, list] = {}
    for t in transcripts:
        for spec in tasks_for(t):
            by_task.setdefault(spec.task_id, []).append(t)
    rng = random.Random(seed)
    picked = {tid: rng.choice(ts) for tid, ts in by_task.items()}
    seen, out = set(), []
    for t in picked.values():
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out, sorted(by_task)


def main() -> None:
    """Load transcripts, prompt each subject's task, write one responses.json."""
    args = parse_args()
    log_header(f"GENERATE RESPONSES ({args.model})")

    prompt_path = args.out_dir / "prompt_dataset.json"
    if not prompt_path.exists():
        raise SystemExit(
            f"No prompt dataset at {prompt_path} — run generate_prompt_datasets.py first"
        )

    transcript_ds = TranscriptDataset.from_json(prompt_path)
    transcripts = transcript_ds.subjects
    if args.test:
        transcripts, task_ids = _one_per_task(transcripts, seed=args.seed)
        log(f"[response] --test: {len(transcripts)} random subjects covering {len(task_ids)} tasks: {task_ids}")
    elif args.subsample < 1.0:
        stride = max(1, int(1 / args.subsample))
        transcripts = transcripts[::stride]
        log(f"[response] loaded {len(transcripts)} transcripts (subsample) from {prompt_path}")
    else:
        log(f"[response] loaded {len(transcripts)} transcripts from {prompt_path}")

    scaffolds = load_scaffolds(args.scaffolding) if args.scaffolding else None
    if scaffolds:
        log(f"[response] scaffolding: WITH and WITHOUT {list(scaffolds)}")

    with P("run_task_responses"):
        responses = run_task_responses(
            transcripts,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            scaffolds=scaffolds,
        )

    dataset = TaskResponseDataset(
        model=args.model,
        prompt_dataset_id=transcript_ds.dataset_id,
        responses=responses,
    )

    log_section("summary")
    parsed = sum(1 for r in responses if r.parsed)
    log(f"  responses: {len(responses)}  parsed: {parsed}")
    log(f"  by task:     {dict(Counter(r.task_id for r in responses))}")
    if scaffolds:
        log(f"  by scaffold: {dict(Counter(r.scaffold for r in responses))}")

    out_dir = args.out_dir / dataset.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("test_responses.json" if args.test else "responses.json")
    dataset.save_as_json(out_path)
    log(f"[response] wrote {out_path}")


if __name__ == "__main__":
    main()
