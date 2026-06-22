# Characterizing Prompt Scaffolding for Medical Diagnosis in Spanish-Prompted LLMs

**Purpose.** Measure the impact of *scaffolds* — short natural-language preambles
prepended to a prompt — on **medical / mental-health diagnoses made by
Spanish-prompted LLMs**. Using the MentalRiskES corpus, models are asked the real
clinical detection tasks (depression, anxiety, eating disorders, suicidal
ideation, gambling risk, addiction type) in Spanish, **with and without** each
scaffold, in order to:

- **measure** how much each scaffold shifts diagnostic accuracy, per task;
- **characterize** which scaffolds work best — and where they help vs. hurt;
- **investigate** whether the same effect can be reached by other means
  (activation **steering**, **fine-tuning**, …) instead of prompt-time scaffolding.

Concretely, the model is prompted for the **actual MentalRiskES task** each
subject was annotated for (suffer/control, disorder type, context, suicidal
ideation, gambling risk level, addiction type) — in Spanish — and we measure how a
**scaffold** (a preamble placed at the top of the prompt) changes its answers, per
task.

Pipeline: **load data → prompt the model per task (with/without each scaffold) →
plot the effect.**

```
scripts/generate_prompt_datasets.py     load ALL data  -> out/prompt_dataset.json   (transcripts + gold)
scripts/generate_response_datasets.py   ask the model  -> out/<model>/responses.json (one row per subject×task×scaffold)
scripts/visualize_scaffold_effect.py    plot accuracy  -> out/<model>/scaffold_effect/<task>.png
```

## 1. Setup

Requires [`uv`](https://docs.astral.sh/uv/). Create the env and install deps:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch transformers accelerate numpy scipy pyzipper matplotlib
uv pip install mlx mlx_lm        # only for MLX (mlx-community/*) checkpoints on Apple Silicon
```

Gated HuggingFace models (e.g. `meta-llama/Llama-3.2-1B-Instruct`) need access
granted on their model page plus `hf auth login`.

## 2. Get the dataset (request access + place it in `datasets/`)

The corpus is **MentalRiskES** ([sinai-uja/corpusMentalRiskES](https://github.com/sinai-uja/corpusMentalRiskES),
Zenodo DOI `10.5281/zenodo.8055604`). It is **access-restricted** (mental-health
data): request access from the organizers via the
[MentalRiskES site](https://sites.google.com/view/mentalriskes) / the GitHub repo,
agree to the usage terms, and download the corpus.

Unzip it under `datasets/` so the root is `datasets/corpusMentalRiskES/`, holding
the base corpus **and** the three IberLEF editions:

```
datasets/corpusMentalRiskES/
├── corpusMentalRiskES/      # base corpus, by disorder:  processed|raw / {Anxiety,Depress,ED} / {data,gold}
├── mentalriskes2023/        # IberLEF 2023 editions:     <task>/{train,trial,test}/{data,gold}
├── mentalriskes2024/        # IberLEF 2024
└── mentalriskes2025/        # IberLEF 2025 (gambling / addiction type)
```

This folder is **git-ignored** — the data is never committed; you re-download it
locally per the access terms.

## 3. Run a simple scaffold comparison

```bash
# (a) Build the prompt dataset once — every subject, all sources (~5k transcripts).
uv run python scripts/generate_prompt_datasets.py

# (b) Ask a model each task WITH and WITHOUT a scaffold. --test = one random
#     subject per task (a fast smoke); drop it / use --subsample for real numbers.
uv run python scripts/generate_response_datasets.py \
    --model google/gemma-4-E2B-it \
    --scaffolding "Piensa paso a paso y con cuidado antes de responder." \
    --test

# (c) Plot accuracy per task: baseline (none) vs each scaffold.
uv run python scripts/visualize_scaffold_effect.py out/gemma-4-E2B-it/test_responses.json
```

`--scaffolding` accepts inline text, **or** a path to a `.json` (a list, or a
`{name: text}` map of multiple scaffolds) / `.txt` file — every task is then run
without scaffolding and once per scaffold.

## 4. Where the data is

| What | Path |
|------|------|
| Input corpus (local, git-ignored) | `datasets/corpusMentalRiskES/` |
| Prompt dataset (transcripts + gold) | `out/prompt_dataset.json` |
| Model responses (per subject × task × scaffold) | `out/<model>/responses.json` (or `test_responses.json` for `--test`) |
| Per-task scaffold-effect plots | `out/<model>/scaffold_effect/<task>.png` |

`<model>` is the bare checkpoint name, e.g. `out/gemma-4-E2B-it/`. The whole
`out/` directory is git-ignored.
