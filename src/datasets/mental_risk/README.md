# `mental_risk` — load MentalRiskES data, prompt a model for each subject's task

The whole pipeline lives here: **load the data**, then **pass each subject to an
LLM for the task its source defines**. Nothing prompts the model except the task
the data specifies.

## Flow

```
ALL on-disk data ──load──▶ RiskTranscript[]  ──task per source──▶ prompt+generate+parse ──▶ responses.json
(corpus + editions)        (transcript+gold)   (TaskSpec)          (TaskResponse[])
```

## Files

**Load the data**
| File | What |
|------|------|
| `all_data_loader.py` | `load_all_transcripts(root)` — reads the base **corpusMentalRiskES** (by disorder) AND every IberLEF edition (2023/2024/2025, all tasks & splits, including the streaming round-based test splits) into one flat list, namespaced by `source`. |
| `mental_risk_loader.py` / `mental_risk_subject.py` / `mental_risk_message.py` / `mental_risk_disorder.py` / `mental_risk_gold.py` / `risk_label_collapse.py` | The base-corpus reader: subjects, messages, disorders, gold csv parsing, and collapsing gold columns to one risk. |
| `risk_transcript.py` | `RiskTranscript` — one subject's transcript + `source` + `condition` + derived `gold_risk` in [0,1] + lossless raw `gold_labels`. |
| `transcript_dataset.py` | `TranscriptDataset` — the saved `prompt_dataset.json` (transcripts only, no questions). |
| `risk_condition.py` | condition → Spanish/English phrase for the question slot. |

**Prompt the model for each task**
| File | What |
|------|------|
| `risk_task_spec.py` | `tasks_for(transcript)` → the actual MentalRiskES task(s) for that subject: binary at-risk, single-choice disorder/type, or multi-label risk factors. |
| `task_querier.py` | `run_task_responses(...)` — render the Spanish prompt for each (subject, task), generate, strip reasoning, parse the answer per kind. |
| `task_response.py` | `TaskResponse` / `TaskResponseDataset` — the saved `responses.json`. |

## Entry points

```
uv run python scripts/generate_prompt_datasets.py     # -> out/prompt_dataset.json   (all transcripts + gold)
uv run python scripts/generate_response_datasets.py   # -> out/<model>/responses.json (one file, every task)
```
