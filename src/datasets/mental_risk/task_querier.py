"""Prompt a model for each subject's OWN task and parse its free-text answer.

This is the task-appropriate generation path: for every (subject, task) it renders
a Spanish prompt carrying the task's question + answer options, generates the
model's answer, strips the reasoning, and parses the choice(s) against the task
kind (binary / single-choice / multi-label). One model load drives the whole sweep.
"""

from __future__ import annotations

import re
import unicodedata

from src.common.logging import log
from src.common.text import strip_thinking_blocks
from src.inference.backends.model_backend import ModelBackend
from src.inference.model_runner import ModelRunner

from .risk_task_spec import BINARY, MULTI_LABEL, SINGLE_CHOICE, TaskSpec, tasks_for
from .risk_transcript import RiskTranscript
from .task_response import TaskResponse

_INSTR = "Lee la siguiente conversación de una persona y responde a la pregunta."
_OPEN, _CLOSE = "<transcripción>", "</transcripción>"


def render_prompt(transcript: RiskTranscript, spec: TaskSpec, scaffold: str = "") -> str:
    """The Spanish prompt for one subject's task: question + options + cue.

    ``scaffold`` is an optional preamble (e.g. a reasoning/debiasing nudge) placed
    at the VERY TOP, before everything else; empty == no scaffolding.
    """
    options = ", ".join(spec.options)
    if spec.kind == MULTI_LABEL:
        cue = (
            "Responde con TODAS las opciones que apliquen, separadas por comas; "
            "si ninguna aplica, responde «ninguno»."
        )
    else:
        cue = "Responde ÚNICAMENTE con una de las opciones."
    preamble = f"{scaffold.strip()}\n\n" if scaffold.strip() else ""
    return (
        f"{preamble}{_INSTR}\n\n{_OPEN}\n{transcript.transcript}\n{_CLOSE}\n\n"
        f"{spec.question}\nOpciones: {options}\n{cue}\nRespuesta:"
    )


# Chat special / EOS tokens to drop before parsing (e.g. "<|im_end|>", "</s>").
_SPECIAL = re.compile(r"<\|[^|]*\|>|</?s>")
# "none" said any common way (incl. the model's English fallback).
_NONE_RE = re.compile(r"\b(ningun[oa]|none|neither|nada|nothing)\b")


def _norm(s: str) -> str:
    """Lower-case, drop accents, and reduce punctuation to spaces for matching."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _answer_region(text: str) -> str:
    """The reasoning-stripped, special-token-free answer (after 'respuesta'), normalized.

    Normalizing (lower-case, strip accents, punctuation -> spaces) is what lets an
    option like "no sufre (control)" match — a raw \\b word-boundary regex fails on
    the parentheses.
    """
    t = _SPECIAL.sub(" ", strip_thinking_blocks(text))
    low = t.lower()
    if "respuesta" in low:
        t = t[low.rindex("respuesta") + len("respuesta"):]
    return _norm(t)


def parse_answer(text: str, spec: TaskSpec) -> list[str]:
    """Parsed option(s) from a free-text answer, per task kind (``[]`` if none)."""
    region = _answer_region(text)
    if spec.kind == BINARY:
        # accept Spanish and the model's occasional English fallback (accents stripped)
        yes = re.search(r"\b(si|yes|afirmativ\w*)\b", region)
        no = re.search(r"\b(no|not|negativ\w*)\b", region)
        if yes and not no:
            return ["sí"]
        if no and not yes:
            return ["no"]
        if yes and no:  # both stated — the later one is the verdict
            return ["sí"] if yes.start() > no.start() else ["no"]
        return []

    nopts = [(o, _norm(o)) for o in spec.options]
    # whole normalized option phrase appears in the answer
    hits = [o for o, n in nopts if n and re.search(rf"(?:^|\s){re.escape(n)}(?:\s|$)", region)]
    if spec.kind == MULTI_LABEL:
        return hits  # nothing matched (e.g. "ninguno") -> [] is the right answer

    # SINGLE_CHOICE robustness: accept an abbreviated answer (prefix either way),
    # then fall back to mapping a "none" word onto the none/control option.
    if not hits and region:
        hits = [o for o, n in nopts if n and (n.startswith(region) or region.startswith(n))]
    if not hits and _NONE_RE.search(region):
        none_opt = next(
            (o for o, n in nopts if any(w in n for w in ("ninguno", "ninguna", "control", "no sufre"))),
            None,
        )
        if none_opt:
            hits = [none_opt]
    hits.sort(key=lambda o: len(_norm(o)), reverse=True)  # most specific wins
    return hits[:1]


def run_task_responses(
    transcripts: list[RiskTranscript],
    model_name: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    scaffolds: dict[str, str] | None = None,
) -> list[TaskResponse]:
    """Render, generate and parse every (subject, task); one model load.

    ``scaffolds`` maps name -> preamble text. When given, each (subject, task) is
    run WITHOUT scaffolding (``scaffold="none"``) AND once with each scaffold.
    """
    # Always include the no-scaffold baseline; add each provided scaffold after it.
    variants = [("none", "")] + list((scaffolds or {}).items())

    # Pin the backend by name: MLX only for mlx-community checkpoints, HuggingFace
    # for everything else (the auto-recommended backend is MLX once mlx is
    # installed, which would wrongly route HF models through MLX).
    backend = ModelBackend.MLX if "mlx" in model_name.lower() else ModelBackend.HUGGINGFACE
    runner = ModelRunner(model_name=model_name, backend=backend)
    work = [(t, spec) for t in transcripts for spec in tasks_for(t)]
    total = len(work) * len(variants)
    log(
        f"[tasks] {total} generations: {len(work)} (subject, task) x "
        f"{len(variants)} scaffold variant(s) {[n for n, _ in variants]}"
    )

    out: list[TaskResponse] = []
    for transcript, spec in work:
        for scaffold_name, scaffold_text in variants:
            prompt = render_prompt(transcript, spec, scaffold_text)
            raw = runner.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )
            out.append(
                TaskResponse(
                    subject_id=transcript.subject_id,
                    source=transcript.source,
                    condition=transcript.condition,
                    task_id=spec.task_id,
                    kind=spec.kind,
                    scaffold=scaffold_name,
                    question=spec.question,
                    options=spec.options,
                    prompt_text=prompt,
                    response_text=strip_thinking_blocks(raw).strip(),
                    parsed=parse_answer(raw, spec),
                    gold_risk=transcript.gold_risk,
                    gold_labels=transcript.gold_labels,
                    _raw_text=raw,
                )
            )
            if len(out) % 25 == 0:
                log(f"[tasks]   {len(out)}/{total}")
    return out
