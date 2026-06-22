"""Plot how each scaffold changes accuracy, per task (baseline 'none' vs each).

Reads a responses dataset produced by generate_response_datasets.py --scaffolding
(each response carries its `scaffold` and gold), scores every answer against its
gold, and writes ONE bar chart per task comparing accuracy WITHOUT scaffolding
(`none`) against WITH each scaffold.

Usage:
  uv run python scripts/visualize_scaffold_effect.py out/<model>/responses.json
  uv run python scripts/visualize_scaffold_effect.py out/<model>/test_responses.json --out-dir figs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# Spanish answer option -> gold value, per task (gold is English/numeric on disk).
_DISORDER = {"depresión": "depression", "ansiedad": "anxiety", "ninguno": "none"}
_ATTITUDE = {
    "a favor del trastorno": "suffer+in favour",
    "en contra del trastorno": "suffer+against",
    "otra": "suffer+other",
    "no sufre (control)": "control",
}
_TYPE = {"apuestas": "betting", "juego en línea": "onlinegaming", "trading": "trading", "lootboxes": "lootboxes"}
_CONTEXT = {
    "adicción": "addiction", "emergencia": "emergency", "familia": "family",
    "trabajo": "work", "social": "social", "otro": "other",
}
_CONTEXT_COLS = ("addiction", "emergency", "family", "work", "social", "other")


def is_correct(r: dict) -> bool | None:
    """Whether one response matches its gold (None == no gold to score against)."""
    parsed, gl, task, kind = r["parsed"], r["gold_labels"], r["task_id"], r["kind"]
    if kind == "binary":
        if r["gold_risk"] is None:
            return None
        if not parsed:
            return False
        return (1 if "sí" in parsed else 0) == (1 if r["gold_risk"] >= 0.5 else 0)
    if task == "context":  # multi-label: exact set match
        pred = {_CONTEXT.get(p) for p in parsed}
        gold = {c for c in _CONTEXT_COLS if gl.get(c, 0) == 1}
        return pred == gold
    if not parsed:
        return False
    p = parsed[0]
    if task == "disorder_type":
        return _DISORDER.get(p) == str(gl.get("label", "")).lower()
    if task == "depression_attitude":
        return _ATTITUDE.get(p) == str(gl.get("category", "")).lower()
    if task == "gambling_risk_level":
        return (1 if p == "alto riesgo" else 0) == int(gl.get("risk_level", 0))
    if task == "addiction_type":
        return _TYPE.get(p) == str(gl.get("type", "")).lower()
    return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot scaffold effect on accuracy, per task")
    parser.add_argument("responses", type=Path, help="Path to a (test_)responses.json")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write the per-task PNGs (default: <responses dir>/scaffold_effect)",
    )
    return parser.parse_args()


def main() -> None:
    """Score responses by (task, scaffold) and write one accuracy bar chart per task."""
    args = parse_args()
    data = json.loads(args.responses.read_text(encoding="utf-8"))
    model = data.get("model", "model").split("/")[-1]  # bare name (fits the title)
    responses = data["responses"]

    scaffolds = sorted({r["scaffold"] for r in responses})
    if scaffolds == ["none"]:
        sys.exit("These responses have no scaffolds — re-run with --scaffolding to compare.")
    # 'none' (baseline) first, then the rest alphabetically.
    order = ["none"] + [s for s in scaffolds if s != "none"]

    # (task, scaffold) -> [correct bools], skipping no-gold (None).
    scored: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in responses:
        c = is_correct(r)
        if c is not None:
            scored[r["task_id"]][r["scaffold"]].append(c)

    # A distinct, consistent colour per scaffold (none stays grey baseline), so the
    # same scaffold reads the same across every per-task plot.
    palette = ["#2a7fff", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
    scaffold_color = {"none": "#888888"}
    for i, s in enumerate(s for s in order if s != "none"):
        scaffold_color[s] = palette[i % len(palette)]

    out_dir = args.out_dir or args.responses.parent / "scaffold_effect"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for task in sorted(scored):
        accs, ns, labels = [], [], []
        for s in order:
            vals = scored[task].get(s, [])
            labels.append(s)
            ns.append(len(vals))
            accs.append(sum(vals) / len(vals) if vals else 0.0)

        fig, ax = plt.subplots(figsize=(max(4.5, 1.8 + 1.2 * len(order)), 4.2))
        bars = ax.bar(labels, accs, color=[scaffold_color[s] for s in labels])
        for bar, acc, n in zip(bars, accs, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                    f"{acc:.2f}\nn={n}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("accuracy vs gold")
        ax.set_title(model, fontsize=9, color="#666")
        fig.suptitle(f"{task} — scaffold effect", fontsize=12, y=0.99)
        ax.axhline(accs[0], color="#888888", ls="--", lw=1)  # baseline reference line
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = out_dir / f"{task}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
        print(f"[viz] {task}: " + "  ".join(f"{s}={a:.2f}(n{n})" for s, a, n in zip(labels, accs, ns)))

    print(f"\n[viz] wrote {len(written)} per-task plots to {out_dir}")


if __name__ == "__main__":
    main()
