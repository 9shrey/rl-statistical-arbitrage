"""Markdown leaderboard generation."""
from __future__ import annotations

from pathlib import Path

from arb.backtest.engine import FoldResult, aggregate


def write_leaderboard(
    results_by_policy: dict[str, list[FoldResult]],
    out_path: str | Path,
) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Backtest Leaderboard\n")
    lines.append("Aggregated metrics across walk-forward folds (mean over folds).\n")
    lines.append("")
    headers = ["policy", "folds", "mean_total_return", "mean_sharpe", "mean_sortino",
               "mean_calmar", "mean_max_drawdown", "mean_hit_rate", "mean_turnover"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for name, results in results_by_policy.items():
        agg = aggregate(results)
        row = [name, str(agg.get("folds", 0))]
        for k in headers[2:]:
            v = agg.get(k, 0.0)
            row.append(f"{v:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
