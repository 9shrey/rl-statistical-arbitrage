"""Generate a tiny, deterministic statistical-arbitrage backtest artifact set.

The example is intentionally dependency-free so a reviewer can run it in a
fresh checkout before installing the full RL stack. It does not train PPO; it
creates a small synthetic pair, compares an adaptive policy proxy against two
baselines, and writes the artifacts referenced by the README.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Day:
    index: int
    spread: float
    zscore: float
    regime: str


def synthetic_pair(n_days: int = 120) -> list[Day]:
    rows: list[Day] = []
    trailing: list[float] = []
    for i in range(n_days):
        seasonal = math.sin(i / 5.0) * 0.7
        shock = math.sin(i / 17.0) * 0.35
        spread = seasonal + shock
        trailing.append(spread)
        window = trailing[max(0, len(trailing) - 20) :]
        mu = mean(window)
        sigma = pstdev(window) or 1.0
        zscore = (spread - mu) / sigma
        regime = "mean_reverting" if abs(math.sin(i / 31.0)) > 0.35 else "trending"
        rows.append(Day(i, spread, zscore, regime))
    return rows


def position_for(day: Day, policy: str) -> int:
    if policy == "buy_hold_pair":
        return 1
    threshold = 1.0
    if policy == "adaptive_rl_proxy" and day.regime == "trending":
        threshold = 1.35
    if day.zscore > threshold:
        return -1
    if day.zscore < -threshold:
        return 1
    return 0


def run_policy(rows: list[Day], policy: str, cost_bps: float = 5.0) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    equity = 1.0
    last_position = 0
    for idx, day in enumerate(rows):
        position = position_for(day, policy)
        prev_spread = rows[idx - 1].spread if idx else day.spread
        spread_return = -(day.spread - prev_spread) * 0.012
        turnover = abs(position - last_position)
        cost = turnover * cost_bps / 10000.0
        pnl = position * spread_return - cost
        equity *= 1.0 + pnl
        out.append(
            {
                "day": day.index,
                "policy": policy,
                "regime": day.regime,
                "zscore": round(day.zscore, 4),
                "position": position,
                "daily_return": round(pnl, 6),
                "equity": round(equity, 6),
            }
        )
        last_position = position
    return out


def metrics(series: list[dict[str, float | int | str]]) -> dict[str, float]:
    returns = [float(row["daily_return"]) for row in series]
    equity = [float(row["equity"]) for row in series]
    avg = mean(returns)
    vol = pstdev(returns) or 1e-12
    downside = [min(0.0, ret) for ret in returns]
    downside_vol = pstdev(downside) or 1e-12
    peak = equity[0]
    max_drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)
    total_return = equity[-1] - 1.0
    return {
        "total_return": round(total_return, 4),
        "sharpe": round(avg / vol * math.sqrt(252), 3),
        "sortino": round(avg / downside_vol * math.sqrt(252), 3),
        "max_drawdown": round(max_drawdown, 4),
        "calmar": round(total_return / abs(max_drawdown or 1e-12), 3),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def drawdown_rows(series: list[dict[str, float | int | str]]) -> list[dict[str, float | int]]:
    peak = 1.0
    rows: list[dict[str, float | int]] = []
    for row in series:
        equity = float(row["equity"])
        peak = max(peak, equity)
        rows.append({"day": int(row["day"]), "drawdown": round(equity / peak - 1.0, 6)})
    return rows


def write_svg(path: Path, title: str, rows: list[tuple[int, float]], color: str) -> None:
    width, height = 720, 300
    pad = 42
    xs = [x for x, _ in rows]
    ys = [y for _, y in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_y = max(max_y - min_y, 1e-9)
    points = []
    for x, y in rows:
        px = pad + (x - min_x) / (max_x - min_x) * (width - 2 * pad)
        py = height - pad - (y - min_y) / span_y * (height - 2 * pad)
        points.append(f"{px:.1f},{py:.1f}")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <rect width="100%" height="100%" fill="#0f172a"/>
  <text x="{pad}" y="28" fill="#e2e8f0" font-family="Arial" font-size="18">{title}</text>
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#475569"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#475569"/>
  <polyline fill="none" stroke="{color}" stroke-width="3" points="{' '.join(points)}"/>
  <text x="{pad}" y="{height-12}" fill="#94a3b8" font-family="Arial" font-size="12">day {min_x}</text>
  <text x="{width-pad-54}" y="{height-12}" fill="#94a3b8" font-family="Arial" font-size="12">day {max_x}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_markdown_table(path: Path, rows: list[dict[str, object]]) -> None:
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = synthetic_pair()
    policies = ["adaptive_rl_proxy", "zscore_baseline", "buy_hold_pair"]
    runs = {policy: run_policy(rows, policy) for policy in policies}
    write_csv(RESULTS / "equity_curve.csv", runs["adaptive_rl_proxy"])
    write_csv(RESULTS / "drawdown.csv", drawdown_rows(runs["adaptive_rl_proxy"]))

    metric_rows = [{"policy": policy, **metrics(series)} for policy, series in runs.items()]
    write_markdown_table(RESULTS / "sharpe_sortino_table.md", metric_rows)
    write_csv(RESULTS / "baseline_comparison.csv", metric_rows)
    (RESULTS / "metrics.json").write_text(json.dumps(metric_rows, indent=2) + "\n", encoding="utf-8")

    sensitivity = []
    for cost in [0, 2, 5, 10, 20]:
        result = metrics(run_policy(rows, "adaptive_rl_proxy", cost_bps=cost))
        sensitivity.append({"cost_bps": cost, **result})
    write_csv(RESULTS / "transaction_cost_sensitivity.csv", sensitivity)

    write_svg(
        RESULTS / "equity_curve.svg",
        "Tiny backtest equity curve",
        [(int(row["day"]), float(row["equity"])) for row in runs["adaptive_rl_proxy"]],
        "#38bdf8",
    )
    write_svg(
        RESULTS / "drawdown.svg",
        "Tiny backtest drawdown",
        [(int(row["day"]), float(row["drawdown"])) for row in drawdown_rows(runs["adaptive_rl_proxy"])],
        "#fb7185",
    )

    summary = [
        "# Tiny Backtest Artifact Summary",
        "",
        "Generated by `python examples/tiny_backtest.py`.",
        "",
        "- Synthetic pair length: 120 daily bars",
        "- Policies: adaptive RL proxy, fixed z-score baseline, buy-and-hold pair",
        "- Transaction cost sensitivity: 0, 2, 5, 10, 20 bps per turnover unit",
        "- Outputs: equity curve, drawdown, metrics, baseline comparison, cost sensitivity",
        "",
        "This is a smoke-sized evidence fixture, not a production backtest.",
    ]
    (RESULTS / "run_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
