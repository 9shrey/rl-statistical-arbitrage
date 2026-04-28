"""Top-level CLI: ``arb ...`` (Typer)."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from arb.config import load_config
from arb.utils.logging import get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True, help="RL Statistical Arbitrage CLI.")
log = get_logger("arb.cli")


def _pipeline(config_path: str) -> dict:
    """End-to-end: load -> data -> features -> walk-forward -> metrics dict."""
    from arb.backtest.engine import aggregate, run_walk_forward
    from arb.data.ingest import align_pair, load_pair
    from arb.signals.features import build_features
    from arb.utils.seeding import seed_everything

    cfg = load_config(config_path)
    seed_everything(cfg.seed)

    bars = load_pair(
        source=cfg.data.source,
        cache_dir=cfg.data.cache_dir,
        symbols=cfg.data.symbols,
        start=str(cfg.data.start),
        end=str(cfg.data.end),
        seed=cfg.seed,
    )
    pair = align_pair(bars)
    feats = build_features(pair, cfg.spread.hedge_window_days, cfg.spread.zscore_window_days)
    results = run_walk_forward(feats, cfg)
    agg = aggregate(results)
    return {
        "config": str(config_path),
        "pair": [cfg.data.symbols[0], cfg.data.symbols[1]],
        "n_bars": len(feats),
        "n_folds": len(results),
        "aggregate": agg,
        "per_fold": [
            {"fold_id": r.fold_id, "metrics": r.metrics} for r in results
        ],
    }


@app.command()
def smoke(config: str = typer.Option("configs/smoke.yaml", "--config", "-c")) -> None:
    """Run end-to-end smoke pipeline (data -> features -> walk-forward backtest)."""
    out = _pipeline(config)
    art_dir = Path(load_config(config).artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "smoke_metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info(
        "smoke ok | folds=%d | mean_sharpe=%.3f | mean_total_return=%.3f",
        out["n_folds"],
        out["aggregate"].get("mean_sharpe", 0.0),
        out["aggregate"].get("mean_total_return", 0.0),
    )
    typer.echo(json.dumps(out["aggregate"], indent=2))


@app.command()
def train(config: str = typer.Option("configs/default.yaml", "--config", "-c")) -> None:
    """Train policy and run backtest end-to-end (alias of smoke for v1)."""
    smoke(config=config)


@app.command()
def backtest(config: str = typer.Option("configs/default.yaml", "--config", "-c")) -> None:
    """Run walk-forward backtest with the configured policy."""
    smoke(config=config)


@app.command()
def report(config: str = typer.Option("configs/default.yaml", "--config", "-c")) -> None:
    """Generate a leaderboard comparing baselines vs configured policy."""
    from arb.backtest.engine import run_walk_forward
    from arb.config.schema import AppConfig
    from arb.data.ingest import align_pair, load_pair
    from arb.report.leaderboard import write_leaderboard
    from arb.signals.features import build_features
    from arb.utils.seeding import seed_everything

    cfg = load_config(config)
    seed_everything(cfg.seed)
    bars = load_pair(
        source=cfg.data.source,
        cache_dir=cfg.data.cache_dir,
        symbols=cfg.data.symbols,
        start=str(cfg.data.start),
        end=str(cfg.data.end),
        seed=cfg.seed,
    )
    pair = align_pair(bars)
    feats = build_features(pair, cfg.spread.hedge_window_days, cfg.spread.zscore_window_days)
    out: dict = {}
    for algo in ["zscore_baseline", "grid_baseline", "buy_hold"]:
        cfg_local = AppConfig(**{**cfg.model_dump(), "train": {**cfg.train.model_dump(), "algo": algo}})
        out[algo] = run_walk_forward(feats, cfg_local)
    art_dir = Path(cfg.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    p = write_leaderboard(out, art_dir / "leaderboard.md")
    typer.echo(f"Leaderboard written to {p}")


if __name__ == "__main__":  # pragma: no cover
    app()
