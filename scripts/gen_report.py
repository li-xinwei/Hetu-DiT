#!/usr/bin/env python3
"""Render the mentor-facing eval report from an aggregated run set.

Inputs (under ``--root``):
  - per_run.csv  (columns described in scripts/aggregate_runs.py)
  - per_run.json (richer payload, same data)

Outputs (under ``--root/report/``):
  - report.md             — markdown summary with tables + plot refs + recommendation
  - plot_slo_vs_pool.png  — x=pool_size, y=slo_attainment, one curve per workload
  - plot_cost_vs_slo.png  — the decision plot: Pareto cost vs SLO

Matplotlib is the only external dep; if unavailable, plots are skipped and the
report still renders (text tables alone are sufficient for a first review).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(v: str, default: float = float("nan")) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def render_pool_sweep_table(rows: List[Dict[str, str]]) -> str:
    """Per-workload pool-size sweep: rows = (workload, pool_size)."""
    if not rows:
        return "_(no rows)_\n"
    lines = [
        "| Workload | Pool | Warm | N served | P50 (ms) | P95 (ms) | P99 (ms) | Warm-hit | SLO attainment | $/req |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda x: (x.get("workload_pattern", ""), _int(x.get("pool_size", "0")))):
        cost = _float(r.get("cost_per_request_usd", ""), float("nan"))
        cost_str = f"${cost:.3f}" if cost == cost else "-"
        lines.append(
            "| {} | {} | {} | {} | {:.0f} | {:.0f} | {:.0f} | {:.0%} | {:.0%} | {} |".format(
                r.get("workload_pattern", "?"),
                r.get("pool_size", "?"),
                r.get("warm_replicas", "?"),
                r.get("n_requests_served", "?"),
                _float(r.get("p50_ms", ""), 0.0),
                _float(r.get("p95_ms", ""), 0.0),
                _float(r.get("p99_ms", ""), 0.0),
                _float(r.get("warm_or_l2_hit_rate", "0"), 0.0),
                _float(r.get("slo_attainment", "0"), 0.0),
                cost_str,
            )
        )
    return "\n".join(lines) + "\n"


def render_hit_breakdown_table(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    lines = [
        "| Run | Ready | Warm | L2 | Cold | Unknown |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda x: x.get("run_id", "")):
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                r.get("run_id", "?"),
                r.get("ready_hits", "0"),
                r.get("warm_hits", "0"),
                r.get("l2_hits", "0"),
                r.get("cold_hits", "0"),
                r.get("unknown_hits", "0"),
            )
        )
    return "\n".join(lines) + "\n"


def plot_slo_vs_pool(rows: List[Dict[str, str]], out_path: Path) -> bool:
    if not _HAS_MPL:
        return False
    by_workload: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_workload[r.get("workload_pattern", "?")].append(r)
    fig, ax = plt.subplots(figsize=(7, 5))
    for workload, ws in by_workload.items():
        ws_sorted = sorted(ws, key=lambda x: _int(x.get("pool_size", "0")))
        xs = [_int(r.get("pool_size", "0")) for r in ws_sorted]
        ys = [_float(r.get("slo_attainment", "0"), 0.0) for r in ws_sorted]
        ax.plot(xs, ys, marker="o", label=workload)
    ax.set_xlabel("pool_size (head + warm_replicas)")
    ax.set_ylabel("SLO attainment")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("SLO attainment vs pool size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_cost_vs_slo(rows: List[Dict[str, str]], out_path: Path) -> bool:
    if not _HAS_MPL:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    seen_workloads = set()
    for r in rows:
        cost = _float(r.get("cost_per_request_usd", ""), float("nan"))
        slo = _float(r.get("slo_attainment", "0"), 0.0)
        workload = r.get("workload_pattern", "?")
        if cost != cost:  # skip NaN/inf
            continue
        marker = "o"
        ax.scatter(cost, slo, marker=marker, s=80,
                   label=workload if workload not in seen_workloads else None)
        seen_workloads.add(workload)
        label = f"warm={r.get('warm_replicas','?')}"
        ax.annotate(label, (cost, slo), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel("cost per request (USD)")
    ax.set_ylabel("SLO attainment")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Cost vs SLO (Pareto frontier — decision plot)")
    ax.grid(True, alpha=0.3)
    if seen_workloads:
        ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def render_recommendation(rows: List[Dict[str, str]]) -> str:
    """Heuristic recommendation block. Human should edit before sending."""
    by_workload: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_workload[r.get("workload_pattern", "?")].append(r)

    lines = ["### Recommendation (auto-generated, human-edit before sending)\n"]
    for workload, ws in sorted(by_workload.items()):
        # Find the Pareto-best cell: highest slo, then lowest cost as tiebreak.
        best = None
        for r in ws:
            slo = _float(r.get("slo_attainment", "0"), 0.0)
            cost = _float(r.get("cost_per_request_usd", "inf"), float("inf"))
            if best is None:
                best = (slo, -cost, r)
                continue
            cand = (slo, -cost, r)
            if cand > best:
                best = cand
        if best is None:
            continue
        slo, _, r = best
        lines.append(
            f"- **{workload}**: best Pareto point is warm_replicas="
            f"{r.get('warm_replicas','?')} (SLO {slo:.0%}, "
            f"${_float(r.get('cost_per_request_usd','0'), 0.0):.3f}/req)."
        )
    lines.append("")
    return "\n".join(lines)


def render_report(rows: List[Dict[str, str]], plots: Dict[str, bool]) -> str:
    summary = (
        f"Eval set summary: **{len(rows)}** runs aggregated.\n"
        f"Cost model assumes RunPod H100 SXM at $4/GPU-hour and 8 GPUs/pod (override via `--cost-rate-usd-per-gpu-hour` / `--gpu-per-pod` on the aggregator).\n"
        f"SLO definition: latency ≤ 2.5 × reference (TridentServe convention).\n"
    )
    plot_section = []
    if plots.get("slo_vs_pool"):
        plot_section.append("![SLO vs pool size](plot_slo_vs_pool.png)")
    if plots.get("cost_vs_slo"):
        plot_section.append("![Cost vs SLO (decision plot)](plot_cost_vs_slo.png)")
    plot_block = "\n\n".join(plot_section) if plot_section else "_matplotlib unavailable — no plots_"

    parts = [
        "# Warm-pool eval report",
        "",
        summary,
        "## Pool-size sweep",
        "",
        render_pool_sweep_table(rows),
        "## Plots",
        "",
        plot_block,
        "",
        "## Hit-type breakdown (per run)",
        "",
        render_hit_breakdown_table(rows),
        render_recommendation(rows),
    ]
    return "\n".join(parts)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="eval root containing per_run.csv + per_run.json")
    args = parser.parse_args(argv[1:])

    csv_path = args.root / "per_run.csv"
    if not csv_path.exists():
        print(f"missing {csv_path}; run aggregate_runs.py first", file=sys.stderr)
        return 1

    rows = load_rows(csv_path)
    report_dir = args.root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        "slo_vs_pool": plot_slo_vs_pool(rows, report_dir / "plot_slo_vs_pool.png"),
        "cost_vs_slo": plot_cost_vs_slo(rows, report_dir / "plot_cost_vs_slo.png"),
    }
    md = render_report(rows, plots)
    (report_dir / "report.md").write_text(md, encoding="utf-8")
    print(
        f"wrote {report_dir/'report.md'}; plots: "
        f"slo_vs_pool={'on' if plots['slo_vs_pool'] else 'OFF (no matplotlib)'} "
        f"cost_vs_slo={'on' if plots['cost_vs_slo'] else 'OFF (no matplotlib)'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
