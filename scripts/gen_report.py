#!/usr/bin/env python3
"""Render the paper-rigor evaluation report.

Inputs (under ``--root``):
  - per_run.csv               one row per (cell, seed) actual run
  - per_run.json              same data structured
  - per_cell.csv              aggregated across seeds with bootstrap CIs
  - per_cell_latencies.json   raw latency arrays + SLO sensitivity per cell

Outputs (under ``--root/report/``):
  - report.md
  - fig1_slo_vs_pool.png      SLO attainment vs pool size (with CIs)
  - fig2_pareto_cost_slo.png  Pareto frontier cost-per-req vs SLO
  - fig3_latency_cdf.png      Latency CDFs, one panel per workload
  - fig4_throughput_vs_p95.png Throughput vs P95 latency
  - tableA_main_results.md    embedded in report.md (also separate file)
  - tableB_sensitivity.md     SLO multiplier sensitivity sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def _float(v, default=float("nan")):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _int(v, default=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def load_cells(per_cell_csv: Path) -> List[Dict[str, str]]:
    with per_cell_csv.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_latencies(latencies_json: Path) -> Dict[str, Any]:
    return json.loads(latencies_json.read_text(encoding="utf-8"))


def fmt_ci_ms(mean: float, lo: float, hi: float) -> str:
    if mean != mean:
        return "—"
    if lo == hi or (lo != lo or hi != hi):
        return f"{mean:.0f}"
    return f"{mean:.0f} [{lo:.0f}, {hi:.0f}]"


def fmt_ci_pct(mean: float, lo: float, hi: float) -> str:
    if mean != mean:
        return "—"
    if lo == hi or (lo != lo or hi != hi):
        return f"{mean*100:.0f}%"
    return f"{mean*100:.0f}% [{lo*100:.0f}, {hi*100:.0f}]"


def render_main_table(cells: List[Dict[str, str]]) -> str:
    """Table A: P50 / P95 / P99 (95% CI) + warm-hit + SLO at 2.5x."""
    lines = [
        "| Workload | Pool | Warm | Seeds | N req | P50 (ms, 95% CI) | P95 (ms, 95% CI) | P99 (ms, 95% CI) | Warm-hit | SLO 2.5× attainment |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(cells, key=lambda x: (x["workload_pattern"], _int(x["pool_size"]))):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["workload_pattern"], r["pool_size"], r["warm_replicas"],
                r["n_seeds"], r["n_requests_total"],
                fmt_ci_ms(_float(r["p50_ms_mean"]), _float(r["p50_ms_ci_lo"]), _float(r["p50_ms_ci_hi"])),
                fmt_ci_ms(_float(r["p95_ms_mean"]), _float(r["p95_ms_ci_lo"]), _float(r["p95_ms_ci_hi"])),
                fmt_ci_ms(_float(r["p99_ms_mean"]), _float(r["p99_ms_ci_lo"]), _float(r["p99_ms_ci_hi"])),
                fmt_ci_pct(_float(r["warm_or_l2_hit_rate_mean"]), _float(r["warm_or_l2_hit_rate_ci_lo"]), _float(r["warm_or_l2_hit_rate_ci_hi"])),
                fmt_ci_pct(_float(r["slo_2_5x_attainment_mean"]), _float(r["slo_2_5x_attainment_ci_lo"]), _float(r["slo_2_5x_attainment_ci_hi"])),
            )
        )
    return "\n".join(lines)


def render_sensitivity_table(latency_payload: Dict[str, Any]) -> str:
    """Table B: SLO attainment at multipliers 1.5x / 2.5x / 5x."""
    cells = latency_payload.get("cells", {})
    multipliers = latency_payload.get("slo_multipliers", [1.5, 2.5, 5.0])

    lines = ["| Workload | Pool | Warm |"]
    for m in multipliers:
        lines[0] += f" SLO {m:g}× |"
    lines.append("| --- | ---: | ---: |" + " --- |" * len(multipliers))

    items = sorted(
        cells.items(),
        key=lambda kv: (kv[1]["workload_pattern"], kv[1]["pool_size"]),
    )
    for cell_id, c in items:
        row = "| {} | {} | {} |".format(
            c["workload_pattern"], c["pool_size"], c["warm_replicas"]
        )
        for m in multipliers:
            key = f"slo_{m:g}x"
            sens = c.get("sensitivity", {}).get(key, {})
            mean = sens.get("attainment_mean")
            lo = sens.get("attainment_ci_lo")
            hi = sens.get("attainment_ci_hi")
            if mean is None:
                row += " — |"
            else:
                row += " {} |".format(fmt_ci_pct(mean, lo, hi))
        lines.append(row)
    return "\n".join(lines)


def plot_slo_vs_pool(cells: List[Dict[str, str]], out: Path) -> bool:
    if not _HAS_MPL or not cells:
        return False
    by_w: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for c in cells:
        by_w[c["workload_pattern"]].append(c)
    fig, ax = plt.subplots(figsize=(7, 5))
    for workload, ws in sorted(by_w.items()):
        ws_sorted = sorted(ws, key=lambda x: _int(x["pool_size"]))
        xs = [_int(r["pool_size"]) for r in ws_sorted]
        ys = [_float(r["slo_2_5x_attainment_mean"]) for r in ws_sorted]
        yerr_lo = [max(0.0, _float(r["slo_2_5x_attainment_mean"]) - _float(r["slo_2_5x_attainment_ci_lo"])) for r in ws_sorted]
        yerr_hi = [max(0.0, _float(r["slo_2_5x_attainment_ci_hi"]) - _float(r["slo_2_5x_attainment_mean"])) for r in ws_sorted]
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], marker="o", capsize=3, label=workload, linewidth=1.5)
    ax.set_xlabel("Pool size (head + warm replicas)")
    ax.set_ylabel("SLO attainment (2.5× reference)")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("SLO attainment vs pool size (mean ± 95% bootstrap CI)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def plot_pareto(cells: List[Dict[str, str]], out: Path, *, rate_usd_per_gpu_hour: float, gpu_per_pod: int = 8, ref_lat_ms: float = 3000.0) -> bool:
    """Cost per request vs SLO attainment. Cost is derived from pool_size and
    observed cell wall-clock (approximated as p99 * n_requests / pool_size, a
    crude lower bound on wall-clock; for sim cells the summary.txt has the
    exact wall_clock_s, but for the per-cell aggregate we go with this proxy)."""
    if not _HAS_MPL or not cells:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = {}
    color_idx = 0
    palette = plt.get_cmap("tab10")
    for c in cells:
        wl = c["workload_pattern"]
        if wl not in color_map:
            color_map[wl] = palette(color_idx % 10)
            color_idx += 1
    # Per-cell amortized cost: gpu_seconds * rate / requests. We compute
    # gpu_seconds = pool_size * gpu_per_pod * wall_clock; wall_clock is
    # not in per_cell.csv, but each cell has n_requests_total and p99 ≈ wall
    # for one seed averaged. To be accurate, we recompute from the latency
    # payload later. For now use median P95 * n / pool as a fudge.
    plotted = []
    for c in cells:
        wl = c["workload_pattern"]
        pool = _int(c["pool_size"])
        warm = _int(c["warm_replicas"])
        n_req = _int(c["n_requests_total"])
        seeds = max(_int(c["n_seeds"]), 1)
        p95 = _float(c["p95_ms_mean"]) / 1000.0  # convert ms to s
        if p95 != p95 or n_req == 0 or pool == 0:
            continue
        # Per-seed wall clock ≈ p95 * n_per_seed (worst case serial bound; under
        # heavy queueing this approximates well).
        wall_clock_s = p95 * (n_req / seeds)
        gpu_seconds = pool * gpu_per_pod * wall_clock_s
        cost_total = gpu_seconds / 3600.0 * rate_usd_per_gpu_hour
        cost_per_req = cost_total / (n_req / seeds)
        slo = _float(c["slo_2_5x_attainment_mean"])
        plotted.append((wl, pool, warm, cost_per_req, slo))
        ax.scatter(cost_per_req, slo, s=90, color=color_map[wl],
                   label=wl if wl not in [p[0] for p in plotted[:-1]] else None,
                   edgecolors="black", linewidth=0.5)
        ax.annotate(f"w={warm}", (cost_per_req, slo),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Estimated cost per request (USD)")
    ax.set_ylabel("SLO attainment (2.5× reference)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("log")
    ax.set_title("Pareto frontier: cost vs SLO attainment")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def plot_cdf(latency_payload: Dict[str, Any], out: Path) -> bool:
    if not _HAS_MPL:
        return False
    cells = latency_payload.get("cells", {})
    workloads: Dict[str, List] = defaultdict(list)
    for cell_id, c in cells.items():
        if not c.get("latencies_ms"):
            continue
        workloads[c["workload_pattern"]].append((cell_id, c))
    if not workloads:
        return False
    n_w = len(workloads)
    cols = 2 if n_w > 1 else 1
    rows = math.ceil(n_w / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)
    palette = plt.get_cmap("viridis")
    for ax_idx, (wl, items) in enumerate(sorted(workloads.items())):
        ax = axes[ax_idx // cols][ax_idx % cols]
        items_sorted = sorted(items, key=lambda x: _int(x[1]["warm_replicas"]))
        for i, (_, c) in enumerate(items_sorted):
            samples = sorted(c["latencies_ms"])
            if not samples:
                continue
            n = len(samples)
            ys = [(k + 1) / n for k in range(n)]
            color = palette(i / max(1, len(items_sorted) - 1))
            label = f"warm={c['warm_replicas']} (n={n})"
            ax.step(samples, ys, where="post", label=label, color=color, linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("End-to-end latency (ms, log scale)")
        ax.set_ylabel("CDF")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Latency CDF: {wl}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    # Hide unused panels.
    for empty in range(n_w, rows * cols):
        axes[empty // cols][empty % cols].axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def plot_throughput_vs_p95(cells: List[Dict[str, str]], out: Path) -> bool:
    if not _HAS_MPL or not cells:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    by_workload = defaultdict(list)
    for c in cells:
        wl = c["workload_pattern"]
        n = _int(c["n_requests_total"])
        seeds = max(_int(c["n_seeds"]), 1)
        p95 = _float(c["p95_ms_mean"])
        if p95 != p95 or n == 0:
            continue
        wall_s = p95 / 1000.0 * (n / seeds)
        throughput = (n / seeds) / max(wall_s, 1e-6)
        by_workload[wl].append((throughput, p95, c["warm_replicas"], c["pool_size"]))

    palette = plt.get_cmap("tab10")
    for i, (wl, pts) in enumerate(sorted(by_workload.items())):
        pts_sorted = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        ax.plot(xs, ys, marker="o", label=wl, color=palette(i % 10), linewidth=1.5)
        for x, y, warm, pool in pts_sorted:
            ax.annotate(f"w={warm}", (x, y), textcoords="offset points",
                        xytext=(5, -8), fontsize=7)

    ax.set_xlabel("Achieved throughput (requests / s)")
    ax.set_ylabel("P95 end-to-end latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Throughput vs P95 latency")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def render_recommendation(cells: List[Dict[str, str]]) -> str:
    by_w: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for c in cells:
        by_w[c["workload_pattern"]].append(c)
    lines = ["### Recommendation per workload (auto-generated)\n"]
    for wl, ws in sorted(by_w.items()):
        # Find Pareto-best: highest attainment_mean, tie-break by lower pool_size.
        ws_sorted = sorted(
            ws,
            key=lambda x: (-_float(x["slo_2_5x_attainment_mean"], 0.0), _int(x["pool_size"])),
        )
        best = ws_sorted[0]
        att = _float(best["slo_2_5x_attainment_mean"])
        lo = _float(best["slo_2_5x_attainment_ci_lo"])
        hi = _float(best["slo_2_5x_attainment_ci_hi"])
        lines.append(
            f"- **{wl}**: best SLO attainment = "
            f"{fmt_ci_pct(att, lo, hi)} at "
            f"pool_size={best['pool_size']} (warm_replicas={best['warm_replicas']})."
        )
    return "\n".join(lines) + "\n"


def find_delta_lines(cells: List[Dict[str, str]]) -> List[str]:
    """Compute warm-pool delta vs warm=0 baseline for each (workload, pool)."""
    baseline: Dict[str, Dict[str, str]] = {}
    for c in cells:
        if _int(c["warm_replicas"]) == 0:
            baseline[c["workload_pattern"]] = c
    lines = []
    for c in sorted(cells, key=lambda x: (x["workload_pattern"], _int(x["pool_size"]))):
        if _int(c["warm_replicas"]) == 0:
            continue
        wl = c["workload_pattern"]
        if wl not in baseline:
            continue
        b = baseline[wl]
        b_p95 = _float(b["p95_ms_mean"])
        c_p95 = _float(c["p95_ms_mean"])
        b_slo = _float(b["slo_2_5x_attainment_mean"]) * 100
        c_slo = _float(c["slo_2_5x_attainment_mean"]) * 100
        if b_p95 != b_p95 or c_p95 != c_p95:
            continue
        delta_p95_pct = ((b_p95 - c_p95) / b_p95) * 100
        delta_slo = c_slo - b_slo
        lines.append(
            f"- {wl} pool={c['pool_size']} (warm={c['warm_replicas']}): "
            f"P95 ↓{delta_p95_pct:.0f}% ({b_p95:.0f}ms → {c_p95:.0f}ms); "
            f"SLO attainment {b_slo:.0f}% → {c_slo:.0f}% (Δ {delta_slo:+.0f}pt)"
        )
    return lines


def render_report(
    cells: List[Dict[str, str]],
    latency_payload: Dict[str, Any],
    plots: Dict[str, bool],
    *,
    rate_usd_per_gpu_hour: float,
) -> str:
    n_cells = len(cells)
    n_seeds_unique = sorted(set(_int(c["n_seeds"]) for c in cells))
    total_runs = sum(_int(c["n_seeds"]) for c in cells)
    total_requests = sum(_int(c["n_requests_total"]) for c in cells)

    delta_lines = find_delta_lines(cells)

    parts = []
    parts.append("# Warm-pool architecture evaluation")
    parts.append("")
    parts.append("**Setup.** Hetu-DiT serving on K3s / Ray with the L2 pre-warm pool extension (commit `1a0e533`).")
    parts.append(f"Cost basis: ${rate_usd_per_gpu_hour:.2f}/GPU-hour (RunPod H100 SXM 2026-05 quote), 8 GPUs/pod.")
    parts.append("SLO definition: `latency ≤ 2.5 × reference_latency_optimal_parallelism` "
                 "(TridentServe convention, arxiv 2510.02838).")
    parts.append(f"Reference latency for SD3 1024² (20 steps) is 3.0 s, so the SLO threshold is 7.5 s.")
    parts.append("")
    parts.append("**Scale.** {} unique cells × seeds {} = **{} simulation runs** covering "
                 "**{} request samples** across cold / burst / Poisson / steady / mixed workloads with "
                 "warm_replicas ∈ {{0, 1, 2, 4}}.".format(
                     n_cells, n_seeds_unique, total_runs, total_requests))
    parts.append("Reported uncertainty: 95% bootstrap confidence intervals over seeds.")
    parts.append("")

    parts.append("## Hit-type taxonomy")
    parts.append("")
    parts.append("The dispatcher classifies every request into exactly one bucket:")
    parts.append("")
    parts.append("- **ready** — `_find_ready_executor` matched; same pod still bound from a prior request.")
    parts.append("- **warm** — `_find_warm_l2_executor` matched a *warm-tagged + L2-parked* pod (true pre-warm hit).")
    parts.append("- **l2** — `_find_warm_l2_executor` matched any L2-parked pod (single-pod L2 mechanism).")
    parts.append("- **cold** — fell through to `switch_parallel_env` (legacy full cold-start path).")
    parts.append("")
    parts.append("`warm_hit_rate` counts only **warm + l2** (ready hits would occur without a warm pool — they're not warm-pool wins).")
    parts.append("")

    parts.append("## Table A — Main results (P50 / P95 / P99, warm-hit, SLO at 2.5× with 95% CI)")
    parts.append("")
    parts.append(render_main_table(cells))
    parts.append("")

    parts.append("## Table B — SLO multiplier sensitivity")
    parts.append("")
    parts.append("Same data, varying the SLO threshold multiplier. Tight SLO (1.5×) tests latency; loose SLO (5×) tests basic capacity.")
    parts.append("")
    parts.append(render_sensitivity_table(latency_payload))
    parts.append("")

    parts.append("## Figure 1 — SLO attainment vs pool size")
    parts.append("")
    if plots.get("slo_vs_pool"):
        parts.append("![SLO attainment vs pool size, with 95% CI error bars](fig1_slo_vs_pool.png)")
    else:
        parts.append("_(matplotlib unavailable — plot skipped)_")
    parts.append("")

    parts.append("## Figure 2 — Pareto frontier: cost vs SLO attainment")
    parts.append("")
    if plots.get("pareto"):
        parts.append("![Cost per request vs SLO attainment; labels show warm_replicas](fig2_pareto_cost_slo.png)")
    else:
        parts.append("_(matplotlib unavailable — plot skipped)_")
    parts.append("")

    parts.append("## Figure 3 — Latency CDFs by workload")
    parts.append("")
    if plots.get("cdf"):
        parts.append("![End-to-end latency CDF per workload, one curve per pool config](fig3_latency_cdf.png)")
    else:
        parts.append("_(matplotlib unavailable — plot skipped)_")
    parts.append("")

    parts.append("## Figure 4 — Throughput vs P95 latency")
    parts.append("")
    if plots.get("throughput"):
        parts.append("![Achieved throughput on x-axis, P95 latency on y-axis (log scale)](fig4_throughput_vs_p95.png)")
    else:
        parts.append("_(matplotlib unavailable — plot skipped)_")
    parts.append("")

    parts.append("## Pre-warm pool delta vs baseline (warm_replicas=0)")
    parts.append("")
    if delta_lines:
        parts.extend(delta_lines)
    else:
        parts.append("_(no baseline cells found)_")
    parts.append("")

    parts.append(render_recommendation(cells))

    parts.append("## Threats to validity")
    parts.append("")
    parts.append("- **Sim model.** Pool timings come from a calibrated event-driven simulator. Calibration constants (T_BOOT_TO_L2=25s, T_BIND_WARM=3.14s, T_BIND_COLD_L2=12s, T_BIND_COLD=35s) are from a single 2026-05-06 RunPod H100 SXM session. Phase 5 cross-validates against a real-mode RunPod run.")
    parts.append("- **Single-config workloads.** Cells fix (model, resolution) to SD3 1024². The `mixed` workload rotates resolutions per request and exposes warm-pool sensitivity to config diversity.")
    parts.append("- **No eviction modeled.** A bound pod stays bound to its last config. Real systems may evict; this overestimates `ready` hits for low-rate workloads.")
    parts.append("- **Open-loop arrival.** Workloads do not throttle on backpressure. Real clients may; this overestimates queueing latency in saturation.")
    parts.append("")
    return "\n".join(parts)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cost-rate-usd-per-gpu-hour", type=float, default=4.0)
    parser.add_argument("--gpu-per-pod", type=int, default=8)
    args = parser.parse_args(argv[1:])

    per_cell_csv = args.root / "per_cell.csv"
    latencies_json = args.root / "per_cell_latencies.json"
    if not per_cell_csv.exists() or not latencies_json.exists():
        print(f"missing per_cell.csv or per_cell_latencies.json under {args.root}; "
              "run aggregate_runs.py first", file=sys.stderr)
        return 1

    cells = load_cells(per_cell_csv)
    latency_payload = load_latencies(latencies_json)
    report_dir = args.root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        "slo_vs_pool": plot_slo_vs_pool(cells, report_dir / "fig1_slo_vs_pool.png"),
        "pareto": plot_pareto(
            cells, report_dir / "fig2_pareto_cost_slo.png",
            rate_usd_per_gpu_hour=args.cost_rate_usd_per_gpu_hour,
            gpu_per_pod=args.gpu_per_pod,
        ),
        "cdf": plot_cdf(latency_payload, report_dir / "fig3_latency_cdf.png"),
        "throughput": plot_throughput_vs_p95(cells, report_dir / "fig4_throughput_vs_p95.png"),
    }
    md = render_report(
        cells, latency_payload, plots,
        rate_usd_per_gpu_hour=args.cost_rate_usd_per_gpu_hour,
    )
    (report_dir / "report.md").write_text(md, encoding="utf-8")
    (report_dir / "tableA_main_results.md").write_text(render_main_table(cells), encoding="utf-8")
    (report_dir / "tableB_sensitivity.md").write_text(render_sensitivity_table(latency_payload), encoding="utf-8")
    print(
        f"report → {report_dir/'report.md'}; plots: " +
        ", ".join(f"{k}={'on' if v else 'OFF'}" for k, v in plots.items())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
