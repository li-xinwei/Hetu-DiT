#!/usr/bin/env python3
"""Walk a results root and emit per-run + aggregate CSV/JSON for the report.

Each run is a subdirectory containing at minimum ``cstrace.log``. Optionally
``summary.txt`` (key=value lines) provides metadata that's hard to recover
from logs alone (workload_pattern, pool_size, model, seed, n_requests_planned,
mode).

For each run we:
1. Invoke ``scripts/parse-coldstart-trace.py --json`` to get a structured event
   summary including the per-request list with hit_type classifications.
2. Compute latency percentiles, hit-rate breakdown, SLO attainment (using the
   reference latencies from ``data/optimal_latencies.json``), and a cost
   estimate via ``scripts/cost_model.py``.
3. Write one row per run to ``<root>/per_run.csv`` plus a richer JSON to
   ``<root>/per_run.json`` for the report generator to consume.

The summary.txt fields are best-effort — if missing, columns default to
sensible NaN / -1 so downstream consumers can flag the gap.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_paper_stats():
    spec = importlib.util.spec_from_file_location(
        "paper_stats", REPO_ROOT / "scripts" / "paper_stats.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paper_stats"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_cost_model():
    spec = importlib.util.spec_from_file_location(
        "cost_model", REPO_ROOT / "scripts" / "cost_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cost_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def _percentile(samples: List[float], pct: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    idx = int(round((pct / 100.0) * (len(s) - 1)))
    return s[idx]


def parse_summary_txt(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def load_optimal_latencies(path: Path) -> Dict[Tuple[str, int, int], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, int, int], float] = {}
    for k, v in data.get("latencies_ms", {}).items():
        model, h, w = k.split(",")
        out[(model, int(h), int(w))] = float(v)
    return out


def run_parser_json(parser_path: Path, cstrace_log: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        ["python3", str(parser_path), str(cstrace_log), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


@dataclass
class RunRow:
    run_id: str
    workload_pattern: str
    mode: str
    model: str
    pool_size: int
    head_size: int
    warm_replicas: int
    n_requests_planned: int
    n_requests_served: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    median_queue_ms: float
    ready_hits: int
    warm_hits: int
    l2_hits: int
    cold_hits: int
    unknown_hits: int
    warm_hit_rate: float        # warm / served
    warm_or_l2_hit_rate: float  # (warm + l2 + ready) / served
    cold_rate: float            # cold / served
    slo_target_ms: float
    slo_violations: int
    slo_attainment: float
    wall_clock_s: float
    gpu_seconds: float
    cost_usd: float
    cost_per_request_usd: float
    cstrace_path: str = ""


def aggregate_run(
    run_dir: Path,
    *,
    parser_path: Path,
    cost_model,
    optimal_latencies: Dict[Tuple[str, int, int], float],
    slo_multiplier: float,
    rate_usd_per_gpu_hour: float,
    gpu_per_pod: int,
) -> Optional[Tuple["RunRow", Dict[str, Any]]]:
    cstrace_log = run_dir / "cstrace.log"
    if not cstrace_log.exists():
        return None
    meta = parse_summary_txt(run_dir / "summary.txt")
    try:
        parsed = run_parser_json(parser_path, cstrace_log)
    except subprocess.CalledProcessError:
        return None

    requests = parsed.get("requests", [])
    latencies = [r["latency_ms"] for r in requests if "latency_ms" in r]
    queues = [r["queue_ms"] for r in requests if "queue_ms" in r]

    by_hit = {"ready": 0, "warm": 0, "l2": 0, "cold": 0, "unknown": 0}
    for r in requests:
        h = r.get("hit_type", "unknown")
        by_hit[h] = by_hit.get(h, 0) + 1

    served = len(requests)

    # Determine SLO target from default model/resolution; fall back to sd3-1024.
    model = meta.get("model", "sd3")
    height = int(meta.get("default_height", 1024) or 1024)
    width = int(meta.get("default_width", 1024) or 1024)
    try:
        slo_thresh = cost_model.slo_target_ms(
            model, height, width, optimal_latencies, multiplier=slo_multiplier
        )
    except KeyError:
        slo_thresh = float("inf")

    violations, total, attainment = cost_model.slo_attainment(latencies, slo_thresh)

    wall_clock_s = float(meta.get("wall_clock_s", 0.0) or 0.0)
    if wall_clock_s <= 0 and requests:
        first = min(r.get("start_ts", 0.0) for r in requests)
        last = max(r.get("done_ts", 0.0) for r in requests)
        wall_clock_s = max(0.0, last - first)

    head_size = int(meta.get("head_size", 1) or 1)
    warm_replicas = int(meta.get("warm_replicas", 0) or 0)
    pool_size = int(meta.get("pool_size", head_size + warm_replicas) or head_size + warm_replicas)
    if pool_size <= 0:
        pool_size = head_size + warm_replicas

    cost = cost_model.compute(
        cost_model.CostInputs(
            pool_size=pool_size,
            wall_clock_s=wall_clock_s,
            gpu_per_pod=gpu_per_pod,
            rate_usd_per_gpu_hour=rate_usd_per_gpu_hour,
            requests_served=served,
        )
    )

    row = RunRow(
        run_id=run_dir.name,
        workload_pattern=meta.get("workload_pattern", "unknown"),
        mode=meta.get("mode", "real"),
        model=model,
        pool_size=pool_size,
        head_size=head_size,
        warm_replicas=warm_replicas,
        n_requests_planned=int(meta.get("n_requests_planned", served) or served),
        n_requests_served=served,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        median_queue_ms=median(queues) if queues else float("nan"),
        ready_hits=by_hit.get("ready", 0),
        warm_hits=by_hit.get("warm", 0),
        l2_hits=by_hit.get("l2", 0),
        cold_hits=by_hit.get("cold", 0),
        unknown_hits=by_hit.get("unknown", 0),
        warm_hit_rate=(by_hit.get("warm", 0) / served) if served else 0.0,
        warm_or_l2_hit_rate=(
            (by_hit.get("warm", 0) + by_hit.get("l2", 0)) / served
            if served else 0.0
        ),
        cold_rate=(by_hit.get("cold", 0) / served) if served else 0.0,
        slo_target_ms=slo_thresh,
        slo_violations=violations,
        slo_attainment=attainment,
        wall_clock_s=wall_clock_s,
        gpu_seconds=cost.gpu_seconds,
        cost_usd=cost.cost_usd,
        cost_per_request_usd=cost.cost_per_request_usd,
        cstrace_path=str(cstrace_log.relative_to(REPO_ROOT)) if cstrace_log.is_relative_to(REPO_ROOT) else str(cstrace_log),
    )
    raw = {
        "run_id": run_dir.name,
        "workload_pattern": row.workload_pattern,
        "pool_size": row.pool_size,
        "warm_replicas": row.warm_replicas,
        "head_size": row.head_size,
        "model": row.model,
        "seed": int(meta.get("seed", -1) or -1),
        "latencies_ms": latencies,
        "queue_ms": queues,
        "hit_types": [r.get("hit_type", "unknown") for r in requests],
    }
    return row, raw


def aggregate_cells(
    raw_runs: List[Dict[str, Any]],
    *,
    ps,  # paper_stats module
    cost_model,
    optimal_latencies: Dict[Tuple[str, int, int], float],
    slo_multipliers: Sequence[float],
    rate_usd_per_gpu_hour: float,
    gpu_per_pod: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Group raw runs by (workload_pattern, pool_size) and compute per-cell stats.

    Returns:
      - per_cell_rows: list of dicts suitable for per_cell.csv
      - per_cell_payload: cell_id -> {meta + raw latencies + sensitivity}
        suitable for per_cell_latencies.json (used by gen_report.py for CDF)
    """
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in raw_runs:
        key = (r["workload_pattern"], r["pool_size"], r["warm_replicas"], r["model"])
        groups[key].append(r)

    per_cell_rows: List[Dict[str, Any]] = []
    payload: Dict[str, Dict[str, Any]] = {}

    for (workload, pool_size, warm, model), group in sorted(groups.items()):
        cell_id = f"{workload}-p{pool_size}-w{warm}"
        all_lat: List[float] = []
        all_hit: List[str] = []
        all_queue: List[float] = []
        per_seed_p50: List[float] = []
        per_seed_p95: List[float] = []
        per_seed_p99: List[float] = []
        per_seed_warm_or_l2: List[float] = []
        per_seed_slo_at_25: List[float] = []
        per_seed_cost: List[float] = []
        for r in group:
            all_lat.extend(r["latencies_ms"])
            all_hit.extend(r["hit_types"])
            all_queue.extend(r["queue_ms"])
            lats = r["latencies_ms"]
            if lats:
                per_seed_p50.append(ps.percentile(lats, 50))
                per_seed_p95.append(ps.percentile(lats, 95))
                per_seed_p99.append(ps.percentile(lats, 99))
                hit_counts = defaultdict(int)
                for h in r["hit_types"]:
                    hit_counts[h] += 1
                served = len(r["hit_types"])
                per_seed_warm_or_l2.append(
                    (hit_counts.get("warm", 0) + hit_counts.get("l2", 0)) / served
                    if served else 0.0
                )

        # SLO sensitivity sweep: at each multiplier, compute attainment per seed,
        # then mean+CI across seeds.
        h = 1024
        w = 1024
        sensitivity: Dict[str, Dict[str, float]] = {}
        for mult in slo_multipliers:
            try:
                thresh = cost_model.slo_target_ms(
                    model, h, w, optimal_latencies, multiplier=mult
                )
            except KeyError:
                continue
            per_seed_att: List[float] = []
            for r in group:
                _, _, att = cost_model.slo_attainment(r["latencies_ms"], thresh)
                per_seed_att.append(att)
            point, lo, hi = ps.mean_ci(per_seed_att, n_boot=500, seed=42)
            sensitivity[f"slo_{mult:g}x"] = {
                "threshold_ms": thresh,
                "attainment_mean": point,
                "attainment_ci_lo": lo,
                "attainment_ci_hi": hi,
            }

        # 2.5x is the headline metric.
        slo25 = sensitivity.get("slo_2.5x", {})

        def _ci_for(values: List[float]):
            if not values:
                return (float("nan"),) * 3
            return ps.mean_ci(values, n_boot=500, seed=42)

        p50_pt, p50_lo, p50_hi = _ci_for(per_seed_p50)
        p95_pt, p95_lo, p95_hi = _ci_for(per_seed_p95)
        p99_pt, p99_lo, p99_hi = _ci_for(per_seed_p99)
        warm_pt, warm_lo, warm_hi = _ci_for(per_seed_warm_or_l2)

        per_cell_rows.append({
            "cell_id": cell_id,
            "workload_pattern": workload,
            "pool_size": pool_size,
            "head_size": (group[0]["head_size"] if group else 1),
            "warm_replicas": warm,
            "model": model,
            "n_seeds": len(group),
            "n_requests_total": len(all_lat),
            "p50_ms_mean": p50_pt,
            "p50_ms_ci_lo": p50_lo,
            "p50_ms_ci_hi": p50_hi,
            "p95_ms_mean": p95_pt,
            "p95_ms_ci_lo": p95_lo,
            "p95_ms_ci_hi": p95_hi,
            "p99_ms_mean": p99_pt,
            "p99_ms_ci_lo": p99_lo,
            "p99_ms_ci_hi": p99_hi,
            "warm_or_l2_hit_rate_mean": warm_pt,
            "warm_or_l2_hit_rate_ci_lo": warm_lo,
            "warm_or_l2_hit_rate_ci_hi": warm_hi,
            "slo_2_5x_attainment_mean": slo25.get("attainment_mean", float("nan")),
            "slo_2_5x_attainment_ci_lo": slo25.get("attainment_ci_lo", float("nan")),
            "slo_2_5x_attainment_ci_hi": slo25.get("attainment_ci_hi", float("nan")),
            "slo_2_5x_threshold_ms": slo25.get("threshold_ms", float("nan")),
        })
        payload[cell_id] = {
            "workload_pattern": workload,
            "pool_size": pool_size,
            "warm_replicas": warm,
            "model": model,
            "n_seeds": len(group),
            "latencies_ms": all_lat,
            "queue_ms": all_queue,
            "hit_types": all_hit,
            "sensitivity": sensitivity,
        }
    return per_cell_rows, payload


def write_csv(rows: List[RunRow], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            d = asdict(r)
            # convert inf to a string CSV consumers can interpret
            for k, v in list(d.items()):
                if isinstance(v, float) and (v == float("inf") or v != v):
                    d[k] = ""
            writer.writerow(d)


def write_json(rows: List[RunRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "eval-v1",
        "n_runs": len(rows),
        "rows": [asdict(r) for r in rows],
    }
    # Replace inf with None for valid JSON.
    def _scrub(o):
        if isinstance(o, dict):
            return {k: _scrub(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_scrub(x) for x in o]
        if isinstance(o, float) and (o == float("inf") or o != o):
            return None
        return o
    out_path.write_text(json.dumps(_scrub(payload), indent=2), encoding="utf-8")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="results root to walk (contains run subdirs)")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="output CSV path (default: <root>/per_run.csv)")
    parser.add_argument("--out-json", type=Path, default=None,
                        help="output JSON path (default: <root>/per_run.json)")
    parser.add_argument("--slo-multiplier", type=float, default=2.5)
    parser.add_argument("--slo-multipliers", type=str, default="1.5,2.5,5.0",
                        help="comma-separated SLO multipliers for sensitivity analysis")
    parser.add_argument("--cost-rate-usd-per-gpu-hour", type=float, default=4.0)
    parser.add_argument("--gpu-per-pod", type=int, default=8)
    parser.add_argument("--optimal-latencies", type=Path,
                        default=REPO_ROOT / "data" / "optimal_latencies.json")
    parser.add_argument("--parser", type=Path,
                        default=REPO_ROOT / "scripts" / "parse-coldstart-trace.py")
    args = parser.parse_args(argv[1:])

    cost_model = _load_cost_model()
    ps = _load_paper_stats()
    optimal_latencies = load_optimal_latencies(args.optimal_latencies)

    rows: List[RunRow] = []
    raws: List[Dict[str, Any]] = []
    for child in sorted(args.root.iterdir()) if args.root.is_dir() else []:
        if not child.is_dir():
            continue
        result = aggregate_run(
            child,
            parser_path=args.parser,
            cost_model=cost_model,
            optimal_latencies=optimal_latencies,
            slo_multiplier=args.slo_multiplier,
            rate_usd_per_gpu_hour=args.cost_rate_usd_per_gpu_hour,
            gpu_per_pod=args.gpu_per_pod,
        )
        if result is not None:
            row, raw = result
            rows.append(row)
            raws.append(raw)

    out_csv = args.out_csv or (args.root / "per_run.csv")
    out_json = args.out_json or (args.root / "per_run.json")
    write_csv(rows, out_csv)
    write_json(rows, out_json)

    # Per-cell aggregation with CIs across seeds.
    slo_multipliers = [float(s) for s in args.slo_multipliers.split(",") if s.strip()]
    per_cell_rows, per_cell_payload = aggregate_cells(
        raws,
        ps=ps,
        cost_model=cost_model,
        optimal_latencies=optimal_latencies,
        slo_multipliers=slo_multipliers,
        rate_usd_per_gpu_hour=args.cost_rate_usd_per_gpu_hour,
        gpu_per_pod=args.gpu_per_pod,
    )
    per_cell_csv = args.root / "per_cell.csv"
    per_cell_latencies_json = args.root / "per_cell_latencies.json"

    if per_cell_rows:
        with per_cell_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_cell_rows[0].keys()))
            writer.writeheader()
            for r in per_cell_rows:
                clean = {}
                for k, v in r.items():
                    if isinstance(v, float) and (v == float("inf") or v != v):
                        clean[k] = ""
                    else:
                        clean[k] = v
                writer.writerow(clean)

    def _scrub_inf(o):
        if isinstance(o, dict):
            return {k: _scrub_inf(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_scrub_inf(x) for x in o]
        if isinstance(o, float) and (o == float("inf") or o != o):
            return None
        return o

    per_cell_latencies_json.write_text(
        json.dumps(
            {"schema_version": "eval-v1",
             "slo_multipliers": slo_multipliers,
             "cells": _scrub_inf(per_cell_payload)},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"aggregated {len(rows)} runs → {out_csv} + {out_json}; "
        f"{len(per_cell_rows)} cells → {per_cell_csv} + {per_cell_latencies_json}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
