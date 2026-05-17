#!/usr/bin/env python3
"""Generate workload trace files for the eval harness.

Emits one ``Request(...)`` line per request in the exact format parsed by
``examples/api_benchmark_example_with_trace.py:parse_request_line``. The
trace-replay client then fires the requests via aiohttp at the times specified.

The five supported patterns model the workloads we want to characterize the
warm-pool architecture against:

* cold      — single request on a fresh cluster (reproduces 2026-05-06 baseline)
* burst     — N simultaneous requests (stresses warm-pool exhaustion + queueing)
* poisson   — exponential inter-arrival, rate λ (paper / AlpaServe regime)
* steady    — fixed 1/λ inter-arrival (deterministic upper bound on warm hit rate)
* mixed     — Poisson with rotating (height, width) to force re-dispatch

Usage::

    python scripts/workload_gen.py --pattern poisson --rate 1.0 \
        --duration 60 --seed 0 --model sd3 --out /tmp/w.trace

The output trace is consumed by
``examples/api_benchmark_example_with_trace.py --trace /tmp/w.trace``.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass
class Request:
    req_id: str
    prompt: str
    negative_prompt: str
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    timestamp: float

    def as_trace_line(self) -> str:
        return (
            f"Request(request_id_local={self.req_id!r}, "
            f"prompt={self.prompt!r}, "
            f"negative_prompt={self.negative_prompt!r}, "
            f"height={self.height}, "
            f"width={self.width}, "
            f"num_frames={self.num_frames}, "
            f"num_inference_steps={self.num_inference_steps}, "
            f"timestamp={self.timestamp})"
        )


@dataclass
class ModelDefaults:
    name: str
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    sample_prompts: Sequence[str] = field(default_factory=tuple)


_PROMPT_BANK = (
    "A photorealistic portrait of an astronaut riding a horse on Mars.",
    "Cinematic shot of a cyberpunk city at night, neon rain on glass.",
    "A serene forest clearing with golden hour light through tall pines.",
    "Studio macro photograph of a single dewdrop on a green leaf.",
    "Oil painting of a Mediterranean fishing village at dawn.",
    "Wide-angle landscape of snow-capped mountains reflected in a glacial lake.",
    "Detailed concept art of a steampunk airship hovering over Victorian London.",
    "Close-up of a hummingbird mid-flight feeding from a red flower.",
)

_MODEL_DEFAULTS = {
    "sd3": ModelDefaults("sd3", 1024, 1024, 0, 20, _PROMPT_BANK),
    "flux": ModelDefaults("flux", 1024, 1024, 0, 20, _PROMPT_BANK),
    "cogvideox": ModelDefaults("cogvideox", 480, 720, 49, 20, _PROMPT_BANK),
    "hunyuanvideo": ModelDefaults("hunyuanvideo", 720, 1280, 41, 30, _PROMPT_BANK),
}


def _new_request(
    idx: int,
    timestamp: float,
    model: ModelDefaults,
    rng: random.Random,
    *,
    height: int = None,
    width: int = None,
) -> Request:
    return Request(
        req_id=f"req-{idx:06d}",
        prompt=rng.choice(model.sample_prompts),
        negative_prompt="",
        height=height if height is not None else model.height,
        width=width if width is not None else model.width,
        num_frames=model.num_frames,
        num_inference_steps=model.num_inference_steps,
        timestamp=round(timestamp, 6),
    )


def gen_cold(model: ModelDefaults, rng: random.Random) -> List[Request]:
    return [_new_request(0, 0.0, model, rng)]


def gen_burst(n: int, model: ModelDefaults, rng: random.Random) -> List[Request]:
    return [_new_request(i, i * 0.001, model, rng) for i in range(n)]


def gen_poisson(
    rate: float, duration: float, model: ModelDefaults, rng: random.Random
) -> List[Request]:
    if rate <= 0:
        raise ValueError("--rate must be positive for poisson pattern")
    requests: List[Request] = []
    t = 0.0
    idx = 0
    while True:
        t += rng.expovariate(rate)
        if t > duration:
            break
        requests.append(_new_request(idx, t, model, rng))
        idx += 1
    return requests


def gen_steady(
    rate: float, duration: float, model: ModelDefaults, rng: random.Random
) -> List[Request]:
    if rate <= 0:
        raise ValueError("--rate must be positive for steady pattern")
    inter_arrival = 1.0 / rate
    requests: List[Request] = []
    t = 0.0
    idx = 0
    while t <= duration:
        requests.append(_new_request(idx, t, model, rng))
        t += inter_arrival
        idx += 1
    return requests


def gen_mixed(
    rate: float,
    duration: float,
    model: ModelDefaults,
    configs: Sequence[Tuple[int, int]],
    rng: random.Random,
) -> List[Request]:
    if not configs:
        raise ValueError("--configs must be non-empty for mixed pattern")
    base = gen_poisson(rate, duration, model, rng)
    out = []
    for i, req in enumerate(base):
        h, w = configs[i % len(configs)]
        req.height = h
        req.width = w
        out.append(req)
    return out


def write_trace(reqs: Iterable[Request], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for req in reqs:
            f.write(req.as_trace_line())
            f.write("\n")
            count += 1
    return count


def _parse_configs(spec: str) -> List[Tuple[int, int]]:
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" not in chunk:
            raise argparse.ArgumentTypeError(
                f"--configs entries must look like HxW (got {chunk!r})"
            )
        h, w = chunk.split("x", 1)
        out.append((int(h), int(w)))
    return out


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        required=True,
        choices=("cold", "burst", "poisson", "steady", "mixed"),
    )
    parser.add_argument("--rate", type=float, default=1.0,
                        help="requests/sec for poisson / steady / mixed")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="total trace duration in seconds")
    parser.add_argument("--n", type=int, default=8,
                        help="number of requests (burst pattern only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        choices=tuple(_MODEL_DEFAULTS.keys()),
        default="sd3",
    )
    parser.add_argument(
        "--configs",
        type=_parse_configs,
        default=[(1024, 1024), (2048, 2048), (768, 768)],
        help="comma-separated HxW list for mixed pattern (default 1024x1024,2048x2048,768x768)",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv[1:])

    rng = random.Random(args.seed)
    model = _MODEL_DEFAULTS[args.model]

    if args.pattern == "cold":
        reqs = gen_cold(model, rng)
    elif args.pattern == "burst":
        reqs = gen_burst(args.n, model, rng)
    elif args.pattern == "poisson":
        reqs = gen_poisson(args.rate, args.duration, model, rng)
    elif args.pattern == "steady":
        reqs = gen_steady(args.rate, args.duration, model, rng)
    elif args.pattern == "mixed":
        reqs = gen_mixed(args.rate, args.duration, model, args.configs, rng)
    else:
        raise AssertionError(f"unhandled pattern {args.pattern}")

    count = write_trace(reqs, args.out)
    print(
        f"wrote {count} requests to {args.out} "
        f"(pattern={args.pattern} model={args.model} seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
