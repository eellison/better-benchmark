"""Input arms for custom-kernel repros, and the sensitivity gate.

A custom-kernel repro launches one kernel at the shapes a captured graph
called it with. Its floating-point inputs are random. Its integer structure
inputs only have to be valid, yet their values can move the kernel's time:
attention over jagged rows does work that grows faster than the row length,
cross attention pairs query rows with key/value rows, and gathers, scatters
and atomics depend on which ids repeat and how they are ordered. This module
draws those inputs from named *arms* and decides whether a kernel's time
depends on which arm it gets.

Length arms (jagged lengths/offsets) keep the graph's row count and total and
never exceed the graph's per-row cap:

- ``uniform``: the gaps between sorted uniform cut points
  (``jagged_inputs``' default, the same values a full-graph replay feeds);
- ``balanced``: every row the same length;
- ``lognormal1`` / ``lognormal2``: lognormal weights, sigma 1 and 2;
- ``powerlaw``: Pareto weights, alpha 1.5;
- ``max_skew``: as many rows at the cap as the total allows (leaving every
  other row at least the minimum length); the other rows share the rest.

For paired inputs (query rows attending to key/value rows), the secondary
input is either independent of the primary (``indep``: a random order) or
comonotone with it (``comono``: the longest query row goes with the longest
key/value row).

Index arms keep the graph's id count and bound and vary the profile that
``index_profile`` reports (unique fraction, sortedness, run length, top-1%
mass): ``uniform`` (independent uniform ids, the replay default), ``unique``
(no id repeats), ``zipf`` (Zipf(1) ids, hot ids scattered), ``zipf_sorted``
(the same ids sorted), ``runs8`` (sorted ids, each repeated 8 times, like
request-to-candidate broadcasts) and ``single`` (one id everywhere: the most
contended atomics).

The sensitivity gate times every arm, interleaved over rounds. A kernel
*qualifies* for captured inputs when the spread of the arm medians exceeds
``max(noise, 5%)``, where noise is the largest same-arm spread across rounds,
and is more than 5 us (timer resolution, and too small to matter).
No arm is ruled out as unrealistic. What a qualifying kernel needs is read
off the arms: if only the marginal arms differ, a histogram (or index
profile) of each input; if ``indep`` and ``comono`` differ, a joint histogram
of the pair.

Inputs come in tiers, resolved as ``real -> stats -> synthetic``. Only the
synthetic tier exists so far; asking for another prints why it fell back.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NOISE_FLOOR_PCT = 5.0
# Below this a difference is timer resolution and launch jitter, and too
# small to matter, whatever its percentage.
NOISE_FLOOR_MS = 0.005
DEFAULT_MIN_LEN = 1
ARMS = ("balanced", "uniform", "lognormal1", "lognormal2", "powerlaw", "max_skew")
PAIRINGS = ("indep", "comono")
INDEX_ARMS = ("uniform", "unique", "zipf", "zipf_sorted", "runs8", "single")
TIERS = ("synthetic", "stats", "real")
STATS_SCHEMA_VERSION = 1


# Length families. Each fills ``count`` rows adding up to ``total`` with every
# row in [min_len, max_len]; ``jagged_inputs.make_lengths`` validates the result.


def _fit(weights: Any, total: int, max_len: int, min_len: int) -> Any:
    """Integer lengths proportional to ``weights``, adding up to ``total``,
    each in [min_len, max_len]."""
    import torch

    count = weights.numel()
    if not count * min_len <= total <= count * max_len:
        raise ValueError(
            f"{count} rows cannot hold {total} with lengths in [{min_len}, {max_len}]"
        )
    x = weights.double().clamp(min=1e-12)
    x = x * (total / x.sum())
    for _ in range(200):
        x = x.clamp(min_len, max_len)
        free = (x > min_len) & (x < max_len)
        gap = total - x.sum().item()
        if abs(gap) < 0.5 or not free.any():
            break
        x[free] *= 1 + gap / x[free].sum().item()
    lengths = x.floor().long().clamp(min_len, max_len)
    remainder = total - int(lengths.sum())
    order = torch.argsort(x - lengths, descending=True)
    while remainder:
        room = lengths[order] < max_len if remainder > 0 else lengths[order] > min_len
        rows = order[room][: abs(remainder)]
        lengths[rows] += 1 if remainder > 0 else -1
        remainder -= int(math.copysign(rows.numel(), remainder))
    return lengths


def _bounds(total: int, count: int, params: dict) -> tuple[int, int]:
    min_len = int(params.get("min_len", DEFAULT_MIN_LEN))
    max_len = params.get("max_len")
    return min_len, int(max_len) if max_len is not None else max(total, min_len)


def _balanced(count: int, total: int, params: dict) -> Any:
    import torch

    min_len, max_len = _bounds(total, count, params)
    return _fit(torch.ones(count), total, max_len, min_len)


def _lognormal(count: int, total: int, params: dict) -> Any:
    import torch

    min_len, max_len = _bounds(total, count, params)
    sigma = float(params.get("sigma", 1.0))
    return _fit(torch.exp(sigma * torch.randn(count)), total, max_len, min_len)


def _powerlaw(count: int, total: int, params: dict) -> Any:
    import torch

    min_len, max_len = _bounds(total, count, params)
    alpha = float(params.get("alpha", 1.5))
    weights = (1 - torch.rand(count, dtype=torch.float64)) ** (-1 / alpha)
    return _fit(weights, total, max_len, min_len)


def _max_skew(count: int, total: int, params: dict) -> Any:
    import torch

    min_len, max_len = _bounds(total, count, params)
    full = min(count, max((total - count * min_len) // max(max_len - min_len, 1), 0))
    weights = torch.full((count,), float(min_len))
    weights[torch.randperm(count)[:full]] = float(max_len)
    return _fit(weights, total, max_len, min_len)


_FAMILIES: dict[str, Callable[[int, int, dict], Any]] = {
    "balanced": _balanced,
    "lognormal": _lognormal,
    "powerlaw": _powerlaw,
    "max_skew": _max_skew,
}


def register_families() -> None:
    """Add the arm families that ``jagged_inputs`` does not define yet."""
    import jagged_inputs

    for name, fn in _FAMILIES.items():
        if name not in jagged_inputs.LENGTHS_DISTRIBUTIONS:
            jagged_inputs.register_lengths_distribution(name, fn)


def arm_distribution(arm: str, max_len: int | None, min_len: int) -> str | dict:
    """``jagged_inputs`` distribution spec of an arm."""
    bounds = {"max_len": max_len, "min_len": min_len}
    specs: dict[str, str | dict] = {
        "uniform": "uniform",
        "balanced": {"family": "balanced", **bounds},
        "lognormal1": {"family": "lognormal", "sigma": 1.0, **bounds},
        "lognormal2": {"family": "lognormal", "sigma": 2.0, **bounds},
        "powerlaw": {"family": "powerlaw", "alpha": 1.5, **bounds},
        "max_skew": {"family": "max_skew", **bounds},
    }
    if arm not in specs:
        raise ValueError(f"unknown arm {arm!r}; choose from {', '.join(ARMS)}")
    return specs[arm]


# Structure inputs.


@dataclass(frozen=True)
class JaggedInput:
    """One integer structure input of a kernel launch.

    ``kind`` is "offsets" (count + 1 values from 0 to total) or "lengths"
    (count values adding up to total). ``pair_with`` names the primary input
    a secondary input is paired with, row by row."""

    name: str
    kind: str
    count: int
    total: int
    max_len: int | None
    dtype: Any
    pair_with: str | None = None

    def describe(self) -> str:
        cap = f", cap {self.max_len}" if self.max_len is not None else ""
        pair = f", paired with {self.pair_with}" if self.pair_with else ""
        return f"{self.name}: {self.kind} of {self.count} rows, total {self.total}{cap}{pair}"


def _seed(*parts: Any) -> int:
    return zlib.crc32(json.dumps(parts).encode())


def pair_lengths(primary: Any, secondary: Any, pairing: str) -> Any:
    """``secondary`` reordered against ``primary``: a random order (indep) or
    rank for rank (comono)."""
    import torch

    if pairing == "indep":
        return secondary[torch.randperm(secondary.numel())]
    if pairing == "comono":
        rank = torch.argsort(torch.argsort(primary, stable=True), stable=True)
        return torch.sort(secondary).values[rank]
    raise ValueError(f"unknown pairing {pairing!r}; choose from {', '.join(PAIRINGS)}")


def structure_lengths(
    inputs: Sequence[JaggedInput],
    arm: str,
    pairing: str = "indep",
    *,
    seed: int = 0,
    min_len: int = DEFAULT_MIN_LEN,
) -> dict[str, Any]:
    """CPU int64 lengths of every input for one arm, primaries first."""
    import torch
    from jagged_inputs import make_lengths

    register_families()
    by_name = {spec.name: spec for spec in inputs}
    order = sorted(inputs, key=lambda spec: spec.pair_with is not None)
    lengths: dict[str, Any] = {}
    for spec in order:
        floor = min(min_len, spec.total // max(spec.count, 1))
        torch.manual_seed(_seed(seed, arm, spec.name))
        drawn = make_lengths(
            spec.count, spec.total, spec.max_len, arm_distribution(arm, spec.max_len, floor)
        )
        if spec.pair_with in by_name:
            if by_name[spec.pair_with].count != spec.count:
                raise ValueError(f"{spec.name} and {spec.pair_with} differ in rows")
            torch.manual_seed(_seed(seed, arm, spec.name, pairing))
            drawn = pair_lengths(lengths[spec.pair_with], drawn, pairing)
        lengths[spec.name] = drawn
    return lengths


def to_tensor(kind: str, lengths: Any, dtype: Any, device: Any) -> Any:
    """Lengths as a "lengths" or "offsets" input tensor."""
    import torch

    values = lengths
    if kind == "offsets":
        values = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    return values.to(dtype=dtype, device=device)


def length_summary(lengths: Any) -> str:
    import torch

    q = torch.quantile(lengths.double(), torch.tensor([0.5, 0.99], dtype=torch.float64))
    return (
        f"min {int(lengths.min())} p50 {q[0].item():.0f} p99 {q[1].item():.0f} "
        f"max {int(lengths.max())}"
    )


# Index arms.


def make_indices(n: int, low: int, high: int, arm: str, *, seed: int = 0) -> Any:
    """CPU int64: ``n`` ids in [low, high) drawn from an index arm."""
    import torch

    rows = max(high - low, 1)
    g = torch.Generator().manual_seed(_seed(seed, arm, n, low, high))
    if arm == "uniform":
        ids = torch.randint(0, rows, (n,), generator=g)
    elif arm == "unique":
        ids = torch.randperm(rows, generator=g)[:n] if n <= rows else torch.randperm(n, generator=g) % rows
    elif arm in ("zipf", "zipf_sorted"):
        # Zipf(1): log-uniform ranks; a random permutation scatters the hot ids.
        u = torch.rand(n, generator=g, dtype=torch.float64)
        ranks = (torch.exp(u * math.log(rows + 1)) - 1).long().clamp(0, rows - 1)
        ids = torch.randperm(rows, generator=g)[ranks] if rows <= 1 << 26 else ranks * 2654435761 % rows
        if arm == "zipf_sorted":
            ids = torch.sort(ids).values
    elif arm == "runs8":
        base = torch.sort(torch.randint(0, rows, (math.ceil(n / 8),), generator=g)).values
        ids = base.repeat_interleave(8)[:n]
    elif arm == "single":
        ids = torch.full((n,), int(torch.randint(0, rows, (1,), generator=g)))
    else:
        raise ValueError(f"unknown index arm {arm!r}; choose from {', '.join(INDEX_ARMS)}")
    return ids + low


def index_profile(ids: Any) -> dict[str, float]:
    """Unique fraction, sorted fraction (adjacent pairs in order), mean run
    length (equal adjacent ids) and the share of ids hitting the 1% most
    frequent ids."""
    import torch

    flat = ids.reshape(-1).long().cpu()
    n = max(flat.numel(), 1)
    counts = torch.unique(flat, return_counts=True)[1]
    top = torch.sort(counts, descending=True).values[: max(1, counts.numel() // 100)]
    steps = flat.diff()
    runs = int((steps != 0).sum()) + 1 if flat.numel() else 0
    return {
        "unique_fraction": counts.numel() / n,
        "sorted_fraction": float((steps >= 0).float().mean()) if steps.numel() else 1.0,
        "mean_run_length": n / max(runs, 1),
        "top1pct_mass": float(top.sum()) / n,
    }


def format_index_profile(ids: Any) -> str:
    p = index_profile(ids)
    return (
        f"unique {p['unique_fraction']:.3f} sorted {p['sorted_fraction']:.2f} "
        f"run {p['mean_run_length']:.1f} top1% {p['top1pct_mass']:.3f}"
    )


# Input tiers.


def load_stats(path: Path) -> dict[str, Any]:
    """An aggregate-stats file: per graph input, a log2 histogram of row
    lengths (cells below k rows merged) and optionally a joint histogram with
    the input it is paired with::

        {"schema_version": 1, "graph": "<model>/<graph>", "k": 100,
         "inputs": {"<input>": {"log2_hist": [[cell, rows], ...],
                                "tail_rows": <rows in merged cells>,
                                "joint": {"with": "<input>",
                                          "log2_hist": [[cell, cell, rows], ...]}}}}
    """
    stats = json.loads(Path(path).read_text())
    if stats.get("schema_version") != STATS_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported stats schema {stats.get('schema_version')!r}")
    if not isinstance(stats.get("inputs"), dict):
        raise ValueError(f"{path}: no inputs")
    return stats


def resolve_tier(requested: str, stats_path: Path | None) -> tuple[str, list[str]]:
    """The tier the run uses and why it differs from ``requested``."""
    if requested not in TIERS:
        raise ValueError(f"unknown input tier {requested!r}; choose from {', '.join(TIERS)}")
    notes = []
    if requested == "real":
        notes.append(
            "real inputs are not available yet (no exact captures are stored); "
            "falling back to stats"
        )
        requested = "stats"
    if requested == "stats":
        if stats_path is not None:
            load_stats(stats_path)
            notes.append(
                f"stats inputs are not available yet ({stats_path} parsed, but no "
                "histogram sampler exists); falling back to synthetic"
            )
        else:
            notes.append(
                "stats inputs are not available yet (no stats file for this graph); "
                "falling back to synthetic"
            )
    return "synthetic", notes


# Timing.


def gpu_clock_sample() -> str:
    """SM clock and active throttle reasons of the current GPU, or ''."""
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    index = visible.split(",")[torch.cuda.current_device()] if visible else str(
        torch.cuda.current_device()
    )
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", index.strip(),
             "--query-gpu=clocks.sm,clocks_event_reasons.active",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def time_arms(
    cases: Mapping[str, Callable[[], Callable[[], Any]]],
    bench: Callable[[Callable[[], Any]], float],
    *,
    rounds: int = 3,
    log: Callable[[str], None] = print,
) -> dict[str, list[float]]:
    """``bench`` ms of each case per round, the cases interleaved in each
    round. A case is a factory that sets up its state (e.g. a forward pass a
    backward needs) and returns the function to time; its state is freed
    after timing. ``bench`` is the caller's locked timer."""
    import torch

    times: dict[str, list[float]] = {name: [] for name in cases}
    for r in range(rounds):
        for name, setup in cases.items():
            fn = setup()
            ms = bench(fn)
            del fn
            torch.cuda.empty_cache()
            times[name].append(ms)
            log(f"  round {r} {name:<22} {ms:9.3f} ms  [{gpu_clock_sample()}]")
    return times


def _pct(hi: float, lo: float) -> float:
    return 100.0 * (hi - lo) / lo if lo > 0 else math.inf


def verdict(times: Mapping[str, Sequence[float]], pairs: Mapping[str, tuple[str, str]] = {}) -> dict:
    """The gate's numbers. ``pairs`` maps an arm to its (indep, comono) cases
    for paired kernels; every other case is a marginal arm, an index arm when
    its name starts with "index:"."""
    medians = {name: sorted(ts)[len(ts) // 2] for name, ts in times.items()}
    noise = {name: _pct(max(ts), min(ts)) for name, ts in times.items()}
    threshold = max(max(noise.values()), NOISE_FLOOR_PCT)
    spread = _pct(max(medians.values()), min(medians.values()))
    marginal = [indep for indep, _ in pairs.values()] or list(medians)
    marginal_spread = _pct(
        max(medians[name] for name in marginal), min(medians[name] for name in marginal)
    )
    pairing = {
        arm: 100.0 * (medians[comono] - medians[indep]) / medians[indep]
        for arm, (indep, comono) in pairs.items()
    }

    def differs(names: list[str]) -> bool:
        if len(names) < 2:
            return False
        hi, lo = max(medians[n] for n in names), min(medians[n] for n in names)
        return _pct(hi, lo) > threshold and hi - lo > NOISE_FLOOR_MS

    needs = []
    if differs([n for n in marginal if not n.startswith("index:")]):
        needs.append("marginal histogram")
    if differs([n for n in marginal if n.startswith("index:")]):
        needs.append("index profile")
    if any(differs(list(pair)) for pair in pairs.values()):
        needs.append("joint histogram of the pair")
    return {
        "median_ms": medians,
        "noise_pct": noise,
        "threshold_pct": threshold,
        "spread_pct": spread,
        "marginal_spread_pct": marginal_spread,
        "pairing_effect_pct": pairing,
        "qualifies": differs(list(medians)),
        "needs": needs or ["valid inputs only (synthetic is enough)"],
    }


def format_verdict(result: Mapping[str, Any]) -> str:
    medians = result["median_ms"]
    fastest = min(medians.values())
    lines = [f"  {'case':<22} {'median ms':>10} {'vs fastest':>11} {'noise':>7}"]
    for name, ms in medians.items():
        lines.append(
            f"  {name:<22} {ms:10.3f} {_pct(ms, fastest):10.1f}% {result['noise_pct'][name]:6.1f}%"
        )
    lines.append(
        f"  spread {result['spread_pct']:.1f}% across all cases "
        f"(marginal arms {result['marginal_spread_pct']:.1f}%); "
        f"threshold max(noise, {NOISE_FLOOR_PCT:.0f}%) = {result['threshold_pct']:.1f}% "
        f"and > {NOISE_FLOOR_MS * 1000:.0f} us"
    )
    for arm, effect in result["pairing_effect_pct"].items():
        lines.append(f"  pairing effect at {arm}: comono vs indep {effect:+.1f}%")
    lines.append(
        f"  verdict: {'QUALIFIES' if result['qualifies'] else 'does not qualify'}"
        f" for captured structure inputs; needs: {', '.join(result['needs'])}"
    )
    return "\n".join(lines)
