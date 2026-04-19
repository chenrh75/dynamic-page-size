#!/usr/bin/env python3
"""
extract_gem5_tlb_metrics.py

Extract the most relevant metrics for a CoLT-style TLB coalescing /
de-coalescing project from one or more gem5 stats.txt files.

What it reports:
- Overall performance: simInsts, cycles, IPC, CPI, simTicks, simSeconds
- L1 DTLB / L1 ITLB / shared L2 TLB hits, misses, accesses, miss rates
- MPKI for DTLB / ITLB / combined L1 TLB / L2 TLB
- Page-walker activity and walker MPKI
- Automatically surfaces any custom counters with names containing:
  coales, coalesce, decoal, decoalesce, split

Usage:
    python extract_gem5_tlb_metrics.py m5out/stats.txt
    python extract_gem5_tlb_metrics.py baseline/stats.txt colt/stats.txt
    python extract_gem5_tlb_metrics.py *.txt -o summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

STAT_LINE_RE = re.compile(
    r"""^\s*
    (?P<key>[A-Za-z0-9_.:]+)
    \s+
    (?P<value>
        [-+]?nan
        |
        [-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?
    )
    (?:\s+\#.*)?$
    """,
    re.VERBOSE | re.IGNORECASE,
)

CUSTOM_COUNTER_PATTERNS = (
    "coales",
    "coalesce",
    "decoal",
    "decoalesce",
    "split",
)


def parse_value(raw: str):
    raw_l = raw.lower()
    if raw_l == "nan":
        return float("nan")
    try:
        val = float(raw)
    except ValueError:
        return raw
    if val.is_integer():
        return int(val)
    return val


def parse_stats_file(path: str) -> dict[str, float]:
    stats: dict[str, float] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = STAT_LINE_RE.match(line)
                if not m:
                    continue
                key = m.group("key")
                val = parse_value(m.group("value"))
                stats[key] = val
    except OSError as e:
        raise RuntimeError(f"Could not read '{path}': {e}") from e
    return stats


def get_first(stats: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        if key in stats:
            return stats[key]
    return None


def safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    if den == 0:
        return None
    return num / den


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:.4f}%"


def fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if abs(x) >= 1_000_000:
            return f"{x:,.0f}"
        if abs(x) >= 1000:
            return f"{x:,.3f}"
        return f"{x:.6f}"
    return f"{x:,}"


def mpki(misses: float | None, insts: float | None) -> float | None:
    if misses is None or insts is None or insts == 0:
        return None
    return (misses * 1000.0) / insts


def extract_tlb_block(
    stats: dict[str, float], prefix: str, kind: str
) -> dict[str, float | None]:
    """
    kind:
      - 'dtb'
      - 'itb'
      - 'l2'
    """
    if kind == "dtb":
        hits = get_first(stats, f"{prefix}.hits")
        misses = get_first(stats, f"{prefix}.misses")
        accesses = get_first(stats, f"{prefix}.accesses")
        read_hits = get_first(stats, f"{prefix}.readHits")
        read_misses = get_first(stats, f"{prefix}.readMisses")
        write_hits = get_first(stats, f"{prefix}.writeHits")
        write_misses = get_first(stats, f"{prefix}.writeMisses")
        read_accesses = get_first(stats, f"{prefix}.readAccesses")
        write_accesses = get_first(stats, f"{prefix}.writeAccesses")
        inserts = get_first(stats, f"{prefix}.inserts")
        flushed = get_first(stats, f"{prefix}.flushedEntries")
        flushes = get_first(stats, f"{prefix}.flushTlb")
        return {
            "hits": hits,
            "misses": misses,
            "accesses": accesses,
            "miss_rate": safe_div(misses, accesses),
            "read_hits": read_hits,
            "read_misses": read_misses,
            "write_hits": write_hits,
            "write_misses": write_misses,
            "read_accesses": read_accesses,
            "write_accesses": write_accesses,
            "inserts": inserts,
            "flushed_entries": flushed,
            "flush_requests": flushes,
        }

    if kind == "itb":
        hits = get_first(stats, f"{prefix}.hits")
        misses = get_first(stats, f"{prefix}.misses")
        accesses = get_first(stats, f"{prefix}.accesses")
        inst_hits = get_first(stats, f"{prefix}.instHits")
        inst_misses = get_first(stats, f"{prefix}.instMisses")
        inst_accesses = get_first(stats, f"{prefix}.instAccesses")
        inserts = get_first(stats, f"{prefix}.inserts")
        flushed = get_first(stats, f"{prefix}.flushedEntries")
        flushes = get_first(stats, f"{prefix}.flushTlb")
        return {
            "hits": hits,
            "misses": misses,
            "accesses": accesses,
            "miss_rate": safe_div(misses, accesses),
            "inst_hits": inst_hits,
            "inst_misses": inst_misses,
            "inst_accesses": inst_accesses,
            "inserts": inserts,
            "flushed_entries": flushed,
            "flush_requests": flushes,
        }

    if kind == "l2":
        hits = get_first(stats, f"{prefix}.hits")
        misses = get_first(stats, f"{prefix}.misses")
        accesses = get_first(stats, f"{prefix}.accesses")
        inserts = get_first(stats, f"{prefix}.inserts")
        flushed = get_first(stats, f"{prefix}.flushedEntries")
        flushes = get_first(stats, f"{prefix}.flushTlb")
        return {
            "hits": hits,
            "misses": misses,
            "accesses": accesses,
            "miss_rate": safe_div(misses, accesses),
            "inserts": inserts,
            "flushed_entries": flushed,
            "flush_requests": flushes,
        }

    raise ValueError(f"Unknown TLB kind: {kind}")


def extract_custom_counters(stats: dict[str, float]) -> dict[str, float]:
    out = {}
    for key, value in stats.items():
        k = key.lower()
        if any(pattern in k for pattern in CUSTOM_COUNTER_PATTERNS):
            out[key] = value
    return dict(sorted(out.items()))


def summarize(
    path: str, stats: dict[str, float]
) -> tuple[dict[str, float | None], dict[str, float]]:
    sim_insts = get_first(
        stats, "simInsts", "system.cpu.commitStats0.numInsts"
    )
    sim_ops = get_first(stats, "simOps", "system.cpu.commitStats0.numOps")
    cycles = get_first(stats, "system.cpu.numCycles")
    ipc = get_first(stats, "system.cpu.ipc", "system.cpu.commitStats0.ipc")
    cpi = get_first(stats, "system.cpu.cpi", "system.cpu.commitStats0.cpi")
    sim_ticks = get_first(stats, "simTicks", "finalTick")
    sim_seconds = get_first(stats, "simSeconds")
    host_seconds = get_first(stats, "hostSeconds")
    num_mem_refs = get_first(stats, "system.cpu.commitStats0.numMemRefs")

    dtb = extract_tlb_block(stats, "system.cpu.mmu.dtb", "dtb")
    itb = extract_tlb_block(stats, "system.cpu.mmu.itb", "itb")
    l2 = extract_tlb_block(stats, "system.cpu.mmu.l2_shared", "l2")

    walker_total = get_first(stats, "system.cpu.mmu.walker.walks")
    walker_inst_s1 = get_first(
        stats, "system.cpu.mmu.walker.instructionWalksS1"
    )
    walker_inst_s2 = get_first(
        stats, "system.cpu.mmu.walker.instructionWalksS2"
    )
    walker_data_s1 = get_first(stats, "system.cpu.mmu.walker.dataWalksS1")
    walker_data_s2 = get_first(stats, "system.cpu.mmu.walker.dataWalksS2")

    l1_tlb_misses = None
    l1_tlb_accesses = None
    if dtb["misses"] is not None or itb["misses"] is not None:
        l1_tlb_misses = (dtb["misses"] or 0) + (itb["misses"] or 0)
    if dtb["accesses"] is not None or itb["accesses"] is not None:
        l1_tlb_accesses = (dtb["accesses"] or 0) + (itb["accesses"] or 0)

    row: dict[str, float | None] = {
        "file": os.path.basename(path),
        "simInsts": sim_insts,
        "simOps": sim_ops,
        "numCycles": cycles,
        "ipc": ipc,
        "cpi": cpi,
        "simTicks": sim_ticks,
        "simSeconds": sim_seconds,
        "hostSeconds": host_seconds,
        "numMemRefs": num_mem_refs,
        "dtlb_hits": dtb["hits"],
        "dtlb_misses": dtb["misses"],
        "dtlb_accesses": dtb["accesses"],
        "dtlb_miss_rate": dtb["miss_rate"],
        "dtlb_mpki": mpki(dtb["misses"], sim_insts),
        "itlb_hits": itb["hits"],
        "itlb_misses": itb["misses"],
        "itlb_accesses": itb["accesses"],
        "itlb_miss_rate": itb["miss_rate"],
        "itlb_mpki": mpki(itb["misses"], sim_insts),
        "l1_tlb_misses_total": l1_tlb_misses,
        "l1_tlb_accesses_total": l1_tlb_accesses,
        "l1_tlb_miss_rate_total": safe_div(l1_tlb_misses, l1_tlb_accesses),
        "l1_tlb_mpki_total": mpki(l1_tlb_misses, sim_insts),
        "l2_tlb_hits": l2["hits"],
        "l2_tlb_misses": l2["misses"],
        "l2_tlb_accesses": l2["accesses"],
        "l2_tlb_miss_rate": l2["miss_rate"],
        "l2_tlb_mpki": mpki(l2["misses"], sim_insts),
        "walker_walks_total": walker_total,
        "walker_inst_s1": walker_inst_s1,
        "walker_inst_s2": walker_inst_s2,
        "walker_data_s1": walker_data_s1,
        "walker_data_s2": walker_data_s2,
        "walker_mpki": mpki(walker_total, sim_insts),
    }

    return row, extract_custom_counters(stats)


def print_summary(
    row: dict[str, float | None], custom: dict[str, float]
) -> None:
    print("=" * 88)
    print(f"FILE: {row['file']}")
    print("-" * 88)
    print("CORE PERFORMANCE")
    print(f"  simInsts        : {fmt(row['simInsts'])}")
    print(f"  numCycles       : {fmt(row['numCycles'])}")
    print(f"  IPC             : {fmt(row['ipc'])}")
    print(f"  CPI             : {fmt(row['cpi'])}")
    print(f"  simTicks        : {fmt(row['simTicks'])}")
    print(f"  simSeconds      : {fmt(row['simSeconds'])}")
    print(f"  hostSeconds     : {fmt(row['hostSeconds'])}")

    print("\nTLB METRICS")
    print(
        f"  L1 DTLB misses  : {fmt(row['dtlb_misses'])}  "
        f"(accesses={fmt(row['dtlb_accesses'])}, miss_rate={pct(row['dtlb_miss_rate'])}, MPKI={fmt(row['dtlb_mpki'])})"
    )
    print(
        f"  L1 ITLB misses  : {fmt(row['itlb_misses'])}  "
        f"(accesses={fmt(row['itlb_accesses'])}, miss_rate={pct(row['itlb_miss_rate'])}, MPKI={fmt(row['itlb_mpki'])})"
    )
    print(
        f"  L1 total misses : {fmt(row['l1_tlb_misses_total'])}  "
        f"(accesses={fmt(row['l1_tlb_accesses_total'])}, miss_rate={pct(row['l1_tlb_miss_rate_total'])}, MPKI={fmt(row['l1_tlb_mpki_total'])})"
    )
    print(
        f"  L2 TLB misses   : {fmt(row['l2_tlb_misses'])}  "
        f"(accesses={fmt(row['l2_tlb_accesses'])}, miss_rate={pct(row['l2_tlb_miss_rate'])}, MPKI={fmt(row['l2_tlb_mpki'])})"
    )

    print("\nPAGE WALKER")
    print(
        f"  total walks     : {fmt(row['walker_walks_total'])}  (MPKI={fmt(row['walker_mpki'])})"
    )
    print(f"  inst walks S1   : {fmt(row['walker_inst_s1'])}")
    print(f"  inst walks S2   : {fmt(row['walker_inst_s2'])}")
    print(f"  data walks S1   : {fmt(row['walker_data_s1'])}")
    print(f"  data walks S2   : {fmt(row['walker_data_s2'])}")

    if custom:
        print("\nCUSTOM COALESCING / DE-COALESCING COUNTERS")
        for key, value in custom.items():
            print(f"  {key:50s} {fmt(value)}")
    else:
        print("\nCUSTOM COALESCING / DE-COALESCING COUNTERS")
        print("  none detected by name pattern")


def write_csv(rows: list[dict[str, float | None]], out_path: str) -> None:
    if not rows:
        return

    fieldnames = [
        "file",
        "simInsts",
        "simOps",
        "numCycles",
        "ipc",
        "cpi",
        "simTicks",
        "simSeconds",
        "hostSeconds",
        "numMemRefs",
        "dtlb_hits",
        "dtlb_misses",
        "dtlb_accesses",
        "dtlb_miss_rate",
        "dtlb_mpki",
        "itlb_hits",
        "itlb_misses",
        "itlb_accesses",
        "itlb_miss_rate",
        "itlb_mpki",
        "l1_tlb_misses_total",
        "l1_tlb_accesses_total",
        "l1_tlb_miss_rate_total",
        "l1_tlb_mpki_total",
        "l2_tlb_hits",
        "l2_tlb_misses",
        "l2_tlb_accesses",
        "l2_tlb_miss_rate",
        "l2_tlb_mpki",
        "walker_walks_total",
        "walker_inst_s1",
        "walker_inst_s2",
        "walker_data_s1",
        "walker_data_s2",
        "walker_mpki",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_comparison(rows: list[dict[str, float | None]]) -> None:
    if len(rows) < 2:
        return

    baseline = rows[0]
    print("\n" + "=" * 88)
    print(f"COMPARISON VS BASELINE: {baseline['file']}")
    print("=" * 88)

    compare_fields = [
        ("numCycles", "cycles", True),
        ("simTicks", "simTicks", True),
        ("ipc", "IPC", False),
        ("cpi", "CPI", True),
        ("l1_tlb_misses_total", "L1 TLB misses", True),
        ("l2_tlb_misses", "L2 TLB misses", True),
        ("walker_walks_total", "page walks", True),
    ]

    for row in rows[1:]:
        print(f"\n{row['file']}")
        for field, label, lower_is_better in compare_fields:
            base = baseline.get(field)
            new = row.get(field)
            if base is None or new is None or base == 0:
                print(f"  {label:16s}: n/a")
                continue

            delta = (new - base) / base
            if lower_is_better:
                improvement = -delta
            else:
                improvement = delta

            print(
                f"  {label:16s}: base={fmt(base):>12s}  new={fmt(new):>12s}  "
                f"relative_change={delta * 100:+8.3f}%  "
                f"{'improvement' if improvement >= 0 else 'regression'}={abs(improvement) * 100:.3f}%"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract CoLT-relevant performance and TLB metrics from gem5 stats.txt files."
    )
    parser.add_argument(
        "stats_files", nargs="+", help="One or more gem5 stats.txt files"
    )
    parser.add_argument("-o", "--output-csv", help="Optional output CSV path")
    args = parser.parse_args()

    rows: list[dict[str, float | None]] = []
    customs: list[dict[str, float]] = []

    for path in args.stats_files:
        try:
            stats = parse_stats_file(path)
            row, custom = summarize(path, stats)
            rows.append(row)
            customs.append(custom)
            print_summary(row, custom)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    if args.output_csv:
        write_csv(rows, args.output_csv)
        print(f"\nWrote CSV summary to: {args.output_csv}")

    print_comparison(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
