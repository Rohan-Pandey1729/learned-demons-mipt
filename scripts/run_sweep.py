#!/usr/bin/env python3
"""Sweep (L, p) grid for the baseline random-placement MIPT.

Usage:
  python scripts/run_sweep.py --L 16 32 64 --shots 800 400 200 \
      --p 0.05 0.08 0.10 0.12 0.14 0.15 0.16 0.17 0.18 0.20 0.24 0.30 \
      --out results/baseline.json [--workers N] [--seed 7] [--T-mult 4]

Each (L, p) cell is an independent job; results append into one JSON file.
Safe to re-run with more Ls/ps: existing cells are kept unless --overwrite.
"""

import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mipt.sim import run_ensemble  # noqa: E402


def cell_key(L, p):
    return f"L{L}_p{p:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", required=True)
    ap.add_argument("--shots", type=int, nargs="+", required=True,
                    help="one value per L (same order)")
    ap.add_argument("--p", type=float, nargs="+", required=True)
    ap.add_argument("--T-mult", type=int, default=4)
    ap.add_argument("--mode", choices=["random", "sweep"], default="random",
                    help="measurement placement: iid random or lawnmower sweep")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    assert len(args.shots) == len(args.L)

    existing = {}
    if os.path.exists(args.out) and not args.overwrite:
        with open(args.out) as f:
            existing = {cell_key(c["L"], c["p"]): c for c in json.load(f)}

    jobs = []
    for L, shots in zip(args.L, args.shots):
        for p in args.p:
            if cell_key(L, p) in existing:
                continue
            # deterministic per-cell seed
            seed = args.seed * 1_000_003 + L * 1009 + int(round(p * 10_000))
            jobs.append((L, p, shots, args.T_mult, seed, args.mode))

    print(f"{len(jobs)} cells to run, {len(existing)} cached", flush=True)
    t0 = time.time()
    results = list(existing.values())
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_ensemble, L, p, shots, T_mult, seed, mode): (L, p)
                for (L, p, shots, T_mult, seed, mode) in jobs}
        for i, fut in enumerate(as_completed(futs)):
            c = fut.result()
            results.append(c)
            print(f"[{i+1}/{len(jobs)}] L={c['L']} p={c['p']:.3f} "
                  f"S={c['S_half_mean']:.3f}±{c['S_half_sem']:.3f} "
                  f"I3={c['I3_mean']:.3f}±{c['I3_sem']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    results.sort(key=lambda c: (c["L"], c["p"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out} ({len(results)} cells, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
