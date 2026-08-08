#!/usr/bin/env python3
"""Merge per-cell JSON files (from the SLURM array) into one results file.

Usage: python scripts/merge_results.py results/cells/*.json --out results/baseline_hyak.json
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", required=True)
args = ap.parse_args()

cells = []
for fn in args.files:
    with open(fn) as f:
        data = json.load(f)
    cells.extend(data if isinstance(data, list) else [data])

# dedupe on (L, p), last write wins
seen = {}
for c in cells:
    seen[(c["L"], round(c["p"], 6))] = c
merged = sorted(seen.values(), key=lambda c: (c["L"], c["p"]))
with open(args.out, "w") as f:
    json.dump(merged, f, indent=1)
print(f"merged {len(args.files)} files -> {args.out} ({len(merged)} cells)")
