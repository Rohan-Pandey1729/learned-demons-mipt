#!/usr/bin/env python3
"""Phase 2 opening figure: the demon's ledger.

Every stabilizer measurement outcome is deterministic (0 bits) or random
(exactly 1 bit), so the mean number of random outcomes IS the record's Shannon
entropy — the minimum memory the demon must hold (and, by Landauer, must pay
kT ln2 per bit to erase).

Left: information efficiency — final S(L/2) vs record entropy per site-layer,
parametric in p, for random vs sweep placement at L = 64. Down-and-left is
better (less entanglement for fewer recorded bits).

Right: the 'price of a look' — record entropy per measurement vs p. In the
volume-law phase nearly every look costs a full bit; structured placement
changes what a look is worth.

Usage: python scripts/plot_ledger.py --random results/random_ledger.json \
    --sweep results/lawnmower.json --L 64 --out figures/fig7_ledger.png
"""

import argparse, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def series(data, L):
    cells = sorted([c for c in data if c["L"] == L and "record_entropy_mean" in c],
                   key=lambda c: c["p"])
    p = np.array([c["p"] for c in cells])
    S = np.array([c["S_half_mean"] for c in cells])
    Se = np.array([c["S_half_sem"] for c in cells])
    H = np.array([c["record_entropy_mean"] for c in cells])
    He = np.array([c["record_entropy_sem"] for c in cells])
    T = np.array([c["T"] for c in cells])
    nmeas = np.array([c["meas_per_site_layer"] for c in cells]) * L * T
    return p, S, Se, H, He, T, nmeas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", default="results/random_ledger.json")
    ap.add_argument("--sweep", default="results/lawnmower.json")
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--out", default="figures/fig7_ledger.png")
    args = ap.parse_args()

    rnd = series(json.load(open(args.random)), args.L)
    swp = series(json.load(open(args.sweep)), args.L)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8))

    for (p, S, Se, H, He, T, nm), color, name, ls in [
            (rnd, plt.cm.Blues(0.8), "random", "-"),
            (swp, plt.cm.Oranges(0.8), "sweep", "--")]:
        x = H / (args.L * T)  # recorded bits per site-layer
        ax1.errorbar(x, S, yerr=Se, color=color, lw=1.8, ls=ls, marker="o",
                     ms=4, capsize=2, label=name)
        for pi, xi, si in zip(p, x, S):
            if pi in (0.02, 0.05, 0.1, 0.16, 0.3):
                ax1.annotate(f"{pi:g}", (xi, si), textcoords="offset points",
                             xytext=(5, 4), fontsize=7.5, color=color)
        ax2.errorbar(p, H / np.maximum(nm, 1), yerr=He / np.maximum(nm, 1),
                     color=color, lw=1.8, ls=ls, marker="o", ms=4, capsize=2,
                     label=name)

    ax1.set_xlabel("record entropy per site-layer  [bits]")
    ax1.set_ylabel(r"$\langle S(L/2)\rangle$  [bits]")
    ax1.set_title(f"Information efficiency frontier (L = {args.L})", fontsize=11)
    ax1.legend()

    ax2.set_xlabel("measurement rate  p")
    ax2.set_ylabel("record entropy per measurement  [bits]")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax2.set_title("The price of a look", fontsize=11)
    ax2.legend()

    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
