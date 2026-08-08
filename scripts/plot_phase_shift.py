#!/usr/bin/env python3
"""Figure: how measurement-placement geometry moves the entanglement transition.

Left: I3 vs p for random vs lawnmower placement (same L set, same budget in
expectation). Right: S(L/2) vs L at fixed p in both modes, showing volume law
surviving under random placement while the sweep collapses it.

Usage: python scripts/plot_phase_shift.py --random results/baseline.json \
    --sweep results/lawnmower.json --out figures/fig5_phase_shift.png
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

BLUES = lambda n: [plt.cm.Blues(x) for x in np.linspace(0.45, 0.95, n)]
ORANGES = lambda n: [plt.cm.Oranges(x) for x in np.linspace(0.45, 0.95, n)]


def curves(data, L):
    cells = sorted([c for c in data if c["L"] == L], key=lambda c: c["p"])
    return ([c["p"] for c in cells], [c["I3_mean"] for c in cells],
            [c["I3_sem"] for c in cells], [c["S_half_mean"] for c in cells],
            [c["S_half_sem"] for c in cells])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", default="results/baseline.json")
    ap.add_argument("--sweep", default="results/lawnmower.json")
    ap.add_argument("--out", default="figures/fig5_phase_shift.png")
    ap.add_argument("--p-slice", type=float, default=0.10,
                    help="fixed p for the S vs L panel")
    args = ap.parse_args()

    rnd = json.load(open(args.random))
    swp = json.load(open(args.sweep))
    Ls = sorted({c["L"] for c in rnd} & {c["L"] for c in swp})
    cr, cs = dict(zip(Ls, BLUES(len(Ls)))), dict(zip(Ls, ORANGES(len(Ls))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8))

    for L in Ls:
        p, i3, i3e, *_ = curves(rnd, L)
        ax1.errorbar(p, i3, yerr=i3e, color=cr[L], lw=1.8, marker="o", ms=3.5,
                     capsize=2, label=f"random, L={L}")
        p, i3, i3e, *_ = curves(swp, L)
        ax1.errorbar(p, i3, yerr=i3e, color=cs[L], lw=1.8, marker="s", ms=3.5,
                     capsize=2, ls="--", label=f"sweep, L={L}")
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.set_xlabel("measurement rate  p")
    ax1.set_ylabel(r"$I_3$  [bits]")
    ax1.set_ylim(-6, 0.4)
    ax1.set_title("Same budget, different geometry", fontsize=11)
    ax1.legend(fontsize=7.5, ncol=2)

    # S vs L at fixed p
    for data, cmap, name, ls in [(rnd, plt.cm.Blues(0.8), "random", "-"),
                                 (swp, plt.cm.Oranges(0.8), "sweep", "--")]:
        Ls_all = sorted({c["L"] for c in data})
        xs, ys, es = [], [], []
        for L in Ls_all:
            cells = [c for c in data if c["L"] == L
                     and abs(c["p"] - args.p_slice) < 1e-9]
            if cells:
                xs.append(L); ys.append(cells[0]["S_half_mean"])
                es.append(cells[0]["S_half_sem"])
        ax2.errorbar(xs, ys, yerr=es, color=cmap, lw=2, ls=ls, marker="o",
                     ms=5, capsize=2, label=name)
    ax2.set_xlabel("system size  L")
    ax2.set_ylabel(r"$\langle S(L/2)\rangle$  [bits]")
    ax2.set_xscale("log", base=2)
    ax2.set_title(f"S vs L at p = {args.p_slice} (below random-placement $p_c$)",
                  fontsize=11)
    ax2.legend()

    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
