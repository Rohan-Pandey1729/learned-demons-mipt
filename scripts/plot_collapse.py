#!/usr/bin/env python3
"""Data collapse test for the ballistic no-transition hypothesis under sweep
placement: S(L/2) = (1/p) * f(pL), with f(x) ~ a·x for small x (volume law at
sizes ≪ 1/p) and f(x) → const for large x (L-independent saturation).

Left: raw S vs L per p. Right: p·S vs p·L — collapse onto one curve supports
the claim that the only scale is 1/p (no critical point).

Usage: python scripts/plot_collapse.py --in results/lawnmower.json \
    --out figures/fig6_collapse.png
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/lawnmower.json")
    ap.add_argument("--out", default="figures/fig6_collapse.png")
    ap.add_argument("--p-min", type=float, default=0.0199,
                    help="drop p below this (transient-dominated at T=4L)")
    args = ap.parse_args()

    data = [c for c in json.load(open(args.inp)) if c["p"] >= args.p_min]
    ps = sorted({c["p"] for c in data})
    colors = {p: plt.cm.Oranges(x)
              for p, x in zip(ps, np.linspace(0.35, 0.95, len(ps)))}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8))

    for p in ps:
        cells = sorted([c for c in data if c["p"] == p], key=lambda c: c["L"])
        Ls = [c["L"] for c in cells]
        S = [c["S_half_mean"] for c in cells]
        E = [c["S_half_sem"] for c in cells]
        ax1.errorbar(Ls, S, yerr=E, color=colors[p], lw=1.6, marker="o", ms=4,
                     capsize=2, label=f"p = {p:g}")
        ax2.errorbar(np.array(Ls) * p, np.array(S) * p,
                     yerr=np.array(E) * p, color=colors[p], lw=1.6,
                     marker="o", ms=4, capsize=2, label=f"p = {p:g}")

    ax1.set_xlabel("system size  L"); ax1.set_xscale("log", base=2)
    ax1.set_ylabel(r"$\langle S(L/2)\rangle$  [bits]")
    ax1.set_title("Sweep placement: raw curves", fontsize=11)
    ax1.legend(fontsize=8, ncol=2)

    ax2.set_xlabel(r"$p\,L$"); ax2.set_xscale("log", base=2)
    ax2.set_ylabel(r"$p\,\langle S(L/2)\rangle$")
    ax2.set_title(r"Ballistic scaling test:  $S = \frac{1}{p} f(pL)$",
                  fontsize=11)
    ax2.legend(fontsize=8, ncol=2)

    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
