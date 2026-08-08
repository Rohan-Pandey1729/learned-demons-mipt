#!/usr/bin/env python3
"""Plot baseline MIPT sweep: S(L/2) vs p and I3 vs p, and estimate p_c from
pairwise I3 curve crossings (linear interpolation between grid points).

Usage: python scripts/plot_results.py --in results/baseline.json --outdir figures
"""

import argparse, json, itertools
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- style: recessive axes/grid, sequential single-hue ramp over L (magnitude) ---
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def ramp(n):
    """Sequential blue ramp, light -> dark with increasing L."""
    return [plt.cm.Blues(x) for x in np.linspace(0.45, 0.95, n)]


def crossings(data, key="I3_mean"):
    """Pairwise crossing points of I3(p) curves for consecutive L, by linear
    interpolation of the difference on the shared p grid."""
    Ls = sorted({c["L"] for c in data})
    out = []
    for L1, L2 in itertools.combinations(Ls, 2):
        c1 = sorted([c for c in data if c["L"] == L1], key=lambda c: c["p"])
        c2 = sorted([c for c in data if c["L"] == L2], key=lambda c: c["p"])
        ps = np.array([c["p"] for c in c1])
        d = np.array([a[key] for a in c1]) - np.array([b[key] for b in c2])
        for i in range(len(ps) - 1):
            if d[i] == 0:
                out.append((L1, L2, ps[i])); continue
            if d[i] * d[i + 1] < 0:
                t = d[i] / (d[i] - d[i + 1])
                out.append((L1, L2, ps[i] + t * (ps[i + 1] - ps[i])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/baseline.json")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    with open(args.inp) as f:
        data = json.load(f)
    Ls = sorted({c["L"] for c in data})
    colors = dict(zip(Ls, ramp(len(Ls))))

    # Figure 1: half-cut entropy vs p
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for L in Ls:
        cells = sorted([c for c in data if c["L"] == L], key=lambda c: c["p"])
        ps = [c["p"] for c in cells]
        ax.errorbar(ps, [c["S_half_mean"] for c in cells],
                    yerr=[c["S_half_sem"] for c in cells],
                    color=colors[L], lw=2, marker="o", ms=4, capsize=2, label=f"L = {L}")
    ax.set_xlabel("measurement rate  p")
    ax.set_ylabel(r"$\langle S(L/2)\rangle$  [bits]")
    ax.set_title("Half-cut entanglement entropy, random placement", fontsize=11)
    ax.legend()
    fig.savefig(f"{args.outdir}/fig1_entropy_vs_p.png")

    # Figure 2: I3 vs p with crossing region
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for L in Ls:
        cells = sorted([c for c in data if c["L"] == L], key=lambda c: c["p"])
        ps = [c["p"] for c in cells]
        ax.errorbar(ps, [c["I3_mean"] for c in cells],
                    yerr=[c["I3_sem"] for c in cells],
                    color=colors[L], lw=2, marker="o", ms=4, capsize=2, label=f"L = {L}")
    xs = crossings(data)
    if xs:
        pc = np.mean([x[2] for x in xs]); spread = np.std([x[2] for x in xs])
        ax.axvline(pc, color="0.4", ls="--", lw=1)
        ax.annotate(rf"$p_c \approx {pc:.3f} \pm {spread:.3f}$",
                    xy=(pc, ax.get_ylim()[0]), xytext=(6, 8),
                    textcoords="offset points", fontsize=10, color="0.25")
        print("pairwise crossings:", [(a, b, round(p, 4)) for a, b, p in xs])
        print(f"p_c = {pc:.4f} +/- {spread:.4f} (crossing spread)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("measurement rate  p")
    ax.set_ylabel(r"$I_3(A{:}B{:}C)$  [bits]")
    ax.set_title("Tripartite mutual information, contiguous quarters (PBC)", fontsize=11)
    ax.legend()
    fig.savefig(f"{args.outdir}/fig2_I3_crossing.png")

    # Figure 3: entropy scaling at extremes (volume vs area law)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for p_sel, marker in [(min(c["p"] for c in data), "o"), (max(c["p"] for c in data), "s")]:
        cells = sorted([c for c in data if abs(c["p"] - p_sel) < 1e-9], key=lambda c: c["L"])
        ax.errorbar([c["L"] for c in cells], [c["S_half_mean"] for c in cells],
                    yerr=[c["S_half_sem"] for c in cells], color="#1f5c99" if marker == "o" else "#7fa8cc",
                    lw=2, marker=marker, ms=5, capsize=2, label=f"p = {p_sel:.2f}")
    ax.set_xlabel("system size  L")
    ax.set_ylabel(r"$\langle S(L/2)\rangle$  [bits]")
    ax.set_title("Volume law vs area law", fontsize=11)
    ax.legend()
    fig.savefig(f"{args.outdir}/fig3_scaling.png")
    print(f"figures written to {args.outdir}/")


if __name__ == "__main__":
    main()
