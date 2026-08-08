#!/usr/bin/env python3
"""Showcase figure: entanglement 'spacetime' S(x, t) heatmaps with measurement
events overlaid, comparing placement policies at matched budget.

For each layer t we compute S([0, x)) for every cut x — the entanglement profile
— giving a heatmap over (x, t). Measurement events are drawn on top. One panel
per policy. `--dark` renders the hero (dark-mode) version.

Usage:
  python scripts/spacetime_fig.py --L 24 --budget 3 --T 96 \
      --theta results/phase1_L24b3.json --out figures/spacetime.png [--dark]
"""

import argparse, json, os, sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mipt.env import AdaptiveMIPTEnv  # noqa: E402
from mipt.sim import stabilizer_matrix, region_entropy  # noqa: E402
from mipt.policies import (RandomPolicy, LinearPolicy, OracleGreedyPolicy,  # noqa: E402
                           StalestPolicy)


def profile(env: AdaptiveMIPTEnv) -> np.ndarray:
    mat = stabilizer_matrix(env.sim, env.L)
    return np.array([region_entropy(mat, np.arange(x), env.L)
                     for x in range(1, env.L)], dtype=float)


def run_with_profiles(policy, L, budget, T, seed, oracle=False):
    env = AdaptiveMIPTEnv(L, budget, T, seed=seed, oracle=oracle)
    if isinstance(policy, OracleGreedyPolicy):
        policy.env = env
    obs = env._obs()
    profiles, events = [profile(env)], []
    while not env.done:
        sites = policy.choose(obs, budget)
        events.extend((env.t, int(q)) for q in sites)
        obs, _ = env.step(sites)
        profiles.append(profile(env))
    return np.array(profiles), events, env.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--theta", type=str, required=True,
                    help="phase1 results JSON containing theta_final")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()

    with open(args.theta) as f:
        pdata = json.load(f)
    theta = np.array(pdata["theta_final"])
    ens = pdata.get("eval", {})
    ens_key = {"random": "random", "learned demon": "learned",
               "oracle greedy": "oracle-greedy", "lawnmower": "stalest-stable"}

    policies = [
        ("random", RandomPolicy(seed=1), False),
        ("learned demon", LinearPolicy(theta, seed=1), False),
        ("oracle greedy", OracleGreedyPolicy(None, seed=1), True),
        ("lawnmower", StalestPolicy(), False),
    ]

    if args.dark:
        plt.style.use("dark_background")
        surface, ink, event_c = "#0d1117", "#e6edf3", "#7ee8fa"
        cmap = "magma"
    else:
        surface, ink, event_c = "white", "#222222", "#00b4d8"
        cmap = "magma"

    fig, axes = plt.subplots(1, len(policies), figsize=(15.5, 4.2), sharey=True,
                             facecolor=surface)
    vmax = None
    runs = []
    for name, pol, oracle in policies:
        profs, events, final = run_with_profiles(
            pol, args.L, args.budget, args.T, seed=args.seed, oracle=oracle)
        runs.append((name, profs, events, final))
        vmax = max(vmax or 0, profs.max())

    for ax, (name, profs, events, final) in zip(axes, runs):
        im = ax.imshow(profs, aspect="auto", origin="lower", cmap=cmap,
                       vmin=0, vmax=vmax, interpolation="nearest",
                       extent=[0.5, args.L - 0.5, 0, len(profs)])
        if events:
            ts = [t for (t, q) in events]
            qs = [q + 0.5 for (t, q) in events]  # site q sits between cuts q, q+1
            ax.scatter(qs, ts, s=4, c=event_c, alpha=0.85, linewidths=0,
                       label="measurement")
        e = ens.get(ens_key.get(name, ""), None)
        sub = (rf"$\langle S(L/2)\rangle$ = {e['S_half']:.2f} $\pm$ {e['S_half_sem']:.2f} bits"
               if e else rf"final $S(L/2)$ = {final['S_half']:.0f} bits")
        ax.set_title(f"{name}\n{sub}", fontsize=10, color=ink)
        ax.set_xlabel("cut position  x", color=ink)
        ax.tick_params(colors=ink)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("circuit layer  t", color=ink)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("entanglement entropy  S([0, x))  [bits]", color=ink)
    cb.ax.tick_params(colors=ink)
    cb.outline.set_visible(False)
    fig.suptitle(
        f"Where the demon looks — measurement placement at matched budget "
        f"(L = {args.L}, k = {args.budget}/layer)",
        color=ink, fontsize=12, y=1.0)
    fig.savefig(args.out, facecolor=surface, dpi=180, bbox_inches="tight")
    print("wrote", args.out)
    for name, _, _, final in runs:
        print(f"  {name:14s} S_half = {final['S_half']}, "
              f"record H ≈ {final['record_entropy_naive']:.0f} bits")


if __name__ == "__main__":
    main()
