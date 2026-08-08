#!/usr/bin/env python3
"""Inspect a trained PPO demon: rollout spacetime pattern + eval vs baselines.

Usage: python scripts/inspect_ppo.py --ckpt results/ppo_L16b2_cloud.pt \
    --L 16 --budget 2 --T 64 --shots 100 --out figures/fig8_ppo_pattern.png
"""

import argparse, os, sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mipt.policies import evaluate  # noqa: E402
from train_ppo import DemonPolicy, sample_sites  # noqa: E402
from spacetime_fig import run_with_profiles  # noqa: E402


class PPOAdapter:
    """Wrap DemonPolicy in the .choose(obs, k) interface (stateful GRU)."""

    name = "ppo"

    def __init__(self, ckpt: str, greedy: bool = False, seed: int = 0):
        self.net = DemonPolicy()
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.net.eval()
        self.h = self.net.init_hidden(1)
        self.greedy = greedy
        torch.manual_seed(seed)

    def choose(self, obs: np.ndarray, k: int) -> np.ndarray:
        ot = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _, self.h = self.net(ot, self.h)
        if self.greedy:
            return torch.topk(logits[0], k).indices.numpy()
        sites, _ = sample_sites(logits[0], k)
        return np.array(sites)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--out", default="figures/fig8_ppo_pattern.png")
    args = ap.parse_args()

    # ensemble eval (stochastic + greedy)
    for greedy in (False, True):
        r = evaluate(lambda s: PPOAdapter(args.ckpt, greedy=greedy, seed=s),
                     args.L, args.budget, args.T, args.shots, seed=5)
        print(f"ppo greedy={greedy}: S_half = {r['S_half']:.3f} ± "
              f"{r['S_half_sem']:.3f}, record H = {r['record_entropy_exact']:.1f} bits")

    # spacetime pattern (one trajectory, stochastic)
    profs, events, final = run_with_profiles(
        PPOAdapter(args.ckpt, seed=1), args.L, args.budget, args.T, seed=7)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(profs, aspect="auto", origin="lower", cmap="magma",
                   vmin=0, interpolation="nearest",
                   extent=[0.5, args.L - 0.5, 0, len(profs)])
    ts = [t for (t, q) in events]; qs = [q + 0.5 for (t, q) in events]
    ax.scatter(qs, ts, s=6, c="#00b4d8", alpha=0.9, linewidths=0)
    ax.set_xlabel("cut position  x"); ax.set_ylabel("circuit layer  t")
    ax.set_title(f"PPO demon pattern (final S = {final['S_half']:.0f} bits)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="S([0,x))  [bits]")
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
