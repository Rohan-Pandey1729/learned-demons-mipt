#!/usr/bin/env python3
"""Train the LinearPolicy with the cross-entropy method (CEM) to MINIMIZE final
half-cut entanglement entropy at fixed measurement budget, and compare against
baselines at exactly matched budget.

This is the laptop-scale Phase 1 driver; a PPO/torch version for HYAK (same env,
same features, richer policy class) is the next step — see README roadmap.

Usage:
  python scripts/train_cem.py --L 16 --budget 3 --T 64 --iters 12 \
      --pop 32 --elite 8 --shots-eval 200 --out results/phase1_L16.json
"""

import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mipt.policies import (LinearPolicy, RandomPolicy, PeriodicPolicy,  # noqa: E402
                           StalestPolicy, OracleGreedyPolicy, rollout, evaluate)


def fitness(args):
    theta, L, budget, T, shots, seed = args
    vals = []
    for s in range(shots):
        pol = LinearPolicy(theta, seed=seed + s)
        r = rollout(pol, L, budget, T, seed=seed * 77_003 + s)
        vals.append(r["S_half"])
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--shots-train", type=int, default=24)
    ap.add_argument("--shots-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_params = LinearPolicy().n_params
    mu, sigma = np.zeros(n_params), np.ones(n_params)
    history = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for it in range(args.iters):
            thetas = mu + sigma * rng.standard_normal((args.pop, n_params))
            jobs = [(thetas[i], args.L, args.budget, args.T, args.shots_train,
                     1000 * args.seed + 137 * it + i) for i in range(args.pop)]
            scores = list(ex.map(fitness, jobs))
            order = np.argsort(scores)  # minimizing S_half
            elite = thetas[order[: args.elite]]
            mu = elite.mean(axis=0)
            sigma = elite.std(axis=0) + 0.05  # noise floor
            best = scores[order[0]]
            history.append({"iter": it, "best": best,
                            "mean": float(np.mean(scores)),
                            "mu": mu.tolist()})
            print(f"iter {it:2d}  best S_half = {best:.3f}  "
                  f"pop mean = {np.mean(scores):.3f}", flush=True)

    # ---- final evaluation at matched budget --------------------------------
    L, budget, T, shots = args.L, args.budget, args.T, args.shots_eval
    results = {"config": vars(args), "history": history,
               "theta_final": mu.tolist(), "eval": {}}

    evals = {
        "random": lambda s: RandomPolicy(seed=s),
        "periodic": lambda s: PeriodicPolicy(),
        "stalest": lambda s: StalestPolicy(),
        "learned": lambda s: LinearPolicy(np.array(mu), seed=s),
        "oracle-greedy": lambda s: OracleGreedyPolicy(None, seed=s),
    }
    for name, factory in evals.items():
        r = evaluate(factory, L, budget, T, shots, seed=args.seed + 7,
                     oracle=(name == "oracle-greedy"))
        results["eval"][name] = r
        print(f"{name:14s} S_half = {r['S_half']:.3f} ± {r['S_half_sem']:.3f}   "
              f"record H ≈ {r['record_entropy_naive']:.1f} bits", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
