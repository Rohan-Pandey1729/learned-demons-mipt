#!/usr/bin/env python3
"""Measure the entanglement growth velocity v_E of the unitary (p = 0)
brickwork random Clifford circuit: S(L/2, t) grows linearly at rate v_E
(bits per layer) before saturating.

The sweep's ballistic argument predicts the collapse plateau
f_inf = p * S_sat to satisfy f_inf ~ v_E / 2 (a cut's time since the sweep
last passed is uniform on [0, 1/p], so the mean regrowth is v_E/(2p)).

Usage: python scripts/measure_velocity.py --L 256 --shots 20 \
    --out results/velocity.json
"""

import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import stim  # noqa: E402
from mipt.sim import _get_clifford_pool, stabilizer_matrix, region_entropy  # noqa: E402


def trajectory(L, T, rng, every=2):
    sim = stim.TableauSimulator(seed=int(rng.integers(2**63)))
    sim.set_num_qubits(L)
    pool = _get_clifford_pool()
    half = np.arange(L // 2)
    ts, Ss = [], []
    for t in range(T):
        offset = t % 2
        for a in range(offset, L, 2):
            b = (a + 1) % L
            sim.do_tableau(pool[int(rng.integers(len(pool)))], [a, b])
        if t % every == 0:
            mat = stabilizer_matrix(sim, L)
            ts.append(t + 1)
            Ss.append(region_entropy(mat, half, L))
    return np.array(ts), np.array(Ss, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--T", type=int, default=None,
                    help="layers (default 0.6*L, inside the linear regime)")
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--fit-lo", type=float, default=0.1,
                    help="fit window as fraction of max t")
    ap.add_argument("--fit-hi", type=float, default=0.9)
    ap.add_argument("--out", default="results/velocity.json")
    args = ap.parse_args()
    T = args.T or int(0.6 * args.L)

    rng = np.random.default_rng(args.seed)
    curves = []
    for s in range(args.shots):
        ts, Ss = trajectory(args.L, T, rng, args.every)
        curves.append(Ss)
        print(f"shot {s+1}/{args.shots}: S(t={ts[-1]}) = {Ss[-1]:.1f}", flush=True)
    S = np.mean(curves, axis=0)
    Se = np.std(curves, axis=0, ddof=1) / np.sqrt(args.shots)

    lo, hi = int(args.fit_lo * len(ts)), int(args.fit_hi * len(ts))
    A = np.vstack([ts[lo:hi], np.ones(hi - lo)]).T
    coef, res, *_ = np.linalg.lstsq(A, S[lo:hi], rcond=None)
    v = float(coef[0])
    # slope uncertainty from per-shot fits
    vs = []
    for c in curves:
        cf, *_ = np.linalg.lstsq(A, np.array(c)[lo:hi], rcond=None)
        vs.append(cf[0])
    v_err = float(np.std(vs, ddof=1) / np.sqrt(args.shots))

    out = {"L": args.L, "T": T, "shots": args.shots,
           "t": ts.tolist(), "S_mean": S.tolist(), "S_sem": Se.tolist(),
           "v_E": v, "v_E_err": v_err,
           "f_inf_predicted": v / 2,
           "note": "ballistic argument: f_inf ~ v_E/2; measured collapse "
                   "plateau f_inf ~ 0.46"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"v_E = {v:.4f} +/- {v_err:.4f} bits/layer  ->  v_E/2 = {v/2:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
