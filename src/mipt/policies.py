"""Measurement-placement policies: baselines + a small learnable linear policy.

All policies choose exactly `k` distinct sites per layer given the classical
observation matrix (L x N_FEATURES) — except OracleGreedy, which is allowed to
peek at per-site local entropies as an upper-bound reference.
"""

from __future__ import annotations

import numpy as np

from .env import AdaptiveMIPTEnv, N_FEATURES


class RandomPolicy:
    """Uniformly random distinct sites — the exact budget-matched analogue of
    the standard rate-p model."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def choose(self, obs: np.ndarray, k: int) -> np.ndarray:
        return self.rng.choice(len(obs), size=k, replace=False)


class PeriodicPolicy:
    """Sweep the chain left-to-right in blocks of k (a 'lawnmower')."""

    name = "periodic"

    def __init__(self):
        self.ptr = 0

    def choose(self, obs: np.ndarray, k: int) -> np.ndarray:
        L = len(obs)
        sites = (self.ptr + np.arange(k)) % L
        self.ptr = (self.ptr + k) % L
        return sites


class StalestPolicy:
    """Measure the k sites least recently measured (classical, hand-designed).

    Stable sort → with an initially unmeasured chain this reduces to an ordered
    lawnmower (contiguous blocks, sequential). See ShuffledCoveragePolicy for
    the same *coverage* without the spatial structure.
    """

    name = "stalest"

    def choose(self, obs: np.ndarray, k: int) -> np.ndarray:
        # feature 0 = time since last measured; stable sort for determinism
        return np.argsort(-obs[:, 0], kind="stable")[:k]


class ShuffledCoveragePolicy:
    """Round-robin coverage in a random (per-sweep) site order: every site is
    measured exactly once per L/k layers, like the lawnmower, but with NO
    spatial contiguity or sequential structure. Isolates 'coverage' from
    'geometry' as the cause of any adaptive advantage."""

    name = "shuffled-coverage"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.order: np.ndarray | None = None
        self.ptr = 0

    def choose(self, obs: np.ndarray, k: int) -> np.ndarray:
        L = len(obs)
        if self.order is None or self.ptr + k > L:
            self.order = self.rng.permutation(L)
            self.ptr = 0
        sites = self.order[self.ptr:self.ptr + k]
        self.ptr += k
        return sites


class OracleGreedyPolicy:
    """Measure the k sites with the largest local entanglement entropy.

    Uses quantum-side information (env.oracle_local_entropy) — NOT a legal
    demon; serves as an upper-bound reference for how much adaptivity could
    possibly buy with full state knowledge at this action granularity.
    """

    name = "oracle-greedy"

    def __init__(self, env: AdaptiveMIPTEnv, seed: int = 0):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def choose(self, obs: np.ndarray, k: int) -> np.ndarray:
        s = self.env.oracle_local_entropy()
        s = s + 1e-9 * self.rng.random(len(s))  # random tie-break
        return np.argsort(-s)[:k]


class LinearPolicy:
    """Score each site with a linear function of its features (+ bias over the
    global mean feature vector); measure the top-k. Trainable by CEM/ES —
    ~2*N_FEATURES parameters, so it trains in minutes on a laptop."""

    name = "linear"

    def __init__(self, theta: np.ndarray | None = None, seed: int = 0):
        self.n_params = 2 * N_FEATURES
        self.theta = np.zeros(self.n_params) if theta is None else np.asarray(theta)
        self.rng = np.random.default_rng(seed)

    def choose(self, obs: np.ndarray, k: int) -> np.ndarray:
        w, v = self.theta[:N_FEATURES], self.theta[N_FEATURES:]
        scores = obs @ w + (obs.mean(axis=0) @ v)  # global term = shared bias
        scores = scores + 1e-9 * self.rng.random(len(scores))
        return np.argsort(-scores)[:k]


def rollout(policy, L: int, budget: int, T: int, seed: int,
            oracle: bool = False) -> dict:
    env = AdaptiveMIPTEnv(L, budget, T, seed=seed, oracle=oracle)
    if isinstance(policy, OracleGreedyPolicy):
        policy.env = env
    obs = env._obs()  # env is fresh from __init__ (reset already ran)
    while not env.done:
        obs, _ = env.step(policy.choose(obs, budget))
    return env.finish()


def evaluate(policy_factory, L: int, budget: int, T: int, shots: int,
             seed: int = 0, oracle: bool = False) -> dict:
    """Average `shots` rollouts; policy_factory(seed) -> fresh policy."""
    res = []
    for s in range(shots):
        pol = policy_factory(seed * 100_003 + s)
        res.append(rollout(pol, L, budget, T, seed=seed * 999_331 + s,
                           oracle=oracle))
    keys = ["S_half", "I3", "record_entropy_naive", "record_entropy_exact"]
    out = {k: float(np.mean([r[k] for r in res])) for k in keys}
    out.update({k + "_sem": float(np.std([r[k] for r in res], ddof=1)
                                  / np.sqrt(shots)) for k in keys})
    out["shots"] = shots
    return out
