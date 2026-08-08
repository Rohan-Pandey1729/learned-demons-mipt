"""Phase 1: adaptive-measurement environment.

A monitored brickwork Clifford circuit where, instead of measuring each site
independently at rate p, an *agent* allocates a fixed per-layer measurement
budget k = round(p * L) to sites of its choosing. Comparisons against the
random-placement baseline are therefore at exactly matched budget.

Observation design (the demon constraint): the agent sees only *classical*
information derivable from the measurement record — no peeking at the quantum
state. Per site features:

  0. time since this site was last measured (normalized by L)
  1. last measurement outcome at this site (+1/-1, 0 if never measured)
  2. fraction of past measurements at this site returning +1
  3. time since either neighbor was measured (normalized, PBC)
  4. current layer parity (broadcast)

An *oracle* flag exposes cheap quantum info (per-site local entropy) for
upper-bound heuristics only — clearly marked, never for the demon itself.

API is deliberately gym-like but dependency-free:

    env = AdaptiveMIPTEnv(L=16, budget=3, T=64, seed=0)
    obs = env.reset()
    while not env.done:
        obs, info = env.step(policy.choose(obs, env.budget))
    final = env.finish()   # -> dict with S_half, I3, record entropy, ...
"""

from __future__ import annotations

import numpy as np
import stim

from .sim import _get_clifford_pool, stabilizer_matrix, region_entropy

N_FEATURES = 5


class AdaptiveMIPTEnv:
    def __init__(self, L: int, budget: int, T: int, seed: int = 0,
                 oracle: bool = False):
        assert L % 4 == 0
        self.L, self.budget, self.T, self.oracle = L, budget, T, oracle
        self.rng = np.random.default_rng(seed)
        self.pool = _get_clifford_pool()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.sim = stim.TableauSimulator(seed=int(self.rng.integers(2**63)))
        self.sim.set_num_qubits(self.L)
        self.t = 0
        self.done = False
        self.last_meas_time = np.full(self.L, -1, dtype=int)
        self.last_outcome = np.zeros(self.L, dtype=int)      # +1/-1/0
        self.ones_count = np.zeros(self.L, dtype=int)
        self.meas_count = np.zeros(self.L, dtype=int)
        self.record: list[tuple[int, int, int]] = []
        self._apply_gate_layer()  # agent first acts after seeing one layer
        return self._obs()

    # ------------------------------------------------------------------
    def _apply_gate_layer(self):
        offset = self.t % 2
        for a in range(offset, self.L, 2):
            b = (a + 1) % self.L
            g = self.pool[int(self.rng.integers(len(self.pool)))]
            self.sim.do_tableau(g, [a, b])

    def _obs(self) -> np.ndarray:
        L, t = self.L, self.t
        f = np.zeros((L, N_FEATURES))
        never = self.last_meas_time < 0
        f[:, 0] = np.where(never, t + 1, t - self.last_meas_time) / L
        f[:, 1] = self.last_outcome
        with np.errstate(invalid="ignore"):
            frac = np.where(self.meas_count > 0,
                            self.ones_count / np.maximum(self.meas_count, 1), 0.5)
        f[:, 2] = frac
        left = np.roll(self.last_meas_time, 1)
        right = np.roll(self.last_meas_time, -1)
        nb = np.maximum(left, right)
        f[:, 3] = np.where(nb < 0, t + 1, t - nb) / L
        f[:, 4] = t % 2
        return f

    def oracle_local_entropy(self) -> np.ndarray:
        """Per-site entanglement entropy S({i}) — quantum info, for oracle
        baselines only."""
        assert self.oracle, "enable oracle=True to use quantum-side features"
        mat = stabilizer_matrix(self.sim, self.L)
        return np.array([region_entropy(mat, np.array([i]), self.L)
                         for i in range(self.L)], dtype=float)

    # ------------------------------------------------------------------
    def step(self, sites) -> tuple[np.ndarray, dict]:
        """Measure the chosen sites (must be <= budget, unique), then apply the
        next gate layer. Returns (next_obs, info)."""
        assert not self.done
        sites = np.unique(np.asarray(sites, dtype=int))
        assert len(sites) <= self.budget, f"budget {self.budget} exceeded"
        for q in sites:
            out = self.sim.measure(int(q))
            self.record.append((self.t, int(q), int(out)))
            self.last_meas_time[q] = self.t
            self.last_outcome[q] = 1 if out else -1
            self.ones_count[q] += int(out)
            self.meas_count[q] += 1
        self.t += 1
        if self.t >= self.T:
            self.done = True
            return self._obs(), {"t": self.t, "done": True}
        self._apply_gate_layer()
        return self._obs(), {"t": self.t, "done": False}

    # ------------------------------------------------------------------
    def finish(self) -> dict:
        """Final observables + the demon-side ledger."""
        L = self.L
        mat = stabilizer_matrix(self.sim, L)
        q = L // 4
        A, B, C = (np.arange(0, q), np.arange(q, 2 * q), np.arange(2 * q, 3 * q))
        S = lambda reg: region_entropy(mat, reg, L)
        i3 = (S(A) + S(B) + S(C) - S(np.concatenate([A, B]))
              - S(np.concatenate([A, C])) - S(np.concatenate([B, C]))
              + S(np.concatenate([A, B, C])))
        outcomes = np.array([o for (_, _, o) in self.record], dtype=float)
        # naive record entropy: sum of per-bit binary entropies with empirical
        # bias per site (upper bound on the true record entropy; Phase 2 will
        # refine with conditional/compression estimates)
        rec_H = 0.0
        for i in range(L):
            n = self.meas_count[i]
            if n == 0:
                continue
            f = self.ones_count[i] / n
            if 0 < f < 1:
                rec_H += n * (-f * np.log2(f) - (1 - f) * np.log2(1 - f))
        return {
            "S_half": float(S(np.arange(L // 2))),
            "I3": float(i3),
            "n_meas": len(self.record),
            "record_entropy_naive": float(rec_H),
            "record_len": len(outcomes),
        }
