"""Core MIPT simulation: brickwork random Clifford circuits with rate-p projective
Z measurements, on a 1D chain with periodic boundary conditions.

Entanglement entropies are computed from the stabilizer tableau via GF(2) rank
(Fattal et al., quant-ph/0406168): for a pure stabilizer state on n qubits with
generator matrix G (n rows, symplectic binary columns), the entanglement entropy
of a region A is

    S_A = rank_GF2(G restricted to A) - |A|      (in bits, log base 2)

This module is deliberately framework-free (numpy + stim only) so the same code
runs locally, on HYAK, and inside an RL environment loop later (Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import stim

# ---------------------------------------------------------------------------
# GF(2) linear algebra
# ---------------------------------------------------------------------------


def gf2_rank(mat: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2).

    Rows are bit-packed into python ints for speed (fast for n <= few thousand).
    """
    rows = []
    ncols = mat.shape[1] if mat.ndim == 2 else 0
    if ncols == 0 or mat.shape[0] == 0:
        return 0
    # pack each row into an int
    for r in mat:
        v = 0
        for b in np.flatnonzero(r):
            v |= 1 << int(b)
        rows.append(v)
    rank = 0
    for _ in range(len(rows)):
        # find pivot: nonzero row with lowest set bit not yet eliminated
        pivot_row = None
        for i, v in enumerate(rows):
            if v:
                pivot_row = i
                break
        if pivot_row is None:
            break
        pivot = rows.pop(pivot_row)
        low = pivot & -pivot  # lowest set bit
        rows = [v ^ pivot if v & low else v for v in rows]
        rank += 1
    return rank


# ---------------------------------------------------------------------------
# Stabilizer -> binary matrix
# ---------------------------------------------------------------------------


def stabilizer_matrix(sim: stim.TableauSimulator, n: int) -> np.ndarray:
    """Return the (n x 2n) binary symplectic matrix [X | Z] of the current
    canonical stabilizer generators."""
    stabs = sim.canonical_stabilizers()
    mat = np.zeros((n, 2 * n), dtype=np.uint8)
    for i, ps in enumerate(stabs):
        xs, zs = ps.to_numpy()
        mat[i, :n] = xs
        mat[i, n:] = zs
    return mat


def region_entropy(mat: np.ndarray, region: np.ndarray, n: int) -> int:
    """Entanglement entropy (bits) of `region` (array of qubit indices) for the
    pure stabilizer state whose generator matrix is `mat` ([X|Z], n x 2n)."""
    cols = np.concatenate([region, region + n])
    sub = mat[:, cols]
    return gf2_rank(sub) - len(region)


# ---------------------------------------------------------------------------
# Random two-qubit Clifford pool
#
# stim.Tableau.random() is not seedable per-call, so for reproducible
# trajectories we pre-generate a pool of random 2-qubit Cliffords once and draw
# indices from our own seeded Generator. The 2-qubit Clifford group has 11,520
# elements; a pool of 4096 gives dense, unbiased-in-practice coverage. (If exact
# uniform seeded sampling ever matters, swap in a canonical-form sampler.)
# ---------------------------------------------------------------------------

_CLIFFORD_POOL_SIZE = 4096
_clifford_pool: list | None = None


def _get_clifford_pool() -> list:
    global _clifford_pool
    if _clifford_pool is None:
        _clifford_pool = [stim.Tableau.random(2) for _ in range(_CLIFFORD_POOL_SIZE)]
    return _clifford_pool


# ---------------------------------------------------------------------------
# Circuit dynamics
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryResult:
    half_cut_entropy: float
    i3: float
    n_measurements: int
    record: list = field(default_factory=list)  # (layer, site, outcome)


def run_trajectory(
    L: int,
    p: float,
    T: int,
    rng: np.random.Generator,
    keep_record: bool = False,
) -> TrajectoryResult:
    """One monitored-circuit trajectory.

    Brickwork of uniformly random 2-qubit Cliffords (PBC), followed each layer by
    projective Z measurement of each site independently with probability p.
    Runs T layers, then computes S(L/2) and the tripartite mutual information
    I3(A:B:C) for four contiguous quarters A,B,C,D.
    """
    assert L % 4 == 0, "L must be divisible by 4 for the I3 quartering"
    sim = stim.TableauSimulator(seed=int(rng.integers(2**63)))
    sim.set_num_qubits(L)
    pool = _get_clifford_pool()

    n_meas = 0
    record: list = []
    for t in range(T):
        # brickwork layer: even bonds on even t, odd bonds (incl. wrap) on odd t
        offset = t % 2
        for a in range(offset, L, 2):
            b = (a + 1) % L
            g = pool[int(rng.integers(len(pool)))]
            sim.do_tableau(g, [a, b])
        # measurement layer
        sites = np.flatnonzero(rng.random(L) < p)
        for q in sites:
            out = sim.measure(int(q))
            n_meas += 1
            if keep_record:
                record.append((t, int(q), int(out)))

    mat = stabilizer_matrix(sim, L)
    q = L // 4
    A = np.arange(0, q)
    B = np.arange(q, 2 * q)
    C = np.arange(2 * q, 3 * q)
    half = np.arange(0, L // 2)

    S = lambda reg: region_entropy(mat, reg, L)
    S_A, S_B, S_C = S(A), S(B), S(C)
    S_AB = S(np.concatenate([A, B]))
    S_AC = S(np.concatenate([A, C]))
    S_BC = S(np.concatenate([B, C]))
    S_ABC = S(np.concatenate([A, B, C]))
    i3 = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC

    return TrajectoryResult(
        half_cut_entropy=float(S(half)),
        i3=float(i3),
        n_measurements=n_meas,
        record=record,
    )


def run_ensemble(
    L: int,
    p: float,
    shots: int,
    T_mult: int = 4,
    seed: int = 0,
) -> dict:
    """Average `shots` trajectories at (L, p). T = T_mult * L layers."""
    rng = np.random.default_rng(seed)
    T = T_mult * L
    s_vals, i3_vals, nm_vals = [], [], []
    for _ in range(shots):
        res = run_trajectory(L, p, T, rng)
        s_vals.append(res.half_cut_entropy)
        i3_vals.append(res.i3)
        nm_vals.append(res.n_measurements)
    s = np.array(s_vals)
    i3 = np.array(i3_vals)
    return {
        "L": L,
        "p": p,
        "shots": shots,
        "T": T,
        "S_half_mean": float(s.mean()),
        "S_half_sem": float(s.std(ddof=1) / np.sqrt(shots)),
        "I3_mean": float(i3.mean()),
        "I3_sem": float(i3.std(ddof=1) / np.sqrt(shots)),
        "meas_per_site_layer": float(np.mean(nm_vals) / (L * T)),
    }
