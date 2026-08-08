"""Physics sanity checks for the MIPT pipeline. Run: python tests/test_sanity.py"""

import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import stim
from mipt.sim import gf2_rank, region_entropy, stabilizer_matrix, run_trajectory, run_ensemble


def test_gf2_rank():
    assert gf2_rank(np.eye(4, dtype=np.uint8)) == 4
    assert gf2_rank(np.zeros((3, 5), dtype=np.uint8)) == 0
    m = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)  # row3 = row1+row2
    assert gf2_rank(m) == 2
    print("ok gf2_rank")


def test_bell_pair_entropy():
    sim = stim.TableauSimulator()
    sim.set_num_qubits(2)
    sim.h(0)
    sim.cnot(0, 1)
    mat = stabilizer_matrix(sim, 2)
    assert region_entropy(mat, np.array([0]), 2) == 1
    print("ok bell pair S_A = 1")


def test_product_state_entropy():
    sim = stim.TableauSimulator()
    sim.set_num_qubits(4)
    mat = stabilizer_matrix(sim, 4)
    assert region_entropy(mat, np.array([0, 1]), 4) == 0
    print("ok product state S_A = 0")


def test_ghz_entropy():
    sim = stim.TableauSimulator()
    sim.set_num_qubits(8)
    sim.h(0)
    for i in range(7):
        sim.cnot(i, i + 1)
    mat = stabilizer_matrix(sim, 8)
    # any bipartition of GHZ has S = 1
    assert region_entropy(mat, np.arange(4), 8) == 1
    assert region_entropy(mat, np.array([2]), 8) == 1
    print("ok GHZ S_A = 1")


def test_limits():
    rng = np.random.default_rng(0)
    # p = 0: unitary random Clifford -> near-maximal (volume-law) half-cut entropy
    res0 = run_trajectory(16, 0.0, 64, rng)
    assert res0.half_cut_entropy >= 6, res0.half_cut_entropy  # max is 8
    # p = 1: measure everything every layer -> product state, S = 0
    res1 = run_trajectory(16, 1.0, 64, rng)
    assert res1.half_cut_entropy == 0, res1.half_cut_entropy
    print(f"ok limits: S(p=0) = {res0.half_cut_entropy}, S(p=1) = {res1.half_cut_entropy}")


def test_speed():
    t0 = time.time()
    run_ensemble(64, 0.16, shots=5, T_mult=4, seed=1)
    dt = (time.time() - t0) / 5
    print(f"ok speed: L=64, T=256 trajectory takes {dt:.2f}s")


if __name__ == "__main__":
    test_gf2_rank()
    test_bell_pair_entropy()
    test_product_state_entropy()
    test_ghz_entropy()
    test_limits()
    test_speed()
    print("ALL SANITY CHECKS PASSED")
