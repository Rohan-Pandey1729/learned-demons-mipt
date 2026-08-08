# learned-demons-mipt

**Learned demons at the measurement transition** — reinforcement-learned adaptive
measurement policies in monitored Clifford circuits, and their
information-thermodynamic cost. Working repo for the paper (Rohan Pandey & Claude,
2026). See `PROPOSAL.md` for the research plan.

## Status

- [x] **Phase 0** — baseline pipeline: brickwork random-Clifford + rate-p Z
  measurements (PBC), stabilizer entanglement entropy via GF(2) rank, tripartite
  mutual information, I3-crossing estimate of p_c. **Result: p_c = 0.162 ± 0.007**
  from I3 crossings at L = 16/32/64 (literature: ≈ 0.16). Pipeline validated.
- [~] **Phase 1** — adaptive placement at matched budget (in progress; first
  results below). Next: PPO/attention policy class on HYAK, budget sweeps
  around p_c, does adaptivity shift the transition?
- [ ] **Phase 2** — demon ledger: record entropy vs entanglement steered, Landauer accounting.
- [ ] **Phase 3** — paper.

### Phase 1 first results (L = 24, k = 3 meas/layer ⇒ p_eff = 0.125 < p_c, T = 96)

Final half-cut entropy at exactly matched measurement budget:

| policy | info used | S(L/2) [bits] |
|---|---|---|
| random placement | none | 4.31 ± 0.08 |
| shuffled coverage (round-robin, random order) | none | 4.12 ± 0.08 |
| oracle greedy (max local entropy — quantum peek!) | quantum | 3.56 ± 0.09 |
| learned linear demon (CEM, classical record only) | classical | 3.29 ± 0.14 |
| **lawnmower sweep (contiguous blocks, in order)** | none | **1.25 ± 0.05** |

Two early findings worth keeping:

1. **Geometry beats coverage, and beats greed.** Equal-coverage round-robin in a
   shuffled order does no better than random, while the same coverage applied as
   a contiguous sequential sweep collapses the entropy by >3×. Even the
   *quantum-cheating* greedy policy loses badly to the dumb sweep: measuring
   where entanglement currently is, is short-sighted; systematically cutting the
   chain is not.
2. **The learned demon "fences."** The CEM-trained linear policy converges to
   pinning measurements at a few fixed sites, quarantining entanglement into a
   protected bubble (see `figures/fig4_spacetime.png`) — locally smart, globally
   suboptimal for compression. A policy class that can express *moving* patterns
   (PPO with recurrence/attention) is the obvious next step, with the lawnmower
   as the baseline to beat from the classical side and the oracle for scale.

## Layout

```
src/mipt/sim.py       core simulation + entropy machinery (numpy + stim only)
scripts/run_sweep.py  (L, p) grid runner, JSON output, resumable
scripts/plot_results.py  figures + p_c from pairwise I3 crossings
scripts/merge_results.py merge per-cell outputs from the SLURM array
slurm/sweep_array.slurm  HYAK (klone) array job for large-L sweeps
tests/test_sanity.py  physics sanity checks (Bell/GHZ entropies, p=0/p=1 limits)
results/, figures/    small local runs live here; big runs stay on HYAK
```

## Quickstart

```bash
pip install stim numpy matplotlib
python tests/test_sanity.py
python scripts/run_sweep.py --L 16 32 64 --shots 800 400 200 \
    --p 0.05 0.08 0.10 0.12 0.14 0.15 0.16 0.17 0.18 0.20 0.24 0.30 \
    --out results/baseline.json
python scripts/plot_results.py --in results/baseline.json --outdir figures
```

On HYAK: edit the account/partition in `slurm/sweep_array.slurm`, then
`sbatch slurm/sweep_array.slurm` and merge with `scripts/merge_results.py`.

## Physics conventions

- Chain of L qubits, periodic boundary conditions, L divisible by 4.
- Brickwork: uniformly random 2-qubit Cliffords; even bonds on even layers, odd
  bonds (with the wrap bond) on odd layers.
- Measurements: each site measured in Z independently with probability p after
  each gate layer; depth T = 4L before observables.
- Entropies in bits. S_A = rank_GF2(G|_A) − |A| (Fattal et al.).
- I3 = I3(A:B:C) for contiguous quarters A,B,C,D; curves for different L cross
  at p_c (Zabalo et al. convention).
- Reproducibility: all randomness flows from one seeded numpy Generator, except
  the 2-qubit Clifford *pool* (4096 gates pre-drawn from stim's unseeded sampler,
  then indexed by the seeded rng) — see note in `src/mipt/sim.py`.
