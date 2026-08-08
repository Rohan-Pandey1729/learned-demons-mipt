# Related-work map (prior-art check, Aug 2026)

**Question:** is the lawnmower result — deterministic sequential-sweep placement
destroying the volume-law phase at matched budget (crossings drift p→0, S
saturates ~O(1/p)) — already known?

**Verdict of the search: plausibly novel.** Nothing found studies a spatially
*sweeping* (time-advancing) measurement pattern in monitored circuits, and
nothing reports destruction of the volume-law phase from restructuring
placement at fixed rate. The structured-placement literature finds the
*opposite* (transition survives with modified criticality), which makes our
result a sharp counterpoint. One paper requires a careful full-text read before
we claim novelty in print (see ⚠ below).

## Closest prior work, in order of proximity

1. ⚠ **arXiv 2411.09784 — "Reinforced Disentanglers on Random Unitary
   Circuits"** (Bao, Furuya, Suer). PPO agent learns WHERE to place projective
   measurements to minimize final entropy with fewest projections; reports
   needing "drastically less" than random-placement thresholds. Different task
   (terminal disentangling, not steady-state phases; no I3/p_c(L) scaling), but
   closest in spirit. **TODO (Rohan): skim the figures — if their learned
   patterns are sweep-like, this is our closest prior art and must be framed
   accordingly.**
2. **arXiv 2308.03844** (Shkolnik et al., PRB 108, 184204) — measurement
   positions on a deterministic quasiperiodic sublattice, static in time:
   transition SURVIVES, new critical family (activated scaling, Luck bound).
3. **arXiv 2607.12386 — "Measurement-induced phase transition in space"** —
   deterministic spatial *gradient of the rate* p(x), placement still random;
   transition realized spatially, volume-law region survives. Cite and
   distinguish explicitly (title collision risk with referees).
4. **arXiv 2205.14002** (Zabalo et al., PRB 107, L220204) — static random
   disorder in measurement rates; infinite-randomness criticality; survives.
5. **arXiv 2304.12965** (PRX Quantum 5, 010309) + **2507.05055** — unitary
   circuit games (entangler vs disentangler); placement random, adaptivity in
   gate choice.
6. **Feedback/absorbing-state class** (Iadecola–Ganeshan–Pixley arXiv
   2211.09100 lineage; O'Dea et al.; Buchhold et al.; Piroli et al.) — in ALL,
   measurement locations are random; adaptivity = outcome-conditioned
   corrective unitaries. Our sweep protocol is outcome-INDEPENDENT — cleaner
   and different. Cite as a class.
7. Color: arXiv 2411.13667 (entanglement regrowth between detection events at a
   single monitored site — echoes our between-visits ballistic regrowth
   argument); arXiv 2412.01917 (temporal-only modulation).

## Framing implication for the paper

The lead claim is not "adaptivity helps" but: **at fixed measurement budget,
the phase diagram of monitored dynamics is a property of the placement
process, not just the rate** — with the sweep as the extreme case where the
volume-law phase vanishes entirely (p_c → 0 as L → ∞), supported by the
ballistic 1/p saturation argument and, contrastingly, the quasiperiodic/static
literature where structure only modifies criticality. The RL demon and the
exact record-entropy ledger then sit on top as the "what does steering cost"
story.

Canonical MIPT citations: Li–Chen–Fisher; Skinner–Ruhman–Nahum; Chan et al.;
Gullans–Huse; Zabalo et al. (I3 conventions); Fisher–Khemani–Nahum–Vijay
review; Fattal et al. (stabilizer entropies); Gidney (stim).
