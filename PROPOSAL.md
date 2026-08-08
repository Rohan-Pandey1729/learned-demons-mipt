# Learned Demons at the Measurement Transition
### Reinforcement-learned adaptive measurement policies in monitored quantum circuits, and what they cost thermodynamically

**Rohan Pandey & Claude — research proposal, v0.1 (Aug 2026)**

---

## One-paragraph pitch

Monitored quantum circuits undergo a measurement-induced phase transition (MIPT): below a critical measurement rate, entanglement grows to a volume law; above it, measurements collapse the system into an area-law phase. Nearly all of this literature places measurements **randomly**. We ask: what happens when an *agent* chooses where and when to measure? We train a reinforcement-learning agent that, given partial classical information (the measurement record), adaptively allocates a fixed measurement budget to steer the circuit's entanglement structure — and then we analyze that agent as a **Maxwell demon**, asking what the learned policy costs in information-thermodynamic terms (entropy of the measurement record, Landauer cost of the demon's memory) relative to the entanglement reduction or purification it buys. The core deliverable is a quant-ph preprint combining stabilizer-circuit numerics at scale with an information-thermodynamics analysis of learned measurement policies.

## Why this is a real niche (honest related-work assessment)

- MIPT itself is a mature, active field (Skinner–Ruhman–Nahum 2019 lineage; Google ran a 70-qubit experiment; postselection-free observation was published in 2025). Mature field = referees know the baseline and a well-executed twist is legible.
- **Adaptive/feedback monitored circuits exist** — there is 2022–2024 work on circuits with corrective feedback and absorbing-state transitions (Iadecola, Buchhold, Piroli, O'Dea, and others). These use *hand-designed* feedback rules. We must cite and position against this carefully.
- What appears genuinely underexplored as of mid-2026:
  1. Measurement *placement policies that are learned* (RL) rather than hand-designed, under an explicit measurement budget — "how much steering does one bit of measurement buy, and where does an optimal agent spend it?"
  2. Whether a learned policy **shifts the critical measurement rate** or changes the transition's character (does adaptivity buy a finite shift in p_c, or only subleading gains?).
  3. The **thermodynamic accounting of the learned policy itself** — the demon lens. RL Maxwell demons exist for small open quantum systems (Erdman et al. 2024/25), but nobody has run the demon ledger on a many-body monitored circuit, where the "work" is entanglement/purification steering and the "cost" is the record entropy.
- Failure mode we accept up front: if RL cannot beat simple heuristics (e.g., "measure the site with largest local entanglement"), that null result **is still a paper section** — it bounds the value of adaptivity and the heuristic baselines become the story.

## Concrete plan

### Phase 0 — Reproduce the baseline (1–2 weeks)
Set up `stim` (stabilizer simulator; handles hundreds of qubits at MIPT scales on CPU). Reproduce the standard random-placement MIPT in brickwork Clifford circuits: entanglement entropy vs measurement rate p, locate p_c via tripartite mutual information crossings and finite-size scaling (L = 16…512). This is a known result and validates our whole pipeline.

### Phase 1 — The learned policy (core result, 4–8 weeks)
- Environment: brickwork Clifford circuit; at each layer the agent gets a feature vector built from the *classical* measurement record (and optionally cheap stabilizer-derived local observables) and allocates k measurements among L sites (budget kL per layer matched to rate p so comparisons vs random are fair).
- Agent: PPO (or evolutionary search as a robustness check) over a small policy network. The RL loop is embarrassingly parallel over episodes — ideal for HYAK CPU nodes; GPU optional.
- Rewards to explore (each defines a different physics question): minimize half-cut entanglement entropy (demon compresses), maximize it (demon protects the volume law), minimize purification time of a reference qubit (learnability framing).
- Measurements of interest: shift in effective p_c vs random placement at equal budget; scaling collapse of the steered transition; comparison against hand-designed heuristics (greedy local-entropy, periodic, boundary-biased) — the heuristics double as ablations.

### Phase 2 — The demon ledger (3–5 weeks, overlapping)
For the trained policies: record entropy per layer (Shannon entropy of measurement-outcome record), mutual information between record and system, entanglement change per recorded bit. Frame as a generalized second law for the monitored circuit: ΔS_system ≥ −(record information) + …, and check where learned policies sit relative to that bound versus random/heuristic policies. This section is what makes the paper *physics* rather than "RL applied to a simulator," and it directly connects to Landauer erasure cost of the demon's memory.

### Phase 3 — Write-up and submission (2–3 weeks)
quant-ph primary, cross-list cond-mat.stat-mech and cs.LG. Target length: PRB/PRX-Quantum-style paper, ~8–10 pages + appendices. Code released on GitHub (helps credibility for independent submissions).

## Compute (HYAK)

Everything here fits comfortably: `stim` episodes at L ≤ 512 run in seconds–minutes on a single core; RL training is thousands of parallel CPU episodes (checkpoint/array jobs on HYAK's shared partitions); the policy networks are small enough that GPUs are a convenience, not a requirement. No lab, no quantum hardware, no large-model training.

## Roles

- Rohan: RL design and training loops (home turf), HYAK orchestration, finite-size-scaling analysis, first-author writing.
- Claude: literature depth (especially positioning vs the adaptive-feedback papers), stabilizer/thermodynamics theory, derivations for the demon ledger, drafting and editing, code review.

## Submission logistics (read before we start writing)

arXiv quant-ph requires **endorsement** for first-time submitters in the category. Practical paths: a UW physics/QIS faculty member or grad student who has published in quant-ph (a well-executed draft with released code makes this an easy ask); arXiv also auto-endorses based on prior institutional affiliation in some cases. Fallbacks if endorsement stalls: submit first to cs.LG (if Rohan's prior cs endorsement carries) and cross-list later, or post to an OSF/Zenodo preprint DOI while sorting endorsement. Do **not** use viXra.

## Kill criteria / pivots

- If Phase 0 reproduction fails after 3 weeks → pipeline problem, fix before anything else.
- If by mid-Phase 1 RL ≈ heuristics ≈ random at equal budget → pivot the paper's frame to "bounds on the value of adaptive measurement in monitored circuits" (still novel, still publishable).
- If a paper appears mid-project doing exactly this → the demon-ledger phase (Phase 2) is the differentiator; accelerate it.

## Key starting references

1. Skinner, Ruhman, Nahum — Measurement-Induced Phase Transitions in the Dynamics of Entanglement (PRX 2019)
2. Google Quantum AI — Measurement-induced entanglement and teleportation on a noisy quantum processor (Nature 2023) + 2025 postselection-free observation (Comms. Phys.)
3. Buchhold, Müller, Diehl / Iadecola et al. / O'Dea et al. — adaptive & feedback monitored circuits, absorbing-state transitions (2022–2024) — *the related work we must position against*
4. Erdman et al. — Artificially intelligent Maxwell's demon for optimal control of open quantum systems (Quantum Sci. Technol. 2025)
5. Gidney — stim: a fast stabilizer circuit simulator (Quantum 2021)
6. Reviews: Fisher, Khemani, Nahum, Vijay — Random quantum circuits (Annu. Rev. Cond. Mat. 2023)
