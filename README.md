# Formal Verification of Geometric Deep Learning for Scientific Simulation

A simulator-agnostic framework for giving scientists provable, quantified
worst-case guarantees about physical properties of ML simulators — over
the whole input space, not sampled inputs.

This started as a research project exploring whether geometric inductive
biases make neural scientific surrogates more amenable to formal
verification; the results below are being written up as a paper.

---

## Supporting results: geometric structure and formal verifiability

Three architectures were compared across two physical domains:

| Architecture | Structure | Particle system (ε) | SIR epidemic (ε) |
|---|---|---|---|
| Model A — plain MLP | none | 1.0 | 0.1 |
| Model C — soft geometric | translation-invariant input | 1e-2 | 1e-2 |
| Model B — hard geometric | conservation law by construction | 1e-5 | ≤ 1e-6 |

ε is the tightest conservation-law violation bound formally proven UNSAT.
The A < C < B ordering holds on both domains (3–5 orders of magnitude).
VGT tightens Model C a further 10× on both domains (ε=1e-2 → ε=1e-3).

---

## Calorimeter case study: synthetic vs real data

The calorimeter pipeline (`case_studies/calorimeter/`) was run on both
synthetic data and real CaloChallenge Dataset 1 photon showers
(Zenodo DOI 10.5281/zenodo.8099322, CC-BY-4.0, 121k events, 256 MeV–4 TeV).

| Property | Synthetic (step5) | Real (step5b) |
|---|---|---|
| voxel_non_negativity | PROVEN | PROVEN |
| energy_upper_bound (CROWN) | 3989 | **443** |
| energy_upper_bound (IBP) | 11362 | **2260** |
| energy_conservation tightest ε | none (all INCONCLUSIVE) | PROVEN at ε=226.65 |

The real model's CROWN upper bound is ~9× tighter because real preprocessed
targets are in [0, 1.4] (log1p-normalised fractions) rather than exponential
scale=10.  The `run_verification_real.py` script reproduces the full pipeline.

---

## Marabou backend: numerical tolerance fix

Marabou's SAT tolerance (~1e-5) caused false-positive COUNTEREXAMPLE results
for ReLU-ending networks.  The solver returned SAT with output = 0.0 (the
post-ReLU boundary) when the query threshold was -1e-9, because it accepted
0.0 as satisfying ≤ -1e-9 within floating-point precision.

Fix: threshold raised to -1e-5 and a secondary CE verification step
cross-checks any SAT result against the PyTorch model.  Regression test:
`tests/test_backend_marabou.py::TestMarabouReLUEndingNetwork`.

Note: Marabou's ONNX parser correctly maps `outputVars` to post-ReLU
variables — the bug was numerical, not structural.

---

## Repo layout

| Folder | What it is |
|---|---|
| [`verifier/`](verifier/) | Core framework: ModelSpec, Property (NonNegativity, Conservation, RangeBound), LiRPA and Marabou backends, runner, PGD falsifier. |
| [`case_studies/`](case_studies/) | Per-domain pipelines: `calorimeter/` (CaloChallenge Dataset 1, deep case study) and `sir_spring/` (generality test, wraps existing trained models). |
| [`tests/`](tests/) | Framework unit tests — 144 passing. |
| [`results/`](results/) | JSON results written here per domain (`results/<domain>/<run_id>.json`). |
| [`verified-scientific-ml/`](verified-scientific-ml/) | Supporting experiment: simulators, trained surrogates (Models A/B/C), Marabou epsilon sweeps, VGT loop — particle system and SIR domains. |
| [`verification-tools-pilot/`](verification-tools-pilot/) | Tool selection pilot comparing `auto_LiRPA`, `alpha-beta-CROWN`, and `Marabou` on a toy grid generator. Recommends auto_LiRPA primary, Marabou for spot-checks. |

## Status

All steps complete. The framework covers PGD falsification (Step 7), Marabou
spot-checks cross-validating LiRPA results (Step 8), architecture variant
experiments (Step 8b), and a full real-data calorimeter pipeline (Step 5b).
