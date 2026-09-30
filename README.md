# Thesis: Formal Verification of Geometric Deep Learning for Scientific Simulation

A simulator-agnostic framework for giving scientists provable, quantified
worst-case guarantees about physical properties of ML simulators — over
the whole input space, not sampled inputs.

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

## Repo layout

| Folder | What it is |
|---|---|
| [`verifier/`](verifier/) | Core framework: ModelSpec, Property (NonNegativity, Conservation, RangeBound), LiRPA and Marabou backends, runner, PGD falsifier. |
| [`case_studies/`](case_studies/) | Per-domain pipelines: `calorimeter/` (CaloChallenge Dataset 1, deep case study) and `sir_spring/` (generality test, wraps existing trained models). |
| [`tests/`](tests/) | Framework unit tests — 143 passing. |
| [`results/`](results/) | JSON results written here per domain (`results/<domain>/<run_id>.json`). |
| [`verified-scientific-ml/`](verified-scientific-ml/) | Supporting experiment: simulators, trained surrogates (Models A/B/C), Marabou epsilon sweeps, VGT loop — particle system and SIR domains. |
| [`verification-tools-pilot/`](verification-tools-pilot/) | Tool selection pilot comparing `auto_LiRPA`, `alpha-beta-CROWN`, and `Marabou` on a toy grid generator. Recommends auto_LiRPA primary, Marabou for spot-checks. |

## What's next

All steps complete. The framework covers PGD falsification (Step 7), Marabou spot-checks cross-validating LiRPA results (Step 8), and architecture variant experiments showing how the final ReLU enables formal non-negativity proofs.
