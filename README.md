# Descartes: Formal Verification of Geometric Deep Learning for Scientific Simulation

**Research question**: does building geometric/physical inductive biases into
a neural scientific simulator make it more amenable to *formal*
verification — not just more accurate — compared to an unstructured
baseline with the same job?

The eventual target is geometric deep learning models used as **generative
surrogates for calorimeter showers** in particle physics (a latent vector
in, a grid of cell energies out) — but before committing a verification
approach to that, this repo builds the question up from first principles on
small, fully-understood physical systems, then pilots candidate
verification tools on a toy version of the real target.

## Repo layout

| Folder | What it is |
|---|---|
| [`verified-scientific-ml/`](verified-scientific-ml/) | The main pipeline (Steps 1-5, see below): physics simulators, trained neural surrogates, and formal verification of their properties with [Marabou](https://github.com/NeuralNetworkVerification/Marabou). |
| [`verification-tools-pilot/`](verification-tools-pilot/) | A separate, self-contained pilot comparing verification tools (`auto_LiRPA`, `alpha-beta-CROWN`, `Marabou`) on a toy latent-to-grid generator shaped like the real calorimeter-shower target, to decide which tool to commit to before building that model for real. |

Each has its own README with setup/run instructions. This file is just the
map between them and the overall narrative.

## Where things stand

### `verified-scientific-ml/` — Steps 1-9 (done)

Built up in stages, each verified end-to-end before moving on:

1. **Harmonic oscillator simulator** — a known-analytic physical system and
   dataset generator, no ML yet.
2. **MLP surrogate** — a plain feedforward network trained to predict the
   oscillator's one-step dynamics. Accurate step-to-step, but its energy
   visibly drifts under long rollout — nothing in the architecture or loss
   knows energy should be conserved.
3. **First formal verification pass (Marabou)** — proves simple linear
   input/output bounds on that surrogate exactly (not sampled), and
   contrasts that with random testing, which can only ever say "no
   violation found," never prove one doesn't exist.
4. **A system with a *linear* conservation law** — two particles connected
   by a spring, where total momentum is conserved and, unlike energy, can
   be stated as an exact linear property Marabou can check directly. A
   plain-MLP baseline (Model A) is trained and verified: Marabou proves a
   loose momentum-conservation bound, and finds a genuine counterexample at
   a tight one.
5. **A geometric surrogate (Model B)** — same task, but momentum
   conservation is made an *exact algebraic identity of the architecture*
   (relative-state input, shared force applied ±F to each particle).
   Marabou proves the *tight* momentum bound UNSAT — the property Model A
   provably violates — about 500x faster than Model A's easier query.
6. **Soft-geometric surrogate (Model C)** — an intermediate architecture:
   translation-invariant input encoding but no hard conservation constraint.
   Verifiable to ε=1e-2 (100× tighter than Model A, 1,000× looser than B).
7. **Verification-Guided Training (VGT)** — Marabou counterexamples used as
   training signal: each worst-case input found by the solver is added to
   the training set with physics-simulator labels, then the model is
   fine-tuned and re-verified. VGT pushes Model C from ε=1e-2 → ε=1e-3
   (10× improvement) without any architecture change.
8-9. **SIR epidemiological model** — the full Steps 4-7 pipeline replicated
   on a completely different simulation domain (SIR ODE, β=0.3, γ=0.1,
   conservation law s+i+r=1). The A < C < B ordering holds identically:
   A verifiable to ε=0.1, C to ε=1e-2, B to ε≤1e-6. VGT again tightens
   C from ε=1e-2 → ε=1e-3 (13 iterations, 2,727 counterexample states).

**Headline result**: the geometric bias → verifiability relationship is
not an artefact of one physical system. It generalises across domains
(particle physics and epidemiology) and VGT produces the same 10×
tightening in both. Cross-domain comparison table and full numbers:
[`verified-scientific-ml/README.md`](verified-scientific-ml/README.md).

### `verification-tools-pilot/` — tool selection pilot (done)

Steps 1-5 all used Marabou. The real target — a generative model outputting
a whole grid of values, not a small fixed-size state vector — is a
different shape of problem, so before building it for real, this pilot
tested three candidate verification tools on a toy stand-in (latent vector
→ 8x8/16x16/32x32 grid of non-negative "energy" cells) against two
properties: output non-negativity and a bounded total sum (approximating
conservation).

**Finding**: `auto_LiRPA` is the recommended primary tool — PyTorch-native,
trivial to set up, and bounds an entire output grid in one near-instant
call regardless of grid size. `alpha-beta-CROWN` computes identical bounds
(same underlying algorithm) but has a much heavier setup and is kept in
reserve for when plain CROWN bounds are too loose to prove something.
`Marabou` doesn't scale to checking every cell (one solver query per cell),
but its complete search can still *prove* properties CROWN's relaxation is
too loose for — worth keeping for targeted spot-checks. Full comparison and
reasoning: [`verification-tools-pilot/REPORT.md`](verification-tools-pilot/REPORT.md).

## What's next

Apply the tool choice above to a real geometric generative surrogate for
calorimeter shower simulation — the actual target this whole pipeline has
been building toward — and re-run the same standard-vs-geometric
verifiability comparison (now established on two domains) on that model.
