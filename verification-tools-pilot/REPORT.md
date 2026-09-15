# Verification Tools Pilot: Report

## What was tested

A toy generator (latent vector, dim 8, uniform in `[-1,1]^8` -> 2D grid of
"cell energies", 3 hidden ReLU layers, <45k params at every grid size) was
trained briefly (300 Adam steps, MSE to synthetic Gaussian-blob targets;
see `data.py`/`train.py`) at three output grid sizes: 8x8, 16x16, 32x32.
The final layer has **no non-negativity-enforcing activation** (no
ReLU/softplus on the output) on purpose, so non-negativity is a genuine
bound-propagation question, not a tautology.

Two properties were checked on all three tools, at all three grid sizes:

- **Non-negativity**: for all `z` in `[-1,1]^8`, every output cell `>= 0`.
- **Bounded sum**: for all `z` in `[-1,1]^8`, `4 <= sum(output) <= 20`
  (a generous band around the training target sum range of 8-12), as a
  stand-in for approximate energy conservation.

Both properties are checked as UNSAT-means-proven queries: the tool
searches for a counterexample to the *negation* of the claim. UNSAT means
the property is proven over the whole box; SAT means a counterexample was
found (the property is genuinely violated); TIMEOUT means neither was
established in the time budget.

All three tools ended up fully working end-to-end. Raw results are in
`results/*.json`.

## (a) Ease of setup

| Tool | Setup effort | Notes |
|---|---|---|
| **auto_LiRPA** | Low, ~2 min, one hiccup | `pip install auto-LiRPA` fails: the PyPI release (0.2/0.3) pins `torch<1.13`, incompatible with any current torch. Fix: install from GitHub source (`pip install --no-deps git+https://github.com/Verified-Intelligence/auto_LiRPA.git`) and add `appdirs graphviz tqdm packaging` by hand. After that, imports and runs cleanly on torch 2.14 despite the repo's own `pyproject.toml` pinning `torch<2.12.0` (pip warns about the conflict but nothing actually breaks at runtime). |
| **alpha-beta-CROWN** | High, ~15-18 min, many hiccups | Not pip-installable as a single package. `git clone` + `pip install --no-deps -e .` (the package pins `torch==2.11.0` exactly, which conflicts with the torch already installed, so `--no-deps` was required). Then import-time `ModuleNotFoundError` chased one dependency at a time: `pyyaml`, then `onnx2pytorch` (installed from a pinned git commit, no PyPI release), which itself imports `torchvision` even though we never touch images or NMS ops, then `skl2onnx`, which imports `scikit-learn`. Then `gurobipy` (an optional MIP/LP solver backend) was needed at class-definition time for `LiRPANet` even though we never asked for MIP-based verification. All of these installed cleanly from PyPI/GitHub once identified, but the dependency surface is much larger than the actual bound-propagation code path we used (its own high-level `compute_bounds` API) needed. |
| **Marabou** | Low, ~1 min | `pip install maraboupy` just works; ships a prebuilt wheel with the C++ core included. No source build needed. |

**Headline**: auto_LiRPA and Marabou are both quick to get running (minutes,
one dependency hiccup each). alpha-beta-CROWN's actual verification code
is a thin layer over the *same* auto_LiRPA bound propagation, but its
packaging drags in a large, mostly-unrelated dependency tree (computer
vision, classical ML, commercial MIP solver bindings) because it is built
to also handle CNN/vision benchmarks, MIP-based complete verification, and
VNN-COMP tooling that a small MLP pilot never touches. It did **not**
exceed the 15-20 minute setup budget as a hard failure, but it was close,
and needed several manual dependency-chasing round-trips rather than
converging in one `pip install -r requirements.txt`.

## (b) PyTorch compatibility

- **auto_LiRPA**: native. Takes an `nn.Module` directly (`BoundedModule`
  wraps it in place); no ONNX round-trip needed. Handles our
  `Linear -> ReLU` stack and the appended all-ones `Linear` sum-head with
  no special-casing.
- **alpha-beta-CROWN**: also native PyTorch through its high-level API
  (`ABCrownSolver(model, x, y, ...)`), since it's built directly on
  auto_LiRPA's `BoundedModule`. No ONNX export needed for this path either
  (there is a separate ONNX-based CLI/VNNLIB workflow for VNN-COMP-style
  benchmarks, which we did not use).
- **Marabou**: ONNX-only. Requires `torch.onnx.export(...)` first (see
  `export_onnx.py`; the legacy `dynamo=False` exporter was needed, the
  default dynamo exporter wants `onnxscript`). This is an extra step and
  an extra place for things to silently diverge (we verified the ONNX
  export matches the PyTorch model to ~1e-7 max abs difference before
  trusting Marabou's results).

## (c) Property expressiveness: per-neuron only, or sum-bounds too?

This was the most interesting comparison point.

- **auto_LiRPA / alpha-beta-CROWN**: both compute sound lower/upper bounds
  for *every* requested output objective in a single `compute_bounds()`
  call. Non-negativity over all N output cells is one call, not N calls.
  Bounded sum is expressed with the standard trick: append a fixed,
  non-trainable `Linear(N, 1)` layer with all-ones weights (see
  `model.GeneratorWithSum`) so the sum becomes a single extra output
  neuron with its own CROWN bound. No new API surface needed -- it's just
  another linear layer in the graph.
- **Marabou**: no first-class "bound every neuron in one pass" call.
  Non-negativity needs one full solver query *per cell* (see below for why
  this matters at scale). Bounded sum, however, turned out to be
  **directly expressible** despite our expectation (per the task
  description) that it might need awkward extra plumbing: Marabou's
  `network.addInequality(vars, coeffs, scalar)` API (already used in this
  repo's `verified-scientific-ml/verification/verify_particle.py` for a
  2-variable momentum combination) accepts an arbitrary-length list of
  output variables and coefficients, so `addInequality(all_64_output_vars,
  [1.0]*64, SUM_MAX)` encodes the sum constraint directly as one linear
  equation over the network's own variables -- no auxiliary sum-node
  needed in the ONNX graph. This is a genuine linear equation, not an
  approximation, and needed no new mechanism beyond what the momentum
  property already used in Step 4/5 of the main pipeline.

**Headline**: auto_LiRPA/CROWN's real expressiveness advantage over
Marabou is *per-neuron bound propagation at no extra query cost* (one
call bounds all N cells), not linear-combination properties specifically
-- Marabou handles linear combinations of outputs (sums, differences)
natively via `addInequality`, just not "all neurons, one shot."

## (d) Scaling behavior (8x8 / 16x16 / 32x32)

**Marabou, concretely** (raw numbers in `results/marabou_*.json`; sample
size was 24 cells at 8x8 and reduced to 10 cells at 16x16/32x32 once the
8x8 run's ~9-minute wall time made a same-size sample impractical at three
grid sizes within this pilot's time budget -- see `verify_marabou.py`):

- **8x8** (24-cell sample): non-negativity sample TIMEOUT overall (9 SAT /
  15 TIMEOUT / 0 UNSAT across the sample, avg 22.9s/query, 551.0s total for
  the sample); bounded-sum TIMEOUT (both one-sided queries hit the 30s
  timeout, 61.5s total).
- **16x16** (10-cell sample): non-negativity sample TIMEOUT overall (9 SAT
  / 1 TIMEOUT, avg 7.5s/query, 75.0s total); bounded-sum TIMEOUT again
  (both one-sided queries timed out, 61.1s total).
- **32x32** (10-cell sample): non-negativity sample TIMEOUT overall (9 SAT
  / 1 TIMEOUT, avg 3.9s/query, 39.2s total) -- but **bounded-sum is UNSAT
  (proven!)** in only 4.2s (above-max UNSAT in 3.1s, below-min UNSAT in
  1.1s). This is the only UNSAT (proven, not just "no counterexample
  found within budget") result anywhere in this pilot.

That last result is the single most interesting number in this pilot.
32x32 has the *narrowest* hidden layer of the three sizes (32 units, vs.
128 at 8x8 and 96 at 16x16, since `model.hidden_width` shrinks the hidden
layer to keep total params under 50k as the output grid grows) --
fewer ReLU units means fewer branch-and-bound case splits, so Marabou's
complete search is actually **faster** at 32x32 than at 8x8, both for
non-negativity (avg 3.9s/query vs 22.9s/query) and dramatically so for
bounded-sum (4.2s vs timing out at 60+s). And because Marabou is complete,
its UNSAT at 32x32 is a real proof that `4 <= sum(output) <= 20` holds
everywhere in `[-1,1]^8` -- tighter than what auto_LiRPA/alpha-beta-CROWN
could show at the same grid size (their CROWN relaxation bounds were
`[-1.94, 45.50]`, i.e. SAT/"violated" against the same `[4,20]` claim,
because the relaxation is too loose to prove it, not because the property
is actually false). **Network width, not output grid size, is what
determines whether complete search stays tractable** -- a genuinely
useful scaling finding for choosing the real calorimeter architecture if
Marabou-style complete verification is ever needed on it.

**auto_LiRPA / alpha-beta-CROWN**: both stayed near-instant at every grid
size (well under 2 seconds total for both properties combined, even at
32x32 / 1024 output cells) -- the whole point of bound propagation is that
its cost scales with network size (forward/backward passes through the
graph), not with the number of properties checked. alpha-beta-CROWN was
consistently ~2-5x slower than auto_LiRPA on identical queries (its
high-level API adds config/branch-and-bound scaffolding even when
`complete_verifier=skip` restricts it to incomplete alpha-CROWN bounds
only), but both are trivially fast at this network scale. As a
correctness check: **auto_LiRPA and alpha-beta-CROWN produced numerically
identical bounds** at every grid size (e.g. at 32x32, both report
non-negativity lower bound -0.9745 and sum bounds [-1.94, 45.50]) --
expected, since alpha-beta-CROWN's incomplete mode is the same CROWN
algorithm auto_LiRPA implements, and a useful sanity check that both
integrations in this pilot are wired correctly.

**Marabou**: complete SMT-based search does not scale like bound
propagation, and -- as the 32x32 result above shows -- it does not scale
with output grid size the way one might expect either; network width
dominates. A first attempt at exhaustively checking all 64 cells at 8x8
with plain single-threaded search timed out at 60s on the very first
query. Switching to Split-and-Conquer mode (`snc=True, numWorkers=4`)
brought individual queries down to single digits of seconds in the best
case, but per-query time was highly variable (observed range: ~1s to a
full timeout) and averaged ~23s per single-neuron query at 8x8 (128-unit
hidden layer) even with 4 parallel workers, dropping to ~7.5s/query at
16x16 (96-unit hidden layer) and ~4s/query at 32x32 (32-unit hidden
layer) -- so **exhaustive per-cell enumeration was not attempted at any
grid size**, not just 32x32 as anticipated. All three grid sizes use a
representative sample (corners, edges, center, plus random cells)
instead; the sample was 24 cells at 8x8, reduced to 10 cells at
16x16/32x32 once the 8x8 run's ~9-minute wall time made a same-size
sample impractical for a pilot at three grid sizes. This is reported as
a sample, not an exhaustive per-cell proof.

## Summary table

| Tool | Setup | PyTorch-native | Per-neuron bounds (all at once) | Sum/linear bounds | 8x8 total time | 16x16 total time | 32x32 total time |
|---|---|---|---|---|---|---|---|
| auto_LiRPA | easy (1 fix) | yes | yes, 1 call | yes (sum-head trick) | ~0.31s | ~0.33s | ~0.38s |
| alpha-beta-CROWN | hard (~15-18 min, many fixes) | yes | yes, 1 call | yes (sum-head trick) | ~1.73s | ~1.20s | ~1.90s |
| Marabou | easy | no (ONNX only) | no -- 1 query/cell | yes (`addInequality`) | ~612s (24-cell sample: TIMEOUT/TIMEOUT) | ~136s (10-cell sample: TIMEOUT/TIMEOUT) | ~43s (10-cell sample: TIMEOUT/**UNSAT**) |

## Recommendation

**Commit to auto_LiRPA as the primary tool for the real calorimeter
generator pipeline**, for these reasons:

1. It is PyTorch-native (no ONNX export/round-trip risk), trivially easy
   to install and keep working, and computes bounds for every output cell
   of a full calorimeter grid in a single near-instant call -- this is
   exactly the shape of the real properties (non-negativity and
   conservation over potentially large grids, e.g. real calorimeters can
   have thousands of cells).
2. Its bound-propagation core is identical to alpha-beta-CROWN's
   incomplete mode (verified empirically here: identical numeric bounds),
   so nothing about verification *quality* is lost by not using
   alpha-beta-CROWN at this stage.
3. **Keep alpha-beta-CROWN in reserve, not adopted now**: once the real
   architecture is large/hard enough that plain CROWN bounds are too loose
   to prove anything useful (as they already are here -- see below), its
   branch-and-bound refinement and alpha-optimized bounds are the natural
   next step, and the migration cost is low since it reuses the exact same
   `IOConstraints`/`compute_bounds` call shape as this pilot's
   auto_LiRPA usage. Budget real setup time for it, though -- 15-20
   minutes of dependency-chasing per environment is a real, recurring cost
   if the CI/dev-environment story isn't nailed down.
4. **Keep Marabou for a narrow, specific role**: spot-checking a small
   number of cells (e.g. a handful of high-energy or boundary cells that
   matter most physically) with a complete solver as an independent
   cross-check against auto_LiRPA's bounds, not as the primary tool. It
   does not scale to "every cell, every property" the way this pipeline
   will need, and its ONNX-only interface adds an extra translation step.
   Its ability to express linear combinations directly via
   `addInequality` (not just per-neuron boxes) is a genuine plus over a
   naive per-neuron-only understanding of Marabou, and worth remembering
   if a future property needs a custom linear combination auto_LiRPA's
   sum-head trick doesn't naturally cover. This pilot also found a
   concrete reason to keep it around rather than dropping it entirely:
   at 32x32 (the narrowest hidden layer of the three sizes here, 32
   units), Marabou's complete search **proved** bounded-sum in 4.2s where
   auto_LiRPA/alpha-beta-CROWN's relaxation was too loose to show anything
   ([-1.94, 45.50] against a [4, 20] claim). If the real architecture ends
   up with narrow-enough hidden layers for Marabou to stay fast (this
   pilot's data suggests network width drives Marabou's cost far more than
   output grid size does), a targeted Marabou pass is worth trying
   whenever auto_LiRPA's bound is too loose to prove something that
   matters.

### An honest caveat about the results themselves

Non-negativity was SAT (violated) on every tool at every grid size, and
this is a **real, expected violation**, not a tool artifact: the toy
generator was deliberately built with no final activation enforcing
non-negativity, and Marabou's SAT results (independent, complete search,
not a relaxation) confirm actual counterexamples exist -- e.g. output
cells pushed to exactly the query's negative epsilon boundary -- not just
loose bounds. Bounded-sum was SAT (loosely, via a very imprecise CROWN
relaxation: `[-24.6, 144.7]` against a claim of `[4, 20]`, on a sum
trained to sit near `[8, 12]`) for auto_LiRPA/alpha-beta-CROWN at every
grid size, and TIMEOUT for Marabou at 8x8/16x16 -- **except at 32x32,
where Marabou's complete search proved bounded-sum UNSAT in 4.2s** (see
the scaling section above). So the pilot did get one genuine UNSAT
("proven") result, and it came from the tool with the weaker per-neuron
API, on the property CROWN's relaxation was least able to say anything
useful about -- a real illustration of complete search's value when
relaxation-based bounds are too loose, provided the network is small
enough for search to stay fast (see the width-vs-grid-size finding above).

For the real calorimeter pipeline, the practical implication is that an
architecture that must satisfy non-negativity provably will likely need
either (a) an architecturally-enforced final activation (softplus/ReLU
output, sum-normalization) so the property becomes provable rather than
merely plausible by fast bound propagation, or (b) tighter verification
(alpha-CROWN's optimized bounds, branch-and-bound, or a targeted Marabou
complete-search pass on a narrow enough network) where the architecture
cannot be changed. We did not train a variant with such a final
activation for this pilot (kept in scope as a possible follow-up, not
built here) -- see `model.py`'s docstring for where to add one (a
`nn.Softplus()` or `nn.ReLU()` as the last layer of `Generator.net` would
make property 1 provable directly, likely trivially, by all three tools).
