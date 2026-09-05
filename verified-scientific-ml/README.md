# Verified Scientific ML — Step 1: Harmonic Oscillator Simulator

This is Step 1 of a larger research project on whether geometric inductive
biases improve the formal verifiability of neural scientific simulators.
This step only builds a clean, reproducible physics simulator and generates
a dataset from it. No neural networks, geometric deep learning, or formal
verification are implemented yet.

## The physics

The simple harmonic oscillator (unit mass, undamped, unforced) obeys:

```
dx/dt = v
dv/dt = -k * x
```

- `x` — position (displacement from equilibrium)
- `v` — velocity
- `k` — spring constant (`k = omega^2`, where `omega` is the angular frequency)

Total mechanical energy (unit mass):

```
E = 0.5 * v^2 + 0.5 * k * x^2
```

is conserved exactly in continuous time. It is not conserved exactly by a
numerical integrator, so tracking `E(t)` is a useful sanity check on
integration accuracy.

## Numerical method

Trajectories are integrated with `scipy.integrate.solve_ivp` using the
explicit Runge-Kutta 4(5) method (`RK45`) with tight tolerances
(`rtol=atol=1e-8`), evaluated at evenly spaced time points via `t_eval`.

## What the generated dataset represents

`data/generate_data.py` generates many independent trajectories from random
initial conditions `(x0, v0)` (uniform in `[-2, 2]`) with a fixed spring
constant `k` and fixed timestep `dt`. Each trajectory is integrated for a
number of steps, and every consecutive pair of states along it is recorded
as one training example:

```
[x_t, v_t] -> [x_(t+1), v_(t+1)]
```

This is the raw material for a Step 2 neural surrogate: a model that learns
to predict the next state given the current state, for this fixed `k` and
`dt`.

The dataset is saved to `data/harmonic_oscillator_dataset.npz` with arrays:

- `states_t`, `states_tp1` — shape `(N, 2)`, columns `[x, v]`
- `energy_t`, `energy_tp1` — shape `(N,)`, the energy at each state
- `k`, `dt`, `seed` — scalars recording how the dataset was generated

Generation uses a fixed random seed (`42`) so the dataset is reproducible.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the simulator

From the `verified-scientific-ml/` directory:

```python
from simulator.harmonic_oscillator import simulate

result = simulate(x0=1.0, v0=0.0, k=1.0, t_span=(0.0, 10.0), n_points=200)
# result["t"], result["x"], result["v"], result["k"]
```

## Generate the dataset

From the `verified-scientific-ml/` directory:

```bash
python -m data.generate_data
```

This writes `data/harmonic_oscillator_dataset.npz` and an example
trajectory plot to `data/plots/example_trajectory.png` (position, velocity,
phase space, and energy vs time).

## Run tests

From the `verified-scientific-ml/` directory:

```bash
pytest
```

Tests check known analytic solutions, energy conservation, dataset shapes,
and that generated data contains no NaN/inf values.

## Step 2: neural-network surrogate model

Step 2 trains a small PyTorch MLP to learn the one-step map produced by
Step 1's dataset:

```
[x_t, v_t] -> [x_(t+1), v_(t+1)]
```

This does not change the simulator — it is purely a model trained on the
data Step 1 already generated.

### Architecture

A small fully-connected network (`models/surrogate.py`):

```
Linear(2 -> 64) -> ReLU -> Linear(64 -> 64) -> ReLU -> Linear(64 -> 2)
```

No attention, recurrence, convolutions, or physics baked into the
architecture — deliberately the simplest possible baseline to compare
later against geometric/equivariant models.

### Training configuration (fixed, for reproducibility)

| setting            | value                                   |
|---------------------|------------------------------------------|
| seed                | 42                                        |
| split               | 70% train / 15% val / 15% test (random split of the 100k pairs) |
| normalization       | standardize `(x, v)` using train-set mean/std only; the same normalizer is reused for inputs and targets, since `x_t` and `x_(t+1)` are drawn from the same stationary physical distribution |
| hidden width         | 64                                        |
| loss                | MSE                                       |
| optimizer           | Adam, lr = 1e-3                           |
| batch size          | 256                                       |
| epochs              | 200                                       |
| model selection     | checkpoint with lowest validation loss    |

### Run training

```bash
python -m models.train
```

Saves to `models/checkpoints/`:
- `surrogate.pt` — trained model weights (best validation loss)
- `normalizer.npz` — mean/std used to standardize inputs and outputs (required for inference)
- `splits.npz` — the raw val/test splits and `k`/`dt`/`seed` metadata
- `loss_history.npz` — per-epoch train/val loss

### Run evaluation

```bash
python -m models.evaluate
```

Computes test-set MSE/MAE/max error, performs a 200-step autoregressive
rollout from `(x0, v0) = (1.5, 0.0)` compared against the true simulator,
and compares energy over that rollout. Writes:

- `models/checkpoints/evaluation_report.txt` — all numerical results
- `models/plots/test_predictions.png` — predicted vs true `x_next`/`v_next`, error histograms
- `models/plots/loss_curve.png` — train/val loss vs epoch
- `models/plots/rollout_comparison.png` — NN rollout vs true trajectory (position, velocity, phase space, error growth)
- `models/plots/energy_comparison.png` — NN rollout energy vs true simulator energy

### Results (this run)

Test-set (physical units, single-step prediction):

| metric | x_next | v_next | overall |
|---|---|---|---|
| MSE | 1.3e-07 | 2.2e-07 | 1.8e-07 |
| MAE | 2.7e-04 | 3.6e-04 | 3.2e-04 |
| max error | 3.5e-03 | 3.8e-03 | 3.8e-03 |

200-step rollout from `(x0, v0) = (1.5, 0.0)`, `dt = 0.05` (10 time units, ~1.6 periods):

- Final position error `|x_nn - x_true|`: 1.3e-02
- Max position error over the rollout: 3.3e-02
- True simulator energy drift: -7.0e-08 (numerically conserved, as expected)
- **NN rollout energy drift: +8.9e-03** (visibly non-conservative)

### What this tells us

The one-step predictions are highly accurate — MAE around 3e-4 on
quantities of order 1, so the network has clearly learned the local
dynamics well. But single-step accuracy is not the same as long-horizon
correctness: chaining thousands of tiny prediction errors together during
rollout causes the NN trajectory to visibly drift from the true one, and
critically, its energy is not conserved — it drifts by about 0.8% over
just 1.6 oscillation periods, while the true simulator's energy stays flat
to within numerical noise (`~1e-7`).

This is exactly the failure mode motivating the rest of the thesis: a
standard MLP has no notion that this is a conservative physical system, so
nothing stops it from slowly "leaking" energy (or gaining it) as it rolls
forward. The later comparison against a geometric/equivariant model —
one that is structurally constrained to respect the system's underlying
symmetries — will test whether that inductive bias produces trajectories
that stay closer to the true one and conserve energy better, and whether
that in turn makes the model's behavior easier to formally verify.

### Run tests

```bash
pytest
```

`tests/test_surrogate.py` adds shape and normalization round-trip checks
for the model code (in addition to the Step 1 simulator tests).

## Step 3: formal verification (Marabou)

Step 3 takes the trained Step 2 MLP and asks a fundamentally different
kind of question about it than testing does:

- **Testing**: "we tried many inputs and didn't find a violation."
- **Formal verification**: "within the specified input domain, there
  does not exist an input that violates the specified property."

This does not modify the Step 1 simulator or the Step 2 model — it only
exports the already-trained network and checks properties of it.

### What is being verified

The verifier is [Marabou](https://github.com/NeuralNetworkVerification/Marabou),
an SMT-based complete verifier for piecewise-linear (ReLU) networks. It is
given the network as an ONNX graph plus a set of linear input/output
bounds, and it either:

- proves **no input in the domain violates the property** (**UNSAT** —
  the negated property is unsatisfiable), or
- returns a **concrete counterexample input** that does violate it
  (**SAT** — the negated property is satisfiable).

**Exact mathematical domain**, identical for every property:

```
x in [-2, 2]
v in [-2, 2]
```

This is exactly the domain `data/generate_data.py` sampled initial
conditions from in Step 1, so it is the region the network was actually
trained on, not an arbitrary or wider region.

**Properties checked** (`verification/properties.py`), each of the form
"for all (x, v) in the domain, `<output> <bound_type> <value>`":

| name | claim | expected |
|---|---|---|
| `x_next_upper_loose` | `x_next <= 3.0` | UNSAT (verified) |
| `x_next_lower_loose` | `x_next >= -3.0` | UNSAT (verified) |
| `v_next_upper_loose` | `v_next <= 3.0` | UNSAT (verified) |
| `v_next_lower_loose` | `v_next >= -3.0` | UNSAT (verified) |
| `x_next_upper_violated` | `x_next <= 1.0` | SAT (deliberately false) |
| `v_next_lower_violated` | `v_next >= -1.0` | SAT (deliberately false) |

The "loose" bounds come from the physics: energy conservation bounds the
true amplitude to `sqrt(2 * E_max) = sqrt(2 * 0.5 * (2^2 + 2^2)) = sqrt(8)
≈ 2.83` for the true simulator, and Step 2's evaluation showed the
network's single-step error is at most ~4e-3 on this domain — so 3.0
leaves comfortable margin. The two "violated" properties use bounds that
are deliberately too tight (e.g. `x=2, v=0` alone already pushes `x_next`
close to 2), included specifically to demonstrate that Marabou *can* find
a real counterexample, not just report UNSAT every time.

Marabou works with non-strict inequalities, so in practice each property
is checked by asking Marabou to solve the **negation**:

- claim `output <= U` -> search for `output >= U`
- claim `output >= L` -> search for `output <= L`

UNSAT on that negated query proves the original claim across the whole
domain; SAT returns an input satisfying the negation, i.e. a
counterexample to the claim.

### What SAT and UNSAT mean here

- **UNSAT**: Marabou has searched the entire (continuous, infinite)
  input domain and proven no point in it violates the property. This is
  a formal proof, not a sample-based estimate.
- **SAT**: Marabou found one specific input in the domain, returned as a
  counterexample, where the property does not hold.

### Why this is different from random testing

`verification/random_search.py` runs the identical properties over
200,000 random `(x, v)` samples drawn from the same domain (seed 42) and
checks each one against the model directly (no solver). Run:

```bash
python -m verification.random_search
```

Results from this run (`verification/artifacts/random_testing_results.json`):

| property | Marabou (exhaustive) | random testing (200,000 samples) |
|---|---|---|
| `x_next_upper_loose` | **UNSAT** — proven, no counterexample exists | 0/200,000 violated — "not found," never proof |
| `x_next_lower_loose` | **UNSAT** — proven | 0/200,000 violated |
| `v_next_upper_loose` | **UNSAT** — proven | 0/200,000 violated |
| `v_next_lower_loose` | **UNSAT** — proven | 0/200,000 violated |
| `x_next_upper_violated` | **SAT** — counterexample found | 49,960/200,000 violated |
| `v_next_lower_violated` | **SAT** — counterexample found | 50,061/200,000 violated |

For the two deliberately-violated properties, both methods agree a
violation exists (random testing finds it easily because the false bound
is violated over roughly a quarter of the domain, not just an edge case).
The important contrast is on the four "loose" properties: random testing
can only ever say *no violation was found in this sample*, which is
consistent with the property being true but is **not proof** — a
violation could in principle exist in one of the uncountably many points
never sampled. Marabou's UNSAT result is what actually rules that out,
by construction (it performs a complete search using SMT + LP bound
tightening over the piecewise-linear network, not sampling).

### Exporting the model (ONNX)

```bash
python -m verification.export_onnx
```

The trained `models/checkpoints/surrogate.pt` + `normalizer.npz` are
loaded and wrapped as `FullSurrogate`
(`verification/export_onnx.py`), which fuses input normalization and
output de-normalization into the network itself as extra `Linear`
layers with diagonal weight matrices. This means the exported ONNX
graph (`verification/artifacts/surrogate.onnx`) takes a **physical**
`[x, v]` state as input and produces a **physical** `[x_next, v_next]`
state as output — so the domain and properties above can be stated
directly in physical units instead of normalized ones. Before exporting,
the script checks the fused model's output against
`models.evaluate.predict()` (the original inference path) over random
samples and asserts the max difference is below `1e-4` (observed:
`~5e-7`, i.e. floating-point-level agreement) — confirming the export
preserves inference behavior.

### Running the verifier

```bash
python -m verification.verify
```

For every property this records input bounds, the property, the
SAT/UNSAT result, the counterexample (if SAT), and verification time to
`verification/artifacts/verification_results.json`.

### Results (this run)

All 6 properties matched their expected result:

| property | result | time (s) | counterexample |
|---|---|---|---|
| `x_next_upper_loose` | UNSAT | 0.097 | — |
| `x_next_lower_loose` | UNSAT | 0.067 | — |
| `v_next_upper_loose` | UNSAT | 0.089 | — |
| `v_next_lower_loose` | UNSAT | 0.118 | — |
| `x_next_upper_violated` | **SAT** | 0.042 | x=1.040201, v=0.232774 -> x_next=1.050022 |
| `v_next_lower_violated` | **SAT** | 0.040 | x=0.395056, v=-0.981592 -> v_next=-1.000000 |

### What has and has not been formally established

**Established (proven, for exactly the stated domain):**
- For every `(x, v)` with `x in [-2,2]` and `v in [-2,2]`, this specific
  trained network satisfies `-3.0 <= x_next <= 3.0` and
  `-3.0 <= v_next <= 3.0`. This is a complete proof over that domain, not
  an empirical observation.

**Not established:**
- Nothing about behavior **outside** `[-2,2] x [-2,2]` — the network is
  unconstrained there and these results say nothing about it.
- Nothing about **energy conservation** — that is nonlinear
  (`E = 0.5*v^2 + 0.5*k*x^2`) and out of scope for this linear-bounds
  pipeline (explicitly deferred, see limitations).
- Nothing about **rotational/phase-space equivariance** — also deferred.
- Nothing about **multi-step rollout behavior** — only the one-step map
  is verified; Step 2 already showed rollout error and energy drift
  compound over many steps, and this verification says nothing about
  that regime.
- These results are specific to **this exact trained checkpoint**;
  retraining the model (even with the same code) would require
  re-verification.

### Limitations

- Only simple linear (box) input bounds and linear output bounds were
  checked — no nonlinear properties (e.g. energy conservation) yet.
- Marabou's inequalities are non-strict, so a SAT counterexample
  technically satisfies `output >= U` (not strictly `> U`); the boundary
  case `output == U` is measure-zero and not practically distinguishable
  here.
- The verified domain (`[-2,2] x [-2,2]`) is deliberately the same
  domain the model was trained on; verifying a much larger domain (where
  the network was never trained) was not attempted here and would likely
  fail, since nothing constrains the network's behavior outside its
  training distribution.
- The "violated" properties were chosen by hand to be false, specifically
  to demonstrate Marabou's SAT/counterexample path — they are a pipeline
  sanity check, not a property anyone would expect to hold.

### Run tests

```bash
pytest
```

`tests/test_verification.py` checks the ONNX export matches the original
model's inference behavior, that a loose property returns UNSAT with no
counterexample, that a deliberately violated property returns SAT with a
counterexample that is independently confirmed against the real model,
and that random search behaves consistently with both cases.

## Step 4 — Formal Verification of Physical Conservation

Step 3 verified simple linear bounds on the harmonic oscillator's
one-step outputs. Step 4 asks a scientifically sharper question: can we
formally verify a real **conservation law**, not just an arbitrary output
bound? This introduces a second, independent experiment (the harmonic
oscillator and its Step 1–3 code are untouched) built specifically so
that conservation law is *linear*.

### 1. Why a new physical system

The harmonic oscillator's natural invariant is energy,
`E = 0.5*k*x^2 + 0.5*v^2`, which is quadratic in the state. Marabou
verifies networks against *linear* arithmetic (plus piecewise-linear
ReLUs) — a nonlinear property like energy would need to be linearized or
bounded indirectly, which Step 4 explicitly defers. So Step 4 introduces
a system whose conservation law is linear by construction: two equal-mass
particles connected by an ideal spring in 2D, with no external forces
(`simulator/two_particle_system.py`). State:

```
[x1, y1, vx1, vy1, x2, y2, vx2, vy2]
```

The spring exerts equal-and-opposite forces on the two particles
(Newton's third law), so the net external force is always zero.

### 2. Why momentum conservation is scientifically meaningful

Total linear momentum,

```
Px = vx1 + vx2
Py = vy1 + vy2
```

(masses are 1, so momentum = velocity sum) is conserved whenever the net
external force is zero — true here by construction. It is a genuine,
non-trivial physical invariant, exactly analogous to energy conservation
in the oscillator, and (unlike energy) it does not disappear or become
harder to state when you formally verify it.

### 3. Why momentum over energy for the first Marabou conservation experiment

Momentum is **linear** in the state: `Px_next - Px` is just a sum and
difference of four network input/output variables. This can be encoded
as an exact Marabou linear inequality with no approximation:

```
|Px_next - Px| <= epsilon    and    |Py_next - Py| <= epsilon
```

Checked as two one-sided linear queries per axis (negating the claim):

```
upper violation:  Px_next - Px >= epsilon
lower violation:  Px  - Px_next >= epsilon
```

(symmetrically for `Py`). Energy conservation would require Marabou to
reason about `x^2` and `v^2` terms, which is out of scope for this first
conservation-law experiment (see `Do NOT` items below and Step 5+).

### 4. The exact property being verified

For a tolerance `epsilon` and domain `D` (below):

```
for every state in D:  |Px_next - Px| <= epsilon
for every state in D:  |Py_next - Py| <= epsilon
```

where `Px, Py` come from the network's *input* state and `Px_next,
Py_next` from its *output* state for that same input — i.e. this is a
property of the trained `models/particle_mlp.py` surrogate's one-step
map, not of the true simulator (which conserves momentum by construction,
see the sanity checks in `tests/test_two_particle_system.py`).

### 5. The verification domain

`verification/particle_properties.py` defines `D = box ∩ separation
constraints`:

```
box:  x1,y1 in [-1,1]      vx1,vy1 in [-1,1]
      x2,y2 in [-2.4,2.4]  vx2,vy2 in [-1,1]

separation constraints:  |x2 - x1| <= 1.4   and   |y2 - y1| <= 1.4
```

`box` is imported directly from `data/generate_particle_data.py`'s
`STATE_BOUNDS` — the same per-dimension ranges that script's sampling
scheme draws training data from, not a separately chosen region. But
that sampling scheme does *not* place particle 2 independently of
particle 1: it draws a random separation `d0` (in `[0.6, 1.4]`) and angle
around particle 1 (see that module's docstring), so an axis-aligned box
alone is a substantial **overapproximation** — box corners like
`x1=-1,y1=-1,x2=2.4,y2=2.4` (separation ≈ 4.8) were never within ~3× the
training data's actual separation range. The two extra linear separation
constraints above tighten `D` to much more closely match the true
sampled region while remaining exactly linear (no approximation of the
underlying Euclidean distance is needed — see the file for the derivation).
A residual gap remains (the box+diamond's diagonal corners still allow
separation up to `1.4*sqrt(2) ≈ 1.98`, versus the training disk's max of
1.4) — this is a known, documented limitation, not swept under the rug
(see Limitations below).

### 6. Random testing vs. formal verification

`verification/random_search_particle.py` samples 200,000 states from
exactly the same domain `D` (rejection-sampled to respect the separation
constraints) and checks the same two tolerances directly against the
model — no solver involved. Results from this run:

| property | Marabou (exhaustive) | random testing (200,000 samples) |
|---|---|---|
| `momentum_x_conserved_loose` (eps=1.0) | **UNSAT** — proven in 6.3s | 0/200,000 violated |
| `momentum_y_conserved_loose` (eps=1.0) | **UNSAT** — proven in 5.1s | 0/200,000 violated |
| `momentum_x_conserved_tight` (eps=1e-4) | **SAT** — counterexample in 0.011s | 193,456/200,000 violated |
| `momentum_y_conserved_tight` (eps=1e-4) | **SAT** — counterexample in 0.011s | 189,227/200,000 violated |

As in Step 3: 200,000 samples finding zero violations of the loose
property is consistent with the property being true, but it is **not a
proof** — some point among the uncountably many never sampled could
still violate it. Marabou's UNSAT is what actually rules that out, via a
complete search over the piecewise-linear network, not sampling. The
tight property shows the opposite case: since ~97% of random samples
already violate it, both methods agree easily — this property was chosen
specifically to demonstrate Marabou's SAT/counterexample path
(`expect="SAT"` in `particle_properties.py`), not because anyone expected
it to hold.

### Running the pipeline

```bash
python -m data.generate_particle_data      # dataset + train/val/test split
python -m models.train_particle            # train the MLP surrogate
python -m models.evaluate_particle         # test metrics, momentum analysis, rollout, plots
python -m verification.export_particle_onnx
python -m verification.verify_particle     # Marabou momentum verification
python -m verification.random_search_particle
pytest                                     # includes all Step 4 tests
```

### Baseline surrogate and momentum-conservation results (this run)

`models/particle_mlp.py` is a plain MLP, `8 -> 16 -> 16 -> 8` with ReLU
activations — deliberately smaller than an initial `8 -> 128 -> 128 -> 8`
attempt (see "Verification scalability" below) and, like Step 2's
oscillator model, has no conservation law built into its architecture.

Test-set state prediction (all 8 dims, physical units): MSE `7.5e-05`,
MAE `5.7e-03`, max error `6.9e-02`.

Momentum-conservation analysis on the test set (`N=15,000`):

| | mean\|.\| | max\|.\| | std |
|---|---|---|---|
| delta_Px | 1.40e-03 | 3.48e-02 | 1.25e-03 |
| delta_Py | 9.84e-04 | 2.12e-02 | 1.27e-03 |
| magnitude | 1.85e-03 | 3.52e-02 | 1.08e-03 |

Percentage of test examples with momentum error within tolerance:

| tolerance | Px | Py | magnitude |
|---|---|---|---|
| 1e-3 | 39.16% | 58.59% | 20.31% |
| 1e-4 | 3.38% | 6.17% | 0.23% |
| 1e-5 | 0.31% | 0.67% | 0.00% |

200-step autoregressive rollout from `state0 = [-0.5,0,0.4,0.1,0.5,0,0.1,-0.2]`
(true initial momentum `Px0=0.5, Py0=-0.1`):

- True simulator momentum drift over the rollout: `~1e-16` (conserved to
  machine precision, as expected — see `particle_momentum_drift.png`).
- **NN rollout momentum drift: `dPx = 0.300`, `dPy = 0.065`** — a large,
  visible drift from a model that was never told momentum should be
  conserved.

This is the central point of Step 4: the network's single-step MAE
(`5.7e-3`) looks small, and most individual predictions look accurate,
but **accurate state prediction does not imply exact conservation** —
nothing in a plain MLP's loss function or architecture penalizes
momentum-inconsistent predictions, so small per-step errors compound into
a large, visible drift under rollout, exactly as Step 2 found for energy
in the oscillator.

### Verification scalability (a limitation worth stating plainly)

An initial `8 -> 128 -> 128 -> 8` version of this model (per the original
architecture suggestion) could not be exactly verified within minutes —
a single loose-bound query did not resolve even with `timeoutInSeconds`
and 4 parallel workers (`numWorkers=4, snc=True`). The `8 -> 16 -> 16 -> 8`
version used here verifies the same kind of query in single-digit
seconds. This is not a footnote: it is direct evidence that **exact
ReLU-network verification cost scales sharply with network size**, and
it is exactly the kind of scalability question the later
standard-vs-geometric comparison (see below) needs to take seriously —
a geometric model that is more verifiable but must also be larger or
more expressive to match baseline accuracy is not a free win.

### What has and has not been formally established

**Established (proven, for exactly the stated domain D):** for every
state in `D` (see Section 5), this specific trained `particle_mlp.pt`
checkpoint satisfies `|Px_next - Px| <= 1.0` and `|Py_next - Py| <= 1.0`.
This is a complete proof over `D`, not a sample-based estimate — and
Marabou also found genuine counterexamples showing the same network does
**not** satisfy the tighter `epsilon = 1e-4` tolerance anywhere near as
strictly (both directions SAT, concrete counterexamples saved in
`verification/artifacts/particle_verification_results.json`).

**Not established:**
- Nothing about states outside `D` (in particular, states where the true
  particle separation exceeds ~1.4, which `D` still partially admits at
  its diagonal corners up to ~1.98 — see the domain-tightening discussion
  in Section 5).
- Nothing about **energy conservation** for this system — out of scope,
  deferred (quadratic, not linear).
- Nothing about multi-step rollout behavior — only the one-step map is
  verified; the rollout results above already show large momentum drift
  compounding over 200 steps, and this verification says nothing about
  that regime.
- These results are specific to this exact trained checkpoint;
  retraining (even with identical code) would require re-verification.
- Random testing over the same domain never *proves* the loose property
  either, no matter how many samples are used — see Section 6.

### Limitations

- The verification domain `D` is a box-plus-separation-constraints
  polytope, which is a strict (if much tighter than a plain box)
  overapproximation of the true disk-shaped sampled region — see Section
  5. A tighter polytope (more linear facets) or accepting nonlinear
  domain constraints would close this gap further but was out of scope
  here.
- Only momentum (linear) conservation was checked — not energy
  (nonlinear) or any equivariance property.
- The smaller `8->16->16->8` architecture (chosen for Marabou
  tractability) has higher single-step error than Step 2's oscillator
  model; a larger, more accurate network would very likely be harder or
  infeasible to verify exactly with this pipeline (see "Verification
  scalability" above) — this accuracy/verifiability tension is itself a
  finding, not just a limitation.
- `epsilon=1.0` for the "loose" property and `epsilon=1e-4` for the
  "tight" property were chosen from an exploratory sweep to reliably
  produce UNSAT and SAT respectively; they are not derived from any
  formal safety requirement.

### Research framing: preparing for the standard-vs-geometric comparison

Step 4 establishes the benchmark and methodology the rest of this thesis
needs: a physical system with a *linear* conservation law, a trained
baseline surrogate, a domain defined directly from the data-generation
process, and a working Marabou pipeline that can report both proven
(UNSAT) and violated (SAT) conservation properties with saved
counterexamples. The next stage will train a **geometric/equivariant**
model (Model B) on this same two-particle dataset and domain, and compare
it against this plain MLP baseline (Model A) on:

- **Physical conservation** — does a geometric inductive bias reduce
  momentum error the way it is designed to (e.g. structurally, rather
  than only empirically on the test set)?
- **Rollout stability** — does momentum drift less over long
  autoregressive rollouts?
- **Robustness** — does accuracy degrade less outside the training
  distribution (relevant to the domain-overapproximation gap noted
  above)?
- **Formal verifiability** — can the *same* linear momentum property be
  verified (UNSAT) at a *tighter* epsilon for the geometric model than
  for this MLP baseline?
- **Verification scalability** — does achieving comparable accuracy with
  a geometric model require a network that is easier, harder, or about
  the same to verify exactly, compared to the scaling problem already
  observed here (128-hidden intractable, 16-hidden fast)?

This step deliberately stops short of building that geometric model —
Step 4's job was only to prove the benchmark and pipeline work.
