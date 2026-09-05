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
