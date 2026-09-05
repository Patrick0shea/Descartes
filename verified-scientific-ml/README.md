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
