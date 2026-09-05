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
