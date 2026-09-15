# Verification Tools Pilot

A small, self-contained pilot testing which neural-network formal
verification tool to commit to for the thesis's real target: verifying
geometric/generative calorimeter shower simulators (latent vector -> grid
of non-negative "energy" values, roughly conserved in total).

This pilot trains a toy generator with the same input/output shape
(latent -> 2D energy grid) at three grid sizes (8x8, 16x16, 32x32) and
checks two properties -- output non-negativity and a bounded total sum
(a stand-in for energy conservation) -- with three tools: **auto_LiRPA**,
**alpha-beta-CROWN**, and **Marabou**.

See `REPORT.md` for what was found and the tool recommendation. This file
is only "how to run it."

## Setup

```bash
cd verification-tools-pilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # core + auto_LiRPA + Marabou deps (see comments in the file)

# auto_LiRPA: the PyPI release is stale (pins torch<1.13), install from source
pip install --no-deps "git+https://github.com/Verified-Intelligence/auto_LiRPA.git"
```

### alpha-beta-CROWN setup

Not pip-installable as a single package; needs a clone + editable install,
plus a long chain of dependencies pulled in one ImportError at a time (see
REPORT.md for the full story and timing):

```bash
git clone --depth 1 https://github.com/Verified-Intelligence/alpha-beta-CROWN.git /tmp/abcrown
cd /tmp/abcrown
pip install --no-deps -e .   # the pinned torch==2.11.0 dependency conflicts with torch already installed; --no-deps avoids that

# then, back in this venv, the dependencies actually needed at import/run time:
pip install pyyaml pandas psutil sortedcontainers sympy termcolor onnxoptimizer onnxsim pillow torchvision scikit-learn skl2onnx gurobipy
pip install --no-deps "git+https://github.com/Verified-Intelligence/onnx2pytorch@0826948e0d50cd09d909b7f26b2ee9ca2d3d71b3"
```

`verify_alpha_beta_crown.py` imports the `abcrown` package directly (it
adds `complete_verifier` to `sys.path` on import), so once the editable
install above succeeds, no extra `PYTHONPATH` setup is needed in this
project.

## Reproducing the pilot

```bash
source .venv/bin/activate

# per grid size: train, export ONNX, then run each tool standalone
python train.py --grid-size 8
python export_onnx.py --grid-size 8
python verify_auto_lirpa.py --grid-size 8
python verify_alpha_beta_crown.py --grid-size 8   # if installed
python verify_marabou.py --grid-size 8

# or run the full sweep (all grid sizes x all tools x both properties)
python run_pilot.py
```

Each `verify_*.py` script is runnable standalone and writes its own JSON
to `results/`; `run_pilot.py` orchestrates all of them (training/export
included) and additionally writes `results/summary.json`.

## Files

- `model.py` -- toy generator (`Generator`) and the fixed all-ones-sum
  wrapper (`GeneratorWithSum`) used for the bounded-sum property.
- `data.py` -- synthetic calorimeter-shower-like target grids (Gaussian
  bump + noise, clipped non-negative, scaled to a target sum).
- `train.py` -- brief training (300 Adam steps, MSE to synthetic targets).
- `export_onnx.py` -- exports both the plain generator and the sum-headed
  variant to ONNX (`dynamo=False`, see the script for why).
- `verify_auto_lirpa.py`, `verify_alpha_beta_crown.py`, `verify_marabou.py`
  -- one standalone script per tool; each exposes `run(grid_size) -> dict`.
- `run_pilot.py` -- full sweep orchestrator.
- `results/*.json` -- raw results per (tool, grid size); `summary.json`
  from a full `run_pilot.py` run.
- `REPORT.md` -- the actual findings and recommendation.
