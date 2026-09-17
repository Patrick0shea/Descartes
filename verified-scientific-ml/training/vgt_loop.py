"""
Verification-Guided Training (VGT) loop for neural physics surrogates.

The core novel contribution of the thesis. A training procedure that uses
a formal SMT solver (Marabou) in a feedback loop with the physics simulator
to produce surrogates with formally-verified physical properties.

Algorithm
---------
Given a pre-trained Model C (soft geometric surrogate) and an epsilon
schedule [ε₁, ε₂, ...] (in decreasing order of tolerance, i.e. harder
to prove), the loop does:

    for each epsilon:
        for up to max_iters_per_epsilon iterations:
            1. Fine-tune model on D_aug (augmented dataset, starts as D_train)
            2. Export model to temp ONNX
            3. Marabou query: does |ΔPx| ≤ ε AND |ΔPy| ≤ ε hold over D?
               - UNSAT (both axes) → property proven, move to next tighter ε
               - SAT (either axis) → counterexample found:
                   x_ce = worst-case input state from Marabou
                   sample `neighbourhood_size` neighbours near x_ce in D
                   for each state: call physics simulator → get ground truth
                   add all to D_aug
        if did not converge within max_iters → stop (report tightest ε proven)

Why this is better than adversarial training
--------------------------------------------
Adversarial training uses gradient-based attacks, which find LOCAL violations
near training data. Marabou performs a COMPLETE search — if it returns SAT,
the counterexample is the GLOBAL worst case for that epsilon over the entire
domain D. If it returns UNSAT, the property is formally proven. No amount of
empirical sampling achieves this guarantee.

Usage
-----
    from training.vgt_loop import VGTTrainer
    trainer = VGTTrainer(model, train_x, train_y, epsilon_schedule=[1e-3, 1e-4, 1e-5],
                         temp_onnx_path="verification/artifacts/vgt_temp.onnx")
    history = trainer.run()
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.generate_particle_data import DT, K, REST_LENGTH, M1, M2
from models.particle_soft_geometric import ParticleSoftGeometricMLP
from verification.particle_properties import (
    DOMAIN_LOWER,
    DOMAIN_UPPER,
    SEPARATION_BOUND,
    VX1, VX2, VY1, VY2,
    X1, X2, Y1, Y2,
)
from verification.verify_particle import _solve_one_sided
from simulator.two_particle_system import simulate

_DOMAIN_LOWER = np.array(DOMAIN_LOWER, dtype=np.float32)
_DOMAIN_UPPER = np.array(DOMAIN_UPPER, dtype=np.float32)


def _simulator_step(state: np.ndarray) -> np.ndarray:
    """Call the physics simulator for one timestep DT, return next state (shape 8,)."""
    result = simulate(
        state.astype(np.float64),
        k=K,
        rest_length=REST_LENGTH,
        m1=M1,
        m2=M2,
        t_span=(0.0, DT),
        n_points=2,
    )
    return result["states"][1].astype(np.float32)


def _in_domain(state: np.ndarray) -> bool:
    """Return True if state satisfies the box + separation constraints of domain D."""
    if np.any(state < _DOMAIN_LOWER) or np.any(state > _DOMAIN_UPPER):
        return False
    if abs(state[X2] - state[X1]) > SEPARATION_BOUND:
        return False
    if abs(state[Y2] - state[Y1]) > SEPARATION_BOUND:
        return False
    return True


def _sample_neighbourhood(
    x_ce: np.ndarray,
    n: int,
    sigma: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> list[np.ndarray]:
    """
    Sample up to n states near counterexample x_ce that lie within domain D.

    Adds Gaussian noise (std=sigma) to x_ce, clips to the box, and keeps
    samples that also satisfy the separation constraints. Returns fewer than
    n samples if the neighbourhood overlaps the domain boundary heavily.
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = []
    max_attempts = n * 20
    for _ in range(max_attempts):
        if len(samples) >= n:
            break
        noise = rng.normal(0.0, sigma, size=8).astype(np.float32)
        candidate = np.clip(x_ce + noise, _DOMAIN_LOWER, _DOMAIN_UPPER)
        if _in_domain(candidate):
            samples.append(candidate)
    return samples


def _export_onnx(model: ParticleSoftGeometricMLP, onnx_path: str) -> None:
    """Export model to ONNX (legacy exporter, Marabou-compatible)."""
    model.eval()
    dummy = torch.zeros((1, 8), dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["state"],
        output_names=["next_state"],
        opset_version=13,
        dynamo=False,
    )


def _verify_axis(onnx_path: str, epsilon: float, axis: str) -> dict:
    """
    Run both one-sided Marabou queries for one momentum axis.

    Returns dict with keys: result ("UNSAT"/"SAT"), time_s, counterexample (or None).
    """
    axis_vars = (VX1, VX2) if axis == "x" else (VY1, VY2)
    upper = _solve_one_sided(onnx_path, axis_vars, epsilon, "upper")
    lower = _solve_one_sided(onnx_path, axis_vars, epsilon, "lower")
    overall = "SAT" if (upper["result"] == "SAT" or lower["result"] == "SAT") else "UNSAT"
    counterexample = upper["counterexample"] or lower["counterexample"]
    total_time = upper["verification_time_seconds"] + lower["verification_time_seconds"]
    return {"result": overall, "time_s": total_time, "counterexample": counterexample}


class VGTTrainer:
    """
    Verification-Guided Training loop for ParticleSoftGeometricMLP (Model C).

    Parameters
    ----------
    model : ParticleSoftGeometricMLP
        The model to train. Should be pre-loaded with a good starting checkpoint
        (e.g. the standard-trained Model C). Modified IN PLACE.
    train_x : np.ndarray, shape (N, 8)
        Original training inputs (D_train). Used as the base of D_aug.
    train_y : np.ndarray, shape (N, 8)
        Corresponding ground-truth next states.
    epsilon_schedule : list[float]
        Epsilons to try, from least tight to tightest (e.g. [1e-3, 1e-4, 1e-5]).
        The loop stops as soon as it fails to converge at an epsilon.
    temp_onnx_path : str
        File path for the temporary ONNX export written each iteration.
    epochs_per_iter : int
        Fine-tuning epochs on D_aug per VGT iteration (default 50).
    max_iters_per_epsilon : int
        Maximum VGT iterations before giving up on an epsilon (default 15).
    neighbourhood_size : int
        Number of neighbours to sample around each counterexample (default 100).
    lr : float
        Adam learning rate for fine-tuning (default 1e-3).
    batch_size : int
        Mini-batch size for fine-tuning (default 256).
    seed : int
        RNG seed for neighbourhood sampling (default 42).
    """

    def __init__(
        self,
        model: ParticleSoftGeometricMLP,
        train_x: np.ndarray,
        train_y: np.ndarray,
        epsilon_schedule: list[float],
        temp_onnx_path: str,
        epochs_per_iter: int = 50,
        max_iters_per_epsilon: int = 15,
        neighbourhood_size: int = 100,
        lr: float = 1e-3,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.aug_x = list(train_x.astype(np.float32))
        self.aug_y = list(train_y.astype(np.float32))
        self.n_original = len(self.aug_x)
        self.epsilon_schedule = epsilon_schedule
        self.temp_onnx_path = temp_onnx_path
        self.epochs_per_iter = epochs_per_iter
        self.max_iters_per_epsilon = max_iters_per_epsilon
        self.neighbourhood_size = neighbourhood_size
        self.lr = lr
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self._optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
        self._loss_fn = nn.MSELoss()

    def _finetune(self) -> float:
        """Fine-tune model on current D_aug for epochs_per_iter epochs. Returns final epoch loss."""
        x_arr = np.array(self.aug_x, dtype=np.float32)
        y_arr = np.array(self.aug_y, dtype=np.float32)
        x_t = torch.from_numpy(x_arr)
        y_t = torch.from_numpy(y_arr)
        loader = DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.model.train()
        final_loss = 0.0
        for _ in range(self.epochs_per_iter):
            epoch_loss = 0.0
            n = 0
            for xb, yb in loader:
                self._optimizer.zero_grad()
                loss = self._loss_fn(self.model(xb), yb)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
                n += 1
            final_loss = epoch_loss / max(n, 1)
        self.model.eval()
        return final_loss

    def _add_counterexample(self, x_ce: np.ndarray) -> int:
        """
        Add x_ce and its neighbourhood to D_aug, with simulator ground-truth labels.
        Returns the number of points added.
        """
        added = 0
        # Add the counterexample itself
        y_ce = _simulator_step(x_ce)
        self.aug_x.append(x_ce.astype(np.float32))
        self.aug_y.append(y_ce)
        added += 1
        # Add neighbourhood
        neighbours = _sample_neighbourhood(x_ce, self.neighbourhood_size, rng=self.rng)
        for x_nb in neighbours:
            y_nb = _simulator_step(x_nb)
            self.aug_x.append(x_nb)
            self.aug_y.append(y_nb)
            added += 1
        return added

    def run(self) -> dict:
        """
        Run the full VGT loop. Returns a history dict with per-iteration records
        and a summary of which epsilons were formally verified.
        """
        history: dict = {
            "epsilon_results": [],
            "iterations": [],
            "final_epsilon_unsat": None,
            "n_original_train": self.n_original,
        }

        for epsilon in self.epsilon_schedule:
            print(f"\n--- VGT: targeting epsilon={epsilon:.1e} ---")
            converged = False

            for iteration in range(self.max_iters_per_epsilon):
                t_start = time.time()

                # Step 1: fine-tune
                train_loss = self._finetune()

                # Step 2: export ONNX
                _export_onnx(self.model, self.temp_onnx_path)

                # Step 3: verify both axes
                res_x = _verify_axis(self.temp_onnx_path, epsilon, "x")
                res_y = _verify_axis(self.temp_onnx_path, epsilon, "y")

                n_added = 0
                overall = "UNSAT" if (res_x["result"] == "UNSAT" and res_y["result"] == "UNSAT") else "SAT"

                iter_record = {
                    "epsilon": epsilon,
                    "iteration": iteration,
                    "result": overall,
                    "result_x": res_x["result"],
                    "result_y": res_y["result"],
                    "verify_time_x_s": res_x["time_s"],
                    "verify_time_y_s": res_y["time_s"],
                    "train_loss": train_loss,
                    "n_augmented_total": len(self.aug_x) - self.n_original,
                    "n_added_this_iter": 0,
                    "wall_time_s": time.time() - t_start,
                }

                print(
                    f"  iter {iteration:2d}  Px={res_x['result']}  Py={res_y['result']}"
                    f"  loss={train_loss:.3e}  D_aug={len(self.aug_x)}"
                    f"  ({res_x['time_s']:.1f}s + {res_y['time_s']:.1f}s verify)"
                )

                if overall == "UNSAT":
                    history["final_epsilon_unsat"] = epsilon
                    converged = True
                    history["iterations"].append(iter_record)
                    print(f"  -> UNSAT proven at epsilon={epsilon:.1e} after {iteration+1} iterations")
                    break

                # Step 4: add counterexamples to D_aug
                for res in [res_x, res_y]:
                    if res["counterexample"] is not None:
                        x_ce = np.array(res["counterexample"]["input_state"], dtype=np.float32)
                        n_added += self._add_counterexample(x_ce)

                iter_record["n_added_this_iter"] = n_added
                history["iterations"].append(iter_record)

            history["epsilon_results"].append({
                "epsilon": epsilon,
                "converged": converged,
                "iterations_used": iteration + 1,
                "n_augmented_total": len(self.aug_x) - self.n_original,
            })

            if not converged:
                print(
                    f"  -> Did not converge at epsilon={epsilon:.1e} "
                    f"after {self.max_iters_per_epsilon} iterations. Stopping."
                )
                break

        if history["final_epsilon_unsat"] is None:
            print("\nVGT did not achieve UNSAT at any epsilon in the schedule.")
        else:
            print(f"\nVGT complete. Tightest formally-verified epsilon: {history['final_epsilon_unsat']:.1e}")

        return history
