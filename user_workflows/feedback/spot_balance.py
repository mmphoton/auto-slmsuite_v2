"""Closed-loop spot balancing workflow using camera ROI metrics and SpotHologram updates."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

ArrayLike = np.ndarray


@dataclass
class BalanceConfig:
    mode: str
    max_iterations: int
    uniformity_threshold: float
    max_gain_step: float
    roi_radius: int
    background_mode: str
    annulus_inner: int
    annulus_outer: int
    global_percentile: float
    hologram_method: str
    hologram_maxiter: int


def parse_spot_vectors(
    spot_kxy: list[str] | None,
    spot_ij: list[str] | None,
    mode: str,
) -> tuple[str, ArrayLike]:
    if bool(spot_kxy) == bool(spot_ij):
        raise ValueError("Provide exactly one of --spot-kxy or --spot-ij.")

    source = spot_kxy if spot_kxy else spot_ij
    basis = "kxy" if spot_kxy else "ij"

    pts: list[tuple[float, float]] = []
    for item in source or []:
        x_str, y_str = item.split(",")
        pts.append((float(x_str), float(y_str)))

    vectors = np.asarray(pts, dtype=float).T
    if vectors.shape[0] != 2 or vectors.shape[1] < 2:
        raise ValueError("Expected at least two spot coordinates in x,y format.")

    if mode == "two-spot" and vectors.shape[1] != 2:
        raise ValueError("--balance-mode two-spot requires exactly 2 coordinates.")
    if mode == "n-spot" and vectors.shape[1] < 2:
        raise ValueError("--balance-mode n-spot requires at least 2 coordinates.")

    return basis, vectors


def parse_target_weights(weights: str | None, n_spots: int) -> ArrayLike:
    if weights is None:
        out = np.full(n_spots, 1.0 / n_spots)
    else:
        out = np.asarray([float(x) for x in weights.split(",")], dtype=float)
        if out.size != n_spots:
            raise ValueError("Target weight count must match number of spots.")
        if np.any(out < 0):
            raise ValueError("Target weights must be non-negative.")
        s = out.sum()
        if s <= 0:
            raise ValueError("Target weights must sum to a positive value.")
        out = out / s
    return out


def _bounded_slice(center: int, radius: int, limit: int) -> tuple[int, int]:
    return max(0, center - radius), min(limit, center + radius + 1)


def extract_spot_intensities(
    frame: ArrayLike,
    spot_ij: ArrayLike,
    roi_radius: int,
    background_mode: str,
    annulus_inner: int,
    annulus_outer: int,
    global_percentile: float,
) -> ArrayLike:
    frame = np.asarray(frame, dtype=float)
    h, w = frame.shape
    intensities = np.zeros(spot_ij.shape[1], dtype=float)

    if background_mode == "global-percentile":
        global_bg = np.percentile(frame, global_percentile)
    else:
        global_bg = 0.0

    for idx in range(spot_ij.shape[1]):
        cx = int(round(spot_ij[0, idx]))
        cy = int(round(spot_ij[1, idx]))

        y0, y1 = _bounded_slice(cy, roi_radius, h)
        x0, x1 = _bounded_slice(cx, roi_radius, w)
        roi = frame[y0:y1, x0:x1]

        if background_mode == "annulus":
            ay0, ay1 = _bounded_slice(cy, annulus_outer, h)
            ax0, ax1 = _bounded_slice(cx, annulus_outer, w)
            patch = frame[ay0:ay1, ax0:ax1]
            yy, xx = np.indices(patch.shape)
            yy = yy + ay0 - cy
            xx = xx + ax0 - cx
            rr2 = xx * xx + yy * yy
            mask = (rr2 >= annulus_inner**2) & (rr2 <= annulus_outer**2)
            bg = float(np.median(patch[mask])) if np.any(mask) else global_bg
        elif background_mode == "global-percentile":
            bg = global_bg
        else:
            bg = 0.0

        corrected = np.sum(roi - bg)
        intensities[idx] = max(0.0, corrected)

    return intensities


def compute_uniformity(measured: ArrayLike, target_weights: ArrayLike) -> float:
    ratio = measured / np.maximum(target_weights, 1e-12)
    rmin = float(np.min(ratio))
    rmax = float(np.max(ratio))
    denom = rmax + rmin
    if denom <= 0:
        return 0.0
    return 1.0 - (rmax - rmin) / denom


def update_weights(
    current_weights: ArrayLike,
    measured_intensity: ArrayLike,
    target_weights: ArrayLike,
    max_gain_step: float,
) -> ArrayLike:
    measured_power = measured_intensity / max(np.sum(measured_intensity), 1e-12)
    gain = target_weights / np.maximum(measured_power, 1e-12)
    bounded_gain = np.clip(gain, 1.0 / max_gain_step, max_gain_step)
    updated = current_weights * bounded_gain
    updated /= np.sum(updated)
    return updated


def run_balance_loop(
    initial_weights: ArrayLike,
    target_weights: ArrayLike,
    measure_fn: Callable[[ArrayLike], ArrayLike],
    max_iterations: int,
    uniformity_threshold: float,
    max_gain_step: float,
) -> tuple[ArrayLike, list[dict[str, float]], ArrayLike]:
    weights = np.asarray(initial_weights, dtype=float)
    weights = weights / weights.sum()

    history: list[dict[str, float]] = []
    last_measured = np.zeros_like(weights)

    for iteration in range(max_iterations):
        measured = np.asarray(measure_fn(weights), dtype=float)
        last_measured = measured
        measured_norm = measured / max(measured.sum(), 1e-12)
        uniformity = compute_uniformity(measured_norm, target_weights)

        history.append(
            {
                "iteration": iteration,
                "uniformity": uniformity,
                "rmse": float(np.sqrt(np.mean((measured_norm - target_weights) ** 2))),
            }
        )

        if uniformity >= uniformity_threshold:
            break

        weights = update_weights(weights, measured, target_weights, max_gain_step=max_gain_step)

    return weights, history, last_measured




def run_spothologram_balance(
    slm_shape: tuple[int, int],
    spot_vectors: ArrayLike,
    basis: str,
    target_weights: ArrayLike,
    capture_frame: Callable[[], ArrayLike],
    display_phase: Callable[[ArrayLike], None],
    to_camera_ij: Callable[[ArrayLike], ArrayLike],
    config: BalanceConfig,
    cameraslm=None,
) -> tuple[ArrayLike, ArrayLike, list[dict[str, float]], ArrayLike]:
    """Run capture -> metric -> update -> redisplay with SpotHologram weight updates."""
    from slmsuite.holography.algorithms import SpotHologram

    spot_amps = np.sqrt(target_weights / np.sum(target_weights))
    hologram = SpotHologram(
        slm_shape,
        spot_vectors=spot_vectors,
        basis=basis,
        spot_amp=spot_amps,
        cameraslm=cameraslm,
    )

    working_weights = target_weights.copy()
    history: list[dict[str, float]] = []
    measured = np.zeros_like(target_weights)

    for iteration in range(config.max_iterations):
        hologram.spot_amp = np.sqrt(np.maximum(working_weights, 1e-12))
        hologram.external_spot_amp = np.copy(hologram.spot_amp)
        hologram.set_target(reset_weights=True)
        hologram.optimize(
            method=config.hologram_method,
            maxiter=config.hologram_maxiter,
            feedback="computational",
            stat_groups=["computational"],
        )

        phase = np.mod(hologram.get_phase(), 2 * np.pi)
        display_phase(phase)

        frame = capture_frame()
        spot_ij = to_camera_ij(spot_vectors) if basis == "kxy" else spot_vectors
        measured = extract_spot_intensities(
            frame,
            spot_ij=spot_ij,
            roi_radius=config.roi_radius,
            background_mode=config.background_mode,
            annulus_inner=config.annulus_inner,
            annulus_outer=config.annulus_outer,
            global_percentile=config.global_percentile,
        )

        measured_norm = measured / max(np.sum(measured), 1e-12)
        uniformity = compute_uniformity(measured_norm, target_weights)
        history.append(
            {
                "iteration": iteration,
                "uniformity": uniformity,
                "rmse": float(np.sqrt(np.mean((measured_norm - target_weights) ** 2))),
            }
        )

        if uniformity >= config.uniformity_threshold:
            break

        working_weights = update_weights(
            working_weights,
            measured,
            target_weights=target_weights,
            max_gain_step=config.max_gain_step,
        )

    return np.mod(hologram.get_phase(), 2 * np.pi), measured, history, working_weights


def save_balance_outputs(
    output_dir: Path,
    phase: ArrayLike | None,
    measured: ArrayLike,
    history: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if phase is not None:
        np.save(output_dir / "final_phase.npy", phase)

    np.save(output_dir / "final_measured_intensities.npy", measured)

    with (output_dir / "balance_metrics.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["iteration", "uniformity", "rmse"])
        writer.writeheader()
        writer.writerows(history)

    with (output_dir / "balance_metrics.json").open("w") as fp:
        json.dump(history, fp, indent=2)

    if history:
        iterations = [row["iteration"] for row in history]
        uniformity = [row["uniformity"] for row in history]
        rmse = [row["rmse"] for row in history]

        plt.figure(figsize=(7, 4))
        plt.plot(iterations, uniformity, marker="o", label="uniformity")
        plt.plot(iterations, rmse, marker="s", label="rmse")
        plt.xlabel("Iteration")
        plt.ylabel("Metric")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "balance_metrics.png", dpi=150)
        plt.close()


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balance-mode", choices=["two-spot", "n-spot"], default="n-spot")
    parser.add_argument("--spot-kxy", action="append", help="Spot coordinate kx,ky (repeat per spot).")
    parser.add_argument("--spot-ij", action="append", help="Spot coordinate i,j (repeat per spot).")
    parser.add_argument("--target-weights", default=None, help="Comma-separated target weights.")

    parser.add_argument("--roi-radius", type=int, default=5)
    parser.add_argument("--background-mode", choices=["none", "annulus", "global-percentile"], default="annulus")
    parser.add_argument("--annulus-inner", type=int, default=8)
    parser.add_argument("--annulus-outer", type=int, default=12)
    parser.add_argument("--global-percentile", type=float, default=10.0)

    parser.add_argument("--max-gain-step", type=float, default=1.2)
    parser.add_argument("--uniformity-threshold", type=float, default=0.97)
    parser.add_argument("--max-iterations", type=int, default=20)

    parser.add_argument("--hologram-method", default="WGS-Kim")
    parser.add_argument("--hologram-maxiter", type=int, default=20)
    parser.add_argument("--output-dir", default="user_workflows/feedback/outputs")

    parser.add_argument("--simulate", action="store_true", help="Run synthetic closed-loop balance without hardware.")
    parser.add_argument("--simulate-imbalance", default="0.35,1.75", help="Per-spot multiplicative imbalance.")
    parser.add_argument("--simulate-noise", type=float, default=0.0)
    return parser


def _simulate_measurement_factory(imbalance: ArrayLike, noise_sigma: float = 0.0):
    rng = np.random.default_rng(1234)

    def measure(weights: ArrayLike) -> ArrayLike:
        noisy = 1.0 + rng.normal(scale=noise_sigma, size=weights.size)
        return np.maximum(1e-9, weights * imbalance * noisy)

    return measure


def run_simulated(args: argparse.Namespace) -> None:
    basis, spot_vectors = parse_spot_vectors(args.spot_kxy, args.spot_ij, args.balance_mode)
    _ = basis
    target = parse_target_weights(args.target_weights, spot_vectors.shape[1])
    imbalance = np.asarray([float(x) for x in args.simulate_imbalance.split(",")], dtype=float)
    if imbalance.size != spot_vectors.shape[1]:
        raise ValueError("--simulate-imbalance length must match number of spots.")

    initial = np.full_like(target, 1.0 / target.size)
    measure = _simulate_measurement_factory(imbalance, args.simulate_noise)
    final_weights, history, measured = run_balance_loop(
        initial,
        target,
        measure,
        max_iterations=args.max_iterations,
        uniformity_threshold=args.uniformity_threshold,
        max_gain_step=args.max_gain_step,
    )

    save_balance_outputs(Path(args.output_dir), phase=None, measured=measured, history=history)
    print(f"Simulated balance complete. Final weights: {final_weights}")
    print(f"Final uniformity: {history[-1]['uniformity']:.4f}")


def main() -> None:
    args = build_cli().parse_args()

    if args.simulate:
        run_simulated(args)
        return

    raise RuntimeError(
        "Hardware run path is workflow-specific and requires camera/SLM setup. "
        "Use --simulate for validation or integrate these helpers with your lab capture/display stack."
    )


if __name__ == "__main__":
    main()
