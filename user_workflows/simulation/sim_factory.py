"""Factories for simulated SLM/camera/FourierSLM workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.simulated import SimulatedSLM


@dataclass(frozen=True)
class SimulationConfig:
    """Container for deterministic simulation setup."""

    seed: int = 0
    scenario: str = "two-spot-imbalance"
    resolution: tuple[int, int] = (512, 512)
    pitch_um: tuple[float, float] = (8.0, 8.0)
    bitdepth: int = 12
    wav_um: float = 0.78
    include_aberrations: bool = True
    include_noise: bool = True


def _source_for_scenario(config: SimulationConfig) -> dict[str, np.ndarray]:
    ny, nx = config.resolution[1], config.resolution[0]
    yy, xx = np.indices((ny, nx), dtype=float)
    x_norm = (xx - nx / 2) / max(nx / 2, 1)
    y_norm = (yy - ny / 2) / max(ny / 2, 1)

    amplitude = np.exp(-((x_norm / 0.8) ** 2 + (y_norm / 0.8) ** 2))

    if config.scenario == "two-spot-imbalance":
        amplitude *= 1.0 + 0.25 * x_norm
    elif config.scenario == "n-spot-lattice-nonuniform":
        amplitude *= 0.8 + 0.2 * np.cos(5 * np.pi * x_norm) * np.cos(3 * np.pi * y_norm)
    elif config.scenario == "high-noise-failure":
        amplitude *= 0.65 + 0.35 * np.cos(2 * np.pi * y_norm)
    else:
        raise ValueError(f"Unknown simulation scenario '{config.scenario}'.")

    phase = np.zeros_like(amplitude)
    if config.include_aberrations:
        phase += 0.4 * (x_norm**2 - y_norm**2)
        phase += 0.2 * (x_norm * y_norm)

    amplitude = np.clip(amplitude, 1e-4, None)
    amplitude /= np.max(amplitude)
    return {
        "amplitude_sim": amplitude,
        "phase_sim": phase,
        "amplitude": amplitude.copy(),
        "phase": phase.copy(),
    }


def _noise_for_scenario(config: SimulationConfig, rng: np.random.Generator):
    if not config.include_noise:
        return None

    if config.scenario == "high-noise-failure":
        dark_sigma = 0.12
        read_lambda = 0.2
    elif config.scenario == "n-spot-lattice-nonuniform":
        dark_sigma = 0.03
        read_lambda = 0.03
    else:
        dark_sigma = 0.02
        read_lambda = 0.01

    return {
        "dark": lambda img: rng.normal(loc=0.01 * img, scale=dark_sigma * img),
        "read": lambda img: rng.poisson(lam=read_lambda * img),
    }


def build_simulated_fourier_slm(
    *,
    seed: int = 0,
    scenario: str = "two-spot-imbalance",
    include_aberrations: bool = True,
    include_noise: bool = True,
    resolution: tuple[int, int] = (512, 512),
):
    """Build a deterministic simulated FourierSLM stack for user workflows."""
    config = SimulationConfig(
        seed=seed,
        scenario=scenario,
        include_aberrations=include_aberrations,
        include_noise=include_noise,
        resolution=resolution,
    )
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    slm = SimulatedSLM(
        resolution=config.resolution,
        pitch_um=config.pitch_um,
        bitdepth=config.bitdepth,
        wav_um=config.wav_um,
        source=_source_for_scenario(config),
        name=f"sim-slm-{config.scenario}",
    )

    cam = SimulatedCamera(
        slm,
        resolution=config.resolution,
        bitdepth=config.bitdepth,
        noise=_noise_for_scenario(config, rng),
        gain=1,
        name=f"sim-cam-{config.scenario}",
    )

    return FourierSLM(cam, slm)
