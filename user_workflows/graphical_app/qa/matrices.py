"""Backend-to-GUI and simulation/hardware QA matrices."""

from __future__ import annotations

from user_workflows.graphical_app.app.controller import AppController


def backend_to_gui_matrix() -> dict[str, bool]:
    return AppController().functionality_matrix()


def sim_hw_matrix() -> dict[str, dict[str, bool]]:
    return {
        "device_connect": {"simulation": True, "hardware": True},
        "simulate_before_apply": {"simulation": True, "hardware": True},
        "plot_export": {"simulation": True, "hardware": True},
        "layout_persistence": {"simulation": True, "hardware": True},
        "sequence_dry_run": {"simulation": True, "hardware": True},
    }
