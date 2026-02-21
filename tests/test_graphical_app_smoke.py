import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from user_workflows.graphical_app.app.controller import AppController
from user_workflows.graphical_app.app.state import Mode
from user_workflows.graphical_app.qa.matrices import backend_to_gui_matrix, sim_hw_matrix


def test_backend_gui_matrix_complete():
    matrix = backend_to_gui_matrix()
    assert matrix
    assert all(matrix.values())


def test_simulation_and_hardware_modes():
    c = AppController()
    c.set_mode("simulation")
    c.devices.connect()
    p = c.generate_pattern("single-gaussian", {"kx": 0.01, "ky": 0.02})
    assert p.success and p.payload is not None
    outputs = c.simulate_before_apply(p.payload)
    assert outputs.success
    assert "simulated_phase" in outputs.payload
    c.set_mode("hardware")
    c.devices.connect()
    assert c.state.mode == Mode.HARDWARE


def test_sequence_and_optimization():
    c = AppController()
    c.sequence.import_sequence([
        {"pattern": "single-gaussian", "duration_ms": 10},
        {"pattern": "double-gaussian", "duration_ms": 20},
    ])
    timing = c.sequence.dry_run_timing()
    assert timing["steps"] == 2
    c.run_optimization({"iterations": 5, "initial": 1.0})
    assert len(c.optimizer.history()) > 0
    assert sim_hw_matrix()["plot_export"]["hardware"]


def test_run_gui_app_bootstrap_path():
    import importlib

    module = importlib.import_module("user_workflows.run_gui_app")
    repo_root = module._ensure_repo_root_on_path()
    assert str(repo_root) in sys.path


def test_gui_capability_checklist_has_phase_gates():
    checklist = Path("user_workflows/graphical_app/qa/gui_capability_checklist.md")
    content = checklist.read_text(encoding="utf-8")

    assert "gui-integration" in content
    assert "gui-phase-1-foundation" in content
    assert "gui-phase-2-devices" in content
    assert "gui-phase-3-patterns-and-plots" in content
    assert "gui-phase-4-optimization-calibration-sequences" in content
    assert "gui-phase-5-persistence-and-release-readiness" in content
    assert "Smoke tests pass" in content
    assert "Manual UI checklist" in content
