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


def test_blaze_composition_pipeline_and_snapshot_metadata(tmp_path):
    c = AppController()
    base = c.generate_pattern("single-gaussian", {"kx": 0.01, "ky": 0.0}).payload
    c.configure_blaze({"enabled": True, "kx": 0.2, "ky": -0.1, "offset": 0.3, "scale": 0.8})

    sim = c.simulate_before_apply(base)
    assert sim.success
    composed = sim.payload["simulated_phase"]
    assert composed.shape == base.shape
    assert not (composed == base).all()

    assert c.apply_pattern(base).success
    assert c.queue_pattern(base).success

    c.start_run("run-with-blaze", {"purpose": "test"})
    assert c.state.active_run is not None
    assert c.state.active_run.blaze["enabled"] is True
    snapshot = tmp_path / "session_snapshot.json"
    c.save_session_snapshot(str(snapshot))
    payload = c.persistence.load_json(snapshot)
    assert payload["blaze"]["enabled"] is True
    assert payload["blaze"]["kx"] == 0.2
