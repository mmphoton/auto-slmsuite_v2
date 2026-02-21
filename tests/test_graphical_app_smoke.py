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


def test_camera_settings_schema_and_snapshot_metadata(tmp_path):
    c = AppController()
    schema_result = c.camera_settings_schema()
    assert schema_result.success
    assert "shutter_mode" in schema_result.payload

    settings = {
        "exposure_ms": 22.5,
        "gain": 2.0,
        "roi_x": 4,
        "roi_y": 6,
        "roi_width": 96,
        "roi_height": 88,
        "binning": 2,
        "trigger_mode": "software",
        "shutter_mode": "global",
        "fps": 18.0,
        "acquisition_mode": "continuous",
    }
    result = c.configure_camera(settings)
    assert result.success
    assert result.payload["roi"] == [4, 6, 96, 88]
    assert c.state.settings_snapshots.camera["shutter_mode"] == "global"

    c.start_run("run-camera-meta", {"purpose": "camera"})
    assert c.state.active_run is not None
    assert c.state.active_run.camera_settings["trigger_mode"] == "software"

    snapshot = tmp_path / "session_snapshot.json"
    c.save_session_snapshot(str(snapshot))
    payload = c.persistence.load_json(snapshot)
    assert payload["settings_snapshots"]["camera"]["fps"] == 18.0
    assert payload["active_run"]["camera_settings"]["shutter_mode"] == "global"


def test_camera_temperature_logging_occurs_on_refresh():
    c = AppController()
    before = len(c.state.logs)
    result = c.camera_telemetry()
    assert result.success
    assert len(c.state.logs) >= before + 1
    camera_logs = [entry for entry in c.state.logs if entry.source == "camera"]
    assert camera_logs
    assert "Camera temperature update" in camera_logs[-1].message


def test_optimization_controls_and_export(tmp_path):
    c = AppController()
    c.start_run("run-opt", {"purpose": "optimization"})
    result = c.start_optimization({"wgs": {"max_iterations": 6, "gain": 0.15}})
    assert result.success

    progress = c.optimization_progress()
    assert progress.success
    assert progress.payload["iteration"] == 6

    assert c.pause_optimization().success
    assert c.resume_optimization().success
    assert c.stop_optimization().success

    out = tmp_path / "optimization_history.csv"
    export = c.export_optimization_history(str(out))
    assert export.success
    assert out.exists()
    sidecar = out.with_suffix(".json")
    assert sidecar.exists()
    payload = c.persistence.load_json(sidecar)
    assert payload["run_metadata"]["run_id"] == "run-opt"


def test_ratio_target_optimization_and_export(tmp_path):
    c = AppController()
    config = {
        "wgs": {"max_iterations": 4, "gain": 0.1},
        "ratio_mode": "simulation",
        "target_definition": {
            "beam_count": 3,
            "beam_positions": [[32, 32], [64, 64], [96, 96]],
            "desired_ratios": [0.2, 0.3, 0.5],
            "lattice": {"geometry": "square", "spacing": 8.0, "rotation_deg": 0.0},
        },
        "ratio_targets": {"desired_ratios": [0.2, 0.3, 0.5]},
        "objective_weights": {"intensity": 1.0, "ratio": 0.6, "regularization": 0.05},
    }
    result = c.start_optimization(config)
    assert result.success
    assert "ratio_metrics" in result.payload

    ratio_plot = c.plots.get_plot_model("ratio_targets_vs_measured")
    assert ratio_plot.data.shape[0] == 2

    out = tmp_path / "ratio_history.csv"
    exported = c.export_optimization_history(str(out))
    assert exported.success
    payload = c.persistence.load_json(out.with_suffix(".json"))
    assert payload["ratio_mode"] == "simulation"
    assert payload["target_definition"]["beam_count"] == 3
    assert payload["objective_weights"]["ratio"] == 0.6
