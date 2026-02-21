import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from user_workflows.graphical_app.app.controller import AppController
from user_workflows.graphical_app.app.state import Mode
from user_workflows.graphical_app.qa.matrices import backend_to_gui_matrix, release_freeze_ready, sim_hw_matrix


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
    assert sim_hw_matrix()["plot_export"]["hardware"]["status"] == "PASS"


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
    c.configure_output(folder=str(tmp_path), template="{date}_{session}_{run_id}_{artifact}", collision_policy="increment")
    c.start_run("run-opt", {"purpose": "optimization"})
    result = c.start_optimization({"wgs": {"max_iterations": 6, "gain": 0.15}})
    assert result.success

    progress = c.optimization_progress()
    assert progress.success
    assert progress.payload["iteration"] == 6

    assert c.pause_optimization().success
    assert c.resume_optimization().success
    assert c.stop_optimization().success

    export = c.export_optimization_history(run_id="run-opt")
    assert export.success
    out = Path(export.payload["csv"])
    assert out.exists()
    sidecar = out.with_suffix(".json")
    assert sidecar.exists()
    payload = c.persistence.load_json(sidecar)
    assert payload["run_metadata"]["run_id"] == "run-opt"
    assert payload["run_metadata"]["software_version"] == "0.1.0"
    assert payload["run_metadata"]["blaze"]["enabled"] is False


def test_ratio_target_optimization_and_export(tmp_path):
    c = AppController()
    c.configure_output(folder=str(tmp_path), template="{date}_{session}_{run_id}_{artifact}", collision_policy="increment")
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

    exported = c.export_optimization_history(run_id="ratio-run")
    assert exported.success
    out = Path(exported.payload["csv"])
    payload = c.persistence.load_json(out.with_suffix(".json"))
    assert payload["ratio_mode"] == "simulation"
    assert payload["target_definition"]["beam_count"] == 3
    assert payload["objective_weights"]["ratio"] == 0.6


def test_calibration_load_validate_apply_and_active_run_metadata(tmp_path):
    c = AppController()
    c.start_run("run-cal", {"purpose": "calibration"})

    profile = {
        "name": "sim-default",
        "mode": "simulation",
        "slm_model": "simulatedslm",
        "camera_model": "simulatedcamera",
        "matrix": [[1.0, 0.0], [0.0, 1.0]],
    }
    path = tmp_path / "profile.json"
    c.persistence.save_json(path, profile)

    loaded = c.load_calibration_profile(str(path))
    assert loaded.success
    validated = c.validate_calibration_profile(loaded.payload)
    assert validated.success
    assert validated.payload["compatibility"]["compatible"] is True

    applied = c.apply_calibration_profile(str(path))
    assert applied.success
    assert applied.payload["metrics"]["before_rmse"] >= 0.0
    assert c.state.settings_snapshots.calibration["profile_name"] == "sim-default"
    assert c.state.active_run is not None
    assert c.state.active_run.calibration_profile == "sim-default"
    assert "calibration_metrics" in c.state.active_run.parameters


def test_calibration_apply_blocked_on_incompatible_profile(tmp_path):
    c = AppController()
    incompatible = {
        "name": "hw-profile",
        "mode": "hardware",
        "slm_model": "hardwareslm",
        "camera_model": "hardwarecamera",
        "matrix": [[1.0, 0.0], [0.0, 1.0]],
    }
    path = tmp_path / "bad_profile.json"
    c.persistence.save_json(path, incompatible)

    validated = c.validate_calibration_profile(incompatible)
    assert validated.success
    assert validated.payload["compatibility"]["compatible"] is False

    applied = c.apply_calibration_profile(str(path))
    assert applied.success is False
    assert "compatibility failed" in applied.message.lower()


def test_naming_engine_applies_to_plot_data_and_snapshot_exports(tmp_path):
    c = AppController()
    configured = c.configure_output(
        folder=str(tmp_path),
        template="{date}_{session}_{run_id}_{artifact}",
        collision_policy="increment",
    )
    assert configured.success

    generated = c.generate_pattern("single-gaussian", {"kx": 0.01, "ky": 0.02})
    assert generated.success
    c.simulate_before_apply(generated.payload)

    plot_export = c.export_plot("simulated_phase", run_id="abc")
    assert plot_export.success
    image_path = Path(plot_export.payload["image_path"])
    assert image_path.exists()
    assert "plot_simulated_phase" in image_path.parent.name

    opt_export = c.export_optimization_history(run_id="abc")
    assert opt_export.success
    opt_path = Path(opt_export.payload["csv"])
    assert opt_path.exists()
    assert "data_optimization_history" in opt_path.name

    snapshot = c.save_session_snapshot(run_id="abc")
    assert snapshot.success
    snap_path = Path(snapshot.payload)
    assert snap_path.exists()
    snap_payload = c.persistence.load_json(snap_path)
    assert "metadata_snapshot" in snap_payload
    assert snap_payload["metadata_snapshot"]["software_version"] == "0.1.0"


def test_collision_policy_error_blocks_rewrite(tmp_path):
    c = AppController()
    assert c.configure_output(folder=str(tmp_path), template="{date}_{session}_{run_id}_{artifact}", collision_policy="error").success
    first = c.save_session_snapshot(run_id="same")
    assert first.success
    second = c.save_session_snapshot(run_id="same")
    assert second.success is False
    assert "already exists" in second.message.lower()


def test_smoke_mode_switching_connect_and_release_actions():
    c = AppController()
    assert c.set_mode("simulation").success
    assert c.connect_devices().success
    assert c.release_slm().success
    assert c.release_camera().success
    assert c.set_mode("hardware").success
    assert c.connect_devices().success
    assert c.release_both().success


def test_smoke_pattern_parameter_form_parity():
    from user_workflows.graphical_app.app.patterns import PatternService
    from user_workflows.graphical_app.ui.pattern_form import parity_check_for_schema

    service = PatternService()
    for pattern in service.available_patterns():
        schema = service.schema_for(pattern)
        represented = {param["name"] for param in schema["parameters"]}
        assert parity_check_for_schema(schema, represented) == []


def test_smoke_blaze_always_applied():
    c = AppController()
    base = c.generate_pattern("single-gaussian", {"kx": 0.01, "ky": 0.0}).payload
    c.configure_blaze({"enabled": True, "kx": 0.2, "ky": 0.1, "offset": 0.05, "scale": 0.9})
    simulated = c.simulate_before_apply(base)
    assert simulated.success
    assert not (simulated.payload["simulated_phase"] == base).all()


def test_smoke_plot_popout_edit_and_export_behavior(tmp_path):
    from user_workflows.graphical_app.persistence.store import PersistenceStore

    c = AppController()
    generated = c.generate_pattern("single-gaussian", {"kx": 0.03, "ky": 0.02})
    c.simulate_before_apply(generated.payload)

    assert c.configure_plot("simulated_phase", {"cmap": "viridis"}).success
    assert c.zoom_plot("simulated_phase", 1.1).success
    assert c.pan_plot("simulated_phase", 0.1, 0.1).success
    assert c.reset_plot("simulated_phase").success

    exported = c.export_plot("simulated_phase", output_dir=str(tmp_path), run_id="smoke")
    assert exported.success
    assert Path(exported.payload["image_path"]).exists()

    store = PersistenceStore()
    model = store.default_layout_model()
    model["popout_plots"] = [{"plot_name": "simulated_phase", "geometry": "700x560"}]
    path = tmp_path / "layout.json"
    store.save_layout_model(path, model)
    loaded = store.load_layout_model(path)
    assert loaded["popout_plots"] == model["popout_plots"]


def test_smoke_optimization_controls_and_outputs(tmp_path):
    c = AppController()
    c.configure_output(folder=str(tmp_path), template="{date}_{session}_{run_id}_{artifact}", collision_policy="increment")

    started = c.start_optimization({"wgs": {"max_iterations": 4, "gain": 0.1}})
    assert started.success
    assert started.payload["history"]

    progress = c.optimization_progress()
    assert progress.success
    assert "iteration" in progress.payload
    assert c.pause_optimization().success
    assert c.resume_optimization().success
    assert c.stop_optimization().success

    exported = c.export_optimization_history(run_id="smoke-opt")
    assert exported.success
    assert Path(exported.payload["csv"]).exists()
    assert Path(exported.payload["json"]).exists()


def test_smoke_calibration_load_and_validate(tmp_path):
    c = AppController()
    profile = {
        "name": "sim-default",
        "mode": "simulation",
        "slm_model": "simulatedslm",
        "camera_model": "simulatedcamera",
        "matrix": [[1.0, 0.0], [0.0, 1.0]],
    }
    path = tmp_path / "profile.json"
    c.persistence.save_json(path, profile)

    loaded = c.load_calibration_profile(str(path))
    assert loaded.success
    validated = c.validate_calibration_profile(loaded.payload)
    assert validated.success
    assert validated.payload["compatibility"]["compatible"] is True


def test_smoke_layout_persistence(tmp_path):
    from user_workflows.graphical_app.persistence.store import PersistenceStore

    store = PersistenceStore()
    model = store.default_layout_model()
    model["visibility"]["Logs"] = False
    model["columns"]["left"] = ["Device", "Session"]

    path = tmp_path / "layout.json"
    store.save_layout_model(path, model)
    loaded = store.load_layout_model(path)
    assert loaded["visibility"]["Logs"] is False
    assert loaded["columns"]["left"] == ["Device", "Session"]


def test_sim_hw_matrix_has_explicit_pass_fail_reporting():
    matrix = sim_hw_matrix()
    assert matrix
    for capability, modes in matrix.items():
        for mode_name, report in modes.items():
            assert mode_name in {"simulation", "hardware"}
            assert report["status"] in {"PASS", "FAIL"}
            assert isinstance(report["supported"], bool)
            assert report["reason"]


def test_release_freeze_requires_matrix_coverage_and_smoke_pass():
    blocked_for_smoke = release_freeze_ready(smoke_suite_passed=False)
    assert blocked_for_smoke["coverage_complete"] is True
    assert blocked_for_smoke["release_freeze_ready"] is False

    ready = release_freeze_ready(smoke_suite_passed=True)
    assert ready["coverage_complete"] is True
    assert ready["smoke_suite_passed"] is True
    assert ready["release_freeze_ready"] is True
