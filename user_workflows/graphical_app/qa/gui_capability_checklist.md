# GUI Capability Checklist

This is the single source of truth for GUI implementation phases, branch flow, feature-to-code mapping, and QA gates.

## Branching and Merge Model

- Integration branch: `gui-integration`.
- Phase implementation branches (merged sequentially into `gui-integration`):
  1. `gui-phase-1-foundation`
  2. `gui-phase-2-devices`
  3. `gui-phase-3-patterns-and-plots`
  4. `gui-phase-4-optimization-calibration-sequences`
  5. `gui-phase-5-persistence-and-release-readiness`
- Merge order is strict. A phase branch may start only after the previous phase has been merged into `gui-integration`.

## Phase Done Criteria (must all pass before merge)

### Phase 1 — Foundation (`gui-phase-1-foundation`)

- Required UI behavior:
  - Main window loads and basic controls initialize.
  - Mode selector can switch between simulation and hardware labels without crash.
- Required backend behavior:
  - `AppController` boots with valid default `AppState`.
  - Functionality matrix reports available baseline capabilities.
- Required tests:
  - `tests/test_graphical_app_smoke.py::test_run_gui_app_bootstrap_path`
  - `tests/test_graphical_app_smoke.py::test_backend_gui_matrix_complete`
- Required metadata outputs:
  - Functionality matrix output from `user_workflows.graphical_app.qa.matrices.backend_to_gui_matrix`.

### Phase 2 — Devices (`gui-phase-2-devices`)

- Required UI behavior:
  - Connect/disconnect device controls operate in both simulation and hardware modes.
  - Device status and telemetry values refresh after connection events.
- Required backend behavior:
  - Device manager handles mode-aware connect and reconnect lifecycle.
  - Simulation and hardware adapters expose compatible APIs for camera/SLM operations.
- Required tests:
  - `tests/test_graphical_app_smoke.py::test_simulation_and_hardware_modes`
- Required metadata outputs:
  - Simulation/hardware capability matrix from `user_workflows.graphical_app.qa.matrices.sim_hw_matrix`.

### Phase 3 — Patterns and Plots (`gui-phase-3-patterns-and-plots`)

- Required UI behavior:
  - Pattern selection + parameter editing drives simulate-before-apply workflow.
  - Plot selector can render required plot families.
- Required backend behavior:
  - Pattern generator supports registered pattern set.
  - Plot backend exports data/image/settings metadata payload.
- Required tests:
  - `tests/test_graphical_app_smoke.py::test_simulation_and_hardware_modes`
  - Pattern registry tests under `slmsuite-main/testing/user_workflows/patterns/`.
- Required metadata outputs:
  - Plot export metadata bundle (data path, image path, settings payload).

### Phase 4 — Optimization, Calibration, Sequences (`gui-phase-4-optimization-calibration-sequences`)

- Required UI behavior:
  - Optimization run controls support start/pause/resume/stop.
  - Calibration profile load/save/apply workflow is accessible.
  - Sequence editor supports import/export and dry-run timing preview.
- Required backend behavior:
  - Optimization runner stores and exposes convergence history.
  - Calibration tooling enforces profile compatibility checks.
  - Sequence runner generates runtime step metadata.
- Required tests:
  - `tests/test_graphical_app_smoke.py::test_sequence_and_optimization`
- Required metadata outputs:
  - Optimization history export.
  - Calibration profile metadata.
  - Sequence runtime metadata.

### Phase 5 — Persistence and Release Readiness (`gui-phase-5-persistence-and-release-readiness`)

- Required UI behavior:
  - Layout/session persistence controls restore previous user state.
  - Final UI manual checklist passes with no blocking issues.
- Required backend behavior:
  - Persistence store can save/load layout and session artifacts.
  - Output naming and run metadata are stable across repeated runs.
- Required tests:
  - Full `tests/test_graphical_app_smoke.py` suite.
- Required metadata outputs:
  - Session snapshot metadata.
  - Layout persistence metadata.
  - Run naming metadata.

## Required Gate Before Starting Next Phase

Before any next phase branch is created, the current phase must satisfy both:

1. Smoke tests pass for phase-required test set.
2. Manual UI checklist is completed and attached to the phase merge record.

No exceptions: failing either gate blocks the next phase.

## Feature-to-Code-and-Test Mapping

| Requested feature | Primary code paths | Required tests |
|---|---|---|
| Simulation/hardware mode switching | `user_workflows/graphical_app/app/state.py`, `user_workflows/graphical_app/app/controller.py`, `user_workflows/graphical_app/devices/manager.py` | `tests/test_graphical_app_smoke.py::test_simulation_and_hardware_modes` |
| Device connect/reconnect lifecycle | `user_workflows/graphical_app/devices/manager.py`, `user_workflows/graphical_app/devices/adapters.py` | `tests/test_graphical_app_smoke.py::test_simulation_and_hardware_modes` |
| Pattern generation + simulate-before-apply | `user_workflows/graphical_app/app/patterns.py`, `user_workflows/graphical_app/app/controller.py`, `user_workflows/graphical_app/ui/main_window.py` | `tests/test_graphical_app_smoke.py::test_simulation_and_hardware_modes`, `slmsuite-main/testing/user_workflows/patterns/test_pattern_registry.py` |
| Required plots + export metadata | `user_workflows/graphical_app/plotting/backend.py`, `user_workflows/graphical_app/ui/main_window.py` | `tests/test_graphical_app_smoke.py::test_sequence_and_optimization` |
| Optimization workflow | `user_workflows/graphical_app/optimization/runner.py`, `user_workflows/graphical_app/app/controller.py` | `tests/test_graphical_app_smoke.py::test_sequence_and_optimization` |
| Calibration tools/profile workflow | `user_workflows/graphical_app/calibration/tools.py`, `user_workflows/graphical_app/app/controller.py` | `tests/test_graphical_app_smoke.py::test_sequence_and_optimization` |
| Sequence import/export + dry run | `user_workflows/graphical_app/sequence/runner.py`, `user_workflows/graphical_app/app/controller.py` | `tests/test_graphical_app_smoke.py::test_sequence_and_optimization` |
| Persistence (layout/session) | `user_workflows/graphical_app/persistence/store.py` | `tests/test_graphical_app_smoke.py` |
| Backend/GUI capability reporting | `user_workflows/graphical_app/qa/matrices.py` | `tests/test_graphical_app_smoke.py::test_backend_gui_matrix_complete` |

## Manual UI Checklist (execute each phase)

- [ ] Open GUI and confirm main window + controls render without exception.
- [ ] Verify mode toggle simulation ↔ hardware updates visible state.
- [ ] Verify connect/disconnect controls and status labels update correctly.
- [ ] Run simulate-before-apply for at least one pattern.
- [ ] Confirm required plots are selectable and render.
- [ ] Execute one optimization run and confirm convergence data appears.
- [ ] Load and apply one calibration profile.
- [ ] Import a sequence, run dry-run preview, and confirm timing display.
- [ ] Save and reload layout/session artifacts.
