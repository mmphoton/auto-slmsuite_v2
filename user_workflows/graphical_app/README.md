# Graphical Application Architecture

This package provides a production-oriented GUI architecture organized into:

- `ui/`: Tkinter dockable/resizable windows and controls.
- `app/`: central controller, interfaces, and app state.
- `devices/`: mode-aware device manager and simulation/hardware adapters.
- `plotting/`: plot backend with export of data/image/settings metadata.
- `persistence/`: sessions, layouts, snapshots, naming templates.

## Included capabilities

- Unified simulation/hardware mode handling and reconnect-safe lifecycle controls.
- Schema-driven pattern editing and simulate-before-apply workflow.
- Camera controls (exposure, gain, ROI, binning, trigger/FPS, acquisition mode) and telemetry temperature warning.
- Required plots: simulated phase/intensity, experimental intensity, optimization convergence.
- Optimization runner with start/pause/resume/stop and history export.
- Calibration profile save/load/apply compatibility checks.
- Sequence runner with import/export, dry-run timing, runtime step metadata.
- Backend-to-GUI functionality matrix and simulation/hardware QA matrix.

Run GUI:

```bash
python user_workflows/run_gui_app.py
```


### Spyder / IDE launch note
If you run `run_gui_app.py` directly from inside the `user_workflows/` folder, the launcher now bootstraps the repository root onto `sys.path`, so `user_workflows` imports resolve correctly.
