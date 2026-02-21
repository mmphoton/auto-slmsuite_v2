from pathlib import Path

import numpy as np

from user_workflows.graphical_app.plotting.backend import PlotBackend


def test_required_plots_exist_by_default() -> None:
    backend = PlotBackend()
    assert "simulated_phase" in backend.data
    assert "simulated_intensity" in backend.data


def test_export_includes_settings_metadata(tmp_path: Path) -> None:
    backend = PlotBackend()
    backend.update("experimental_intensity", np.arange(100, dtype=float).reshape(10, 10))
    backend.configure(
        "experimental_intensity",
        {
            "autoscale": False,
            "xlim": (2, 7),
            "ylim": (1, 5),
            "scale": "log",
            "colormap": "gray",
        },
    )
    export = backend.export("experimental_intensity", tmp_path)
    assert export.image_path.exists()
    assert export.data_path.exists()
    text = export.metadata_path.read_text()
    assert '"plot_name": "experimental_intensity"' in text
    assert '"colormap": "gray"' in text
