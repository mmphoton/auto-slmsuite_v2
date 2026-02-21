"""Entry-point for the graphical SLM application.

Supports both:
- `python user_workflows/run_gui_app.py` from repository root.
- IDE execution from inside `user_workflows/` (e.g. Spyder runfile), where the
  repo root may not already be on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> Path:
    """Add repository root to ``sys.path`` when launched as a loose script."""
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


_ensure_repo_root_on_path()

from user_workflows.graphical_app.ui.main_window import launch


if __name__ == "__main__":
    launch()
