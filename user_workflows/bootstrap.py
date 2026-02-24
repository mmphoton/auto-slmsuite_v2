"""Environment bootstrap helpers for repository and HOLOEYE SDK paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_REPO_ROOT = Path(r"C:\Users\admin\Desktop\SLM_Python\v4\auto-slmsuite_v2-main").resolve()
DEFAULT_SDK_ROOT = Path(r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0").resolve()


def bootstrap_runtime(repo_root: str | Path | None = None, sdk_root: str | Path | None = None) -> tuple[Path, Path]:
    """Set python path + HEDS environment using user-overridable roots."""
    resolved_repo_root = Path(repo_root).resolve() if repo_root else DEFAULT_REPO_ROOT
    resolved_sdk_root = Path(sdk_root).resolve() if sdk_root else DEFAULT_SDK_ROOT

    if str(resolved_repo_root) not in sys.path:
        sys.path.insert(0, str(resolved_repo_root))
    os.chdir(resolved_repo_root)

    os.environ["HEDS_4_0_PYTHON"] = str(resolved_sdk_root)
    for sdk_path in (resolved_sdk_root / "examples", resolved_sdk_root / "api" / "python"):
        if sdk_path.exists() and str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))

    return resolved_repo_root, resolved_sdk_root

