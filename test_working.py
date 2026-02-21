"""Manual diagnostic script for projecting an LG30 phase pattern to the SLM.

This file intentionally keeps runtime-only hardware imports inside ``main()`` so
that test discovery (e.g. ``pytest``) does not fail on systems without the
vendor stack installed.
"""

from pathlib import Path
import time

import numpy as np
import scipy.io

from user_workflows.calibration_io import assert_required_calibration_files
from user_workflows.io.output_manager import OutputManager
from user_workflows.io.run_naming import RunNamingConfig


CALIBRATION_ROOT = "user_workflows/calibrations"


def main() -> None:
    # Hardware-dependent imports are kept local so repository tests can run in
    # environments where slmsuite is not installed as a package.
    from slmsuite.hardware.slms.holoeye import Holoeye
    from slmsuite.holography.toolbox import phase
    from user_workflows.patterns.utils import add_blaze_and_wrap, apply_depth_correction

    calibration_paths = assert_required_calibration_files(CALIBRATION_ROOT)
    output = OutputManager(
        RunNamingConfig(run_name="test_working", output_root=Path("user_workflows/output")),
        pattern="diagnostic",
        camera="none",
        metadata={"workflow": "test_working"},
    )

    deep = scipy.io.loadmat("deep_1024.mat").get("deep")
    if deep is None:
        raise ValueError("LUT file must contain variable 'deep'")
    deep = deep.squeeze()

    slm = Holoeye(preselect="index:0")
    ny, nx = slm.shape

    blaze_vector = (0.00, 0.0045)
    lg30_phase = phase.laguerre_gaussian(slm, l=3, p=0)

    phi = np.zeros((ny, nx))
    phi += lg30_phase

    phi_wrapped = add_blaze_and_wrap(base_phase=phi, grid=slm, blaze_vector=blaze_vector)
    corrected_phase = apply_depth_correction(phi_wrapped, deep)

    output.save_phase(corrected_phase)

    slm.set_phase(corrected_phase, settle=True)
    print("Pattern displayed on SLM")
    time.sleep(5)

    slm.set_phase(None, settle=True)
    slm.close()
    print("SLM cleared")


if __name__ == "__main__":
    main()
