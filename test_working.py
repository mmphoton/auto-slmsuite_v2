import numpy as np
import scipy.io
from slmsuite.hardware.slms.holoeye import Holoeye
from slmsuite.holography.toolbox.phase import blaze
from slmsuite.holography import toolbox
from slmsuite.holography.toolbox import phase
from slmsuite.holography.algorithms import Hologram
from slmsuite.holography.algorithms import FeedbackHologram
from slmsuite.holography.algorithms import SpotHologram


import time

# === PARAMETERS ===

# Gaussian beam radius (std dev) in pixels
sigma = 100000000000000000000000000000000000

# Blaze grating strength (normalized kx/k, ky/k)
blaze_kx = 0.00
blaze_ky = 0.0045
blaze_vector = (blaze_kx, blaze_ky)





# Path to your MATLAB LUT file
lut_file = "deep_1024.mat"

# === LOAD LUT ===
mat = scipy.io.loadmat(lut_file)
if 'deep' not in mat:
    raise ValueError("LUT file must contain variable 'deep'")
deep = mat['deep'].squeeze()  # flatten to 1D
nLUT = deep.shape[0]
# === INITIALIZE SLM ===
slm = Holoeye(preselect="index:0")  # change index if needed
ny, nx = slm.shape

# === CREATE GAUSSIAN PHASE ===
x = np.linspace(-nx/2, nx/2, nx)
y = np.linspace(-ny/2, ny/2, ny)
X, Y = np.meshgrid(x, y)

target_amp = np.exp(-(X**2 + Y**2)/(2.0*sigma**2))
indices = np.clip(np.rint(target_amp*(nLUT-1)), 0, nLUT-1).astype(int)
# LUT value for each pixel
F = deep[indices]

#### Lg30 phase
lg30_phase = phase.laguerre_gaussian(slm, l=3,p=0)


# === COMPUTE PHASE + BLAZE ===
phi = np.zeros((ny, nx))
phi += lg30_phase

# # add blaze ramp
blaze_phase = blaze(grid=slm, vector=blaze_vector)
# phi += blaze_phase
phi += blaze_phase

# wrap to 0..2pi
phi_wrapped = np.mod(phi, 2*np.pi)

# === APPLY DEPTH CORRECTION ===
corrected = (phi_wrapped - np.pi) * F + np.pi
# wrap again
corrected_phase = np.mod(corrected, 2*np.pi)

hologram = corrected_phase
# hologram.optimize('WGS-Kim',feedback='computational_spot',stat_groups=['computational_spot'], maxiter=30)




# === SEND TO SLM ===
slm.set_phase(hologram, settle=True)
print("Pattern displayed on SLM")

time.sleep(5)

# # CLEAR
slm.set_phase(None, settle=True)
slm.close()
print("SLM cleared")