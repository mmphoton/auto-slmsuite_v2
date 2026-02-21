# =========================
# REAL main.py (HARDWARE) - TDC001 (thorlabs_apt_device) + Andor iDus (pylablib)
# =========================

# - Stage: Thorlabs TDC001 via thorlabs_apt_device (APTDevice_Motor)
# - Camera: Andor iDus via pylablib (AndorSDK2Camera)
#
# Key requirements satisfied:
# - No "Start velocity" button / no velocity mode UI. Stage velocity is set via parameters and used for moves.
# - Position in mm = counts / DEFAULT_COUNTS_PER_MM.
# - Saving: for single acquisitions (auto+manual) saves .npy + .txt + .json with unique index (no timestamp in filename).
# - Continuous FVB: live plot only, no saving.
# - g1 measurement sweep: includes last step even if it surpasses upper limit; saves each step.
# - Results viewer: slider + grid + thumbnails + batch export (PNG/TIFF images, CSV profiles) without scale/ax errors.
# - Analysis: temporal coherence (Michelson: delay = 2Δx/c; c=3e8), model selection; spatial coherence using Hilbert envelope
#   and selectable measurement image for spatial coherence extraction; baseline poly order "-1" = none works correctly.
# - Keyboard stage jogging ONLY active after pressing "Keyboard movement" toggle.
#
# Dependencies:
#   pip install numpy matplotlib
#   pip install thorlabs-apt-device
#   pip install pylablib
#   pip install pillow
##############################################################################################

import os, re, time, json, csv, math, threading, queue
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# Hardware libs (real)
_HAS_THR = True
_HAS_PYLABLIB = True
try:
    from serial.tools import list_ports
except Exception:
    list_ports = None

try:
    from thorlabs_apt_device.devices.aptdevice_motor import APTDevice_Motor
    from thorlabs_apt_device.utils import from_pos, to_pos, from_vel, from_acc
except Exception:
    _HAS_THR = False

try:
    from pylablib.devices import Andor
except Exception:
    _HAS_PYLABLIB = False


DEFAULT_COUNTS_PER_MM = 34555
TDC001_TIME_UNIT_S = 2048 / 6e6

DEFAULT_OUTDIR = os.path.join(os.path.expanduser("~"), "measurements")
DEFAULT_BASENAME = "meas"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def safe_float(s: str, default: float) -> float:
    try:
        return float(s)
    except Exception:
        return float(default)


def safe_int(s: str, default: int) -> int:
    try:
        return int(s)
    except Exception:
        return int(default)


def format_pos_mm(mm: Optional[float]) -> str:
    return f"{(0.0 if mm is None else float(mm)):.3f}"


def parse_position_mm_from_filename(name: str) -> Optional[float]:
    m = re.search(r"pos(-?\d+(?:\.\d+)?)mm", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def list_npy_files_sorted(folder: str) -> List[str]:
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".npy")]
    items = []
    for p in files:
        pos = parse_position_mm_from_filename(os.path.basename(p))
        items.append((0.0 if pos is None else pos, p))
    items.sort(key=lambda t: t[0])
    return [p for _, p in items]


def make_unique_stem(out_dir: str, stem_prefix: str) -> str:
    ensure_dir(out_dir)
    for idx in range(0, 1000000):
        stem = f"{stem_prefix}_{idx:04d}"
        if not any(os.path.exists(os.path.join(out_dir, stem + ext)) for ext in (
            ".npy", ".txt", ".json", ".csv", ".png", ".tif", ".tiff", ".log"
        )):
            return stem
    raise RuntimeError("Too many files; cannot find unique name.")


def make_unique_dir(parent: str, base: str) -> str:
    ensure_dir(parent)
    for i in range(0, 100000):
        d = os.path.join(parent, f"{base}_{i:04d}")
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            return d
    raise RuntimeError("Too many sweep folders; cannot find unique directory name.")


def save_dataset(out_dir: str, stem_prefix: str, data: np.ndarray, meta: Dict[str, Any]) -> List[str]:
    ensure_dir(out_dir)
    stem = make_unique_stem(out_dir, stem_prefix)
    npy = os.path.join(out_dir, stem + ".npy")
    txt = os.path.join(out_dir, stem + ".txt")
    jsn = os.path.join(out_dir, stem + ".json")

    arr = np.asarray(data)
    np.save(npy, arr)

    if arr.ndim == 1:
        np.savetxt(txt, arr.reshape(-1, 1), fmt="%.6f")
    elif arr.ndim == 2:
        np.savetxt(txt, arr, fmt="%.6f")
    else:
        np.savetxt(txt, arr.reshape(arr.shape[0], -1), fmt="%.6f")

    with open(jsn, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return [npy, txt, jsn]


def save_png_from_array(path_png: str, arr: np.ndarray) -> None:
    a = np.asarray(arr)
    if a.ndim == 2:
        vmin, vmax = float(np.nanmin(a)), float(np.nanmax(a))
        if vmax <= vmin:
            vmax = vmin + 1.0
        img = (255.0 * (a - vmin) / (vmax - vmin)).clip(0, 255).astype(np.uint8)
        if _HAS_PIL:
            Image.fromarray(img, mode="L").save(path_png)
        else:
            import matplotlib.pyplot as plt
            plt.imsave(path_png, img, cmap="gray")
    elif a.ndim == 1:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(a.reshape(-1))
        ax.set_title(os.path.basename(path_png))
        fig.tight_layout()
        fig.savefig(path_png, dpi=150)
        plt.close(fig)


def save_tiff_from_array(path_tif: str, arr: np.ndarray) -> None:
    if not _HAS_PIL:
        raise RuntimeError("Pillow not installed; TIFF export unavailable.")
    a = np.asarray(arr)
    if a.ndim != 2:
        raise RuntimeError("TIFF export only for 2D arrays.")
    vmin, vmax = float(np.nanmin(a)), float(np.nanmax(a))
    if vmax <= vmin:
        vmax = vmin + 1.0
    img = (65535.0 * (a - vmin) / (vmax - vmin)).clip(0, 65535).astype(np.uint16)
    Image.fromarray(img, mode="I;16").save(path_tif)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


@dataclass
class Measurement:
    meas_id: int
    timestamp_s: float
    kind: str
    exposure_s: float
    stage_counts: Optional[int]
    stage_mm: Optional[float]
    saved_paths: Optional[List[str]] = None


@dataclass
class AppSettings:
    out_dir: str = DEFAULT_OUTDIR
    base_name: str = DEFAULT_BASENAME

    counts_per_mm: float = float(DEFAULT_COUNTS_PER_MM)
    max_vel_mm_s: float = 2.0
    acc_mm_s2: float = 10.0
    jog_step_mm: float = 0.05
    settle_wait_s: float = 0.15

    andor_idx: int = 0
    shutter_mode: str = "auto"     # "auto" / "open" / "closed"
    shutter_ttl_mode: int = 1      # 0=low open, 1=high open

    c_m_s: float = 3e8
    pixel_size_um: float = 6.5


class RunLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._lock = threading.RLock()
        ensure_dir(os.path.dirname(log_path))
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n==== NEW SESSION {now_str()} ====\n")

    def write(self, msg: str):
        line = f"[{now_str()}] {msg}\n"
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)


# =========================
# Real Stage (TDC001)
# =========================
class StageController:
    def __init__(self, settings: AppSettings):
        self.s = settings
        self._dev = None
        self._lock = threading.RLock()

    def is_connected(self) -> bool:
        with self._lock:
            return self._dev is not None

    def connect(self, com_port: str) -> None:
        if not _HAS_THR:
            raise RuntimeError("thorlabs_apt_device not available in this environment.")
        with self._lock:
            if self._dev is not None:
                return
            com_port = (com_port or "").strip()
            if not com_port:
                raise RuntimeError("Select a COM port for the TDC001.")
            self._dev = APTDevice_Motor(
                serial_port=com_port,
                serial_number=None,
                home=False,
                invert_direction_logic=True,
                swap_limit_switches=True,
                status_updates="polled",
            )
            try:
                self._dev.set_enabled(True)
            except Exception:
                pass

            self.apply_velocity_params()
            self.apply_jog_params()
            time.sleep(0.1)
            _ = self.get_status()

    def disconnect(self) -> None:
        with self._lock:
            if self._dev is not None:
                try:
                    self.stop(True)
                except Exception:
                    pass
                try:
                    self._dev.close()
                except Exception:
                    pass
            self._dev = None

    # conversions: mm <-> counts
    def _mm_to_counts(self, mm: float) -> int:
        return int(round(from_pos(mm, factor=self.s.counts_per_mm)))

    def _counts_to_mm(self, counts: int) -> float:
        # NOTE: to_pos expects factor in inverse form
        return float(to_pos(counts, factor=1.0 / self.s.counts_per_mm))

    def _mmps_to_countsps(self, mmps: float) -> int:
        return int(round(from_vel(mmps, factor=self.s.counts_per_mm, t=TDC001_TIME_UNIT_S)))

    def _mmpsps_to_countspsps(self, mmpsps: float) -> int:
        return int(round(from_acc(mmpsps, factor=self.s.counts_per_mm, t=TDC001_TIME_UNIT_S)))

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            self._require()
            dev = self._dev

            if hasattr(dev, "status"):
                try:
                    return dict(dev.status)
                except Exception:
                    pass

            if hasattr(dev, "status_"):
                st = getattr(dev, "status_")
                try:
                    return dict(st[0][0])
                except Exception:
                    try:
                        return dict(st[0])
                    except Exception:
                        pass

            if hasattr(dev, "bay0") and hasattr(dev.bay0, "status"):
                try:
                    return dict(dev.bay0.status)
                except Exception:
                    pass

            if hasattr(dev, "bays") and dev.bays:
                try:
                    b0 = dev.bays[0]
                    if hasattr(b0, "status"):
                        return dict(b0.status)
                except Exception:
                    pass

            raise AttributeError("Could not read device status (no status/status_/bay0.status).")

    def get_position_counts(self) -> int:
        s = self.get_status()
        return int(s.get("position", 0))

    def get_position_mm(self) -> float:
        # Requirement: mm = counts / DEFAULT_COUNTS_PER_MM
        return float(self.get_position_counts()) / float(DEFAULT_COUNTS_PER_MM)

    def is_moving(self) -> bool:
        s = self.get_status()
        for k in ("moving_forward", "moving_reverse", "homing", "jogging_forward", "jogging_reverse"):
            if k in s and s.get(k):
                return True
        if "moving" in s:
            return bool(s.get("moving"))
        return False

    def is_homed(self) -> bool:
        s = self.get_status()
        return bool(s.get("homed", False))

    def stop(self, immediate: bool = True) -> None:
        with self._lock:
            self._require()
            try:
                self._dev.stop(immediate=bool(immediate))
                return
            except TypeError:
                pass
            try:
                self._dev.stop(immediate_stop=bool(immediate))
                return
            except TypeError:
                pass
            self._dev.stop()

    def wait_until_stopped(self, timeout_s: float, abort_event: Optional[threading.Event]) -> None:
        t0 = time.time()
        while True:
            if abort_event is not None and abort_event.is_set():
                try:
                    self.stop(True)
                except Exception:
                    pass
                raise RuntimeError("Aborted")
            if not self.is_moving():
                break
            if time.time() - t0 > timeout_s:
                raise TimeoutError("Timed out waiting for stage")
            time.sleep(0.03)
        if self.s.settle_wait_s > 0:
            time.sleep(self.s.settle_wait_s)

    def apply_velocity_params(self) -> None:
        with self._lock:
            self._require()
            acc_c = self._mmpsps_to_countspsps(self.s.acc_mm_s2)
            vel_c = self._mmps_to_countsps(self.s.max_vel_mm_s)
            self._dev.set_velocity_params(acceleration=acc_c, max_velocity=vel_c)

    def apply_jog_params(self) -> None:
        with self._lock:
            self._require()
            step_c = self._mm_to_counts(self.s.jog_step_mm)
            acc_c = self._mmpsps_to_countspsps(self.s.acc_mm_s2)
            vel_c = self._mmps_to_countsps(self.s.max_vel_mm_s)
            self._dev.set_jog_params(size=step_c, acceleration=acc_c, max_velocity=vel_c,
                                     continuous=False, immediate_stop=True)

    def home(self, abort_event: Optional[threading.Event]) -> None:
        with self._lock:
            self._require()
            try:
                self._dev.set_enabled(True)
            except Exception:
                pass
            self.apply_velocity_params()
            self._dev.home()
        self.wait_until_stopped(timeout_s=180.0, abort_event=abort_event)

    def move_absolute_mm(self, mm: float, abort_event: Optional[threading.Event]) -> None:
        with self._lock:
            self._require()
            self.apply_velocity_params()
            counts = self._mm_to_counts(mm)
            self._dev.move_absolute(position=counts, now=True)
        self.wait_until_stopped(timeout_s=60.0, abort_event=abort_event)

    def move_relative_mm(self, dmm: float, abort_event: Optional[threading.Event]) -> None:
        with self._lock:
            self._require()
            self.apply_velocity_params()
            counts = self._mm_to_counts(dmm)
            self._dev.move_relative(distance=counts, now=True)
        self.wait_until_stopped(timeout_s=60.0, abort_event=abort_event)

    def jog_step(self, direction: str, abort_event: Optional[threading.Event]) -> None:
        with self._lock:
            self._require()
            self.apply_velocity_params()
            self.apply_jog_params()
            d = direction.lower().strip()
            self._dev.move_jog(direction=("forward" if d.startswith("+") else "reverse"))
        self.wait_until_stopped(timeout_s=60.0, abort_event=abort_event)

    def get_enabled_error(self) -> Tuple[Optional[bool], Optional[int]]:
        try:
            s = self.get_status()
            enabled = s.get("enabled", None)
            err = s.get("error_code", None)
            if err is None:
                # some firmwares expose "error" / "error_state"
                err = s.get("error", s.get("error_state", None))
            return (None if enabled is None else bool(enabled),
                    None if err is None else int(err))
        except Exception:
            return (None, None)

    def _require(self) -> None:
        if self._dev is None:
            raise RuntimeError("Stage not connected")


# =========================
# Real Andor Camera
# =========================
class AndorCamera:
    def __init__(self, settings: AppSettings):
        self.s = settings
        self.cam = None
        self._lock = threading.RLock()
        self._cont_running = threading.Event()
        self._cont_thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._latest: Optional[np.ndarray] = None

        self.shutter_state = "auto"
        self.acq_state = "idle"

    def is_connected(self) -> bool:
        with self._lock:
            return self.cam is not None and bool(self.cam.is_opened())

    def connect(self) -> None:
        if not _HAS_PYLABLIB:
            raise RuntimeError("pylablib not available in this environment.")
        with self._lock:
            if self.cam is not None and self.cam.is_opened():
                return
            self.cam = Andor.AndorSDK2Camera(idx=int(self.s.andor_idx))
            self.cam.open()

            self.apply_shutter_settings()
            try:
                self.cam.set_trigger_mode("int")
            except Exception:
                pass
            self.acq_state = "idle"

    def disconnect(self) -> None:
        with self._lock:
            try:
                self.stop_continuous()
            except Exception:
                pass
            if self.cam is not None:
                try:
                    self.cam.close()
                except Exception:
                    pass
            self.cam = None
            self.acq_state = "idle"

    def set_shutter(self, mode: str):
        mode = (mode or "auto").strip().lower()
        if mode not in ("auto", "open", "closed"):
            mode = "auto"
        self.s.shutter_mode = mode
        self.apply_shutter_settings()

    def apply_shutter_settings(self) -> None:
        if self.cam is None:
            return
        mode = (self.s.shutter_mode or "auto").lower().strip()
        ttl = int(self.s.shutter_ttl_mode)
        ttl = 0 if ttl not in (0, 1) else ttl
        try:
            self.cam.setup_shutter(mode=mode, ttl_mode=ttl)
        except Exception:
            try:
                self.cam.set_shutter(mode)
            except Exception:
                pass
        self.shutter_state = mode

    def get_temperature(self) -> Optional[float]:
        if self.cam is None:
            return None
        for fn in ("get_temperature", "get_temperature_status"):
            try:
                f = getattr(self.cam, fn, None)
                if callable(f):
                    out = f()
                    if isinstance(out, (tuple, list)) and out:
                        out = out[0]
                    return float(out)
            except Exception:
                continue
        return None

    def snap(self, kind: str, exposure_s: float) -> np.ndarray:
        with self._lock:
            self._require()
            self.acq_state = "snap"
            self.apply_shutter_settings()

            try:
                self.cam.set_trigger_mode("int")
            except Exception:
                pass
            try:
                self.cam.set_exposure(float(exposure_s))
            except Exception:
                pass
            try:
                self.cam.set_read_mode("fvb" if kind == "fvb" else "image")
            except Exception:
                pass

            arr = self.cam.snap(timeout=max(10.0, float(exposure_s) + 8.0))
            self.acq_state = "idle" if not self._cont_running.is_set() else "continuous"
            return np.asarray(arr)

    def start_continuous_fvb(self, exposure_s: float, buffer_frames: int = 200) -> None:
        with self._lock:
            self._require()
            self.apply_shutter_settings()
            self._cont_running.set()
            self._frame_count = 0
            self._latest = None
            self.acq_state = "continuous"

            try:
                self.cam.set_trigger_mode("int")
            except Exception:
                pass
            try:
                self.cam.set_exposure(float(exposure_s))
            except Exception:
                pass
            try:
                self.cam.set_read_mode("fvb")
            except Exception:
                pass

            try:
                self.cam.setup_acquisition(mode="cont", nframes=int(buffer_frames))
            except Exception:
                try:
                    self.cam.setup_acquisition(mode="cont")
                except Exception:
                    pass

            self.cam.start_acquisition()
            self._cont_thread = threading.Thread(target=self._cont_loop, daemon=True)
            self._cont_thread.start()

    def _cont_loop(self) -> None:
        while self._cont_running.is_set():
            try:
                ok = self.cam.wait_for_frame(timeout=0.5)
                if not ok:
                    continue
                try:
                    frame = self.cam.read_newest_image()
                except Exception:
                    frame = self.cam.read_newest()
                self._latest = np.asarray(frame)
                self._frame_count += 1
            except Exception:
                time.sleep(0.05)

    def stop_continuous(self) -> None:
        with self._lock:
            if self.cam is None:
                return
            self._cont_running.clear()
            try:
                self.cam.stop_acquisition()
            except Exception:
                pass
            try:
                self.cam.clear_acquisition()
            except Exception:
                pass
            self.acq_state = "idle"

    def get_latest(self) -> Tuple[int, Optional[np.ndarray]]:
        with self._lock:
            return int(self._frame_count), (None if self._latest is None else np.asarray(self._latest))

    def _require(self) -> None:
        if self.cam is None or not self.cam.is_opened():
            raise RuntimeError("Camera not connected")


# =========================
# Main App (GUI)
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("g1 GUI (REAL) - TDC001 + Andor")
        self.geometry("1650x950")

        self.settings = AppSettings()
        ensure_dir(self.settings.out_dir)

        self.stage = StageController(self.settings)
        self.camera = AndorCamera(self.settings)

        self.abort_event = threading.Event()
        self.ui_q: "queue.Queue[Tuple[str, Any]]" = queue.Queue()

        self._meas_id = 0
        self._last_meas: Optional[Measurement] = None
        self._last_data: Optional[np.ndarray] = None

        self.res_files: List[str] = []
        self.res_pos: List[float] = []
        self.res_data: List[Optional[np.ndarray]] = []

        self._thumb_imgs: List[Any] = []
        self._thumb_paths: List[str] = []

        self.g1_abort = threading.Event()
        self.g1_thread: Optional[threading.Thread] = None
        self.g1_sweep_dir: Optional[str] = None
        self.g1_logger: Optional[RunLogger] = None

        self.cont_running = False
        self.cont_last_frame = -1

        self.an_imgs: List[np.ndarray] = []
        self.an_files: List[str] = []
        self.an_pos_mm: np.ndarray = np.array([])
        self.an_vis: np.ndarray = np.array([])
        self.an_best_idx: int = 0
        self.an_roi: Optional[Tuple[int, int, int, int]] = None
        self._rect_selector = None
        self.spatial_win = None

        # Keyboard movement gating
        self.keyboard_enabled = False
        self._keybind_tags = ["<Left>", "<Right>", "<KP_Left>", "<KP_Right>"]

        # Status variables
        self.stage_status_var = tk.StringVar(value="Stage: disconnected")
        self.cam_status_var = tk.StringVar(value="Camera: disconnected")
        self.cam_temp_var = tk.StringVar(value="Temp: --")
        self.cam_shutter_var = tk.StringVar(value="Shutter: --")
        self.cam_acq_var = tk.StringVar(value="Acq: --")
        self.stage_err_var = tk.StringVar(value="Err: --")
        self.stage_en_var = tk.StringVar(value="Enabled: --")

        # Analysis vars
        self.an_mode = tk.StringVar(value="ROI")
        self.an_line_y = tk.StringVar(value="120")
        self.an_pixel_um = tk.StringVar(value=str(self.settings.pixel_size_um))
        self.an_bw = tk.StringVar(value="5")
        self.an_baseline_order = tk.StringVar(value="-1")
        self.an_vis_method = tk.StringVar(value="FFT")
        self.an_model_sel = tk.StringVar(value="Auto(AIC)")
        self.an_spatial_choice = tk.StringVar(value="Best visibility")

        self._build_menu()

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_motion = ttk.Frame(self.nb)
        self.tab_acq = ttk.Frame(self.nb)
        self.tab_g1 = ttk.Frame(self.nb)
        self.tab_results = ttk.Frame(self.nb)
        self.tab_analysis = ttk.Frame(self.nb)

        self.nb.add(self.tab_motion, text="Motion")
        self.nb.add(self.tab_acq, text="Acquisition")
        self.nb.add(self.tab_g1, text="g1 measurement")
        self.nb.add(self.tab_results, text="Results")
        self.nb.add(self.tab_analysis, text="Analysis")

        self._build_motion_tab()
        self._build_acq_tab()
        self._build_g1_tab()
        self._build_results_tab()
        self._build_analysis_tab()

        # Hardware status panel
        panel = ttk.LabelFrame(self, text="Hardware status")
        panel.pack(fill=tk.X, padx=10, pady=6)
        row = ttk.Frame(panel)
        row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row, textvariable=self.stage_status_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(row, textvariable=self.stage_en_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(row, textvariable=self.stage_err_var).pack(side=tk.LEFT, padx=10)
        ttk.Separator(row, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(row, textvariable=self.cam_status_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(row, textvariable=self.cam_temp_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(row, textvariable=self.cam_shutter_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(row, textvariable=self.cam_acq_var).pack(side=tk.LEFT, padx=10)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=10, pady=6)

        self._set_keyboard_bindings(False)
        self.after(80, self._poll)

    # -------- menu --------
    def _build_menu(self):
        m = tk.Menu(self)
        mf = tk.Menu(m, tearoff=False)
        mf.add_command(label="Settings (output folder)", command=self._settings_dialog)
        mf.add_separator()
        mf.add_command(label="Quit", command=self._quit)
        m.add_cascade(label="File", menu=mf)
        self.config(menu=m)

    def _settings_dialog(self):
        d = filedialog.askdirectory(initialdir=self.settings.out_dir)
        if d:
            self.settings.out_dir = d
            ensure_dir(d)
            self._set_status(f"Output folder: {d}")

    def _quit(self):
        try:
            self.abort_event.set()
            self.g1_abort.set()
            self.stage.disconnect()
        except Exception:
            pass
        try:
            self.camera.disconnect()
        except Exception:
            pass
        self.destroy()

    # -------- helpers --------
    def _set_status(self, s: str):
        self.status_var.set(str(s))

    def _new_meas(self, kind: str, exp: float) -> Measurement:
        self._meas_id += 1
        sc = self.stage.get_position_counts() if self.stage.is_connected() else None
        sm = self.stage.get_position_mm() if self.stage.is_connected() else None
        return Measurement(self._meas_id, time.time(), kind, float(exp), sc, sm, None)

    def _stem_prefix(self, meas: Measurement) -> str:
        base = (self.settings.base_name or "meas").strip()
        mode = "FVB" if meas.kind.lower() == "fvb" else "image"
        pos = format_pos_mm(meas.stage_mm)
        return f"{base}_{mode}_pos{pos}mm_id{meas.meas_id:06d}"

    # =========================
    # Keyboard bindings (only on demand)
    # =========================
    def _set_keyboard_bindings(self, enabled: bool):
        for seq in self._keybind_tags:
            try:
                self.unbind_all(seq)
            except Exception:
                pass
        self.keyboard_enabled = bool(enabled)
        if not enabled:
            return
        self.bind_all("<Left>", lambda e: self._key_jog("-", e))
        self.bind_all("<Right>", lambda e: self._key_jog("+", e))
        self.bind_all("<KP_Left>", lambda e: self._key_jog("-", e))
        self.bind_all("<KP_Right>", lambda e: self._key_jog("+", e))

    def _key_jog(self, direction: str, event: tk.Event):
        if not self.keyboard_enabled:
            return
        if not self.stage.is_connected():
            return
        base = abs(self.settings.jog_step_mm)
        step = base
        if event.state & 0x0001:   # Shift
            step *= 10.0
        if event.state & 0x0004:   # Ctrl
            step /= 10.0
        try:
            self.stage.move_relative_mm((-step if direction.startswith("-") else step), self.abort_event)
            self._set_status(f"Key jog {direction}{step:g} mm")
        except Exception as e:
            self.ui_q.put(("error", f"Key jog failed: {e}"))

    # =========================
    # Motion tab
    # =========================
    def _list_com_ports(self) -> List[str]:
        if list_ports is None:
            return []
        out = []
        try:
            for p in list_ports.comports():
                out.append(p.device)
        except Exception:
            pass
        return out

    def _build_motion_tab(self):
        root = self.tab_motion

        top = ttk.Frame(root)
        top.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top, text="Stage COM port").pack(side=tk.LEFT)
        self.stage_com = tk.StringVar(value="")
        self.stage_com_combo = ttk.Combobox(top, textvariable=self.stage_com, values=self._list_com_ports(), width=12, state="readonly")
        self.stage_com_combo.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Refresh", command=lambda: self.stage_com_combo.configure(values=self._list_com_ports())).pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="Stage Connect", command=self._stage_connect).pack(side=tk.LEFT, padx=10)
        ttk.Button(top, text="Home", command=self._stage_home).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="STOP", command=self._stage_stop).pack(side=tk.LEFT, padx=6)

        self.kb_btn_text = tk.StringVar(value="Keyboard movement: OFF")
        ttk.Button(top, textvariable=self.kb_btn_text, command=self._toggle_keyboard).pack(side=tk.LEFT, padx=12)

        prm = ttk.LabelFrame(root, text="Motion parameters")
        prm.pack(fill=tk.X, padx=10, pady=10)

        self.vel_var = tk.StringVar(value=str(self.settings.max_vel_mm_s))
        self.acc_var = tk.StringVar(value=str(self.settings.acc_mm_s2))
        self.jog_var = tk.StringVar(value=str(self.settings.jog_step_mm))

        row = ttk.Frame(prm); row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row, text="Max vel (mm/s)").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.vel_var, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="Acc (mm/s^2)").pack(side=tk.LEFT, padx=(12,0))
        ttk.Entry(row, textvariable=self.acc_var, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="Jog step (mm)").pack(side=tk.LEFT, padx=(12,0))
        ttk.Entry(row, textvariable=self.jog_var, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Apply", command=self._apply_motion_params).pack(side=tk.LEFT, padx=12)

        mv = ttk.LabelFrame(root, text="Move")
        mv.pack(fill=tk.X, padx=10, pady=10)

        self.move_mode = tk.StringVar(value="absolute")
        self.move_units = tk.StringVar(value="mm")
        self.move_val = tk.StringVar(value="0.1")
        self.move_dir = tk.StringVar(value="+")

        row = ttk.Frame(mv); row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row, text="Mode").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.move_mode, values=["absolute", "relative", "jog"], width=10, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="Units").pack(side=tk.LEFT, padx=(12,0))
        ttk.Combobox(row, textvariable=self.move_units, values=["mm", "counts"], width=8, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="Value").pack(side=tk.LEFT, padx=(12,0))
        ttk.Entry(row, textvariable=self.move_val, width=12).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="Dir").pack(side=tk.LEFT, padx=(12,0))
        ttk.Combobox(row, textvariable=self.move_dir, values=["+", "-"], width=4, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="GO", command=self._move_go).pack(side=tk.LEFT, padx=10)
        ttk.Button(row, text="Jog step", command=self._jog_step_button).pack(side=tk.LEFT, padx=6)

        live = ttk.LabelFrame(root, text="Live readback")
        live.pack(fill=tk.X, padx=10, pady=10)

        self.pos_counts_var = tk.StringVar(value="0")
        self.pos_mm_var = tk.StringVar(value="0.000000")
        self.moving_var = tk.StringVar(value="False")
        self.homed_var = tk.StringVar(value="False")

        r = ttk.Frame(live); r.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(r, text="Counts:").pack(side=tk.LEFT)
        ttk.Label(r, textvariable=self.pos_counts_var, width=12).pack(side=tk.LEFT, padx=6)
        ttk.Label(r, text="mm:").pack(side=tk.LEFT, padx=(12,0))
        ttk.Label(r, textvariable=self.pos_mm_var, width=12).pack(side=tk.LEFT, padx=6)
        ttk.Label(r, text="Moving:").pack(side=tk.LEFT, padx=(12,0))
        ttk.Label(r, textvariable=self.moving_var, width=8).pack(side=tk.LEFT, padx=6)
        ttk.Label(r, text="Homed:").pack(side=tk.LEFT, padx=(12,0))
        ttk.Label(r, textvariable=self.homed_var, width=8).pack(side=tk.LEFT, padx=6)

    def _toggle_keyboard(self):
        new_state = not self.keyboard_enabled
        self._set_keyboard_bindings(new_state)
        self.kb_btn_text.set(f"Keyboard movement: {'ON' if self.keyboard_enabled else 'OFF'}")
        self._set_status(f"Keyboard movement {'enabled' if self.keyboard_enabled else 'disabled'}.")

    def _stage_connect(self):
        try:
            self.stage.connect(self.stage_com.get())
            self._set_status("Stage connected.")
        except Exception as e:
            messagebox.showerror("Stage", str(e))

    def _stage_home(self):
        def worker():
            try:
                self.stage.home(self.abort_event)
                self.ui_q.put(("status", "Home done."))
            except Exception as e:
                self.ui_q.put(("error", f"Home failed: {e}"))
        threading.Thread(target=worker, daemon=True).start()

    def _stage_stop(self):
        try:
            self.stage.stop(True)
            self._set_status("Stop sent.")
        except Exception as e:
            messagebox.showerror("Stage", str(e))

    def _apply_motion_params(self):
        self.settings.max_vel_mm_s = safe_float(self.vel_var.get(), self.settings.max_vel_mm_s)
        self.settings.acc_mm_s2 = safe_float(self.acc_var.get(), self.settings.acc_mm_s2)
        self.settings.jog_step_mm = safe_float(self.jog_var.get(), self.settings.jog_step_mm)
        try:
            if self.stage.is_connected():
                self.stage.apply_velocity_params()
                self.stage.apply_jog_params()
        except Exception:
            pass
        self._set_status("Motion params applied.")

    def _move_go(self):
        mode = self.move_mode.get()
        units = self.move_units.get()
        d = self.move_dir.get()
        v = safe_float(self.move_val.get(), 0.0)

        def worker():
            try:
                if mode == "absolute":
                    mm = (float(int(v)) / float(DEFAULT_COUNTS_PER_MM)) if units == "counts" else float(v)
                    self.stage.move_absolute_mm(mm, self.abort_event)
                elif mode == "relative":
                    mm = (float(int(v)) / float(DEFAULT_COUNTS_PER_MM)) if units == "counts" else float(v)
                    if d == "-":
                        mm = -mm
                    self.stage.move_relative_mm(mm, self.abort_event)
                else:
                    self.stage.jog_step(d, self.abort_event)
                self.ui_q.put(("status", "Move done."))
            except Exception as e:
                self.ui_q.put(("error", f"Move failed: {e}"))
        threading.Thread(target=worker, daemon=True).start()

    def _jog_step_button(self):
        def worker():
            try:
                self.stage.jog_step(self.move_dir.get(), self.abort_event)
                self.ui_q.put(("status", "Jog done."))
            except Exception as e:
                self.ui_q.put(("error", f"Jog failed: {e}"))
        threading.Thread(target=worker, daemon=True).start()

    # =========================
    # Acquisition tab
    # =========================
    def _build_acq_tab(self):
        root = self.tab_acq

        top = ttk.Frame(root); top.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(top, text="Camera Connect", command=self._cam_connect).pack(side=tk.LEFT)
        ttk.Button(top, text="Disconnect", command=self._cam_disconnect).pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="Shutter").pack(side=tk.LEFT, padx=(18, 4))
        self.shutter_sel = tk.StringVar(value=self.settings.shutter_mode)
        ttk.Combobox(top, textvariable=self.shutter_sel, values=["auto", "open", "closed"], width=8, state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Apply", command=self._cam_apply_shutter).pack(side=tk.LEFT, padx=6)

        self.cam_state = tk.StringVar(value="disconnected")
        ttk.Label(top, textvariable=self.cam_state).pack(side=tk.LEFT, padx=12)

        single = ttk.LabelFrame(root, text="Single acquisition (autosave supported)")
        single.pack(fill=tk.X, padx=10, pady=10)

        self.single_exp = tk.StringVar(value="0.05")
        self.single_autosave = tk.BooleanVar(value=True)

        r = ttk.Frame(single); r.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(r, text="Exposure (s)").pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self.single_exp, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(r, text="Auto-save", variable=self.single_autosave).pack(side=tk.LEFT, padx=12)
        ttk.Button(r, text="Acquire IMAGE", command=lambda: self._acquire_once("image")).pack(side=tk.LEFT, padx=6)
        ttk.Button(r, text="Acquire FVB", command=lambda: self._acquire_once("fvb")).pack(side=tk.LEFT, padx=6)
        ttk.Button(r, text="Save last (manual)", command=self._save_last_manual).pack(side=tk.LEFT, padx=12)
        ttk.Button(r, text="Plot last", command=self._plot_last).pack(side=tk.LEFT, padx=6)

        self.last_info = tk.StringVar(value="Last: (none)")
        ttk.Label(single, textvariable=self.last_info, anchor="w").pack(fill=tk.X, padx=8, pady=(0,8))

        cont = ttk.LabelFrame(root, text="Continuous FVB (live plot only; no saving)")
        cont.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.cont_exp = tk.StringVar(value="0.02")
        row = ttk.Frame(cont); row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row, text="Exposure (s)").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.cont_exp, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Start", command=self._cont_start).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Stop", command=self._cont_stop).pack(side=tk.LEFT, padx=6)
        self.cont_frames = tk.StringVar(value="Frames: 0")
        ttk.Label(row, textvariable=self.cont_frames).pack(side=tk.LEFT, padx=12)

        self.cont_fig = Figure()
        self.cont_ax = self.cont_fig.add_subplot(111)
        self.cont_canvas = FigureCanvasTkAgg(self.cont_fig, master=cont)
        self.cont_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        toolbar_frame = ttk.Frame(cont)
        toolbar_frame.pack(fill=tk.X)
        self.cont_toolbar = NavigationToolbar2Tk(self.cont_canvas, toolbar_frame)
        self.cont_toolbar.update()

    def _cam_connect(self):
        try:
            self.camera.connect()
            self._set_status("Camera connected.")
        except Exception as e:
            messagebox.showerror("Camera", str(e))

    def _cam_disconnect(self):
        try:
            self.camera.disconnect()
            self._set_status("Camera disconnected.")
        except Exception as e:
            messagebox.showerror("Camera", str(e))

    def _cam_apply_shutter(self):
        try:
            self.camera.set_shutter(self.shutter_sel.get())
            self._set_status(f"Shutter set: {self.shutter_sel.get()}")
        except Exception as e:
            messagebox.showerror("Camera", str(e))

    def _acquire_once(self, kind: str):
        def worker():
            try:
                exp = safe_float(self.single_exp.get(), 0.05)
                data = self.camera.snap(kind, exp)
                meas = self._new_meas(kind, exp)

                saved = None
                if self.single_autosave.get():
                    saved = save_dataset(self.settings.out_dir, self._stem_prefix(meas), data, {"measurement": asdict(meas)})
                meas.saved_paths = saved

                self._last_meas = meas
                self._last_data = np.asarray(data)
                self.ui_q.put(("last", meas))
                self.ui_q.put(("status", f"Acquired {kind}. shape={np.asarray(data).shape}"))
            except Exception as e:
                self.ui_q.put(("error", f"Acquire failed: {e}"))
        threading.Thread(target=worker, daemon=True).start()

    def _save_last_manual(self):
        if self._last_meas is None or self._last_data is None:
            messagebox.showinfo("Save", "No last acquisition.")
            return
        meas = self._last_meas
        meas.saved_paths = save_dataset(self.settings.out_dir, self._stem_prefix(meas), self._last_data, {"measurement": asdict(meas)})
        self.ui_q.put(("last", meas))
        self._set_status("Saved last acquisition.")

    def _plot_last(self):
        if self._last_meas is None or self._last_data is None:
            return
        win = tk.Toplevel(self)
        win.title("Last plot")
        win.geometry("900x650")
        fig = Figure()
        ax = fig.add_subplot(111)
        canv = FigureCanvasTkAgg(fig, master=win)
        canv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(win); toolbar_frame.pack(fill=tk.X)
        tb = NavigationToolbar2Tk(canv, toolbar_frame); tb.update()

        meas = self._last_meas
        arr = np.asarray(self._last_data)
        ax.clear()
        if meas.kind == "image" and arr.ndim >= 2:
            ax.imshow(arr, aspect="auto")
        else:
            ax.plot(arr.reshape(-1))
        ax.set_title(f"{meas.kind} | exp={meas.exposure_s:.3f}s | pos={format_pos_mm(meas.stage_mm)} mm")
        canv.draw()

    def _cont_start(self):
        def worker():
            try:
                exp = safe_float(self.cont_exp.get(), 0.02)
                self.camera.start_continuous_fvb(exp)
                self.cont_running = True
                self.cont_last_frame = -1
                self.ui_q.put(("status", "Continuous started."))
            except Exception as e:
                self.ui_q.put(("error", f"Continuous start failed: {e}"))
        threading.Thread(target=worker, daemon=True).start()

    def _cont_stop(self):
        try:
            self.camera.stop_continuous()
            self.cont_running = False
            self._set_status("Continuous stopped.")
        except Exception as e:
            messagebox.showerror("Continuous", str(e))

    # =========================
    # g1 measurement tab
    # =========================
    def _build_g1_tab(self):
        root = self.tab_g1

        ctrl = ttk.LabelFrame(root, text="Sweep controls")
        ctrl.pack(fill=tk.X, padx=10, pady=10)

        self.g1_lower = tk.StringVar(value="-1.0")
        self.g1_upper = tk.StringVar(value="1.0")
        self.g1_step = tk.StringVar(value="0.1")
        self.g1_exp = tk.StringVar(value="0.05")
        self.g1_mode = tk.StringVar(value="image")
        self.g1_live_fvb = tk.BooleanVar(value=True)

        r = ttk.Frame(ctrl); r.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(r, text="Lower (mm)").pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self.g1_lower, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Label(r, text="Upper (mm)").pack(side=tk.LEFT, padx=(12,0))
        ttk.Entry(r, textvariable=self.g1_upper, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Label(r, text="Step (mm)").pack(side=tk.LEFT, padx=(12,0))
        ttk.Entry(r, textvariable=self.g1_step, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Label(r, text="Exp (s)").pack(side=tk.LEFT, padx=(12,0))
        ttk.Entry(r, textvariable=self.g1_exp, width=10).pack(side=tk.LEFT, padx=6)

        r2 = ttk.Frame(ctrl); r2.pack(fill=tk.X, padx=8, pady=(0,6))
        ttk.Label(r2, text="Mode").pack(side=tk.LEFT)
        ttk.Combobox(r2, textvariable=self.g1_mode, values=["image", "fvb"], width=10, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(r2, text="Live plot (FVB only)", variable=self.g1_live_fvb).pack(side=tk.LEFT, padx=12)
        ttk.Button(r2, text="Start sweep", command=self._g1_start).pack(side=tk.LEFT, padx=6)
        ttk.Button(r2, text="Stop", command=self._g1_stop).pack(side=tk.LEFT, padx=6)

        self.g1_status = tk.StringVar(value="Idle.")
        ttk.Label(root, textvariable=self.g1_status, anchor="w").pack(fill=tk.X, padx=18, pady=(0,6))

        self.g1_stats = tk.StringVar(value="Step stats: --")
        ttk.Label(root, textvariable=self.g1_stats, anchor="w").pack(fill=tk.X, padx=18, pady=(0,10))

        self.g1_fig = Figure()
        self.g1_ax = self.g1_fig.add_subplot(111)
        self.g1_canvas = FigureCanvasTkAgg(self.g1_fig, master=root)
        self.g1_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        toolbar_frame = ttk.Frame(root); toolbar_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        self.g1_toolbar = NavigationToolbar2Tk(self.g1_canvas, toolbar_frame)
        self.g1_toolbar.update()

    def _g1_start(self):
        if self.g1_thread and self.g1_thread.is_alive():
            messagebox.showinfo("g1", "Sweep already running.")
            return
        if not self.stage.is_connected():
            messagebox.showerror("g1", "Stage not connected.")
            return
        if not self.camera.is_connected():
            messagebox.showerror("g1", "Camera not connected.")
            return

        lower = safe_float(self.g1_lower.get(), 0.0)
        upper = safe_float(self.g1_upper.get(), 0.0)
        step = safe_float(self.g1_step.get(), 0.1)
        exp = safe_float(self.g1_exp.get(), 0.05)
        mode = self.g1_mode.get().strip().lower()
        live = bool(self.g1_live_fvb.get()) and (mode == "fvb")
        if step == 0:
            messagebox.showerror("g1", "Step must be non-zero.")
            return

        self.g1_abort.clear()
        self.g1_sweep_dir = make_unique_dir(self.settings.out_dir, "g1_sweep")
        self.g1_logger = RunLogger(os.path.join(self.g1_sweep_dir, "run.log"))
        self.g1_logger.write(f"START sweep | mode={mode} exp={exp}s | lower={lower} upper={upper} step={step}")

        # include last step even if it surpasses upper limit
        positions = []
        p = lower
        if step > 0:
            while p <= upper:
                positions.append(p)
                p += step
            if not positions:
                positions = [lower]
            if positions[-1] < upper:
                positions.append(positions[-1] + step)
        else:
            while p >= upper:
                positions.append(p)
                p += step
            if not positions:
                positions = [lower]
            if positions[-1] > upper:
                positions.append(positions[-1] + step)

        t_sweep_start = time.time()
        step_times: List[float] = []
        move_times: List[float] = []
        acq_times: List[float] = []
        save_times: List[float] = []

        def worker():
            try:
                for i, pos_mm in enumerate(positions, start=1):
                    if self.g1_abort.is_set():
                        raise RuntimeError("Aborted.")

                    self.g1_status.set(f"Step {i}/{len(positions)}: move {pos_mm:.3f} mm")
                    t0 = time.time()
                    self.stage.move_absolute_mm(pos_mm, self.g1_abort)
                    t_move = time.time() - t0

                    retries = 0
                    t_acq = 0.0
                    data = None
                    while True:
                        try:
                            self.g1_status.set(f"Step {i}/{len(positions)}: acquire {mode} exp={exp}s (try {retries+1})")
                            t1 = time.time()
                            data = self.camera.snap(mode, exp)
                            t_acq = time.time() - t1
                            break
                        except Exception as e:
                            retries += 1
                            if self.g1_logger:
                                self.g1_logger.write(f"ERROR acquire step {i}: {e} | retry={retries}")
                            if retries >= 2:
                                raise
                            time.sleep(0.2)

                    t2 = time.time()
                    meas = self._new_meas(mode, exp)
                    meas.stage_counts = self.stage.get_position_counts()
                    meas.stage_mm = self.stage.get_position_mm()

                    mode_str = "FVB" if mode == "fvb" else "image"
                    stem_prefix = f"{self.settings.base_name}_{mode_str}_pos{format_pos_mm(meas.stage_mm)}mm_id{meas.meas_id:06d}"
                    meas.saved_paths = save_dataset(self.g1_sweep_dir, stem_prefix, data, {"measurement": asdict(meas), "sweep_dir": self.g1_sweep_dir})
                    t_save = time.time() - t2

                    arr = np.asarray(data)
                    metric = f"mean={float(np.mean(arr)):.2f} max={float(np.max(arr)):.2f}"

                    if live:
                        self.g1_ax.clear()
                        self.g1_ax.plot(np.asarray(data).reshape(-1))
                        self.g1_ax.set_title(f"FVB live | pos={meas.stage_mm:.3f} mm")
                        self.g1_canvas.draw()

                    t_step = t_move + t_acq + t_save
                    step_times.append(t_step)
                    move_times.append(t_move)
                    acq_times.append(t_acq)
                    save_times.append(t_save)

                    avg_step = float(np.mean(step_times))
                    remaining = len(positions) - i
                    eta_s = avg_step * remaining

                    avg_move = float(np.mean(move_times))
                    avg_acq = float(np.mean(acq_times))
                    avg_save = float(np.mean(save_times))
                    bottleneck = max(("move", avg_move), ("acq", avg_acq), ("save", avg_save), key=lambda t: t[1])[0]

                    self.g1_stats.set(
                        f"Avg step={avg_step:.2f}s | ETA={eta_s:.1f}s | "
                        f"avg(move/acq/save)={avg_move:.2f}/{avg_acq:.2f}/{avg_save:.2f}s | bottleneck={bottleneck}"
                    )

                    if self.g1_logger:
                        self.g1_logger.write(
                            f"STEP {i}/{len(positions)} pos={meas.stage_mm:.3f}mm "
                            f"t_move={t_move:.3f}s t_acq={t_acq:.3f}s t_save={t_save:.3f}s retries={retries} {metric} "
                            f"files={','.join(os.path.basename(p) for p in meas.saved_paths or [])}"
                        )

                dt_total = time.time() - t_sweep_start
                if self.g1_logger:
                    self.g1_logger.write(f"DONE sweep | total={dt_total:.2f}s | saved_dir={self.g1_sweep_dir}")
                self.g1_status.set(f"Done. Saved to {self.g1_sweep_dir}")
            except Exception as e:
                if self.g1_logger:
                    self.g1_logger.write(f"STOPPED sweep | error={e}")
                self.g1_status.set(f"Stopped: {e}")

        self.g1_thread = threading.Thread(target=worker, daemon=True)
        self.g1_thread.start()

    def _g1_stop(self):
        self.g1_abort.set()
        try:
            self.stage.stop(True)
        except Exception:
            pass
        if self.g1_logger:
            self.g1_logger.write("STOP requested by user.")
        self.g1_status.set("Stop requested.")

    # =========================
    # Results tab (same as dummy)
    # =========================
    def _build_results_tab(self):
        root = self.tab_results

        top = ttk.Frame(root); top.pack(fill=tk.X, padx=10, pady=10)
        self.res_folder = tk.StringVar(value=self.settings.out_dir)

        ttk.Label(top, text="Folder").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.res_folder, width=80).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Browse", command=self._res_browse).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load", command=self._res_load).pack(side=tk.LEFT, padx=6)

        view = ttk.Frame(root); view.pack(fill=tk.X, padx=10, pady=(0,10))
        self.res_view_mode = tk.StringVar(value="slider")
        self.res_cols = tk.StringVar(value="3")

        ttk.Label(view, text="View").pack(side=tk.LEFT)
        ttk.Radiobutton(view, text="Slider", variable=self.res_view_mode, value="slider", command=self._res_refresh).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(view, text="Grid", variable=self.res_view_mode, value="grid", command=self._res_refresh).pack(side=tk.LEFT, padx=6)
        ttk.Label(view, text="Grid cols").pack(side=tk.LEFT, padx=(20,0))
        ttk.Entry(view, textvariable=self.res_cols, width=6).pack(side=tk.LEFT, padx=6)

        ttk.Button(view, text="Build thumbnails", command=self._thumb_build).pack(side=tk.LEFT, padx=12)
        ttk.Button(view, text="Batch export selected", command=self._batch_export).pack(side=tk.LEFT, padx=6)

        mid = ttk.Frame(root)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        mid.columnconfigure(0, weight=3)
        mid.columnconfigure(1, weight=2)
        mid.rowconfigure(0, weight=1)

        left = ttk.Frame(mid)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        right = ttk.Frame(mid)
        right.grid(row=0, column=1, sticky="nsew")

        self.res_fig = Figure()
        self.res_ax = self.res_fig.add_subplot(111)
        self.res_canvas = FigureCanvasTkAgg(self.res_fig, master=left)
        self.res_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = ttk.Frame(left); tb_frame.pack(fill=tk.X)
        self.res_toolbar = NavigationToolbar2Tk(self.res_canvas, tb_frame)
        self.res_toolbar.update()

        bot = ttk.Frame(left); bot.pack(fill=tk.X, pady=6)
        ttk.Label(bot, text="Index").pack(side=tk.LEFT)
        self.res_idx = tk.IntVar(value=0)
        self.res_scale = tk.Scale(bot, from_=0, to=0, orient=tk.HORIZONTAL, variable=self.res_idx, command=lambda *_: self._res_plot_current())
        self.res_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.res_info = tk.StringVar(value="No data loaded.")
        ttk.Label(left, textvariable=self.res_info, anchor="w").pack(fill=tk.X, pady=(0,6))

        selbox = ttk.LabelFrame(right, text="Select files for export (multi-select)")
        selbox.pack(fill=tk.BOTH, expand=True)
        self.res_list = tk.Listbox(selbox, selectmode=tk.EXTENDED)
        self.res_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(selbox, orient=tk.VERTICAL, command=self.res_list.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.res_list.configure(yscrollcommand=sb.set)

        thumb = ttk.LabelFrame(right, text="Thumbnail gallery (2D images only)")
        thumb.pack(fill=tk.BOTH, expand=True, pady=(10,0))

        self.thumb_canvas = tk.Canvas(thumb, height=240)
        self.thumb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.thumb_scroll = ttk.Scrollbar(thumb, orient=tk.VERTICAL, command=self.thumb_canvas.yview)
        self.thumb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.thumb_canvas.configure(yscrollcommand=self.thumb_scroll.set)

        self.thumb_inner = ttk.Frame(self.thumb_canvas)
        self.thumb_canvas.create_window((0, 0), window=self.thumb_inner, anchor="nw")
        self.thumb_inner.bind("<Configure>", lambda e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all")))

        expf = ttk.LabelFrame(right, text="Batch export settings")
        expf.pack(fill=tk.X, pady=(10,0))
        self.batch_profile_mode = tk.StringVar(value="ROI")
        self.batch_line_y = tk.StringVar(value="120")
        self.batch_baseline = tk.StringVar(value="-1")
        self.batch_roi_use_suggested = tk.BooleanVar(value=True)

        rr = ttk.Frame(expf); rr.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(rr, text="Profile").pack(side=tk.LEFT)
        ttk.Combobox(rr, textvariable=self.batch_profile_mode, values=["ROI", "Line"], width=6, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Label(rr, text="Line y").pack(side=tk.LEFT, padx=(12,0))
        ttk.Entry(rr, textvariable=self.batch_line_y, width=6).pack(side=tk.LEFT, padx=6)
        ttk.Label(rr, text="Baseline poly order").pack(side=tk.LEFT, padx=(12,0))
        ttk.Combobox(rr, textvariable=self.batch_baseline, values=["-1", "0", "1", "2", "3", "4", "5"], width=4, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(rr, text="Use analysis ROI if set", variable=self.batch_roi_use_suggested).pack(side=tk.LEFT, padx=10)

    def _res_browse(self):
        d = filedialog.askdirectory(initialdir=self.res_folder.get() or self.settings.out_dir)
        if d:
            self.res_folder.set(d)

    def _res_load(self):
        folder = self.res_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Results", "Select a valid folder.")
            return
        files = list_npy_files_sorted(folder)
        if not files:
            self.res_files, self.res_pos, self.res_data = [], [], []
            self.res_scale.configure(from_=0, to=0)
            self.res_info.set("No .npy files.")
            self._res_plot_blank()
            self.res_list.delete(0, tk.END)
            self._thumb_clear()
            return

        pos, data = [], []
        for p in files:
            pos.append(parse_position_mm_from_filename(os.path.basename(p)) or 0.0)
            try:
                data.append(np.load(p))
            except Exception:
                data.append(None)

        idxs = sorted(range(len(files)), key=lambda i: pos[i])
        self.res_files = [files[i] for i in idxs]
        self.res_pos = [pos[i] for i in idxs]
        self.res_data = [data[i] for i in idxs]

        self.res_scale.configure(from_=0, to=max(0, len(self.res_files) - 1))
        self.res_idx.set(0)
        self.res_info.set(f"Loaded {len(self.res_files)} files (sorted by pos).")
        self._res_refresh()

        self.res_list.delete(0, tk.END)
        for i, p in enumerate(self.res_files):
            self.res_list.insert(tk.END, f"[{i:04d}] {os.path.basename(p)}")

        self._thumb_clear()

    def _res_refresh(self):
        if self.res_view_mode.get() == "slider":
            self.res_scale.configure(state=("normal" if self.res_files else "disabled"))
            self._res_plot_current()
        else:
            self.res_scale.configure(state="disabled")
            self._res_plot_grid()

    def _res_plot_blank(self):
        self.res_ax.clear()
        self.res_ax.set_title("No data")
        self.res_canvas.draw()

    def _res_plot_current(self):
        if not self.res_files:
            self._res_plot_blank()
            return
        i = int(self.res_idx.get())
        i = max(0, min(i, len(self.res_files) - 1))
        arr = self.res_data[i]
        name = os.path.basename(self.res_files[i])
        pos = self.res_pos[i]

        self.res_ax.clear()
        if arr is None:
            self.res_ax.set_title(f"Load error: {name}")
        else:
            a = np.asarray(arr)
            if a.ndim >= 2:
                self.res_ax.imshow(a, aspect="auto")
            else:
                self.res_ax.plot(a.reshape(-1))
            self.res_ax.set_title(f"{name} | pos={pos:.3f} mm")
        self.res_canvas.draw()

    def _res_plot_grid(self):
        if not self.res_files:
            self._res_plot_blank()
            return
        cols = max(1, safe_int(self.res_cols.get(), 3))
        n = len(self.res_files)
        rows = int(math.ceil(n / cols))

        win = tk.Toplevel(self)
        win.title("Grid view")
        win.geometry("1200x900")

        fig = Figure(figsize=(max(6, cols * 3), max(4, rows * 2.5)))
        axes = fig.subplots(rows, cols, squeeze=False)
        for ax in axes.flatten():
            ax.axis("off")

        for i in range(n):
            ax = axes[i // cols][i % cols]
            arr = self.res_data[i]
            ax.axis("on")
            if arr is None:
                ax.axis("off")
                continue
            a = np.asarray(arr)
            if a.ndim >= 2:
                ax.imshow(a, aspect="auto")
            else:
                ax.plot(a.reshape(-1))
            ax.set_title(f"{self.res_pos[i]:.3f} mm", fontsize=8)
            ax.tick_params(labelsize=6)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(win); toolbar_frame.pack(fill=tk.X)
        tb = NavigationToolbar2Tk(canvas, toolbar_frame); tb.update()
        canvas.draw()

    def _thumb_clear(self):
        for w in self.thumb_inner.winfo_children():
            w.destroy()
        self._thumb_imgs.clear()
        self._thumb_paths.clear()

    def _thumb_build(self):
        if not self.res_files:
            return
        self._thumb_clear()
        if not _HAS_PIL:
            messagebox.showinfo("Thumbnails", "Pillow not installed. Thumbnails unavailable.")
            return

        max_show = min(200, len(self.res_files))
        for i in range(max_show):
            arr = self.res_data[i]
            if arr is None:
                continue
            a = np.asarray(arr)
            if a.ndim != 2:
                continue

            vmin, vmax = float(np.nanmin(a)), float(np.nanmax(a))
            if vmax <= vmin:
                vmax = vmin + 1.0
            img8 = (255.0 * (a - vmin) / (vmax - vmin)).clip(0, 255).astype(np.uint8)
            im = Image.fromarray(img8, mode="L")
            im.thumbnail((140, 140))
            ph = ImageTk.PhotoImage(im)
            self._thumb_imgs.append(ph)
            self._thumb_paths.append(self.res_files[i])

            frame = ttk.Frame(self.thumb_inner)
            frame.pack(fill=tk.X, padx=6, pady=4)
            btn = ttk.Button(frame, image=ph, command=lambda p=self.res_files[i]: self._open_file_plot(p))
            btn.pack(side=tk.LEFT)
            txt = ttk.Label(frame, text=os.path.basename(self.res_files[i]), width=30, anchor="w")
            txt.pack(side=tk.LEFT, padx=8)

        self._set_status("Thumbnails built.")

    def _open_file_plot(self, path: str):
        try:
            arr = np.load(path)
        except Exception as e:
            messagebox.showerror("Open", str(e))
            return

        win = tk.Toplevel(self)
        win.title(os.path.basename(path))
        win.geometry("900x650")

        fig = Figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tb_frame = ttk.Frame(win); tb_frame.pack(fill=tk.X)
        tb = NavigationToolbar2Tk(canvas, tb_frame); tb.update()

        a = np.asarray(arr)
        ax.clear()
        if a.ndim >= 2:
            ax.imshow(a, aspect="auto")
        else:
            ax.plot(a.reshape(-1))
        ax.set_title(os.path.basename(path))
        canvas.draw()

    # ---- baseline used in batch export too ----
    def _apply_poly_baseline(self, y: np.ndarray, order: int) -> np.ndarray:
        y = np.asarray(y, float).reshape(-1)
        if order is None or int(order) < 0:
            return y
        order = int(order)
        x = np.arange(y.size, dtype=float)
        if y.size <= order + 2:
            return y - float(np.mean(y))
        coef = np.polyfit(x, y, deg=order)
        base = np.polyval(coef, x)
        return y - base

    def _batch_export(self):
        idxs = list(self.res_list.curselection())
        if not idxs:
            messagebox.showinfo("Batch export", "Select at least one file in the list.")
            return
        out_dir = filedialog.askdirectory(initialdir=self.settings.out_dir)
        if not out_dir:
            return
        ensure_dir(out_dir)

        profile_mode = self.batch_profile_mode.get()
        line_y = safe_int(self.batch_line_y.get(), 120)
        baseline_order = safe_int(self.batch_baseline.get(), -1)
        use_roi = bool(self.batch_roi_use_suggested.get()) and (self.an_roi is not None)

        csv_path = os.path.join(out_dir, "batch_profiles.csv")
        rows_written = 0

        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["file", "pos_mm", "x_index", "profile_value"])

            for i in idxs:
                path = self.res_files[i]
                pos = self.res_pos[i]
                arr = self.res_data[i]
                if arr is None:
                    continue
                a = np.asarray(arr)

                if a.ndim == 2:
                    png = os.path.join(out_dir, os.path.basename(path).replace(".npy", ".png"))
                    save_png_from_array(png, a)
                    if _HAS_PIL:
                        try:
                            tif = os.path.join(out_dir, os.path.basename(path).replace(".npy", ".tif"))
                            save_tiff_from_array(tif, a)
                        except Exception:
                            pass

                prof = None
                if a.ndim == 1:
                    prof = a.astype(float).reshape(-1)
                elif a.ndim == 2:
                    if profile_mode == "Line":
                        y = max(0, min(line_y, a.shape[0] - 1))
                        prof = a[y, :].astype(float)
                    else:
                        if use_roi:
                            x0, x1, y0, y1 = self.an_roi
                        else:
                            h, w0 = a.shape
                            y0, y1 = int(0.4*h), int(0.6*h)
                            x0, x1 = int(0.1*w0), int(0.9*w0)
                        x0, x1 = sorted((max(0, x0), min(a.shape[1]-1, x1)))
                        y0, y1 = sorted((max(0, y0), min(a.shape[0]-1, y1)))
                        roi = a[y0:y1+1, x0:x1+1].astype(float)
                        prof = roi.mean(axis=0)
                else:
                    continue

                prof = self._apply_poly_baseline(prof, baseline_order)
                for xi, val in enumerate(prof):
                    w.writerow([os.path.basename(path), f"{pos:.6f}", xi, f"{float(val):.6f}"])
                rows_written += 1

        messagebox.showinfo(
            "Batch export",
            f"Export complete.\nFolder: {out_dir}\nProfiles CSV: {csv_path}\nFiles exported: {len(idxs)}\nProfiles written: {rows_written}"
        )

    # =========================
    # Analysis tab (same logic as dummy: robust baseline, selectable spatial image, Hilbert envelope)
    # =========================
    def _build_analysis_tab(self):
        root = self.tab_analysis

        top = ttk.Frame(root); top.pack(fill=tk.X, padx=10, pady=10)
        self.an_folder = tk.StringVar(value=self.settings.out_dir)
        ttk.Label(top, text="Folder").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.an_folder, width=80).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Browse", command=self._an_browse).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load dataset", command=self._an_load).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Run analysis", command=self._an_run).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Export CSV+Plots", command=self._an_export).pack(side=tk.LEFT, padx=6)

        sel = ttk.LabelFrame(root, text="Analysis settings")
        sel.pack(fill=tk.X, padx=10, pady=(0,10))

        r = ttk.Frame(sel); r.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(r, text="Region").pack(side=tk.LEFT)
        ttk.Combobox(r, textvariable=self.an_mode, values=["ROI", "Line"], width=6, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Label(r, text="Line y").pack(side=tk.LEFT, padx=(12,0))
        ttk.Entry(r, textvariable=self.an_line_y, width=8).pack(side=tk.LEFT, padx=6)

        ttk.Label(r, text="Pixel size (µm)").pack(side=tk.LEFT, padx=(12,0))
        ttk.Entry(r, textvariable=self.an_pixel_um, width=8).pack(side=tk.LEFT, padx=6)

        ttk.Label(r, text="Band ±bins").pack(side=tk.LEFT, padx=(12,0))
        ttk.Entry(r, textvariable=self.an_bw, width=6).pack(side=tk.LEFT, padx=6)

        r2 = ttk.Frame(sel); r2.pack(fill=tk.X, padx=8, pady=(0,6))
        ttk.Label(r2, text="Baseline poly order (-1 none)").pack(side=tk.LEFT)
        ttk.Combobox(r2, textvariable=self.an_baseline_order, values=["-1", "0", "1", "2", "3", "4", "5"], width=4, state="readonly").pack(side=tk.LEFT, padx=6)

        ttk.Label(r2, text="Visibility method").pack(side=tk.LEFT, padx=(12,0))
        ttk.Combobox(r2, textvariable=self.an_vis_method, values=["FFT", "PeakValley", "Michelson", "Hilbert"], width=12, state="readonly").pack(side=tk.LEFT, padx=6)

        ttk.Label(r2, text="Temporal model").pack(side=tk.LEFT, padx=(12,0))
        ttk.Combobox(
            r2,
            textvariable=self.an_model_sel,
            values=["Auto(AIC)", "Auto(BIC)", "Gaussian", "Lorentzian", "sech2", "Exponential"],
            width=12,
            state="readonly",
        ).pack(side=tk.LEFT, padx=6)

        ttk.Button(r2, text="Select ROI", command=self._an_select_roi).pack(side=tk.LEFT, padx=12)
        ttk.Button(r2, text="Suggest ROI", command=self._an_suggest_roi).pack(side=tk.LEFT, padx=6)

        r3 = ttk.Frame(sel); r3.pack(fill=tk.X, padx=8, pady=(0,6))
        ttk.Label(r3, text="Spatial coherence image").pack(side=tk.LEFT)
        self.an_spatial_combo = ttk.Combobox(r3, textvariable=self.an_spatial_choice, values=["Best visibility"], width=60, state="readonly")
        self.an_spatial_combo.pack(side=tk.LEFT, padx=6)
        ttk.Button(r3, text="Update spatial plot", command=self._update_spatial_from_choice).pack(side=tk.LEFT, padx=8)

        self.an_status = tk.StringVar(value="Load a folder with saved .npy images.")
        ttk.Label(root, textvariable=self.an_status, anchor="w").pack(fill=tk.X, padx=18, pady=(0,10))

        plots = ttk.Frame(root); plots.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        plots.columnconfigure(0, weight=1)
        plots.columnconfigure(1, weight=1)
        plots.rowconfigure(0, weight=1)
        plots.rowconfigure(1, weight=0)

        self.best_fig = Figure()
        self.best_ax = self.best_fig.add_subplot(111)
        self.best_canvas = FigureCanvasTkAgg(self.best_fig, master=plots)
        self.best_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        tf = ttk.Frame(plots); tf.grid(row=1, column=0, sticky="ew", padx=6)
        self.best_toolbar = NavigationToolbar2Tk(self.best_canvas, tf); self.best_toolbar.update()

        self.temp_fig = Figure()
        self.temp_ax = self.temp_fig.add_subplot(111)
        self.temp_canvas = FigureCanvasTkAgg(self.temp_fig, master=plots)
        self.temp_canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        tf2 = ttk.Frame(plots); tf2.grid(row=1, column=1, sticky="ew", padx=6)
        self.temp_toolbar = NavigationToolbar2Tk(self.temp_canvas, tf2); self.temp_toolbar.update()

    def _an_browse(self):
        d = filedialog.askdirectory(initialdir=self.an_folder.get() or self.settings.out_dir)
        if d:
            self.an_folder.set(d)

    def _an_load(self):
        folder = self.an_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Analysis", "Select a valid folder.")
            return
        files = list_npy_files_sorted(folder)
        if not files:
            self.an_status.set("No .npy files found.")
            return

        imgs, pos, kept = [], [], []
        for p in files:
            try:
                a = np.load(p)
            except Exception:
                continue
            if np.asarray(a).ndim < 2:
                continue
            imgs.append(np.asarray(a))
            pos.append(parse_position_mm_from_filename(os.path.basename(p)) or 0.0)
            kept.append(p)

        if not imgs:
            self.an_status.set("No 2D image .npy files found.")
            return

        idxs = sorted(range(len(imgs)), key=lambda i: pos[i])
        self.an_imgs = [imgs[i] for i in idxs]
        self.an_pos_mm = np.array([pos[i] for i in idxs], dtype=float)
        self.an_files = [kept[i] for i in idxs]
        self.an_vis = np.array([])
        self.an_best_idx = 0
        self.an_roi = None
        self._plot_image(0, title="Loaded (index 0)")
        self.an_status.set(f"Loaded {len(self.an_imgs)} images.")

        items = ["Best visibility"]
        for i, p in enumerate(self.an_files):
            items.append(f"[{i:04d}] pos={self.an_pos_mm[i]:.3f} mm | {os.path.basename(p)}")
        self.an_spatial_combo.configure(values=items)
        self.an_spatial_choice.set("Best visibility")

    def _profile_from_image(self, img: np.ndarray) -> np.ndarray:
        mode = self.an_mode.get()
        if mode == "Line":
            y = safe_int(self.an_line_y.get(), img.shape[0] // 2)
            y = max(0, min(y, img.shape[0] - 1))
            return img[y, :].astype(float)
        if self.an_roi is None:
            h, w = img.shape
            y0, y1 = int(0.4*h), int(0.6*h)
            x0, x1 = int(0.1*w), int(0.9*w)
        else:
            x0, x1, y0, y1 = self.an_roi
        x0, x1 = sorted((max(0, x0), min(img.shape[1]-1, x1)))
        y0, y1 = sorted((max(0, y0), min(img.shape[0]-1, y1)))
        roi = img[y0:y1+1, x0:x1+1].astype(float)
        return roi.mean(axis=0)

    def _analytic_signal(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float).reshape(-1)
        N = x.size
        X = np.fft.fft(x)
        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = 1
            h[N//2] = 1
            h[1:N//2] = 2
        else:
            h[0] = 1
            h[1:(N+1)//2] = 2
        return np.fft.ifft(X * h)

    def _visibility(self, profile_raw: np.ndarray) -> float:
        method = self.an_vis_method.get()
        order = safe_int(self.an_baseline_order.get(), -1)

        raw = np.asarray(profile_raw, float).reshape(-1)
        if raw.size < 8:
            return 0.0

        p = self._apply_poly_baseline(raw, order)

        if method in ("PeakValley", "Michelson"):
            q = p.copy()
            k = 5
            q = np.convolve(q, np.ones(k)/k, mode="same")
            vmax = float(np.max(q))
            vmin = float(np.min(q))
            denom = vmax + vmin
            if denom <= 0:
                return 0.0
            return float(max(0.0, min(1.5, (vmax - vmin) / denom)))

        if method == "Hilbert":
            x = p - float(np.mean(p))
            xa = self._analytic_signal(x)
            env = np.abs(xa)
            dc = float(abs(np.mean(raw)) + 1e-12)
            V = float(np.mean(env) / dc)
            return float(max(0.0, min(1.5, V)))

        dc = float(abs(np.sum(raw)) + 1e-12)  # raw DC reference
        x = p - float(np.mean(p))
        F = np.fft.rfft(x)
        mag = np.abs(F)
        if mag.size < 3:
            return 0.0
        k0 = int(np.argmax(mag[1:]) + 1)
        V = 2.0 * float(mag[k0]) / dc
        return float(max(0.0, min(1.5, V)))

    def _dominant_k(self, profile_raw: np.ndarray) -> int:
        order = safe_int(self.an_baseline_order.get(), -1)
        raw = np.asarray(profile_raw, float).reshape(-1)
        p = self._apply_poly_baseline(raw, order)
        x = p - float(np.mean(p))
        F = np.fft.rfft(x)
        mag = np.abs(F)
        if mag.size < 3:
            return 1
        return max(1, int(np.argmax(mag[1:]) + 1))

    def _bandpass_osc(self, profile_raw: np.ndarray, k0: int, bw: int) -> np.ndarray:
        order = safe_int(self.an_baseline_order.get(), -1)
        raw = np.asarray(profile_raw, float).reshape(-1)
        p = self._apply_poly_baseline(raw, order)
        x = p - float(np.mean(p))
        F = np.fft.rfft(x)
        keep = np.zeros_like(F, dtype=complex)
        k0 = max(1, int(k0))
        bw = max(1, int(bw))
        k1 = max(1, k0 - bw)
        k2 = min(F.size - 1, k0 + bw)
        keep[k1:k2+1] = F[k1:k2+1]
        return np.fft.irfft(keep, n=x.size)

    def _spatial_choice_to_index(self) -> Optional[int]:
        s = (self.an_spatial_choice.get() or "").strip()
        if s == "Best visibility":
            return self.an_best_idx if self.an_vis.size > 0 else 0
        m = re.match(r"^\[(\d+)\]", s)
        if not m:
            return None
        try:
            i = int(m.group(1))
        except Exception:
            return None
        if i < 0 or i >= len(self.an_imgs):
            return None
        return i

    def _spatial_coherence(self, img: np.ndarray) -> Dict[str, Any]:
        prof = self._profile_from_image(img)
        k0 = self._dominant_k(prof)
        bw = safe_int(self.an_bw.get(), 5)
        osc = self._bandpass_osc(prof, k0, bw)

        xa = self._analytic_signal(osc)
        env = np.abs(xa)

        k = 11
        env_s = np.convolve(env, np.ones(k)/k, mode="same") if env.size >= k else env.copy()

        floor = float(np.percentile(env_s, 10))
        env_fit_target = np.clip(env_s - floor, 0, None)

        x_px = np.arange(env_fit_target.size, dtype=float)
        if env_fit_target.size < 20:
            return {"ok": False}

        x0_guess = float(x_px[int(np.argmax(env_fit_target))])
        span = float(np.max(x_px) - np.min(x_px))
        sigmas = np.logspace(math.log10(max(2.0, span/400)), math.log10(max(5.0, span/2)), 120)
        w0 = max(5, int(0.12 * env_fit_target.size))
        x0_candidates = np.linspace(max(0, x0_guess - w0), min(env_fit_target.size-1, x0_guess + w0), 41)

        best = {"ok": False, "sse": float("inf")}
        for x0 in x0_candidates:
            dx = (x_px - x0)
            for sigma in sigmas:
                phi = np.exp(-0.5 * (dx / sigma) ** 2)
                A = np.vstack([phi, np.ones_like(phi)]).T
                coef, *_ = np.linalg.lstsq(A, env_fit_target, rcond=None)
                a, b = float(coef[0]), float(coef[1])
                if a < 0:
                    a = 0.0
                if b < 0:
                    b = 0.0
                yhat = a * phi + b
                sse = float(np.mean((env_fit_target - yhat) ** 2))
                if sse < best["sse"]:
                    best = {"ok": True, "A": a, "B": b, "sigma": float(sigma), "x0": float(x0), "sse": sse}

        if not best["ok"]:
            return {"ok": False}

        fwhm_px = 2.0 * math.sqrt(2.0 * math.log(2.0)) * best["sigma"]
        px_um = safe_float(self.an_pixel_um.get(), self.settings.pixel_size_um)
        self.settings.pixel_size_um = px_um

        phi = np.exp(-0.5 * ((x_px - best["x0"]) / best["sigma"]) ** 2)
        env_model = best["A"] * phi + best["B"]

        return {
            "ok": True,
            "x_um": x_px * px_um,
            "env": env_s,
            "env_model": env_model,
            "fit": best,
            "fwhm_um": float(fwhm_px * px_um),
            "k0": int(k0),
            "bw": int(bw),
        }

    def _show_spatial_plot(self, spatial: Dict[str, Any], title_extra: str = ""):
        if not spatial.get("ok"):
            return
        if self.spatial_win is None or not self.spatial_win.winfo_exists():
            self.spatial_win = tk.Toplevel(self)
            self.spatial_win.title("Spatial coherence")
            self.spatial_win.geometry("1000x700")

            self.spat_fig = Figure()
            self.spat_ax = self.spat_fig.add_subplot(111)
            self.spat_canvas = FigureCanvasTkAgg(self.spat_fig, master=self.spatial_win)
            self.spat_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            tf = ttk.Frame(self.spatial_win); tf.pack(fill=tk.X)
            self.spat_tb = NavigationToolbar2Tk(self.spat_canvas, tf); self.spat_tb.update()

        x_um = spatial["x_um"]
        env_plot = spatial["env"]
        env_model = spatial["env_model"]

        self.spat_ax.clear()
        self.spat_ax.plot(x_um, env_plot, linestyle="-", label="Envelope (Hilbert, smoothed)")
        self.spat_ax.plot(x_um, env_model, linestyle="--", label="Gaussian fit")
        t = f"Spatial coherence FWHM = {spatial['fwhm_um']:.2f} µm"
        if title_extra:
            t += " | " + title_extra
        self.spat_ax.set_title(t)
        self.spat_ax.set_xlabel("Spatial coordinate (µm)")
        self.spat_ax.set_ylabel("Envelope amplitude (a.u.)")
        self.spat_ax.grid(True)
        self.spat_ax.legend()
        self.spat_canvas.draw()

    def _update_spatial_from_choice(self):
        if not self.an_imgs:
            messagebox.showinfo("Spatial", "Load dataset first.")
            return
        idx = self._spatial_choice_to_index()
        if idx is None:
            idx = self.an_best_idx if self.an_imgs else 0
        img = self.an_imgs[idx]
        spatial = self._spatial_coherence(img)
        if not spatial.get("ok"):
            self.an_status.set("Spatial coherence failed (insufficient data/fit).")
            return
        extra = f"idx={idx:04d} pos={self.an_pos_mm[idx]:.3f} mm"
        self._show_spatial_plot(spatial, title_extra=extra)
        self.an_status.set(f"Spatial coherence updated from selected image: {extra}")

    def _plot_image(self, idx: int, title: str = ""):
        if not self.an_imgs:
            return
        idx = int(max(0, min(idx, len(self.an_imgs)-1)))
        img = self.an_imgs[idx]
        self.best_ax.clear()
        self.best_ax.imshow(img, aspect="auto")
        name = os.path.basename(self.an_files[idx]) if self.an_files else f"idx={idx}"
        self.best_ax.set_title(title if title else name)
        if self.an_roi is not None:
            x0, x1, y0, y1 = self.an_roi
            self.best_ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=2)
        self.best_canvas.draw()

    def _an_select_roi(self):
        if not self.an_imgs:
            messagebox.showinfo("Analysis", "Load dataset first.")
            return
        self.an_status.set("Drag to select ROI on the image plot. Release to set ROI.")
        if self._rect_selector is not None:
            try:
                self._rect_selector.set_active(False)
            except Exception:
                pass
            self._rect_selector = None

        def onselect(eclick, erelease):
            if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
                return
            x0, y0 = int(round(eclick.xdata)), int(round(eclick.ydata))
            x1, y1 = int(round(erelease.xdata)), int(round(erelease.ydata))
            self.an_roi = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            self._plot_image(self.an_best_idx if self.an_imgs else 0, title="ROI set")
            self.an_status.set(f"ROI set: x[{self.an_roi[0]}:{self.an_roi[1]}], y[{self.an_roi[2]}:{self.an_roi[3]}]")

        self._rect_selector = RectangleSelector(self.best_ax, onselect, useblit=True, button=[1], interactive=True)
        self.best_canvas.draw()

    def _an_suggest_roi(self):
        if not self.an_imgs:
            messagebox.showinfo("Suggest ROI", "Load dataset first.")
            return
        img = self.an_imgs[self.an_best_idx] if (self.an_vis.size > 0) else self.an_imgs[len(self.an_imgs)//2]
        h, w = img.shape
        roi_h = max(20, int(0.20 * h))
        roi_w = max(60, int(0.50 * w))
        ys = np.linspace(0, h - roi_h, 8).astype(int)
        xs = np.linspace(0, w - roi_w, 6).astype(int)

        best = {"score": -1e18, "roi": None}
        for y0 in ys:
            for x0 in xs:
                y1 = y0 + roi_h
                x1 = x0 + roi_w
                roi = img[y0:y1, x0:x1].astype(float)
                prof = roi.mean(axis=0)
                v = self._visibility(prof)
                snr = float(np.mean(roi) / (np.std(roi) + 1e-12))
                score = 1000.0 * v + 0.05 * snr
                if score > best["score"]:
                    best = {"score": score, "roi": (x0, x1-1, y0, y1-1), "v": v, "snr": snr}

        if best["roi"] is None:
            messagebox.showerror("Suggest ROI", "ROI suggestion failed.")
            return
        self.an_roi = best["roi"]
        self._plot_image(self.an_best_idx if self.an_imgs else 0, title="ROI set (suggested)")
        self.an_status.set(f"Suggested ROI set. vis={best['v']:.3f} snr={best['snr']:.1f}")

    def _fit_models(self, delay_fs: np.ndarray, vis: np.ndarray) -> Dict[str, Any]:
        t = np.asarray(delay_fs, float)
        v = np.asarray(vis, float)
        mask = np.isfinite(t) & np.isfinite(v)
        t = t[mask]
        v = np.clip(v[mask], 0, 1.5)
        if t.size < 6:
            return {"ok": False}

        ta = np.abs(t)
        span = max(1e-12, float(np.max(ta) - np.min(ta)))
        taus = np.logspace(math.log10(max(1e-6, span/200)), math.log10(max(1e-3, span*2)), 160)

        def basis(model: str, tau: float) -> np.ndarray:
            x = ta / tau
            if model == "Gaussian":
                return np.exp(-(x**2))
            if model == "Lorentzian":
                return 1.0 / (1.0 + x**2)
            if model == "sech2":
                return 1.0 / (np.cosh(x)**2 + 1e-12)
            if model == "Exponential":
                return np.exp(-x)
            raise ValueError(model)

        def fit_one(model: str) -> Dict[str, Any]:
            best = {"ok": False, "sse": float("inf")}
            for tau in taus:
                phi = basis(model, tau)
                A = np.vstack([phi, np.ones_like(phi)]).T
                coef, *_ = np.linalg.lstsq(A, v, rcond=None)
                a, b = float(coef[0]), float(coef[1])
                b = float(max(0.0, min(0.8, b)))
                if a < 0:
                    a = 0.0
                vhat = a * phi + b
                sse = float(np.sum((v - vhat) ** 2))
                if sse < best["sse"]:
                    best = {"ok": True, "model": model, "A": a, "B": b, "tau": float(tau), "sse": sse}
            if not best["ok"]:
                return {"ok": False}
            n = int(v.size)
            k = 3
            sse = max(1e-12, best["sse"])
            crit = n * math.log(sse / n)
            best["AIC"] = crit + 2 * k
            best["BIC"] = crit + k * math.log(n)
            if best["model"] == "Gaussian":
                best["fwhm"] = best["tau"] * math.sqrt(2.0 * math.log(2.0))
            elif best["model"] == "Lorentzian":
                best["fwhm"] = best["tau"] * 1.0
            elif best["model"] == "sech2":
                best["fwhm"] = best["tau"] * float(np.arccosh(math.sqrt(2.0)))
            elif best["model"] == "Exponential":
                best["fwhm"] = best["tau"] * math.log(2.0)
            return best

        models = ["Gaussian", "Lorentzian", "sech2", "Exponential"]
        fits = [fit_one(m) for m in models if fit_one(m).get("ok")]
        if not fits:
            return {"ok": False}

        sel = self.an_model_sel.get()
        if sel in ("Gaussian", "Lorentzian", "sech2", "Exponential"):
            chosen = min([f for f in fits if f["model"] == sel], key=lambda d: d["sse"], default=fits[0])
        elif sel == "Auto(BIC)":
            chosen = min(fits, key=lambda d: d["BIC"])
        else:
            chosen = min(fits, key=lambda d: d["AIC"])

        return {"ok": True, "fits": fits, "best": chosen}

    def _basis_eval(self, model: str, t_fs: np.ndarray, tau: float) -> np.ndarray:
        ta = np.abs(np.asarray(t_fs, float))
        x = ta / float(tau)
        if model == "Gaussian":
            return np.exp(-(x**2))
        if model == "Lorentzian":
            return 1.0 / (1.0 + x**2)
        if model == "sech2":
            return 1.0 / (np.cosh(x)**2 + 1e-12)
        if model == "Exponential":
            return np.exp(-x)
        raise ValueError(model)

    def _an_run(self):
        if not self.an_imgs:
            messagebox.showinfo("Analysis", "Load dataset first.")
            return

        vis = []
        for img in self.an_imgs:
            prof = self._profile_from_image(img)
            vis.append(self._visibility(prof))
        vis = np.array(vis, dtype=float)
        self.an_vis = vis
        self.an_best_idx = int(np.nanargmax(vis)) if np.any(np.isfinite(vis)) else 0

        pos0 = float(self.an_pos_mm[self.an_best_idx])
        dx_mm = self.an_pos_mm - pos0
        delay_fs = (2.0 * dx_mm * 1e-3) / float(self.settings.c_m_s) * 1e15  # Michelson: 2Δx/c

        fits = self._fit_models(delay_fs, vis)
        if not fits.get("ok"):
            self.an_status.set("Temporal fit failed (not enough data).")
            return
        best = fits["best"]

        self.temp_ax.clear()
        self.temp_ax.plot(delay_fs, np.clip(vis, 0, 1.0), marker="o", linestyle="-", label="data")
        t = np.linspace(float(np.min(delay_fs)), float(np.max(delay_fs)), 400)
        phi = self._basis_eval(best["model"], t, best["tau"])
        vfit = best["A"] * phi + best["B"]
        self.temp_ax.plot(t, np.clip(vfit, 0, 1.0), linestyle="--", label=f"fit: {best['model']}")
        self.temp_ax.set_title("Visibility vs delay (Michelson: 2Δx/c)")
        self.temp_ax.set_xlabel("Delay (fs) relative to max-V position")
        self.temp_ax.set_ylabel("Visibility")
        self.temp_ax.grid(True)
        self.temp_ax.legend()

        txt = f"Chosen: {best['model']} | tau={best['tau']:.2f} fs | FWHM={best['fwhm']:.2f} fs\n"
        txt += "Model scores (lower is better):\n"
        for f in sorted(fits["fits"], key=lambda d: d["AIC"]):
            txt += f"  {f['model']}: AIC={f['AIC']:.1f} BIC={f['BIC']:.1f} sse={f['sse']:.3g}\n"
        self.temp_ax.text(0.02, 0.98, txt, transform=self.temp_ax.transAxes, va="top", ha="left",
                          bbox=dict(boxstyle="round", alpha=0.2), fontsize=9)
        self.temp_canvas.draw()

        self._plot_image(self.an_best_idx, title=f"Best visibility (idx={self.an_best_idx})")
        self._update_spatial_from_choice()

        msg = f"Best: {os.path.basename(self.an_files[self.an_best_idx])} | pos={pos0:.3f} mm | max V={vis[self.an_best_idx]:.3f}"
        msg += f" | model={best['model']} FWHM={best['fwhm']:.1f} fs"
        self.an_status.set(msg)

        self.an_last = {
            "folder": self.an_folder.get().strip(),
            "files": self.an_files,
            "positions_mm": self.an_pos_mm.copy(),
            "visibility": vis.copy(),
            "best_idx": self.an_best_idx,
            "best_file": self.an_files[self.an_best_idx],
            "pos0_mm": pos0,
            "delay_fs": delay_fs,
            "temporal_best": best,
            "temporal_all": fits["fits"],
            "roi": self.an_roi,
            "mode": self.an_mode.get(),
            "pixel_size_um": safe_float(self.an_pixel_um.get(), self.settings.pixel_size_um),
            "baseline_poly_order": safe_int(self.an_baseline_order.get(), -1),
            "visibility_method": self.an_vis_method.get(),
            "spatial_choice": self.an_spatial_choice.get(),
        }

    def _an_export(self):
        if not hasattr(self, "an_last"):
            messagebox.showinfo("Export", "Run analysis first.")
            return
        out_dir = filedialog.askdirectory(initialdir=self.an_folder.get() or self.settings.out_dir)
        if not out_dir:
            return
        ensure_dir(out_dir)

        base = "analysis"
        csv_path = os.path.join(out_dir, base + ".csv")
        tpng = os.path.join(out_dir, base + "_temporal.png")
        bpng = os.path.join(out_dir, base + "_image.png")
        spng = os.path.join(out_dir, base + "_spatial.png")

        last = self.an_last
        best = last["temporal_best"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["# folder", last["folder"]])
            w.writerow(["# best_file", os.path.basename(last["best_file"])])
            w.writerow(["# spatial_choice", last.get("spatial_choice", "")])
            w.writerow(["# region_mode", last["mode"]])
            w.writerow(["# roi", last["roi"]])
            w.writerow(["# pixel_size_um", last["pixel_size_um"]])
            w.writerow(["# baseline_poly_order", last["baseline_poly_order"]])
            w.writerow(["# visibility_method", last["visibility_method"]])
            w.writerow(["# chosen_model", best["model"]])
            w.writerow(["# tau_fs", best["tau"]])
            w.writerow(["# fwhm_fs", best["fwhm"]])
            w.writerow(["# A", best["A"]])
            w.writerow(["# B", best["B"]])
            w.writerow(["position_mm", "delay_fs", "visibility"])
            for p, t, v in zip(last["positions_mm"], last["delay_fs"], last["visibility"]):
                w.writerow([f"{p:.6f}", f"{t:.6f}", f"{v:.6f}"])

        self.temp_fig.savefig(tpng, dpi=200)
        self.best_fig.savefig(bpng, dpi=200)
        if self.spatial_win is not None and self.spatial_win.winfo_exists():
            self.spat_fig.savefig(spng, dpi=200)

        messagebox.showinfo("Export", f"Saved:\n{csv_path}\n{tpng}\n{bpng}\n{spng if os.path.exists(spng) else ''}")

    # =========================
    # Poll loop (status + continuous)
    # =========================
    def _poll(self):
        while True:
            try:
                msg, payload = self.ui_q.get_nowait()
            except queue.Empty:
                break
            if msg == "status":
                self._set_status(payload)
            elif msg == "error":
                self._set_status(payload)
                messagebox.showerror("Error", str(payload))
            elif msg == "last":
                meas: Measurement = payload
                saved = os.path.basename(meas.saved_paths[0]) if meas.saved_paths else ""
                self.last_info.set(
                    f"Last: id={meas.meas_id}, kind={meas.kind}, exp={meas.exposure_s:.3f}s, pos={format_pos_mm(meas.stage_mm)} mm"
                    + (f", saved={saved}" if saved else "")
                )

        # stage status
        if self.stage.is_connected():
            try:
                counts = self.stage.get_position_counts()
                mm = self.stage.get_position_mm()
                moving = self.stage.is_moving()
                homed = self.stage.is_homed()
                enabled, err = self.stage.get_enabled_error()
            except Exception:
                counts, mm, moving, homed, enabled, err = 0, 0.0, False, False, None, None

            self.pos_counts_var.set(str(counts))
            self.pos_mm_var.set(f"{mm:.6f}")
            self.moving_var.set(str(bool(moving)))
            self.homed_var.set(str(bool(homed)))

            self.stage_status_var.set("Stage: connected")
            self.stage_en_var.set(f"Enabled: {('--' if enabled is None else enabled)}")
            self.stage_err_var.set(f"Err: {('--' if err is None else err)}")
        else:
            self.stage_status_var.set("Stage: disconnected")
            self.stage_en_var.set("Enabled: --")
            self.stage_err_var.set("Err: --")
            self.moving_var.set("False")
            self.homed_var.set("False")

        # camera status
        self.cam_state.set("connected" if self.camera.is_connected() else "disconnected")
        if self.camera.is_connected():
            self.cam_status_var.set("Camera: connected")
            tC = self.camera.get_temperature()
            self.cam_temp_var.set(f"Temp: {tC:.2f} C" if tC is not None else "Temp: --")
            self.cam_shutter_var.set(f"Shutter: {self.camera.shutter_state}")
            self.cam_acq_var.set(f"Acq: {self.camera.acq_state}")
        else:
            self.cam_status_var.set("Camera: disconnected")
            self.cam_temp_var.set("Temp: --")
            self.cam_shutter_var.set("Shutter: --")
            self.cam_acq_var.set("Acq: --")

        # continuous FVB
        if self.camera.is_connected():
            frames, arr = self.camera.get_latest()
            self.cont_frames.set(f"Frames: {frames}")
            if self.cont_running and arr is not None and frames != self.cont_last_frame:
                self.cont_last_frame = frames
                self.cont_ax.clear()
                self.cont_ax.plot(np.asarray(arr).reshape(-1))
                self.cont_ax.set_title(f"Continuous FVB | frame {frames}")
                self.cont_canvas.draw()

        self.after(80, self._poll)


if __name__ == "__main__":
    ensure_dir(DEFAULT_OUTDIR)
    App().mainloop()
