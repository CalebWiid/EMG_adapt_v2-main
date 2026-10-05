"""
This script contains the functions to preprocess the data from the EMG signals. 

It includes: 

load data (DB5 for now) --> read and detect gesture --> filter signal --> segment signal 
        --> FFT --> Power spectrum --> Cepstral coefficients --> CCA 

Currently the code uses the data from the DB5 dataset to find the features but this will be replaced with data recored form the MindRove device.

**Note - The CCA values describe the stable, low-frequency activation patterns of the EMG signals that allows for generalization. 

"""
from __future__ import annotations
 
import os
from dataclasses import dataclass
from typing import Optional
 
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

# ----------------------Configuration---------------------- #
#DATA_DIR = r"C:\Users\25861670\Masters_2026\EMG_sensor_data\DB5"
DATA_DIR = '/home/caleb/caleb_work/DB5'
FS = 200 
WINDOW_SIZE_MS = 200
STEP_SIZE_MS = 10
LOG_EPS = 1e-10             #Prevents log(0) 

# Interleaves the two Myo bands so physically adjacent electrodes are neighbours.
# This is a MODELLING choice: the Conv1D slides along the channel axis, so column
# order is the spatial layout the network sees. Pass channel_order=None to load_data
# to keep the native file order instead. Only applied when it matches the channel
# count (so the 8-channel MindRove case is left untouched).
PHYSICAL_CHANNEL_ORDER = [1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16]

# To mimic the Mindroves 8 channel layout, revome the second myo band 
SELECT_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]  

DEFAULT_FILTER_KWARGS = dict(
    lowpass_cutoff=1.0,
    lowpass_order=1, 
    highpass_cutoff=20.0, 
    highpass_order=4, 
    rectify=True,
)


def filename_for(subject, exercise):
    """Build a DB5 file path from subject/exercise numbers (S{subject}_E{exercise}_A1.mat)."""
    return os.path.join(DATA_DIR, f"S{subject}_E{exercise}_A1.mat")


@dataclass
class GestureSegment: 
    """"Find 1 continuous performance of a single gesture"""
    label: int 
    repetition: int 
    emg: np.ndarray

#----------------------Load Data---------------------- #
def load_data(path: str, channel_order: Optional[list] = PHYSICAL_CHANNEL_ORDER, select_channels: Optional[list] = SELECT_CHANNELS):
    """Load data from the DB5 dataset.
    
        returns 
        ____ 
        EMG         : as a float array 
        restimulus  : as a int array
        repetition  : as a int array
    
    """
    mat = loadmat(path)
    emg = np.asarray(mat['emg'], dtype=np.float64)
    restimulus = np.asarray(mat["restimulus"]).squeeze().astype(int)
    rerepetition = np.asarray(mat["rerepetition"]).squeeze().astype(int)

    if emg.ndim != 2:
        raise ValueError(f"expected emg to be 2-D (samples, channels), got {emg.shape}")

    if select_channels is not None and emg.shape[1] >= len(select_channels):
        emg = emg[:, [c-1 for c in select_channels]]

    # Reorder channels only if the order matches the channel count (1-based -> 0-based).
    if channel_order is not None and len(channel_order) == emg.shape[1]:
        emg = emg[:, [c - 1 for c in channel_order]]

    return emg, restimulus, rerepetition

#----------------------Gesture detection---------------------- #
def find_segments(restimulus: np.ndarray, 
                  rerepetition: Optional[np.ndarray] = None, 
                  include_rest: bool = False): 
    """Find the segments of the EMG signal corresponding to each gesture. 
        ignore rest laabels """ 

    labels = np.asarray(restimulus).astype(int)
    change = np.diff(labels) != 0
    if rerepetition is not None:
        change = change | (np.diff(np.asarray(rerepetition).astype(int)) != 0)
    change_idx = np.where(change)[0] + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(labels)]))
 
    segments = []
    for s, e in zip(starts, ends):
        lab = int(labels[s])
        if lab == 0 and not include_rest:
            continue
        rep = int(rerepetition[s]) if rerepetition is not None else -1
        segments.append((lab, rep, int(s), int(e)))
    return segments

def segment_emg(emg: np.ndarray,
                restimulus: np.ndarray, 
                rerepetition: Optional[np.ndarray] = None,
                include_rest: bool = False):

    """Return a list of GestureSegment objects, one per gesture performance."""
    segs = []
    for lab, rep, s, e in find_segments(restimulus, rerepetition, include_rest):
        segs.append(GestureSegment(label=lab, repetition=rep, emg=emg[s:e, :]))
    return segs

#----------------------Filtering---------------------- #
def zero_phase_filter(sos, x): 
    """Zero-phase (forward-backward) filtering with a second-order-section filter.
    padlen is clamped so short segments don't raise."""
    n = x.shape[0]
    default_pad = 3 * (2 * sos.shape[0] + 1)
    padlen = default_pad if n > default_pad else max(0, n - 1)
    return sosfiltfilt(sos, x, axis=0, padlen=padlen)

def preprocess_signal(emg: np.ndarray,
                      fs: float,
                      lowpass_cutoff: Optional[float] = 1.0,
                      lowpass_order: int = 1,
                      highpass_cutoff: Optional[float] = 20.0,
                      highpass_order: int = 4,
                      rectify: bool = True) -> np.ndarray:
    """Apply filters (optional high-pass and recifying) to the EMG signal.""" 
    x = emg.astype(np.float64)
    nyq = fs/2.0 

    if highpass_cutoff is not None:
        sos = butter(highpass_order, highpass_cutoff / nyq, btype="highpass", output="sos")
        x = zero_phase_filter(sos, x)
 
    if rectify:
        x = np.abs(x)
 
    if lowpass_cutoff is not None:
        sos = butter(lowpass_order, lowpass_cutoff / nyq, btype="lowpass", output="sos")
        x = zero_phase_filter(sos, x)
 
    return x 

#----------------------Windowing---------------------- #
def window_signal(emg: np.ndarray, 
                  fs: float, 
                  window_ms: float = WINDOW_SIZE_MS,
                  step_ms: float = STEP_SIZE_MS) -> np.ndarray:
    """Slide window over the processed signal"""

    win = int(round(window_ms * fs / 1000))
    step = int(round(step_ms * fs / 1000))
    n, ch = emg.shape
    if win <= 0 or step <= 0:
        raise ValueError("window/step resolve to <= 0 samples")
    if n < win:
        return np.empty((0, win, ch))
    starts = range(0, n - win + 1, step)
    return np.stack([emg[s:s + win, :] for s in starts], axis=0)

#----------------------Compute CCA---------------------- #
def compute_cepstrum(window: np.ndarray) -> np.ndarray:
    """Compute the cepstral coefficients of a windowed signal."""
    spectrum = np.fft.fft(window, axis=0)
    power = np.abs(spectrum) ** 2
    log_power = np.log(power + LOG_EPS)
    cepstrum = np.fft.ifft(log_power, axis=0).real
    return cepstrum

def compute_cca(window: np.ndarray,
                drop_zero_quefrency: bool = False) -> np.ndarray:
    """ CCA feature for one window: average cepstral coeffs across quefrency.
    
    **Note: the drop_zero_quefrency option is set to false. The n=0 coeeficient reacks the overall log-energy and can dominate the values. 
            if this causes an issue set drop_zero_quefrency to True.
    """
    cepstrum = compute_cepstrum(window)
    if drop_zero_quefrency:
        cepstrum = cepstrum[1:, :]
    return cepstrum.mean(axis=0)

def cca_features(windows: np.ndarray, **kwargs) -> np.ndarray:
    """Compute CCA features for all windows."""
    if windows.ndim != 3: 
        raise ValueError(f"Expected (n_windows, win, ch), got {windows.shape}")
    
    if windows.shape[0] == 0:
        return np.empty((0, windows.shape[2]))

    
    return np.stack([compute_cca(w, **kwargs) for w in windows], axis=0)

#----------------------Segment- and file-level pipelines---------------------- #
def load_and_segment(path: str,
                     fs: float = FS,
                     include_rest: bool = False,
                     filter_kwargs: Optional[dict] = None):
    """Load a file, filter the WHOLE recording, then segment (OPTION B).

    Filtering the entire recording BEFORE segmenting means each gesture's start
    and end are filtered using the real neighbouring samples, so there are no
    per-segment filter edge transients (a 1 Hz zero-phase filter has a long
    settling time relative to a single gesture, so per-segment filtering would
    leave an unreliable boundary region).

    NOTE: the paper does not specify per-segment vs whole-recording filtering.
    Its wording ("filtered signals are then segmented using a sliding window")
    is consistent with this whole-recording approach, and it avoids the edge
    artefact, so we adopt it and document the choice here.
    """
    emg, restim, rerep = load_data(path)
    filter_kwargs = DEFAULT_FILTER_KWARGS if filter_kwargs is None else filter_kwargs
    emg = preprocess_signal(emg, fs, **filter_kwargs)          # filter once, whole signal
    return segment_emg(emg, restim, rerep, include_rest=include_rest)


def process_segment(segment: GestureSegment,
                    fs: float,
                    filter_kwargs: Optional[dict] = None,
                    window_ms: float = WINDOW_SIZE_MS,
                    step_ms: float = STEP_SIZE_MS,
                    cca_kwargs: Optional[dict] = None,
                    apply_filter: bool = True) -> np.ndarray:
    """(optionally filter) -> window gesture -> CCA

    apply_filter : if False, the segment is assumed to be ALREADY filtered
        (Option B: the whole recording was filtered before segmenting), so this
        only windows and computes CCA. Set False when called from process_file /
        load_and_segment; leave True to filter a single raw segment on its own.
    """
    cca_kwargs = cca_kwargs or {}
    emg = segment.emg
    if apply_filter:
        filter_kwargs = DEFAULT_FILTER_KWARGS if filter_kwargs is None else filter_kwargs
        emg = preprocess_signal(emg, fs, **filter_kwargs)
    windows = window_signal(emg, fs, window_ms, step_ms)
    return cca_features(windows, **cca_kwargs)

def process_file(path: str,
                 fs: float = FS,
                 include_rest: bool = False,
                 filter_kwargs: Optional[dict] = None,
                 window_ms: float = WINDOW_SIZE_MS,
                 step_ms: float = STEP_SIZE_MS,
                 cca_kwargs: Optional[dict] = None):
    """Full pipeline for one DB5 file.
 
    Returns
    -------
    X    : (n_total_windows, n_channels) CCA features
    y    : (n_total_windows,) gesture labels
    reps : (n_total_windows,) repetition ids (for leakage-safe splitting)
    """
    segments = load_and_segment(path, fs, include_rest, filter_kwargs)
 
    X_list, y_list, rep_list = [], [], []
    n_ch = None
    for seg in segments:
        n_ch = seg.emg.shape[1]
        # apply_filter=False: load_and_segment already filtered the whole recording
        feats = process_segment(seg, fs, None, window_ms, step_ms, cca_kwargs,
                                apply_filter=False)
        if feats.shape[0] == 0:
            continue
        X_list.append(feats)
        y_list.append(np.full(feats.shape[0], seg.label, dtype=int))
        rep_list.append(np.full(feats.shape[0], seg.repetition, dtype=int))
 
    if not X_list:
        n_ch = n_ch if n_ch is not None else len(PHYSICAL_CHANNEL_ORDER)
        return np.empty((0, n_ch)), np.empty((0,), dtype=int), np.empty((0,), dtype=int)
 
    return (np.concatenate(X_list, axis=0),
            np.concatenate(y_list, axis=0),
            np.concatenate(rep_list, axis=0))

def process_files(paths, **kwargs):
    """Concatenate process_file over several files (e.g. a subject's exercises)."""
    Xs, ys, reps = [], [], []
    for p in paths:
        X, y, r = process_file(p, **kwargs)
        if X.shape[0]:
            Xs.append(X); ys.append(y); reps.append(r)
    if not Xs:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(reps)

#----------------------self-test (for development purposes only, not using DB5)---------------------- #

def _make_synthetic_mat(path, fs=FS, channels=16, seed=0):
    """Write a .mat with the same variables/shapes as a real DB5 file."""
    from scipy.io import savemat
    rng = np.random.default_rng(seed)
    # rest / gesture-1 / rest / gesture-2 / rest / gesture-1 (2nd repetition)
    blocks = [(0, 400), (1, 1000), (0, 400), (2, 1000), (0, 400), (1, 1000)]
    restim, rerep, counts = [], [], {}
    for lab, length in blocks:
        restim += [lab] * length
        if lab == 0:
            rerep += [0] * length
        else:
            counts[lab] = counts.get(lab, 0) + 1
            rerep += [counts[lab]] * length
    restim = np.array(restim, dtype=int)
    rerep = np.array(rerep, dtype=int)
    n = len(restim)
    t = np.arange(n) / fs
 
    emg = rng.normal(0.0, 0.05, (n, channels))              # baseline noise
    active = restim != 0
    emg[active] += rng.normal(0.0, 0.3, (int(active.sum()), channels))  # burst
    lowf = np.sin(2 * np.pi * 0.5 * t)[:, None]             # 0.5 Hz activation
    emg += active[:, None] * (0.2 * restim[:, None]) * lowf
 
    savemat(path, {"emg": emg,
                   "restimulus": restim.reshape(-1, 1),
                   "rerepetition": rerep.reshape(-1, 1)})
    return path
 
 
def _self_test():
    import tempfile
    print("=" * 60)
    print("Preprocessing.py self-test")
    print("=" * 60)
 
    # --- spectral-orientation check: power spectrum peaks at the right bin ---
    fs, N, f0 = 200, 200, 20  # 20 Hz sine, N=200 -> expect bin k=20
    sine = np.sin(2 * np.pi * f0 * np.arange(N) / fs)[:, None]
    power = np.abs(np.fft.fft(sine, axis=0)) ** 2
    peak = int(np.argmax(power[1:N // 2, 0])) + 1
    assert peak == f0, f"FFT orientation wrong: peak bin {peak} != {f0}"
    print(f"[ok] power spectrum of a {f0} Hz sine peaks at bin {peak}")
 
    # --- full pipeline on a synthetic DB5-shaped file ---
    tmp = os.path.join(tempfile.gettempdir(), "synthetic_db5.mat")
    _make_synthetic_mat(tmp, fs=FS, channels=16)
 
    emg, restim, rerep = load_data(tmp)
    print(f"[ok] loaded emg {emg.shape}, restimulus {restim.shape}")
 
    segs = segment_emg(emg, restim, rerep)
    print(f"[ok] detected {len(segs)} gesture segments: "
          f"{[(s.label, s.repetition, s.emg.shape[0]) for s in segs]}")
 
    # window count sanity for one 1000-sample segment at 200 Hz
    win = int(round(WINDOW_SIZE_MS * FS / 1000))
    step = int(round(STEP_SIZE_MS * FS / 1000))
    w = window_signal(segs[0].emg, FS)
    exp = (1000 - win) // step + 1
    assert w.shape == (exp, win, 16), w.shape
    print(f"[ok] windowing: {w.shape} (expected ({exp}, {win}, 16))")
 
    X, y, reps = process_file(tmp, fs=FS)
    print(f"[ok] features X {X.shape}, y {y.shape}, reps unique {np.unique(reps)}")
    assert X.ndim == 2 and X.shape[1] == 16, X.shape
    assert X.shape[0] == y.shape[0] == reps.shape[0]
    assert not np.isnan(X).any() and np.isfinite(X).all()
    print(f"[ok] labels present: {np.unique(y)}  (rest excluded)")
 
    # e=8 swap check (mimics MindRove channel count, arbitrary fs)
    X8, y8, _ = process_file(_make_synthetic_mat(
        os.path.join(tempfile.gettempdir(), "syn8.mat"), fs=500, channels=8),
        fs=500)
    assert X8.shape[1] == 8, X8.shape
    print(f"[ok] channel-count swap: e=8 at 500 Hz -> features {X8.shape}")
 
    print("=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
 
 
if __name__ == "__main__":
    X, y, reps = process_file(filename_for(1, 2))
    print(X.shape, y.shape, reps.shape)
    print(np.unique(y))