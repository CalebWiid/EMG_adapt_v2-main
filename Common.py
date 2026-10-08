"""
Common.py - shared helpers for the MindRove pipeline (Ninapro-matched version).

Imported by Enroll.py, Live_predict.py, Offline_eval.py and Check_embedding.py.
Never run directly.

What makes the live data match the DB5 training data:
  1. RESAMPLE 500 Hz -> 200 Hz before anything else. The CCA feature works out to
     2*log(sum of window)/N, so the number of samples per window (N) changes the
     feature scale. At 200 Hz a 200 ms window has N = 40, exactly as in training.
  2. FILTER WHOLE RECORDINGS before windowing (your Option B), not short snippets.
  3. AMPLITUDE ALIGNMENT: MindRove (uV) and Myo (raw units) differ in gain. In the
     log domain a gain difference is a constant added to every feature, so one
     per-user offset (computed at enrollment) moves the user's features onto the
     DB5 scale. The DB5 reference comes from Check_embedding.py.
  4. CLASSIFY BY MAJORITY VOTE over windows, as in the offline evaluation.
  5. ONE PREDICTION = ONE 3 s CAPTURE. Live, the capture starts when "p" is
     pressed; offline, it starts at the HOLD cue. Both are cut and filtered by
     the same function (capture_features), so they are processed identically.
"""
import os
import sys
import glob
from math import gcd
from collections import Counter

import numpy as np
import torch
from scipy.signal import resample_poly

from Preprocessing import (preprocess_signal, window_signal, cca_features,
                           DEFAULT_FILTER_KWARGS, WINDOW_SIZE_MS, STEP_SIZE_MS,
                           FS as MODEL_FS)
from CNN_model import EMGAdapt

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEVICE_FS = 500                    # MindRove native sampling rate
N_CHANNELS = 8
EMBED_DIM = 128
REST_LABEL = "rest"
REFERENCE_PATH = "db5_reference.npz"   # written by Check_embedding.py
RECORDINGS_DIR = "recordings"          # raw enrollment sessions are saved here

CAPTURE_S = 3.0      # one prediction = this much signal (live: after pressing "p")
PREROLL_S = 2.0      # signal from just before the capture, filtered together with it


# --------------------------------------------------------------------------- #
# MindRove streaming (imported lazily so offline scripts run without the SDK)
# --------------------------------------------------------------------------- #
def _board_shim():
    from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams
    return BoardShim, BoardIds.MINDROVE_WIFI_BOARD, MindRoveInputParams


def connect_board():
    """Open a session and start streaming. Returns the board object."""
    BoardShim, board_id, Params = _board_shim()
    BoardShim.disable_board_logger()
    board = BoardShim(board_id, Params())
    board.prepare_session()
    board.start_stream()
    return board


def release_board(board):
    try:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
    except Exception:
        pass


def flush(board):
    """Discard everything currently buffered so the next grab starts fresh."""
    board.get_board_data()


def grab_all(board):
    """All samples received since the last grab/flush -> (n_samples, 8) at 500 Hz."""
    BoardShim, board_id, _ = _board_shim()
    data = board.get_board_data()                      # (rows, n), clears buffer
    if data.shape[1] == 0:
        return np.empty((0, N_CHANNELS))
    return data[BoardShim.get_emg_channels(board_id), :].T


# --------------------------------------------------------------------------- #
# Signal -> features (identical chain to training, at 200 Hz)
# --------------------------------------------------------------------------- #
def to_model_fs(emg, fs_in=DEVICE_FS, fs_out=MODEL_FS):
    """Resample (n, ch) EMG from fs_in to fs_out (polyphase, anti-aliased)."""
    emg = np.asarray(emg, dtype=np.float64)
    if fs_in == fs_out:
        return emg
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(emg, int(fs_out) // g, int(fs_in) // g, axis=0)


def filter_recording(emg, fs_in=DEVICE_FS, resample=True):
    """Raw (n, 8) recording -> filtered envelope, filtered as ONE piece (Option B).
    Returns (filtered, fs) where fs is the rate of the returned signal."""
    if resample:
        x, fs = to_model_fs(emg, fs_in), MODEL_FS
    else:
        x, fs = np.asarray(emg, dtype=np.float64), fs_in
    return preprocess_signal(x, fs, **DEFAULT_FILTER_KWARGS), fs


def convert_index(idx, fs_from, fs_to):
    """Sample index at fs_from -> the matching index at fs_to."""
    return int(round(idx * fs_to / fs_from))


def interval_features(filtered, fs, start, end):
    """CCA features of every window lying fully inside filtered[start:end]."""
    seg = filtered[max(0, start):max(0, end)]
    windows = window_signal(seg, fs, WINDOW_SIZE_MS, STEP_SIZE_MS)
    return cca_features(windows)


def trimmed(start, end, trim, fs):
    """Shrink an interval [start, end) by `trim` seconds at each end."""
    t = int(round(trim * fs))
    return start + t, end - t


# --------------------------------------------------------------------------- #
# 3 s captures (live prediction, and the identical offline test)
# --------------------------------------------------------------------------- #
def cut_capture(emg, start, end, fs, preroll_s=PREROLL_S):
    """From a longer raw recording: (pre-roll + capture, capture length).
    The capture is emg[start:end]; up to preroll_s of the signal before it is
    included so the filter has settled by the time the capture starts.
    Nothing after `end` is included - live, that signal doesn't exist yet."""
    p0 = max(0, start - int(round(preroll_s * fs)))
    return emg[p0:end], end - start


def capture_features(raw, capture_n, fs_in=DEVICE_FS, resample=True):
    """CCA features of one capture, processed identically live and offline.

    raw = pre-roll followed by the capture (raw device samples); the capture is
    its last capture_n samples. The whole piece is filtered as one, then only the
    windows lying fully inside the capture are kept."""
    filt, fs = filter_recording(raw, fs_in, resample=resample)
    start = convert_index(len(raw) - capture_n, fs_in, fs)
    return interval_features(filt, fs, start, filt.shape[0])


def classify_features(model, feats, offset, labels, protos, device=None):
    """Offset -> embed -> nearest prototype per window -> majority vote.
    Returns (label, vote fraction, per-window prototype indices)."""
    idx = nearest(embed_features(model, feats + offset, device), protos)
    label, frac = vote(idx, labels)
    return label, frac, idx


# --------------------------------------------------------------------------- #
# Amplitude alignment to DB5
# --------------------------------------------------------------------------- #
def load_reference(path=REFERENCE_PATH):
    """Median DB5 CCA feature value (gesture windows, 8 channels)."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run Check_embedding.py once first "
                         f"(it measures the DB5 feature scale).")
    return float(np.load(path)["median"])


def compute_offset(gesture_feats, reference_median):
    """Constant to add to the user's features so their gesture median matches DB5."""
    return float(reference_median - np.median(np.asarray(gesture_feats)))


# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #
def load_embedding(path, e=N_CHANNELS, device=None):
    """Load the features + embedding weights. Fails LOUDLY if they don't load
    (the old strict=False version could silently run with random weights)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = EMGAdapt(e=e, n_classes=2)                 # classifier head unused
    state = torch.load(path, map_location=device)
    if isinstance(state, torch.nn.Module):             # a whole model was saved
        state = state.state_dict()
    sub = {k: v for k, v in state.items()
           if k.startswith("features.") or k.startswith("embedding.")}
    if not sub:
        raise RuntimeError(f"{path}: no 'features.'/'embedding.' weights found. "
                           f"Keys look like: {list(state.keys())[:5]}")
    res = model.load_state_dict(sub, strict=False)
    missing = [k for k in res.missing_keys if not k.startswith("classifier.")]
    if missing:
        raise RuntimeError(f"{path}: embedding weights missing: {missing}")
    return model.to(device).eval()


@torch.no_grad()
def embed_features(model, feats, device=None, batch=2048):
    """(n_windows, 8) CCA features -> (n_windows, 128) embeddings (cpu tensor)."""
    feats = np.asarray(feats)
    if len(feats) == 0:
        return torch.empty((0, EMBED_DIM))
    device = device or next(model.parameters()).device
    x = torch.as_tensor(feats, dtype=torch.float32)
    return torch.cat([model.embed(x[i:i + batch].to(device)).cpu()
                      for i in range(0, len(x), batch)])


# --------------------------------------------------------------------------- #
# Prototypes + classification
# --------------------------------------------------------------------------- #
def build_prototypes(embs_by_label):
    """{label: (n,128) tensor} -> (labels list, (n_labels,128) tensor)."""
    labels = [l for l, e in embs_by_label.items() if len(e)]
    protos = torch.stack([embs_by_label[l].mean(0) for l in labels])
    return labels, protos


def make_prototypes(model, gestures, rests, reps, reference_median=None,
                    include_rest=True, device=None):
    """Prototypes (one per gesture, plus rest) from the given reps of a session.

    gestures/rests come from session_features(). If reference_median is given,
    the amplitude offset is computed from these reps' gesture windows and applied
    to everything; otherwise offset = 0. Returns (labels, protos, offset)."""
    def pick(by_rep):
        parts = [by_rep[r] for r in reps if r in by_rep and len(by_rep[r])]
        return np.concatenate(parts) if parts else np.empty((0, N_CHANNELS))

    gest = {g: pick(by_rep) for g, by_rep in gestures.items()}
    all_gest = np.concatenate([f for f in gest.values() if len(f)])
    offset = 0.0 if reference_median is None else compute_offset(all_gest, reference_median)

    embs = {g: embed_features(model, f + offset, device) for g, f in gest.items()}
    if include_rest:
        embs[REST_LABEL] = embed_features(model, pick(rests) + offset, device)
    labels, protos = build_prototypes(embs)
    return labels, protos, offset


def save_prototypes(path, labels, protos, offset, session_dir=""):
    np.savez(path, labels=np.array(labels), protos=np.asarray(protos),
             offset=np.float64(offset), session_dir=np.array(session_dir))


def load_prototypes(path):
    d = np.load(path, allow_pickle=True)
    if "offset" not in d.files:
        raise SystemExit(f"{path} was made by the old enrollment - run Enroll.py again.")
    labels = [str(l) for l in d["labels"]]
    return labels, torch.as_tensor(d["protos"], dtype=torch.float32), float(d["offset"])


def nearest(embs, protos):
    """Index of the nearest prototype for every window -> (n_windows,) numpy."""
    if len(embs) == 0:
        return np.empty((0,), dtype=int)
    return torch.cdist(embs, protos).argmin(1).numpy()


def vote(indices, labels):
    """Majority vote over window predictions -> (label, fraction of votes)."""
    if len(indices) == 0:
        return None, 0.0
    idx, count = Counter(indices.tolist()).most_common(1)[0]
    return labels[idx], count / len(indices)


# --------------------------------------------------------------------------- #
# Raw recordings (save everything so it can be re-analysed offline)
# --------------------------------------------------------------------------- #
def save_recording(path, emg, fs, gesture, holds, rests):
    """holds and rests: lists of (rep, start, end) sample indices at `fs`.
    A rest belongs to the rep of the hold that follows it (the final rest to the
    last rep), so enroll/test splits by rep keep rest data separate too."""
    np.savez(path, emg=np.asarray(emg), fs=fs, gesture=np.array(gesture),
             holds=np.asarray(holds, dtype=np.int64).reshape(-1, 3),
             rests=np.asarray(rests, dtype=np.int64).reshape(-1, 3))


def load_recording(path):
    d = np.load(path, allow_pickle=True)
    return dict(emg=d["emg"], fs=int(d["fs"]), gesture=str(d["gesture"]),
                holds=[tuple(int(v) for v in h) for h in d["holds"]],
                rests=[tuple(int(v) for v in r) for r in d["rests"]])


def latest_session(root=RECORDINGS_DIR):
    sessions = sorted(glob.glob(os.path.join(root, "*")))
    if not sessions:
        raise SystemExit(f"No sessions in {root}/ - run Enroll.py first.")
    return sessions[-1]


def session_features(session_dir, gesture_names, hold_trim, rest_trim,
                     resample=True):
    """Load every recording in a session and cut it into labelled feature sets.

    Returns {gesture: {rep: (n,8) feats}} and {rep: (n,8) rest feats}.
    Features are NOT offset-aligned yet."""
    gestures, rests = {}, {}
    for name in gesture_names:
        path = os.path.join(session_dir, f"{name}.npz")
        if not os.path.exists(path):
            print(f"  [skip] {path} missing")
            continue
        rec = load_recording(path)
        filt, fs = filter_recording(rec["emg"], rec["fs"], resample=resample)
        gestures[name] = {}
        for rep, s, e in rec["holds"]:
            s, e = trimmed(convert_index(s, rec["fs"], fs),
                           convert_index(e, rec["fs"], fs), hold_trim, fs)
            gestures[name][rep] = interval_features(filt, fs, s, e)
        for rep, s, e in rec["rests"]:
            s, e = trimmed(convert_index(s, rec["fs"], fs),
                           convert_index(e, rec["fs"], fs), rest_trim, fs)
            rests.setdefault(rep, []).append(interval_features(filt, fs, s, e))
    rests = {r: np.concatenate(v) for r, v in rests.items()}
    return gestures, rests


# --------------------------------------------------------------------------- #
# Keyboard: block until the spacebar (raw single-key, Linux)
# --------------------------------------------------------------------------- #
def read_key(allowed):
    """Block until one of the characters in `allowed` is pressed; return it.
    Raises KeyboardInterrupt on Ctrl+C."""
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == '\x03':
                raise KeyboardInterrupt
            if ch.lower() in allowed:
                return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def wait_for_space():
    read_key(" ")