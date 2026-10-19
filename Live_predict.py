"""
Live_predict.py - press-to-predict gesture recognition, plus a cued live test.

Three modes:
  python Live_predict.py                       free use: press p, perform, get a command
  python Live_predict.py --test                cued live test (measures accuracy)
  python Live_predict.py --replay <folder>     re-score a saved test, no armband

PREDICTION (same in every mode):
  * The armband streams the whole time, but nothing is classified until "p".
  * Press "p", then perform the gesture. The next CAPTURE_S (5 s) of signal is
    collected, counted in samples from the moment of the key press.
  * The capture is processed exactly like Offline_eval.py's test captures
    (Common.capture_features): PREROLL_S of signal from before the key press is
    filtered together with it, every 200 ms window inside the capture is classified
    by nearest prototype, and the windows vote.
  * The winning class is the prediction. "rest" means no gesture: no command.

CUED TEST (--test). Trials in random order: every enrolled gesture
TRIALS_PER_GESTURE times, plus REST_TRIALS "stay relaxed" trials. For each trial
the screen shows what to do; press "p" when ready, then perform it (or stay
relaxed) for the whole capture. "q" stops early; the completed trials are still scored.
A trial is correct when the prediction equals the cue, so the results are
directly comparable with Offline_eval.py version C, "with rest class".
Saved to live_tests/<date_time>/: captures.npz (for replay), trials.csv, summary.txt.

CALIBRATION (runs first in free use and --test). With CALIBRATE = "ask" the
script asks at start-up: [c] calibrate or [s] skip. Command-line flags decide
without asking:  --cal  (always calibrate)   --skip-cal  (never calibrate)
  python Live_predict.py --test --cal
  python Live_predict.py --test --skip-cal
After the band has been put back on, the user performs N_CAL_GESTURES enrolled gestures
(CAL_REPS timed holds each, cued like Enroll.py). Their channel patterns are
compared with the same gestures in the enrollment recording, and two numbers
are estimated:
  * rotation: how far the band has turned around the forearm, in channels
    (a fraction of the electrode spacing, within +/- MAX_ROTATION). Every capture is turned back by this
    amount (Common.rotate_channels) before features are computed.
  * level: the change in overall signal level since enrollment (skin contact),
    added to the amplitude offset (CAL_LEVEL = True).
The prototypes are NOT changed: calibration only aligns new data to them.
With --test the calibration recording and result are saved in calibration.npz.

REPLAY (--replay) re-classifies the saved captures with the CURRENT prototypes
and the calibration saved with the test, and saves the result in a
replay_<date_time> subfolder. Add --no-cal to score the same captures without
calibration (rotation 0, enrollment offset): the difference is what the
calibration contributed.
  python Live_predict.py --replay <folder> --no-cal
"""
import os
import sys
import csv
import time
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import Common as lc
from Enroll import (record_protocol, GESTURE_NAMES, ENROLL_REPS, HOLD_TRIM_S)

MODEL_PATH = "deployment_embedding.pt"
PROTO_PATH = "prototypes.npz"

START_KEY = "p"          # press to start a capture
QUIT_KEY = "q"
POLL_S = 0.05            # how often the capture loop collects data (s)
STALL_S = 2.0            # stop if no data arrives for this long during a capture (s)

# map gestures to drone commands here when you get to the drone
COMMANDS = {}            # e.g. {"gesture_1": "takeoff", "gesture_2": "land"}

# --- cued test (--test) ---
TRIALS_PER_GESTURE = 5
REST_TRIALS = 10
SEED = None              # set an int to reuse the same trial order
SHOW_FEEDBACK = False    # show the prediction after every trial (off for report tests)
OUT_DIR = "live_tests"

# --- calibration after putting the band back on ---
CALIBRATE = "ask"        # "ask" = choose at start-up; True = always; False = never
                         # (--cal / --skip-cal on the command line override this)
CAL_GESTURES = None      # e.g. ["gesture_3", "gesture_7"]; None = pick automatically
N_CAL_GESTURES = 4       # how many gestures when picking automatically
CAL_REPS = 1             # timed holds per calibration gesture (5 s each)
MAX_ROTATION = 0.5       # largest rotation searched, in electrode spacings (the band moves less than one)
ROT_STEP = 0.05          # search resolution, in electrode spacings
CAL_LEVEL = True         # also correct the overall signal level (offset)


class Predictor:
    """Embedding model + prototypes: one capture in, one prediction out."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lc.load_embedding(MODEL_PATH, device=self.device)
        self.labels, self.protos, self.offset = lc.load_prototypes(PROTO_PATH)
        self.offset_enroll = self.offset       # as stored by Enroll.py
        self.rotation = 0.0                    # set by calibration
        self.calibrated = False

    def set_calibration(self, rotation, level):
        """Apply a calibration result: turn captures back by `rotation` channels
        and shift the offset by `level`."""
        self.rotation = float(rotation)
        self.offset = self.offset_enroll + float(level)
        self.calibrated = True

    def predict(self, raw, capture_n):
        """raw = pre-roll + capture (500 Hz); the capture is its last capture_n
        samples. Returns (label, vote fraction, number of windows)."""
        feats = lc.capture_features(raw, capture_n, rotation=self.rotation)
        label, frac, idx = lc.classify_features(self.model, feats, self.offset,
                                                self.labels, self.protos, self.device)
        return label, frac, len(idx)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def enrollment_session():
    """The recording folder the stored prototypes were built from."""
    d = np.load(PROTO_PATH, allow_pickle=True)
    session = str(d["session_dir"]) if "session_dir" in d.files else ""
    if not session or not os.path.isdir(session):
        raise SystemExit(f"Calibration needs the enrollment recordings, but "
                         f"'{session}' (named in {PROTO_PATH}) was not found. "
                         f"Set CALIBRATE = False to run without calibration.")
    return session


def pick_cal_gestures(ref, n):
    """The n enrolled gestures whose channel patterns vary most around the ring
    (the easiest to locate after a rotation)."""
    if CAL_GESTURES:
        missing = [g for g in CAL_GESTURES if g not in ref]
        if missing:
            raise SystemExit(f"CAL_GESTURES not in the enrollment: {missing}")
        return list(CAL_GESTURES)
    spread = {g: float(np.std(v - v.mean())) for g, v in ref.items()}
    return sorted(spread, key=spread.get, reverse=True)[:n]


def want_calibration(argv):
    """Calibrate this run? --cal / --skip-cal decide; otherwise CALIBRATE, which
    with "ask" prompts once at start-up."""
    if "--cal" in argv:
        return True
    if "--skip-cal" in argv:
        return False
    if CALIBRATE == "ask":
        print("\nCalibrate the band before starting?   [c] calibrate   [s] skip")
        choice = lc.read_key("cs")
        print("  -> calibrating" if choice == "c" else "  -> skipping calibration "
              "(enrollment used exactly as stored)")
        return choice == "c"
    return bool(CALIBRATE)


def calibrate(board, pred):
    """Record the calibration gestures, estimate rotation + level, apply them to
    `pred`. Returns a dict with the result and the raw recordings."""
    session = enrollment_session()
    ref = lc.enrollment_reference(session, GESTURE_NAMES, ENROLL_REPS, HOLD_TRIM_S)
    gestures = pick_cal_gestures(ref, N_CAL_GESTURES)
    print(f"\nCALIBRATION - {len(gestures)} gestures, {CAL_REPS} x 5 s hold each: "
          f"{', '.join(gestures)}")
    print("Perform each gesture exactly as you did during enrollment.")

    recordings = {}
    for name in gestures:
        while True:
            print(f"\n[{name}]  press SPACE to start")
            lc.wait_for_space()
            emg, holds, _ = record_protocol(board, name, n_reps=CAL_REPS)
            expected = int((CAL_REPS * 8 + 3) * lc.DEVICE_FS)
            print(f"  received {len(emg)} samples (expected ~{expected})")
            print("  [SPACE] keep    [r] redo")
            if lc.read_key(" r") == " ":
                break
        recordings[name] = (emg, [(s, e) for _, s, e in holds])

    rotation, level, costs = lc.estimate_rotation(recordings, ref, HOLD_TRIM_S,
                                                  MAX_ROTATION, ROT_STEP)
    if not CAL_LEVEL:
        level = 0.0
    pred.set_calibration(rotation, level)

    print(f"\ncalibration: rotation {rotation:+.2f} of an electrode spacing ({rotation * 45:+.0f} deg), "
          f"level {level:+.4f}  -> offset {pred.offset:+.4f} "
          f"(enrollment {pred.offset_enroll:+.4f})")
    if abs(rotation) >= MAX_ROTATION - 1e-9:
        print(f"  WARNING: rotation is at the edge of the search range "
              f"(+/-{MAX_ROTATION}). Re-seat the band on the mark and calibrate again.")
    c0, cb = costs.get(0.0), min(costs.values())
    if c0 is not None and c0 > 0:
        print(f"  pattern mismatch: {c0:.4f} without rotation -> {cb:.4f} after "
              f"({100 * (1 - cb / c0):.0f}% smaller)")
    return dict(rotation=rotation, level=level, costs=costs, gestures=gestures,
                recordings=recordings, session=session)


def save_calibration(folder, cal, pred):
    rec = {}
    for g, (emg, holds) in cal["recordings"].items():
        rec[f"emg_{g}"] = np.asarray(emg)
        rec[f"holds_{g}"] = np.asarray(holds, dtype=np.int64).reshape(-1, 2)
    np.savez(os.path.join(folder, "calibration.npz"),
             rotation=cal["rotation"], level=cal["level"],
             offset_enroll=pred.offset_enroll, offset_used=pred.offset,
             gestures=np.array(cal["gestures"]), enrollment_session=cal["session"],
             cost_rotations=np.array(list(cal["costs"].keys())),
             cost_values=np.array(list(cal["costs"].values())), fs=lc.DEVICE_FS, **rec)


class CaptureStream:
    """Keeps the newest PREROLL_S of signal and cuts out a capture on request.

    Between captures the MindRove SDK keeps buffering, so at a key press
    everything since the previous capture is fetched and its tail is the
    pre-roll. The signal is continuous: the samples that arrive after a capture
    ends become the start of the next pre-roll."""

    def __init__(self, board):
        self.board = board
        self.preroll_n = int(round(lc.PREROLL_S * lc.DEVICE_FS))
        self.capture_n = int(round(lc.CAPTURE_S * lc.DEVICE_FS))
        self.history = np.empty((0, lc.N_CHANNELS))

    def _keep_tail(self, new):
        self.history = np.concatenate([self.history, new])[-self.preroll_n:]

    def warm_up(self):
        """Fill the first pre-roll and check that data is arriving."""
        lc.flush(self.board)
        time.sleep(lc.PREROLL_S + 0.5)
        new = lc.grab_all(self.board)
        expected = int((lc.PREROLL_S + 0.5) * lc.DEVICE_FS)
        print(f"stream check: {len(new)} samples (expected ~{expected})")
        if len(new) < 0.5 * expected:
            raise SystemExit("Too little data arriving from the armband - check the "
                             "WiFi connection and that the band is streaming.")
        self._keep_tail(new)

    def capture(self, text):
        """Call right after the key press. Returns raw pre-roll + capture (500 Hz)."""
        self._keep_tail(lc.grab_all(self.board))       # everything up to the key press
        pre = self.history
        chunks, n, last_data = [], 0, time.time()
        while n < self.capture_n:
            time.sleep(POLL_S)
            c = lc.grab_all(self.board)
            if len(c):
                chunks.append(c)
                n += len(c)
                last_data = time.time()
            elif time.time() - last_data > STALL_S:
                raise RuntimeError("no data from the armband for "
                                   f"{STALL_S:.0f} s - check the connection")
            left = max(0, self.capture_n - n) / lc.DEVICE_FS
            print(f"\r  {text:<44} {left:4.1f} s ", end="", flush=True)
        print()
        cap = np.concatenate(chunks)
        self._keep_tail(cap)              # samples after the capture start the next pre-roll
        return np.concatenate([pre, cap[:self.capture_n]])


def on_command(label, frac):
    """Called once per predicted gesture. Send the drone command from here."""
    command = COMMANDS.get(label, label)
    print(f"  >>> {command}   (votes {frac:.0%})")


def cue_text(cue):
    return "STAY RELAXED" if cue == lc.REST_LABEL else f"PERFORM {cue}"


# --------------------------------------------------------------------------- #
# Mode 1: free use
# --------------------------------------------------------------------------- #
def run_free(do_cal=None):
    if not os.path.exists(PROTO_PATH):
        print(f"No gestures enrolled yet ({PROTO_PATH} not found). Run Enroll.py first.")
        return
    pred = Predictor()
    do_cal = want_calibration(sys.argv) if do_cal is None else do_cal
    print(f"loaded {len(pred.labels)} classes: {pred.labels}")
    print(f"amplitude offset {pred.offset:+.4f}")

    board = lc.connect_board()
    try:
        if do_cal:
            calibrate(board, pred)
        stream = CaptureStream(board)
        stream.warm_up()
        while True:
            print(f"\npress [{START_KEY}] and perform a gesture "
                  f"({lc.CAPTURE_S:.0f} s)    [{QUIT_KEY}] quit")
            if lc.read_key(START_KEY + QUIT_KEY) == QUIT_KEY:
                break
            raw = stream.capture(">>> perform now")
            label, frac, n_win = pred.predict(raw, stream.capture_n)
            if label is None or label == lc.REST_LABEL:
                print(f"  rest ({frac:.0%} of {n_win} windows) - no command")
            else:
                on_command(label, frac)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        lc.release_board(board)


# --------------------------------------------------------------------------- #
# Scoring (shared by --test and --replay)
# --------------------------------------------------------------------------- #
def score(cues, results):
    """cues: [cue label]; results: [(predicted label, vote fraction)].
    Returns (trial_rows, summary_dict)."""
    rows = [dict(trial=i, cue=cue, predicted=str(p), vote=round(float(f), 3),
                 correct=int(p == cue))
            for i, (cue, (p, f)) in enumerate(zip(cues, results), start=1)]
    g = [r for r in rows if r["cue"] != lc.REST_LABEL]
    rest = [r for r in rows if r["cue"] == lc.REST_LABEL]
    summary = dict(
        gesture_trials=len(g),
        gesture_accuracy=np.mean([r["correct"] for r in g]) if g else float("nan"),
        predicted_rest=sum(r["predicted"] == lc.REST_LABEL for r in g),
        wrong_gesture=sum(r["predicted"] not in (lc.REST_LABEL, r["cue"]) for r in g),
        rest_trials=len(rest),
        rest_accuracy=np.mean([r["correct"] for r in rest]) if rest else float("nan"),
        false_commands=sum(r["predicted"] != lc.REST_LABEL for r in rest),
        mean_vote=np.mean([r["vote"] for r in rows]) if rows else float("nan"),
    )
    return rows, summary


def format_summary(rows, s, labels):
    out = []
    out.append(f"gesture trials : {s['gesture_trials']}")
    out.append(f"  accuracy     : {s['gesture_accuracy']:.3f}   "
               f"(predicted rest {s['predicted_rest']}, wrong gesture {s['wrong_gesture']})")
    out.append(f"rest trials    : {s['rest_trials']}   accuracy {s['rest_accuracy']:.3f}   "
               f"(false commands {s['false_commands']})")
    out.append(f"mean vote fraction of the prediction: {s['mean_vote']:.2f}")

    # confusion: cued (rows) vs predicted (cols)
    classes = [l for l in labels if l != lc.REST_LABEL] + [lc.REST_LABEL]
    short = lambda l: l.replace("gesture_", "g")
    m = Counter((r["cue"], r["predicted"]) for r in rows)
    out.append("\nconfusion (rows = cued, cols = predicted):")
    out.append("        " + "".join(f"{short(c):>5}" for c in classes))
    for cue in classes:
        if not any(r["cue"] == cue for r in rows):
            continue
        out.append(f"  {short(cue):>5} " + "".join(
            f"{(m[(cue, c)] if m[(cue, c)] else '.'):>5}" for c in classes))
    return "\n".join(out)


def settings_text(pred):
    cal = (f"calibrated: rotation={pred.rotation:+.2f} offset={pred.offset:+.4f} "
           f"(enrollment {pred.offset_enroll:+.4f})" if pred.calibrated else
           f"not calibrated: offset={pred.offset:+.4f} (enrollment)")
    return (f"settings: capture_s={lc.CAPTURE_S} preroll_s={lc.PREROLL_S} "
            f"prototypes={PROTO_PATH}\n{cal}")


def report(folder, pred, cues, results):
    rows, summary = score(cues, results)
    text = settings_text(pred) + "\n\n" + format_summary(rows, summary, pred.labels)
    print("\n" + text)
    with open(os.path.join(folder, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(folder, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nsaved -> {folder}/")


# --------------------------------------------------------------------------- #
# Mode 2: cued live test (--test)
# --------------------------------------------------------------------------- #
def make_trials(labels, rng):
    gestures = [l for l in labels if l != lc.REST_LABEL]
    trials = gestures * TRIALS_PER_GESTURE + [lc.REST_LABEL] * REST_TRIALS
    rng.shuffle(trials)
    return trials


def save_captures(folder, segments, capture_ns, cues, results, pred):
    """Every capture (pre-roll + capture, raw 500 Hz) end to end, plus the
    calibration that was applied, so --replay can re-classify exactly the same
    signal with or without it."""
    np.savez(os.path.join(folder, "captures.npz"),
             emg=np.concatenate(segments),
             seg_ends=np.cumsum([len(s) for s in segments]),
             capture_n=np.array(capture_ns), cues=np.array(cues),
             live_pred=np.array([str(p) for p, _ in results]),
             live_vote=np.array([f for _, f in results]),
             calibrated=pred.calibrated, rotation=pred.rotation, offset_used=pred.offset,
             offset_enroll=pred.offset_enroll,
             fs=lc.DEVICE_FS, capture_s=lc.CAPTURE_S, preroll_s=lc.PREROLL_S)


def run_test(do_cal=None):
    if not os.path.exists(PROTO_PATH):
        print(f"No gestures enrolled yet ({PROTO_PATH} not found). Run Enroll.py first.")
        return
    pred = Predictor()
    do_cal = want_calibration(sys.argv) if do_cal is None else do_cal
    trials = make_trials(pred.labels, np.random.default_rng(SEED))
    print(f"{len(trials)} trials ({TRIALS_PER_GESTURE} per gesture + {REST_TRIALS} rest).")
    print(f"For each trial: read the cue, press [{START_KEY}], then perform it for "
          f"{lc.CAPTURE_S:.0f} s. Relax between trials. [{QUIT_KEY}] stops the test.")

    folder = os.path.join(OUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    segments, capture_ns, cues, results = [], [], [], []

    board = lc.connect_board()
    try:
        if do_cal:
            cal = calibrate(board, pred)
            save_calibration(folder, cal, pred)
            print("\ncalibration done - starting the test.")
        stream = CaptureStream(board)
        stream.warm_up()
        for i, cue in enumerate(trials, start=1):
            print(f"\n[{i}/{len(trials)}] next: {cue_text(cue)}   "
                  f"- press [{START_KEY}] when ready")
            if lc.read_key(START_KEY + QUIT_KEY) == QUIT_KEY:
                print("test stopped - scoring the completed trials.")
                break
            raw = stream.capture(f">>> {cue_text(cue)}")
            label, frac, _ = pred.predict(raw, stream.capture_n)
            segments.append(raw)
            capture_ns.append(stream.capture_n)
            cues.append(cue)
            results.append((label, frac))
            if SHOW_FEEDBACK:
                print(f"  predicted: {label}  ({frac:.0%})")
    except KeyboardInterrupt:
        print("\ntest interrupted - scoring the completed trials.")
    finally:
        lc.release_board(board)

    if not cues:
        print("no completed trials.")
        return
    save_captures(folder, segments, capture_ns, cues, results, pred)
    report(folder, pred, cues, results)


# --------------------------------------------------------------------------- #
# Mode 3: replay a saved test with the current prototypes (--replay)
# --------------------------------------------------------------------------- #
def replay(folder, use_cal=True):
    d = np.load(os.path.join(folder, "captures.npz"), allow_pickle=True)
    pred = Predictor()
    if use_cal and "rotation" in d.files and bool(d["calibrated"] if "calibrated" in d.files else True):
        if not np.isclose(float(d["offset_enroll"]), pred.offset_enroll):
            print("  note: the current prototypes are not the ones this test used "
                  "(different enrollment offset)")
        pred.set_calibration(float(d["rotation"]),
                             float(d["offset_used"]) - float(d["offset_enroll"]))
    emg, ends = d["emg"], d["seg_ends"]
    starts = np.concatenate([[0], ends[:-1]])
    cues = [str(c) for c in d["cues"]]
    results = [pred.predict(emg[s:e], int(n))[:2]
               for s, e, n in zip(starts, ends, d["capture_n"])]
    tag = "replay_" if use_cal else "replay_nocal_"
    out = os.path.join(folder, tag + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    print(f"replaying {folder} with the current prototypes, "
          f"{'with the saved calibration' if use_cal else 'WITHOUT calibration'}")
    report(out, pred, cues, results)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test()
    elif len(sys.argv) > 2 and sys.argv[1] == "--replay":
        replay(sys.argv[2], use_cal="--no-cal" not in sys.argv[3:])
    elif len(sys.argv) > 1:
        print(__doc__.split("PREDICTION")[0])
    else:
        run_free()