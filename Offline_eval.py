"""
Offline_eval.py - accuracy of the system on YOUR saved MindRove recordings,
scored the same way as the live 3 s prediction (no armband needed).

PROTOTYPES are built from ENROLL_REPS of one Enroll.py session, from the full
trimmed holds - exactly as Enroll.py builds them.

TEST: every hold of TEST_REPS becomes one CAPTURE of CAPTURE_S (3 s) starting at
the HOLD cue - the offline equivalent of pressing "p" and then performing the
gesture. Each capture is cut and filtered exactly like a live capture
(Common.cut_capture + Common.capture_features): PREROLL_S of signal before it,
nothing after it. One capture = one prediction (majority vote over its windows).

REST: every rest period of TEST_REPS gives one rest capture. Rest periods are
only 3 s and begin at a release, so a rest capture starts REST_TRIM_S into the
rest and is shorter than 3 s (2.25 s).

If the session only has 3 reps, it falls back to leave-one-rep-out.

Three versions of the feature pipeline are run side by side:
  A  old pipeline      - 500 Hz features, no amplitude alignment
  B  resampled         - 200 Hz features (N = 40 per window, as in training)
  C  resampled+aligned - B plus the per-user amplitude offset (what Enroll uses)

Columns:
  12 gestures      12-way (no rest prototype): window accuracy, capture accuracy
  with rest class  13-way, as used live: gesture captures correct, rest captures correct
Version C "with rest class" is what Live_predict.py --test measures live.

Run:  python Offline_eval.py                 (latest session)
      python Offline_eval.py recordings/20261005_131500
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import Common as lc
from Enroll import (MODEL_PATH, GESTURE_NAMES, ENROLL_REPS,
                    HOLD_TRIM_S, REST_TRIM_S)

TEST_REPS = (2, 5, 6)
assert not set(ENROLL_REPS) & set(TEST_REPS), "enroll and test reps overlap"


def session_captures(session_dir, gesture_names, resample=True):
    """Live-style 3 s test captures of every rep of a session.

    Returns {gesture: {rep: (n,8) feats}} (one capture per hold, starting at the
    HOLD cue) and {rep: [(n,8) feats, ...]} (one capture per rest period).
    Features are NOT offset-aligned yet."""
    caps, rest_caps = {}, {}
    for name in gesture_names:
        path = os.path.join(session_dir, f"{name}.npz")
        if not os.path.exists(path):
            continue
        rec = lc.load_recording(path)
        emg, fs = rec["emg"], rec["fs"]
        cap_n = int(round(lc.CAPTURE_S * fs))
        caps[name] = {}
        for rep, s, e in rec["holds"]:
            raw, n = lc.cut_capture(emg, s, min(s + cap_n, e), fs)
            caps[name][rep] = lc.capture_features(raw, n, fs_in=fs, resample=resample)
        for rep, s, e in rec["rests"]:
            s += int(round(REST_TRIM_S * fs))
            if s >= e:
                continue
            raw, n = lc.cut_capture(emg, s, min(s + cap_n, e), fs)
            rest_caps.setdefault(rep, []).append(
                lc.capture_features(raw, n, fs_in=fs, resample=resample))
    return caps, rest_caps


def evaluate(model, gestures, rests, caps, rest_caps, enroll, test, reference,
             include_rest, device):
    """Prototypes from `enroll` (full trimmed holds), tested on the captures of
    `test`. Returns (rows, offset); rows = (true, prediction, n_correct_windows, n_windows)."""
    labels, protos, offset = lc.make_prototypes(model, gestures, rests, enroll,
                                                reference, include_rest, device)
    rows = []

    def score(feats, true):
        if len(feats) == 0:
            return
        pred, _, idx = lc.classify_features(model, feats, offset, labels, protos, device)
        n_ok = int(sum(labels[i] == true for i in idx))
        rows.append((true, pred, n_ok, len(idx)))

    for g, by_rep in caps.items():
        for r in test:
            if r in by_rep:
                score(by_rep[r], g)
    if include_rest:
        for r in test:
            for feats in rest_caps.get(r, []):
                score(feats, lc.REST_LABEL)
    return rows, offset


def summarise(rows):
    """(window accuracy, capture accuracy) of a set of rows."""
    if not rows:
        return float("nan"), float("nan")
    win = sum(r[2] for r in rows) / max(1, sum(r[3] for r in rows))
    acc = float(np.mean([r[0] == r[1] for r in rows]))
    return win, acc


def folds_for(gestures):
    reps = sorted({r for by_rep in gestures.values() for r in by_rep})
    if set(ENROLL_REPS) | set(TEST_REPS) <= set(reps):
        return [(ENROLL_REPS, TEST_REPS)]
    print(f"  (session has reps {reps}: using leave-one-rep-out)")
    return [(tuple(x for x in reps if x != r), (r,)) for r in reps]


def print_confusion(rows, labels):
    short = {l: (l.replace("gesture_", "g") if l != lc.REST_LABEL else "rest") for l in labels}
    m = defaultdict(int)
    for true, pred, _, _ in rows:
        m[(true, pred)] += 1
    print("\n  capture confusion (rows = true, cols = predicted):")
    print("        " + "".join(f"{short[l]:>5}" for l in labels))
    for t in labels:
        print(f"  {short[t]:>5} " + "".join(
            f"{(m[(t, p)] if m[(t, p)] else '.'):>5}" for p in labels))


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else lc.latest_session()
    print(f"session: {session}")
    print(f"prototypes: reps {ENROLL_REPS}, full holds trimmed {HOLD_TRIM_S} s")
    print(f"test: reps {TEST_REPS}, one {lc.CAPTURE_S:.0f} s capture per hold from the cue "
          f"(pre-roll {lc.PREROLL_S:.0f} s)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lc.load_embedding(MODEL_PATH, device=device)
    reference = lc.load_reference()

    variants = [("A old pipeline     ", False, None),
                ("B resampled        ", True, None),
                ("C resampled+aligned", True, reference)]
    folds, last_rows = None, None
    print(f"\nDB5 reference feature median: {reference:.4f}")
    print(f"{'':22}{'12 gestures':>22}{'with rest class':>26}")
    print(f"{'':22}{'window':>11}{'capture':>11}{'gestures':>13}{'rest':>13}   user median")
    for name, resample, ref in variants:
        gestures, rests = lc.session_features(session, GESTURE_NAMES, HOLD_TRIM_S,
                                              REST_TRIM_S, resample=resample)
        if not gestures:
            raise SystemExit("no recordings found in that session")
        caps, rest_caps = session_captures(session, GESTURE_NAMES, resample=resample)
        folds = folds or folds_for(gestures)

        rows12, rows13 = [], []
        for enroll, test in folds:
            rows12 += evaluate(model, gestures, rests, caps, rest_caps, enroll, test,
                               ref, False, device)[0]
            rows13 += evaluate(model, gestures, rests, caps, rest_caps, enroll, test,
                               ref, True, device)[0]
        win12, acc12 = summarise(rows12)
        acc13_g = summarise([r for r in rows13 if r[0] != lc.REST_LABEL])[1]
        acc13_r = summarise([r for r in rows13 if r[0] == lc.REST_LABEL])[1]
        med = np.median(np.concatenate([f for by_rep in gestures.values()
                                        for f in by_rep.values() if len(f)]))
        print(f"  {name}{win12:>11.3f}{acc12:>11.3f}{acc13_g:>13.3f}{acc13_r:>13.3f}"
              f"   {med:.4f}")
        last_rows = rows13

    n_g = sum(r[0] != lc.REST_LABEL for r in last_rows)
    n_r = sum(r[0] == lc.REST_LABEL for r in last_rows)
    print(f"\n{n_g} gesture captures, {n_r} rest captures")
    labels = [g for g in GESTURE_NAMES if any(r[0] == g for r in last_rows)] + [lc.REST_LABEL]
    print("\nversion C, with rest class:")
    print_confusion(last_rows, labels)
    print("\nRead it like this:")
    print("  Version C 'with rest class' is the offline estimate of Live_predict.py --test:")
    print("  same 3 s captures from the moment the gesture is cued, same processing.")
    print("  The DB5 check votes over whole trimmed holds, so it is not the same test.")


if __name__ == "__main__":
    main()