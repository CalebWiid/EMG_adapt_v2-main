"""
Enroll.py - record your gestures with a Ninapro-style timed protocol and build
prototypes (one per gesture, plus "rest") for every model in Common.MODELS.

Protocol per gesture (one continuous recording, like a DB5 exercise):
    REST 3 s | HOLD 5 s | REST 3 s | HOLD 5 s | ... | HOLD 5 s | REST 3 s
Follow the on-screen cue: hold the gesture for the whole HOLD period, relax for
REST. The first/last HOLD_TRIM_S of every hold are dropped (reaction time and
transitions, a stand-in for DB5's restimulus relabelling).

While TESTING the system: N_REPS = 6. Prototypes are built from reps 1, 3, 4
(ENROLL_REPS); reps 2, 5, 6 are kept for Offline_eval.py (same split as the DB5
evaluation). For DEPLOYMENT (a new user just enrolling): N_REPS = 3.

Every raw recording is saved to recordings/<date_time>/<gesture>.npz so it can be
re-analysed offline without the armband.

PROTOTYPES FOR EVERY MODEL. After recording, prototypes are built for every
embedding in Common.MODELS whose weights file exists (supervised ->
prototypes.npz, reptile -> prototypes_reptile.npz), all from the same session,
so the models are always compared on the same enrollment. Each file stores the
embedding it was built with, and Live_predict.py refuses a mismatched pair.

On start-up, if gestures are already stored:
  [r] reset - delete them (all models) and enroll a NEW user     [q] quit - keep them

REBUILD (no armband): build the prototypes again from a saved session, e.g.
after adding a new embedding to Common.MODELS:
  python Enroll.py --rebuild                                 session of the current prototypes
  python Enroll.py --rebuild <session_folder>                a specific session
  python Enroll.py --rebuild <session_folder> --out <folder> write the files elsewhere
--out leaves the active prototype files alone. Use it for an older session whose
live tests you want to replay (Live_predict.py --replay <test> --protos-dir <folder>).
The amplitude offset does not depend on the model, so every model must print the
same offset, equal to the one stored before.

Run:  python Enroll.py   (run Check_embedding.py once beforehand)
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import Common as lc
from stream import LiveBoard          # live window of the raw EMG

N_GESTURES = 12
GESTURE_NAMES = [f"gesture_{i + 1}" for i in range(N_GESTURES)]
N_REPS = 6                     # 6 while testing (3 enroll + 3 test); 3 for deployment
ENROLL_REPS = (1, 3, 4)        # reps used to build the prototypes


REST_S = 3.0                   # DB5: 3 s rest
HOLD_S = 5.0                   # DB5: 5 s movement
HOLD_TRIM_S = 0.5              # dropped at each end of a hold
REST_TRIM_S = 0.75             # dropped at each end of a rest


# --------------------------------------------------------------------------- #
# Prototype files (one per model)
# --------------------------------------------------------------------------- #
def prototype_files(out_dir=""):
    """{model: prototype path} for every model whose prototype file exists."""
    files = {}
    for name in lc.MODELS:
        path = lc.model_paths(name, out_dir)[1]
        if os.path.exists(path):
            files[name] = path
    return files


def active_session():
    """The recording session the active prototype files were built from ('' if none)."""
    for path in prototype_files().values():
        try:
            session = lc.prototype_info(path)["session_dir"]
        except Exception:
            continue
        if session:
            return session
    return ""


def build_all_prototypes(session_dir, out_dir="", device=None):
    """Prototypes for every model in Common.MODELS from ONE recorded session.
    The CCA features (and so the offset) are computed once and shared, so the only
    difference between the models' prototype files is the embedding."""
    reference = lc.load_reference()
    gestures, rests = lc.session_features(session_dir, GESTURE_NAMES,
                                          HOLD_TRIM_S, REST_TRIM_S)
    if not gestures:
        raise SystemExit(f"no gesture recordings found in {session_dir}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    built = []
    for name in lc.MODELS:
        weights, proto_path = lc.model_paths(name, out_dir)
        if not os.path.exists(weights):
            note = (f" - {proto_path} left unchanged, it may be from an older build"
                    if os.path.exists(proto_path) else "")
            print(f"  [skip] {name}: weights {weights} not found{note}")
            continue
        previous = None
        if os.path.exists(proto_path):
            try:
                previous = float(np.load(proto_path, allow_pickle=True)["offset"])
            except Exception:
                pass
        model = lc.load_embedding(weights, device=device)
        labels, protos, offset = lc.make_prototypes(model, gestures, rests, ENROLL_REPS,
                                                    reference_median=reference,
                                                    device=device)
        lc.save_prototypes(proto_path, labels, protos.numpy(), offset,
                           session_dir, weights)
        was = f"   (previous file: {previous:+.4f})" if previous is not None else ""
        print(f"  {name:<11} offset {offset:+.4f}   {len(labels)} prototypes "
              f"-> {proto_path}{was}")
        built.append(name)
    if not built:
        raise SystemExit("no model weights found - no prototypes were built")
    return built


def handle_existing_gestures():
    """Returns True to continue with a fresh enrollment, False to keep and exit."""
    files = prototype_files()
    if not files:
        print("No stored gestures found - starting a new enrollment.\n")
        return True
    try:
        stored = [str(l) for l in
                  np.load(next(iter(files.values())), allow_pickle=True)["labels"]]
    except Exception:
        stored = ["<unreadable file>"]
    print(f"The system already knows {len(stored)} classes: {stored}")
    print(f"prototype files: {', '.join(files.values())}")
    while True:
        choice = input("[r] reset - delete them (all models) and enroll a NEW user\n"
                       "[q] quit  - keep them and exit\n> ").strip().lower()
        if choice == "q":
            print("Keeping the stored gestures. Exiting.")
            return False
        if choice == "r":
            confirm = input("This permanently deletes the stored gestures. "
                            "Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                for path in files.values():
                    os.remove(path)
                print(f"Deleted {', '.join(files.values())}. "
                      f"Starting a fresh enrollment.\n")
                return True
            print("Reset cancelled.\n")
            continue
        print("Please type r or q.\n")


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def check_stream(board, seconds=1.0):
    """Make sure data is actually arriving before starting the protocol."""
    lc.flush(board)
    time.sleep(seconds)
    n = lc.grab_all(board).shape[0]
    expected = int(seconds * lc.DEVICE_FS)
    print(f"stream check: {n} samples in {seconds:.0f} s (expected ~{expected})")
    if n < 0.5 * expected:
        raise SystemExit("Too little data arriving from the armband - check the "
                         "WiFi connection and that the band is streaming.")


def record_protocol(board, name, n_reps=N_REPS, rest_s=REST_S, hold_s=HOLD_S):
    """Run the timed REST/HOLD cues for one gesture as ONE continuous recording.

    Phase boundaries are taken as the number of samples received at each cue
    change, so they stay correct even if the stream rate wobbles.
    Returns (emg (n,8), holds [(rep,start,end)], rests [(rep,start,end)])."""
    chunks, total = [], 0

    def run_phase(seconds, text):
        nonlocal total
        t_end = time.time() + seconds
        while True:
            remaining = t_end - time.time()
            if remaining <= 0:
                break
            print(f"\r  {text:<40} {remaining:4.1f} s ", end="", flush=True)
            time.sleep(min(0.05, remaining))
            c = lc.grab_all(board)
            if len(c):
                chunks.append(c)
                total += len(c)
        c = lc.grab_all(board)
        if len(c):
            chunks.append(c)
            total += len(c)

    holds, rests = [], []
    lc.flush(board)
    for rep in range(1, n_reps + 1):
        s = total
        run_phase(rest_s, f"rest          (next: rep {rep}/{n_reps})")
        rests.append((rep, s, total))
        s = total
        run_phase(hold_s, f">>> HOLD {name}   rep {rep}/{n_reps}")
        holds.append((rep, s, total))
    s = total
    run_phase(rest_s, "rest          (last one)")
    rests.append((n_reps, s, total))
    print()

    emg = np.concatenate(chunks) if chunks else np.empty((0, lc.N_CHANNELS))
    return emg, holds, rests


# --------------------------------------------------------------------------- #
# Rebuild from a saved session (no armband)
# --------------------------------------------------------------------------- #
def rebuild(argv):
    """--rebuild [session_folder] [--out folder]"""
    i = argv.index("--rebuild")
    session = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else ""
    out_dir = lc.arg_value(argv, "--out", "")
    session = session or active_session()
    if not session:
        raise SystemExit("No session given and no current prototypes to take it from:\n"
                         "  python Enroll.py --rebuild <session_folder>")
    if not os.path.isdir(session):
        raise SystemExit(f"session folder not found: {session}")

    if not out_dir:
        current = active_session()
        if current and os.path.normpath(current) != os.path.normpath(session):
            print(f"The active prototypes come from {current},\n"
                  f"but you are rebuilding from {session}.\n"
                  f"This REPLACES the active prototypes "
                  f"(use --out <folder> to write them somewhere else).")
            if input("Type 'yes' to replace them: ").strip().lower() != "yes":
                print("Cancelled.")
                return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"rebuilding prototypes from {session} -> "
          f"{out_dir or 'the working directory'}  (device: {device})")
    build_all_prototypes(session, out_dir, device)
    print("Every model should show the same offset (it does not depend on the model).")


# --------------------------------------------------------------------------- #
# Enrollment
# --------------------------------------------------------------------------- #
def main():
    if "--rebuild" in sys.argv:
        rebuild(sys.argv)
        return
    if not handle_existing_gestures():
        return

    lc.load_reference()                              # fail early if missing
    available = [m for m in lc.MODELS if os.path.exists(lc.model_paths(m)[0])]
    if lc.ACTIVE_MODEL not in available:
        raise SystemExit(f"weights of the active model '{lc.ACTIVE_MODEL}' "
                         f"({lc.model_paths()[0]}) not found.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"prototypes will be built for: {', '.join(available)} "
          f"(active: {lc.ACTIVE_MODEL}, device: {device})")

    session_dir = os.path.join(lc.RECORDINGS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    per_gesture_s = N_REPS * (REST_S + HOLD_S) + REST_S
    print(f"session -> {session_dir}")
    print(f"{len(GESTURE_NAMES)} gestures x {N_REPS} reps, about {per_gesture_s:.0f} s each.\n")

    board = LiveBoard(lc.connect_board(), "Enroll - raw EMG")
    try:
        check_stream(board)
        for name in GESTURE_NAMES:
            while True:
                print(f"\n[{name}]  get ready. Press SPACE to start "
                      f"({N_REPS} x {HOLD_S:.0f} s holds).")
                lc.wait_for_space()
                emg, holds, rests = record_protocol(board, name)
                expected = int(per_gesture_s * lc.DEVICE_FS)
                print(f"  received {emg.shape[0]} samples (expected ~{expected}, "
                      f"{100 * emg.shape[0] / max(expected, 1):.0f}%)")
                print("  [SPACE] keep and continue    [r] redo this gesture")
                if lc.read_key(" r") == " ":
                    break
            lc.save_recording(os.path.join(session_dir, f"{name}.npz"),
                              emg, lc.DEVICE_FS, name, holds, rests)
    except KeyboardInterrupt:
        print("\nenrollment interrupted - prototypes NOT saved "
              f"(recordings so far are in {session_dir}).")
        return
    finally:
        lc.release_board(board)

    # ---- features -> amplitude alignment -> prototypes (every model) ----
    print("\nbuilding prototypes...")
    build_all_prototypes(session_dir, device=device)
    test_reps = [r for r in range(1, N_REPS + 1) if r not in ENROLL_REPS]
    if test_reps:
        print(f"\nNext: python Offline_eval.py   (tests on reps "
              f"{', '.join(map(str, test_reps))} of this session)")


if __name__ == "__main__":
    main()