"""
Enroll.py - record your gestures with a Ninapro-style timed protocol and build
prototypes (one per gesture, plus "rest").

Protocol per gesture (one continuous recording, like a DB5 exercise):
    REST 3 s | HOLD 5 s | REST 3 s | HOLD 5 s | ... | HOLD 5 s | REST 3 s
Follow the on-screen cue: hold the gesture for the whole HOLD period, relax for
REST. The first/last HOLD_TRIM_S of every hold are dropped (reaction time and
transitions, a stand-in for DB5's restimulus relabelling).

While TESTING the system: N_REPS = 6. Prototypes are built from reps 1-3 only;
reps 4-6 are kept for Offline_eval.py (same split as the DB5 evaluation).
For DEPLOYMENT (a new user just enrolling): N_REPS = 3.

Every raw recording is saved to recordings/<date_time>/<gesture>.npz so it can be
re-analysed offline without the armband.

On start-up, if gestures are already stored:
  [r] reset - delete them and enroll a NEW user     [q] quit - keep them

Run:  python Enroll.py   (run Check_embedding.py once beforehand)
"""
import os
import time
from datetime import datetime

import numpy as np
import torch
import Common as lc

MODEL_PATH = "deployment_embedding.pt"
PROTO_PATH = "prototypes.npz"

N_GESTURES = 12
GESTURE_NAMES = [f"gesture_{i + 1}" for i in range(N_GESTURES)]
N_REPS = 6                     # 6 while testing (3 enroll + 3 test); 3 for deployment
ENROLL_REPS = (1, 3, 4)        # reps used to build the prototypes


REST_S = 3.0                   # DB5: 3 s rest
HOLD_S = 5.0                   # DB5: 5 s movement
HOLD_TRIM_S = 0.5              # dropped at each end of a hold
REST_TRIM_S = 0.75             # dropped at each end of a rest


def handle_existing_gestures(path=PROTO_PATH):
    """Returns True to continue with a fresh enrollment, False to keep and exit."""
    if not os.path.exists(path):
        print("No stored gestures found - starting a new enrollment.\n")
        return True
    try:
        stored = [str(l) for l in np.load(path, allow_pickle=True)["labels"]]
    except Exception:
        stored = ["<unreadable file>"]
    print(f"The system already knows {len(stored)} classes: {stored}")
    while True:
        choice = input("[r] reset - delete them and enroll a NEW user\n"
                       "[q] quit  - keep them and exit\n> ").strip().lower()
        if choice == "q":
            print("Keeping the stored gestures. Exiting.")
            return False
        if choice == "r":
            confirm = input("This permanently deletes the stored gestures. "
                            "Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                os.remove(path)
                print(f"Deleted {path}. Starting a fresh enrollment.\n")
                return True
            print("Reset cancelled.\n")
            continue
        print("Please type r or q.\n")


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


def main():
    if not handle_existing_gestures():
        return

    reference = lc.load_reference()                  # fail early if missing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lc.load_embedding(MODEL_PATH, device=device)
    print(f"loaded embedding <- {MODEL_PATH}  (device: {device})")

    session_dir = os.path.join(lc.RECORDINGS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    per_gesture_s = N_REPS * (REST_S + HOLD_S) + REST_S
    print(f"session -> {session_dir}")
    print(f"{len(GESTURE_NAMES)} gestures x {N_REPS} reps, about {per_gesture_s:.0f} s each.\n")

    board = lc.connect_board()
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

    # ---- features -> amplitude alignment -> prototypes ----
    print("\nbuilding prototypes...")
    gestures, rests = lc.session_features(session_dir, GESTURE_NAMES,
                                          HOLD_TRIM_S, REST_TRIM_S)
    labels, protos, offset = lc.make_prototypes(model, gestures, rests, ENROLL_REPS,
                                                reference_median=reference,
                                                device=device)
    lc.save_prototypes(PROTO_PATH, labels, protos.numpy(), offset, session_dir)
    print(f"amplitude offset = {offset:+.4f} (added to every feature)")
    print(f"saved {len(labels)} prototypes ({', '.join(labels)}) -> {PROTO_PATH}")
    if N_REPS > max(ENROLL_REPS):
        print(f"\nNext: python Offline_eval.py   (tests on reps "
              f"{max(ENROLL_REPS) + 1}-{N_REPS} of this session)")


if __name__ == "__main__":
    main()