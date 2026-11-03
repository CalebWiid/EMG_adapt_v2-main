"""
stream.py - live view of the raw MindRove EMG (all 8 channels).

A visual check only: the window plots the raw samples exactly as the armband
sends them (no filtering, no processing) and nothing is saved.

TWO WAYS TO USE IT

1. Inside Enroll.py / Live_predict.py (the armband allows only one connection,
   so this is the way to watch while recording). Wrap the board once:

       from stream import LiveBoard
       board = LiveBoard(lc.connect_board())

   Everything the script reads from the board is passed on, unchanged, to a
   window that runs as a SEPARATE program. The script gets exactly the same data
   as before, and the window can never slow down, freeze or stop a recording:
   if the window falls behind, it skips samples; if it is closed or fails, the
   script carries on without it. The window closes when the board is released.

2. On its own, when no other script is using the armband (e.g. a contact check
   before enrolling):

       python3 stream.py            # connect to the armband and show the signal
       python3 stream.py --demo     # synthetic signal, no armband needed

   It refuses to connect while another program is using the armband.

Close the window (or press Ctrl+C) to stop.
"""
import os
import sys
import glob
import select
import subprocess
import threading

import numpy as np

SECONDS_SHOWN = 5.0                     # seconds of signal in the window
REFRESH_S = 0.05                        # redraw every 50 ms
N_CH = 8
SAMPLE_BYTES = N_CH * 8                 # one sample = 8 channels as float64
CHUNK = select.PIPE_BUF // SAMPLE_BYTES  # samples per write (writes this small are all-or-nothing)


# --------------------------------------------------------------------------- #
# Used by Enroll.py / Live_predict.py: pass the board's data on to the window
# --------------------------------------------------------------------------- #
class LiveBoard:
    """Wraps a connected board. Behaves exactly like the board (every call is
    passed straight through, returning the same data); in addition, the EMG of
    every get_board_data() call is copied to the live window."""

    def __init__(self, board, title="MindRove raw EMG"):
        self._board = board
        self._proc = None
        try:
            from mindrove.board_shim import BoardShim
            board_id = board.get_board_id()
            self._rows = BoardShim.get_emg_channels(board_id)[:N_CH]
            fs = BoardShim.get_sampling_rate(board_id)
            self._proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--pipe", str(fs), title],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                start_new_session=True)            # Ctrl+C in the script doesn't reach it
            self._fd = self._proc.stdin.fileno()
            os.set_blocking(self._fd, False)       # never wait for the window
            print("live EMG window opened. If key presses stop working, click "
                  "back on this terminal.")
        except Exception as e:
            print(f"[live EMG window not started: {e}]")
            self._proc = None

    def get_board_data(self, *args, **kwargs):
        data = self._board.get_board_data(*args, **kwargs)
        self._send(data)
        return data                                # the same array, untouched

    def release_session(self):
        try:
            return self._board.release_session()
        finally:
            self.close()

    def close(self):
        """Close the window (also happens automatically when the script exits)."""
        if self._proc is not None:
            proc, self._proc = self._proc, None
            try:
                proc.stdin.close()                 # the window sees the end and exits
                proc.wait(timeout=2)
            except Exception:
                proc.kill()

    def __getattr__(self, name):                   # everything else: the real board
        return getattr(self._board, name)

    def _send(self, data):
        if self._proc is None or data.shape[1] == 0:
            return
        try:
            x = np.ascontiguousarray(data[self._rows].T, dtype=np.float64)
            for i in range(0, len(x), CHUNK):
                os.write(self._fd, x[i:i + CHUNK].tobytes())
        except BlockingIOError:
            pass                                   # window is behind: skip, don't wait
        except Exception:
            self.close()                           # window closed or failed: carry on without it


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #
def show_window(get_new, fs, title, finished=lambda: False):
    """Plot the last SECONDS_SHOWN of signal until the window is closed or
    finished() is True. get_new() returns the new samples (n, 8) since last call."""
    import matplotlib
    import matplotlib.pyplot as plt
    for k in [k for k in matplotlib.rcParams if k.startswith("keymap.")]:
        matplotlib.rcParams[k] = []                # no matplotlib keyboard shortcuts
    matplotlib.rcParams["toolbar"] = "None"        # no toolbar (and so no save button)

    n = int(SECONDS_SHOWN * fs)
    buf = np.full((n, N_CH), np.nan)               # nan = not received yet
    t = np.arange(-n, 0) / fs
    fig, axes = plt.subplots(N_CH, 1, sharex=True, figsize=(10, 7), num=title)
    lines = []
    for c, ax in enumerate(axes):
        lines.append(ax.plot(t, buf[:, c], lw=0.6, color=f"C{c}")[0])
        ax.set_ylabel(f"ch{c + 1}", rotation=0, labelpad=15)
    axes[-1].set_xlim(t[0], 0)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    plt.show(block=False)

    while plt.fignum_exists(fig.number) and not finished():
        x = get_new()
        if len(x):
            buf = np.concatenate([buf, x])[-n:]
            for c, (line, ax) in enumerate(zip(lines, axes)):
                y = buf[:, c]                      # raw values, unchanged
                line.set_ydata(y)
                if np.isfinite(y).any():
                    lo, hi = np.nanmin(y), np.nanmax(y)
                    pad = max(0.05 * (hi - lo), 1e-6)
                    ax.set_ylim(lo - pad, hi + pad)
            fig.canvas.draw_idle()
        plt.pause(REFRESH_S)
    plt.close("all")


def run_pipe(fs, title):
    """Window fed by LiveBoard through stdin; exits when the script closes it."""
    pending, lock, done = [], threading.Lock(), threading.Event()

    def reader():
        rest, f = b"", sys.stdin.buffer
        while True:
            b = f.read1(65536)
            if not b:
                done.set()
                return
            rest += b
            k = len(rest) // SAMPLE_BYTES * SAMPLE_BYTES
            if k:
                with lock:
                    pending.append(np.frombuffer(rest[:k], dtype=np.float64)
                                   .reshape(-1, N_CH))
                rest = rest[k:]

    def get_new():
        with lock:
            if not pending:
                return np.empty((0, N_CH))
            x = np.concatenate(pending)
            pending.clear()
        return x

    threading.Thread(target=reader, daemon=True).start()
    show_window(get_new, fs, title, finished=done.is_set)


# --------------------------------------------------------------------------- #
# Standalone use (no other script running)
# --------------------------------------------------------------------------- #
def other_mindrove_users():
    """PIDs + command lines of other processes that have the MindRove SDK loaded."""
    users = []
    for maps in glob.glob("/proc/[0-9]*/maps"):
        pid = maps.split("/")[2]
        if int(pid) == os.getpid():
            continue
        try:
            with open(maps) as f:
                if "libBoardController" not in f.read():
                    continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="ignore").strip()
            users.append((pid, cmd))
        except OSError:                            # process ended / not ours
            pass
    return users


def run_standalone(demo):
    from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams
    BoardShim.disable_board_logger()
    board_id = BoardIds.SYNTHETIC_BOARD if demo else BoardIds.MINDROVE_WIFI_BOARD
    if not demo:
        busy = other_mindrove_users()
        if busy:
            print("The armband is in use by another program - not connecting, so "
                  "that program is not disturbed:")
            for pid, cmd in busy:
                print(f"  pid {pid}: {cmd}")
            print("Use LiveBoard inside that script instead (see the top of stream.py).")
            sys.exit(1)
    board = BoardShim(board_id, MindRoveInputParams())
    board.prepare_session()
    board.start_stream()
    try:
        rows = BoardShim.get_emg_channels(board_id)[:N_CH]
        show_window(lambda: board.get_board_data()[rows].T,
                    BoardShim.get_sampling_rate(board_id),
                    "MindRove raw EMG" + (" (demo)" if demo else ""))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            board.stop_stream()
            board.release_session()
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--pipe":
        try:
            run_pipe(int(float(sys.argv[2])),
                     sys.argv[3] if len(sys.argv) > 3 else "MindRove raw EMG")
        except KeyboardInterrupt:
            pass
    else:
        run_standalone("--demo" in sys.argv)