"""
paper_imp.py - supervised 52-gesture model on ALL 10 DB5 subjects, split by repetition.

WHAT IT DOES
  * Loads every subject (1-10) and every exercise (E1-E3) -> all 52 DB5 movements.
    Rest is excluded by default; --include-rest adds it as ONE extra class (53-way, as
    in the paper). DB5 rest periods have repetition 0, so each rest period gets the rep
    of the movement that follows it (trailing rest: the movement before it) - so rest
    is split into train/test by repetition exactly like the gestures.
  * Uses the SAME preprocessing (Preprocessing.py: 20 Hz HP -> rectify -> 1 Hz LP,
    200 ms windows / 10 ms step, CCA features) and the SAME network
    (CNN_model.EMGAdapt) as the rest of the project.
  * Channels are chosen HERE (--channels 16 | 8, --native-order), not in Preprocessing.py,
    so Preprocessing.py can stay at 8 channels for the MindRove / deployment pipeline.
    16 = both Myo bands, interleaved so neighbouring electrodes are neighbours
    [1, 9, 2, 10, ..., 8, 16]; 8 = first band only (MindRove-like).
  * Trains ONE pooled 52-way model on repetitions 1, 3, 4, 6 of every subject and
    tests it on the held-out repetitions 2 and 5 of every subject
    (the Atzori / Ninapro split used by the paper).

HOW THIS DIFFERS FROM Reptile_learning.py
  * No base/novel gesture split: all 52 gestures are trained AND tested (seen gestures).
  * No subject hold-out: all 10 subjects are in training; only the repetitions differ.
  * --split window replaces the repetition split by a RANDOM split of individual windows
    (all 6 reps, 1/3 test by default). Windows of one performance then end up in both
    train and test; consecutive windows are 10 ms apart and share 95% of their samples,
    so this measures recognition of near-copies of training data. The run prints the
    share of test windows whose direct neighbour is in training. Pooled accuracy then
    votes over the test windows of each segment.
  * Every test window is classified three ways by the same model: softmax head,
    nearest prototype (all subjects), nearest prototype (own subject).
  * Writes nothing that the live pipeline uses (no deployment_embedding*.pt).

USAGE
  python paper_imp.py                          # 16 channels, paper lr 1e-3 / lambda 0.25
  python paper_imp.py --seeds 0 1 2            # repeat with 3 seeds
  python paper_imp.py --include-rest           # 53 classes: 52 gestures + rest
  python paper_imp.py --include-rest --split window   # random WINDOW split instead of reps
  python paper_imp.py --channels 8             # first Myo band only (MindRove-like)
  python paper_imp.py --lr 1e-4 --lambda 0     # the Reptile_learning pre-training settings
  python paper_imp.py --epochs 5 --subjects 1 2 --data-dir /path/to/DB5

OUTPUT
  paper_imp_results/<stamp>/   train_log.csv, per_subject.csv, per_class.csv,
                               confusion_seed<k>_<method>.csv, summary.csv
  paper_imp_models/<stamp>_seed<k>.pt   (new file every run, never overwritten)
  paper_imp_cache/             cached CCA features (delete to force re-processing)
"""
import argparse
import csv
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import Preprocessing as P
from CNN_model import EMGAdapt

# ---- configuration ----
SUBJECTS = tuple(range(1, 11))     # all 10 DB5 subjects
EXERCISES = (1, 2, 3)              # 12 + 17 + 23 = 52 movements
EXP_BASE = 1000                    # global class id = exercise * 1000 + label
TRAIN_REPS = (1, 3, 4, 6)
TEST_REPS = (2, 5)                 # held out for testing

INCLUDE_REST = False               # True = rest as one extra class (53-way, as in the paper)
REST_ID = 0                        # global id of the rest class (same class in all exercises)

CHANNELS = 16                      # 16 = both Myo bands (paper), 8 = first band (MindRove-like)
NATIVE_ORDER = False               # False = interleave the two bands (PHYSICAL_CHANNEL_ORDER)

EPOCHS = 20
LR = 1e-3                          # paper value (Reptile_learning pre-training used 1e-4)
BATCH_SIZE = 256
LAMBDA = 0.25                      # paper hybrid-loss weight (0 = plain cross-entropy)
SEEDS = (0,)

SPLIT = "rep"                      # "rep" = train reps 1,3,4,6 / test reps 2,5 (Atzori split)
                                   # "window" = random split of individual windows, all reps
TEST_FRAC = 1 / 3                  # window split: share of windows in the test set (2 of 6 reps)
SPLIT_SEED = 0                     # window split: fixed, so every training seed sees the same split

CACHE_DIR = "paper_imp_cache"
RESULTS_DIR = "paper_imp_results"
MODELS_DIR = "paper_imp_models"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
_P_LOAD_DATA = P.load_data          # the original, before configure_channels wraps it
CHANNEL_LAYOUT = None               # final column order (original DB5 channel numbers)


def configure_channels(n_channels=CHANNELS, native_order=NATIVE_ORDER):
    """Choose the DB5 channels for this script only, whatever Preprocessing.py's
    SELECT_CHANNELS says. Wraps Preprocessing.load_data (its defaults are fixed when the
    module is imported, so changing P.SELECT_CHANNELS at runtime would have no effect)."""
    global CHANNEL_LAYOUT
    if n_channels == 16:
        select = None                                   # keep all 16 channels
    elif n_channels == 8:
        select = list(range(1, 9))                      # first Myo band
    else:
        raise ValueError("--channels must be 8 or 16")
    order = None if native_order else list(P.PHYSICAL_CHANNEL_ORDER)

    def load_data(path, channel_order=order, select_channels=select):
        return _P_LOAD_DATA(path, channel_order=channel_order, select_channels=select_channels)
    P.load_data = load_data                             # used by P.load_and_segment

    # the layout load_data will produce (same rules as Preprocessing.load_data)
    layout = list(range(1, 17))
    if select is not None:
        layout = [layout[c - 1] for c in select]
    if order is not None and len(order) == len(layout):
        layout = [layout[c - 1] for c in order]
    CHANNEL_LAYOUT = layout
    print(f"channels ({len(layout)}): {layout}")
    return layout


def _cache_path(subjects, exercises, include_rest):
    flt = P.DEFAULT_FILTER_KWARGS
    tag = (f"s{'-'.join(map(str, subjects))}_e{'-'.join(map(str, exercises))}"
           f"_ch{'-'.join(map(str, CHANNEL_LAYOUT))}"
           f"_hp{flt['highpass_cutoff']}_lp{flt['lowpass_cutoff']}_r{int(flt['rectify'])}"
           f"_fs{P.FS}_w{P.WINDOW_SIZE_MS}_st{P.STEP_SIZE_MS}_rest{int(include_rest)}")
    return os.path.join(CACHE_DIR, f"db5v2_{tag}.npz")


def rest_repetitions(segs):
    """DB5 rest periods carry rerepetition 0, so they would fall in neither the train nor
    the test reps. Give each rest period the rep of the NEXT movement (the rest before
    rep k belongs to rep k); a trailing rest with no movement after it gets the rep of the
    movement before it. Rest periods that already have a rep keep it."""
    move_idx = [i for i, s in enumerate(segs) if s.label != 0]
    reps = []
    for i, s in enumerate(segs):
        if s.label != 0 or s.repetition != 0:
            reps.append(s.repetition)
            continue
        later = [j for j in move_idx if j > i]
        earlier = [j for j in move_idx if j < i]
        src = later[0] if later else (earlier[-1] if earlier else None)
        reps.append(segs[src].repetition if src is not None else 0)
    return reps


def process_recording(path, ex, include_rest):
    """Same pipeline as Preprocessing.process_file (filter whole recording -> segment ->
    window -> CCA), but keeps the segment of every window and optionally the rest periods.
    Returns X, y (global id: ex*1000+label, rest = REST_ID), reps, local segment index."""
    segs = P.load_and_segment(path, P.FS, include_rest=include_rest)
    reps_per_seg = rest_repetitions(segs) if include_rest else [s.repetition for s in segs]
    X_l, y_l, r_l, g_l = [], [], [], []
    for k, (seg, rep) in enumerate(zip(segs, reps_per_seg)):
        feats = P.process_segment(seg, P.FS, apply_filter=False)   # recording already filtered
        if feats.shape[0] == 0:
            continue
        gid = REST_ID if seg.label == 0 else ex * EXP_BASE + seg.label
        X_l.append(feats)
        y_l.append(np.full(len(feats), gid, dtype=np.int64))
        r_l.append(np.full(len(feats), rep, dtype=np.int64))
        g_l.append(np.full(len(feats), k, dtype=np.int64))
    if not X_l:
        return None
    return np.concatenate(X_l), np.concatenate(y_l), np.concatenate(r_l), np.concatenate(g_l)


def load_all(subjects=SUBJECTS, exercises=EXERCISES, include_rest=INCLUDE_REST, use_cache=True):
    """Return X (n, e) CCA features, y (n,) global ids, reps (n,), subj (n,),
    seg (n,) = id of the continuous segment (one gesture performance or one rest period)."""
    path = _cache_path(subjects, exercises, include_rest)
    if use_cache and os.path.exists(path):
        d = np.load(path)
        if (str(d["data_dir"]) == os.path.abspath(P.DATA_DIR)
                and "channels" in d and d["channels"].tolist() == CHANNEL_LAYOUT):
            print(f"loaded cached features <- {path}")
            return d["X"], d["y"], d["reps"], d["subj"], d["seg"]
        print("cache was built from a different DATA_DIR / channel layout -> re-processing")

    X_all, y_all, rep_all, subj_all, seg_all = [], [], [], [], []
    n_seg = 0
    t0 = time.perf_counter()
    for s in subjects:
        for ex in exercises:
            try:
                out = process_recording(P.filename_for(s, ex), ex, include_rest)
            except FileNotFoundError:
                print(f"  [skip] missing file: {P.filename_for(s, ex)}")
                continue
            if out is None:
                continue
            X, y, reps, seg = out
            X_all.append(X)
            y_all.append(y)
            rep_all.append(reps)
            subj_all.append(np.full(X.shape[0], s, dtype=np.int64))
            seg_all.append(seg + n_seg)                    # globally unique segment ids
            n_seg += int(seg.max()) + 1
        print(f"  processed S{s}  ({time.perf_counter() - t0:.0f} s)")
    if not X_all:
        raise SystemExit(f"No data loaded - check DATA_DIR ({P.DATA_DIR}).")
    X, y = np.concatenate(X_all), np.concatenate(y_all)
    reps, subj, seg = np.concatenate(rep_all), np.concatenate(subj_all), np.concatenate(seg_all)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, X=X, y=y, reps=reps, subj=subj, seg=seg,
                            data_dir=os.path.abspath(P.DATA_DIR), channels=np.array(CHANNEL_LAYOUT))
        print(f"cached features -> {path}")
    return X, y, reps, subj, seg


def class_name(gid):
    return "rest" if gid == REST_ID else f"E{gid // EXP_BASE}_g{gid % EXP_BASE}"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, X, device, batch=2048):
    """Return (embeddings, predicted class index) for every window."""
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    embs, preds = [], []
    for i in range(0, len(Xt), batch):
        z, logits = model(Xt[i:i + batch].to(device))
        embs.append(z.cpu())
        preds.append(logits.argmax(1).cpu())
    return torch.cat(embs), torch.cat(preds).numpy()


def class_prototypes(model, X, y_idx, n_classes, device):
    embs, _ = predict(model, X, device)
    y_t = torch.as_tensor(y_idx)
    protos = torch.zeros(n_classes, embs.shape[1])
    for c in range(n_classes):
        m = y_t == c
        if m.any():
            protos[c] = embs[m].mean(0)
    return protos.to(device)


# how a test window is classified (all from the SAME trained model)
METHODS = {
    "softmax":       "softmax head (logits.argmax)",
    "proto_global":  "nearest prototype, all subjects",
    "proto_subject": "nearest prototype, own subject",
}


@torch.no_grad()
def prototype_sets(model, X, y_idx, subj, n_classes, device, batch=4096):
    """Class prototypes = mean embedding of the TRAINING windows (reps 1,3,4,6).

    global      : (C, d) mean over all subjects' training windows of each gesture.
    per subject : (S, C, d) each subject's own training windows only - the same idea
                  as enrolling a user (Reptile_learning / MindRove Enroll.py).
    Sums are accumulated batch by batch, so no full embedding matrix is kept in memory.
    """
    model.eval()
    subjects = np.unique(subj)                         # sorted
    d = model.embedding_dim
    key = torch.as_tensor(np.searchsorted(subjects, subj) * n_classes + np.asarray(y_idx))
    sums = torch.zeros(len(subjects) * n_classes, d, dtype=torch.float64)
    counts = torch.zeros(len(subjects) * n_classes, dtype=torch.float64)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    for i in range(0, len(Xt), batch):
        z = model.embed(Xt[i:i + batch].to(device)).cpu().double()
        k = key[i:i + batch]
        sums.index_add_(0, k, z)
        counts.index_add_(0, k, torch.ones(len(k), dtype=torch.float64))
    sums = sums.view(len(subjects), n_classes, d)
    counts = counts.view(len(subjects), n_classes)
    glob = sums.sum(0) / counts.sum(0).clamp_min(1)[:, None]
    per = sums / counts.clamp_min(1)[..., None]
    missing = counts == 0                              # subject never did this gesture
    if missing.any():
        print(f"  note: {int(missing.sum())} (subject, gesture) pairs without training windows "
              f"-> global prototype used for those")
        per[missing] = glob.expand_as(per)[missing]
    return glob.float(), per.float(), subjects


@torch.no_grad()
def predict_all(model, X, subj, glob, per, subjects, device, batch=2048):
    """Predicted class index of every window for each method in METHODS (Euclidean distance)."""
    model.eval()
    assert np.isin(np.unique(subj), subjects).all(), "test subject without prototypes"
    sidx = torch.as_tensor(np.searchsorted(subjects, subj))
    glob, per = glob.to(device), per.to(device)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    out = {m: [] for m in METHODS}
    for i in range(0, len(Xt), batch):
        z, logits = model(Xt[i:i + batch].to(device))
        out["softmax"].append(logits.argmax(1).cpu())
        out["proto_global"].append(torch.cdist(z, glob).argmin(1).cpu())
        P = per[sidx[i:i + batch].to(device)]          # (B, C, d): each window's own subject
        out["proto_subject"].append(((z[:, None, :] - P) ** 2).sum(-1).argmin(1).cpu())
    return {m: torch.cat(v).numpy() for m, v in out.items()}


def fmt_hm(seconds):
    """Seconds -> 'h:mm' (rounded UP to the next minute, so short runs show 0:01, not 0:00)."""
    minutes = int(math.ceil(max(seconds, 0) / 60))
    return f"{minutes // 60}:{minutes % 60:02d}"


def finish_clock(seconds_left):
    """Wall-clock time the run should finish, e.g. '14:35' (or 'Sat 02:10' if not today)."""
    end = datetime.now() + timedelta(seconds=seconds_left)
    return end.strftime("%H:%M") if end.date() == datetime.now().date() else end.strftime("%a %H:%M")


class TrainClock:
    """Estimates the time left for the WHOLE run (all epochs of all seeds).

    * Early estimate: the first WARMUP_BATCHES batches are skipped (they are slower),
      then the next EARLY_BATCHES are timed (so you know within seconds, not after a
      full epoch). With too few batches per epoch it is given after epoch 1 instead.
    * After every epoch: average epoch time so far x epochs still to go.
    """
    WARMUP_BATCHES = 5
    EARLY_BATCHES = 30

    def __init__(self, n_seeds, epochs):
        self.total_epochs = n_seeds * epochs
        self.done_epochs = 0
        self.train_seconds = 0.0          # time spent inside epochs (excludes evaluation/saving)
        self.first_estimate = None

    def _announce(self, total, already_spent):
        self.first_estimate = total
        print(f"  estimated training time: {fmt_hm(total)} (h:mm) for {self.total_epochs} epochs "
              f"-> finishes ~{finish_clock(total - already_spent)}")

    def early(self, timed_seconds, n_timed, n_batches, proto_seconds, epoch_seconds_so_far):
        per_epoch = proto_seconds + timed_seconds / n_timed * n_batches
        self._announce(per_epoch * self.total_epochs, epoch_seconds_so_far)

    def epoch_done(self, epoch_seconds):
        self.done_epochs += 1
        self.train_seconds += epoch_seconds
        if self.first_estimate is None:            # too few batches for the early estimate
            self._announce(epoch_seconds * self.total_epochs, self.train_seconds)
        left = self.train_seconds / self.done_epochs * (self.total_epochs - self.done_epochs)
        return f"{fmt_hm(left)} left, done ~{finish_clock(left)}"

    def summary(self):
        est = f" (first estimate {fmt_hm(self.first_estimate)})" if self.first_estimate else ""
        print(f"total training time: {fmt_hm(self.train_seconds)} (h:mm){est}")


def train(Xtr, ytr, n_classes, e, epochs, lr, batch_size, lambda_, seed, device, log_rows, clock):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EMGAdapt(e=e, n_classes=n_classes).to(device)
    ds = TensorDataset(torch.as_tensor(Xtr, dtype=torch.float32),
                       torch.as_tensor(ytr, dtype=torch.long))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_fn = nn.CrossEntropyLoss()

    n_batches = len(loader)
    for epoch in range(1, epochs + 1):
        t_epoch = time.perf_counter()
        protos = class_prototypes(model, Xtr, ytr, n_classes, device) if lambda_ > 0 else None
        proto_seconds = time.perf_counter() - t_epoch
        model.train()
        tot_loss, correct, n = 0.0, 0, 0
        w, k = clock.WARMUP_BATCHES, clock.EARLY_BATCHES
        early_possible = clock.first_estimate is None and n_batches > w + k
        for b, (xb, yb) in enumerate(loader, 1):
            if early_possible and b == w + 1:
                t_timed = time.perf_counter()            # start timing after warm-up batches
            if early_possible and b == w + k + 1:
                now = time.perf_counter()
                clock.early(now - t_timed, k, n_batches, proto_seconds, now - t_epoch)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            emb, logits = model(xb)
            loss = ce_fn(logits, yb)
            if lambda_ > 0:
                proto = ((emb - protos[yb]) ** 2).sum(1).clamp_min(1e-12).sqrt().mean()
                loss = (1 - lambda_) * loss + lambda_ * proto
            loss.backward()
            opt.step()
            tot_loss += loss.item() * yb.numel()
            correct += (logits.argmax(1) == yb).sum().item()
            n += yb.numel()

        eta = clock.epoch_done(time.perf_counter() - t_epoch)
        print(f"  [seed {seed}] epoch {epoch:>3}/{epochs}  loss {tot_loss / n:.4f}  "
              f"train acc {correct / n:.4f}  | {eta}")
        log_rows.append({"seed": seed, "epoch": epoch, "loss": round(tot_loss / n, 6),
                         "train_acc_running": round(correct / n, 6)})
    return model


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def majority_pooled(preds, y_true, groups):
    """One vote per continuous segment: all windows of one gesture performance, or of
    one rest period (each rest period is its own vote, not merged per rep)."""
    votes, truth = defaultdict(list), {}
    for p, t, g in zip(preds, y_true, groups):
        votes[int(g)].append(int(p))
        truth[int(g)] = int(t)
    correct = sum(Counter(v).most_common(1)[0][0] == truth[k] for k, v in votes.items())
    return correct / max(len(votes), 1), len(votes)


def evaluate(preds, yte, subj_te, reps_te, seg_te, n_classes, rest_idx=None):
    res = {"window_acc": float((preds == yte).mean())}
    res["pooled_acc"], res["n_performances"] = majority_pooled(preds, yte, seg_te)

    if rest_idx is not None:                    # split the score into gestures vs rest
        g = yte != rest_idx
        res["gesture_window_acc"] = float((preds[g] == yte[g]).mean())
        res["gesture_pooled_acc"], res["n_gesture_perf"] = majority_pooled(preds[g], yte[g], seg_te[g])
        res["rest_window_acc"] = float((preds[~g] == yte[~g]).mean())
        res["rest_pooled_acc"], res["n_rest_periods"] = majority_pooled(preds[~g], yte[~g], seg_te[~g])
        res["rest_share"] = float((~g).mean())
        res["gesture_as_rest"] = float((preds[g] == rest_idx).mean())   # missed gestures

    per_class = np.array([(preds[yte == c] == c).mean() if (yte == c).any() else np.nan
                          for c in range(n_classes)])
    res["per_class"] = per_class
    res["balanced_acc"] = float(np.nanmean(per_class))

    res["per_subject"] = {}
    for s in np.unique(subj_te):
        m = subj_te == s
        pooled, n_perf = majority_pooled(preds[m], yte[m], seg_te[m])
        res["per_subject"][int(s)] = {"window_acc": float((preds[m] == yte[m]).mean()),
                                      "pooled_acc": pooled, "n_windows": int(m.sum()),
                                      "n_performances": n_perf}
    res["per_rep"] = {int(r): float((preds[reps_te == r] == yte[reps_te == r]).mean())
                      for r in np.unique(reps_te)}

    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(conf, (yte, preds), 1)
    res["confusion"] = conf
    return res


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Train / test split
# --------------------------------------------------------------------------- #
def make_split(reps, seg, split, train_reps, test_reps, test_frac, split_seed):
    """Return boolean masks (train, test) and a description.

    rep    : whole repetitions -> every window of a performance is on one side.
    window : each window is assigned at random (all 6 reps, all subjects); windows of the
             same performance end up on both sides. Consecutive windows are 10 ms apart and
             share 190 of 200 ms of signal, so most test windows have a near-copy in training.
    """
    if split == "rep":
        assert set(train_reps).isdisjoint(test_reps), "train and test repetitions overlap"
        trm, tem = np.isin(reps, train_reps), np.isin(reps, test_reps)
        dropped = ~(trm | tem)
        if dropped.any():
            print(f"note: {int(dropped.sum()):,} windows have a rep in neither set and are not used")
        assert set(np.unique(seg[trm])).isdisjoint(np.unique(seg[tem])), "a segment is in train AND test"
        return trm, tem, f"reps {list(test_reps)} held out"

    if split == "window":
        n = len(reps)
        order = np.random.default_rng(split_seed).permutation(n)
        tem = np.zeros(n, dtype=bool)
        tem[order[:int(round(n * test_frac))]] = True
        trm = ~tem
        return trm, tem, f"random {test_frac:.1%} of windows, all reps, split seed {split_seed}"

    raise ValueError(f"unknown split '{split}' (use 'rep' or 'window')")


def neighbour_in_train(seg, trm, tem):
    """Share of test windows whose directly neighbouring window (10 ms before or after,
    same segment) is a training window - how close the nearest training copy is."""
    same_next = seg[1:] == seg[:-1]                  # window i and i+1 are neighbours
    nb = np.zeros(len(seg), dtype=bool)
    nb[:-1] |= same_next & trm[1:]                   # next window is in train
    nb[1:] |= same_next & trm[:-1]                   # previous window is in train
    return float(nb[tem].mean())


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(subjects=SUBJECTS, exercises=EXERCISES, train_reps=TRAIN_REPS, test_reps=TEST_REPS,
        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, lambda_=LAMBDA, seeds=SEEDS,
        channels=CHANNELS, native_order=NATIVE_ORDER, include_rest=INCLUDE_REST,
        split=SPLIT, test_frac=TEST_FRAC, split_seed=SPLIT_SEED, use_cache=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | DATA_DIR: {P.DATA_DIR}")
    configure_channels(channels, native_order)

    X, y_gid, reps, subj, seg = load_all(subjects, exercises, include_rest, use_cache)

    # contiguous class ids 0..n-1 over ALL classes present (52 gestures, + rest if included)
    class_ids = sorted(int(c) for c in np.unique(y_gid))
    lookup = {c: i for i, c in enumerate(class_ids)}
    y = np.array([lookup[int(v)] for v in y_gid], dtype=np.int64)
    n_classes, e = len(class_ids), X.shape[1]
    rest_idx = lookup.get(REST_ID) if include_rest else None
    expected = 53 if include_rest else 52
    if n_classes != expected:
        print(f"WARNING: found {n_classes} classes, expected {expected} for full DB5.")

    trm, tem, split_desc = make_split(reps, seg, split, train_reps, test_reps, test_frac, split_seed)
    assert not (trm & tem).any(), "a window is in train AND test"
    Xtr, ytr = X[trm], y[trm]
    Xte, yte, reps_te, subj_te, seg_te = X[tem], y[tem], reps[tem], subj[tem], seg[tem]

    missing = sorted(set(range(n_classes)) - set(np.unique(ytr)))
    if missing:
        print(f"WARNING: classes with no training windows: {[class_name(class_ids[c]) for c in missing]}")
    print(f"subjects {[int(s) for s in np.unique(subj)]} | {n_classes} classes "
          f"({'52 gestures + rest' if include_rest else 'gestures only, no rest'}) | e = {e} channels")
    print(f"split: {split_desc} | train {len(ytr):,} windows | test {len(yte):,} windows")
    nb_share = neighbour_in_train(seg, trm, tem)
    print(f"test windows with a neighbouring window (10 ms away) in training: {nb_share:.1%}")
    if include_rest:
        print(f"rest share: train {np.mean(ytr == rest_idx):.1%}, test {np.mean(yte == rest_idx):.1%} of windows")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_DIR, stamp)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    config = {"split": split, "test_frac": round(test_frac, 6) if split == "window" else "",
              "split_seed": split_seed if split == "window" else "",
              "neighbour_in_train": round(nb_share, 6),
              "include_rest": include_rest, "channels": len(CHANNEL_LAYOUT),
              "native_order": native_order, "epochs": epochs, "lr": lr, "batch_size": batch_size, "lambda": lambda_,
              "train_reps": "-".join(map(str, train_reps)) if split == "rep" else "all",
              "test_reps": "-".join(map(str, test_reps)) if split == "rep" else "all",
              "n_classes": n_classes, "e": e}

    log_rows, subject_rows, class_rows, summary_rows = [], [], [], []
    clock = TrainClock(len(seeds), epochs)
    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        t0 = time.perf_counter()
        model = train(Xtr, ytr, n_classes, e, epochs, lr, batch_size, lambda_, seed, device,
                      log_rows, clock)
        train_min = (time.perf_counter() - t0) / 60

        # prototypes from the TRAINING windows only (test windows are never used to build them)
        glob, per, proto_subjects = prototype_sets(model, Xtr, ytr, subj[trm], n_classes, device)
        preds_by = predict_all(model, Xte, subj_te, glob, per, proto_subjects, device)
        results = {m: evaluate(p, yte, subj_te, reps_te, seg_te, n_classes, rest_idx)
                   for m, p in preds_by.items()}
        names = [class_name(g) for g in class_ids]

        print(f"\nTEST ({split_desc}, {n_classes}-way, chance {1 / n_classes:.3f}, "
              f"{len(yte):,} windows, {results['softmax']['n_performances']} "
              f"{'segments' if include_rest else 'performances'})")
        print(f"  {'method':<34}{'window':>8}{'balanced':>10}{'pooled':>8}   per rep (window)")
        for m, res in results.items():
            reps_txt = ", ".join(f"rep {r}: {a:.4f}" for r, a in res["per_rep"].items())
            print(f"  {METHODS[m]:<34}{res['window_acc']:>8.4f}{res['balanced_acc']:>10.4f}"
                  f"{res['pooled_acc']:>8.4f}   {reps_txt}")
        if include_rest:
            r0 = results["softmax"]
            print(f"  split by true class ({r0['rest_share']:.1%} of test windows are rest; "
                  f"{r0['n_gesture_perf']} gesture performances, {r0['n_rest_periods']} rest periods):")
            print(f"  {'method':<34}{'gest.win':>9}{'gest.pool':>10}{'rest.win':>9}{'rest.pool':>10}"
                  f"{'gest->rest':>11}")
            for m, res in results.items():
                print(f"  {METHODS[m]:<34}{res['gesture_window_acc']:>9.4f}{res['gesture_pooled_acc']:>10.4f}"
                      f"{res['rest_window_acc']:>9.4f}{res['rest_pooled_acc']:>10.4f}"
                      f"{res['gesture_as_rest']:>11.4f}")
        print("  per subject, window / pooled:")
        print("        " + "".join(f"{m:>20}" for m in results))
        for s in results["softmax"]["per_subject"]:
            cells = (f"{results[m]['per_subject'][s]['window_acc']:.4f} / "
                     f"{results[m]['per_subject'][s]['pooled_acc']:.4f}" for m in results)
            print(f"    S{s:>2} " + "".join(c.rjust(20) for c in cells))
        for m, res in results.items():
            worst = np.argsort(res["per_class"])[:5]
            print(f"  5 hardest gestures ({m}): " +
                  ", ".join(f"{names[c]} {res['per_class'][c]:.2f}" for c in worst))

        for m, res in results.items():
            for s, v in res["per_subject"].items():
                subject_rows.append({"seed": seed, "method": m, "subject": s,
                                     "window_acc": round(v["window_acc"], 6),
                                     "pooled_acc": round(v["pooled_acc"], 6),
                                     "n_windows": v["n_windows"], "n_performances": v["n_performances"]})
            ps = res["per_subject"].values()
            subject_rows.append({"seed": seed, "method": m, "subject": "MEAN",
                                 "window_acc": round(float(np.mean([v["window_acc"] for v in ps])), 6),
                                 "pooled_acc": round(float(np.mean([v["pooled_acc"] for v in ps])), 6),
                                 "n_windows": len(yte), "n_performances": res["n_performances"]})
            for c in range(n_classes):
                class_rows.append({"seed": seed, "method": m, "class_index": c, "class": names[c],
                                   "window_acc": round(float(res["per_class"][c]), 6),
                                   "n_test_windows": int((yte == c).sum())})
            with open(os.path.join(out_dir, f"confusion_seed{seed}_{m}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["true \\ pred"] + names)
                for c in range(n_classes):
                    w.writerow([names[c]] + res["confusion"][c].tolist())

            summary_rows.append({"seed": seed, "method": m, **config,
                                 "window_acc": round(res["window_acc"], 6),
                                 "balanced_acc": round(res["balanced_acc"], 6),
                                 "pooled_acc": round(res["pooled_acc"], 6),
                                 **{k: round(res[k], 6) for k in
                                    ("gesture_window_acc", "gesture_pooled_acc", "rest_window_acc",
                                     "rest_pooled_acc", "rest_share", "gesture_as_rest") if k in res},
                                 **{f"rep{r}_window_acc": round(a, 6) for r, a in res["per_rep"].items()},
                                 "train_minutes": round(train_min, 2)})

        ckpt = os.path.join(MODELS_DIR, f"{stamp}_seed{seed}.pt")
        torch.save({"state_dict": model.state_dict(), "e": e, "n_classes": n_classes,
                    "class_ids": class_ids, "class_names": names,
                    "train_reps": list(train_reps), "test_reps": list(test_reps),
                    "split": split, "test_frac": test_frac, "split_seed": split_seed,
                    "subjects": [int(s) for s in np.unique(subj)], "channel_layout": CHANNEL_LAYOUT,
                    "config": config}, ckpt)
        print(f"  saved model -> {ckpt}")

    if len(seeds) > 1:
        print(f"\nMEAN +/- STD over seeds {list(seeds)}:")
        for m in METHODS:
            rows = [r for r in summary_rows if r["method"] == m]
            txt = "  ".join(f"{k} {np.mean([r[k] for r in rows]):.4f} +/- {np.std([r[k] for r in rows]):.4f}"
                            for k in ("window_acc", "balanced_acc", "pooled_acc",
                                      "gesture_window_acc", "rest_window_acc") if k in rows[0])
            print(f"  {METHODS[m]:<34}{txt}")

    clock.summary()
    write_csv(os.path.join(out_dir, "train_log.csv"), log_rows)
    write_csv(os.path.join(out_dir, "per_subject.csv"), subject_rows)
    write_csv(os.path.join(out_dir, "per_class.csv"), class_rows)
    write_csv(os.path.join(out_dir, "summary.csv"), summary_rows)
    print(f"\nsaved results -> {out_dir}/")
    return summary_rows


def main():
    ap = argparse.ArgumentParser(description="52-gesture DB5 model, all subjects, reps 2 & 5 held out")
    ap.add_argument("--data-dir", default=None, help="DB5 folder (default: Preprocessing.DATA_DIR)")
    ap.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS))
    ap.add_argument("--exercises", type=int, nargs="+", default=list(EXERCISES))
    ap.add_argument("--train-reps", type=int, nargs="+", default=list(TRAIN_REPS))
    ap.add_argument("--test-reps", type=int, nargs="+", default=list(TEST_REPS))
    ap.add_argument("--split", choices=("rep", "window"), default=SPLIT,
                    help="rep = hold out whole repetitions (default); window = random window split")
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC,
                    help="window split: share of windows used for testing (default 1/3)")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED,
                    help="window split: seed of the random split (same for all training seeds)")
    ap.add_argument("--include-rest", action="store_true",
                    help="add rest as one extra class (53-way, as in the paper)")
    ap.add_argument("--channels", type=int, choices=(8, 16), default=CHANNELS)
    ap.add_argument("--native-order", action="store_true",
                    help="keep the file's channel order instead of interleaving the two bands")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=LAMBDA)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    if a.data_dir:
        P.DATA_DIR = a.data_dir          # filename_for reads this module global
    run(subjects=tuple(a.subjects), exercises=tuple(a.exercises),
        train_reps=tuple(a.train_reps), test_reps=tuple(a.test_reps),
        epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, lambda_=a.lambda_,
        seeds=tuple(a.seeds), channels=a.channels, native_order=a.native_order,
        include_rest=a.include_rest, split=a.split, test_frac=a.test_frac, split_seed=a.split_seed,
        use_cache=not a.no_cache)


if __name__ == "__main__":
    main()