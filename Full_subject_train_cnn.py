"""
Full_subject_train_cnn.py - step-1 supervised training on ALL DB5 subjects.

POOLED CROSS-SESSION protocol:
    * every subject's repetitions {1,3,4,6} -> train
    * every subject's repetitions {2,5}     -> test
So all subjects appear in both train and test, but never the same repetition
(no leakage). This measures the cross-SESSION question with maximum data
diversity, and is the direct route to the paper's "Ours (w/o adaptation)"
numbers. It is NOT cross-user: the model has seen every test subject during
training. Leave-one-subject-out (cross-user) is a later change - the subject ids
tracked here are what you'll switch the split on when you get to it.

Label consistency: each gesture is encoded as a composite id (exercise*1000 +
restimulus) that is identical for the same gesture across every subject, then the
whole pool is remapped to 0..C-1 once. This avoids any per-subject label drift.

Reports per-window accuracy AND gesture-level accuracy via majority pooling.
Because the pool spans subjects, a gesture performance is keyed by
(subject, repetition, true label). Results are logged to CSV (per-epoch curve +
a per-subject gesture-level breakdown) and the trained model is saved.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv
import os
import time
from datetime import datetime
from collections import defaultdict, Counter

from Preprocessing import process_files, filename_for, FS
from CNN_model import EMGAdapt

TEST_REPS = (2, 5)                       # cross-session test repetitions (DB5)
DB5_SUBJECTS = tuple(range(1, 11))       # DB5 has 10 subjects
EXP_BASE = 1000                          # composite label base: exercise*1000 + gesture


# --------------------------------------------------------------------------- #
# 1. Load every subject into one pool with cross-subject-consistent labels
# --------------------------------------------------------------------------- #
def load_all_subjects(subjects=DB5_SUBJECTS, exercises=(1, 2, 3), fs=FS):
    """Load all (subject, exercise) files into one pool.

    Returns X, y_composite, reps, subj  (all aligned along axis 0), where
    y_composite = exercise*EXP_BASE + restimulus  (consistent across subjects).
    """
    X_all, y_all, rep_all, subj_all = [], [], [], []
    for s in subjects:
        for ex in exercises:
            try:
                X, y, reps = process_files([filename_for(s, ex)], fs=fs)
            except FileNotFoundError:
                print(f"  [skip] missing file: S{s}_E{ex}")
                continue
            if X.shape[0] == 0:
                continue
            comp = ex * EXP_BASE + y                     # same gesture -> same id, all subjects
            X_all.append(X)
            y_all.append(comp)
            rep_all.append(reps)
            subj_all.append(np.full(X.shape[0], s, dtype=np.int64))

    if not X_all:
        raise SystemExit("No data loaded - check DATA_DIR / filenames.")
    return (np.concatenate(X_all, axis=0),
            np.concatenate(y_all, axis=0),
            np.concatenate(rep_all, axis=0),
            np.concatenate(subj_all, axis=0))


def remap_labels(y):
    """Map composite gesture ids to a contiguous 0..C-1 range (done once, on the
    combined pool). Returns (y_mapped, class_ids)."""
    class_ids = np.unique(y)
    lookup = {c: i for i, c in enumerate(class_ids)}
    y_mapped = np.array([lookup[v] for v in y], dtype=np.int64)
    return y_mapped, class_ids


# --------------------------------------------------------------------------- #
# 2. Dataset / DataLoader
# --------------------------------------------------------------------------- #
class CCADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loaders(Xtr, ytr, Xte, yte, batch_size=256):
    train_loader = DataLoader(CCADataset(Xtr, ytr), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(CCADataset(Xte, yte), batch_size=512, shuffle=False)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# 3. Evaluate / majority pooling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device):
    """Per-WINDOW accuracy (fraction in [0,1])."""
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, logits = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.numel()
    return correct / total if total else 0.0


@torch.no_grad()
def accuracy_on(model, X, y, device, batch=512):
    """Per-window accuracy on raw (X, y) arrays, batched."""
    model.eval()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
    correct = 0
    for i in range(0, len(yt), batch):
        xb = Xt[i:i + batch].to(device)
        yb = yt[i:i + batch].to(device)
        _, logits = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
    return correct / len(yt) if len(yt) else 0.0


@torch.no_grad()
def predict_all(model, X, device, batch=512):
    """Per-window predicted labels for the whole array X -> (N,) int array."""
    model.eval()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    preds = []
    for i in range(0, len(Xt), batch):
        _, logits = model(Xt[i:i + batch].to(device))
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def majority_pooled(preds, y_true, subj, reps):
    """GESTURE-LEVEL accuracy via majority pooling across subjects.

    A gesture performance is keyed by (subject, repetition, true label) -- subject
    MUST be in the key here because the pool spans people. Returns
    (overall_acc, per_subject_acc dict, n_performances).
    """
    votes = defaultdict(list)
    truth = {}
    for i in range(len(preds)):
        key = (int(subj[i]), int(reps[i]), int(y_true[i]))
        votes[key].append(int(preds[i]))
        truth[key] = int(y_true[i])

    correct = 0
    subj_correct = defaultdict(int)
    subj_total = defaultdict(int)
    for key, plist in votes.items():
        voted = Counter(plist).most_common(1)[0][0]
        hit = int(voted == truth[key])
        correct += hit
        subj_correct[key[0]] += hit
        subj_total[key[0]] += 1

    overall = correct / len(votes)
    per_subject = {s: subj_correct[s] / subj_total[s] for s in subj_total}
    return overall, per_subject, len(votes)


# --------------------------------------------------------------------------- #
# 4. Train (with per-epoch history for CSV logging)
# --------------------------------------------------------------------------- #
# fixed learning rate of 1e-4 (tuned; the decaying schedule underperformed here)
def train(model, train_loader, val_loader, epochs=20, lr=1e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []                                      # one row per epoch (CSV log)
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            _, logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimiser.step()
            running += loss.item() * yb.numel()
        train_loss = running / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)         # fraction in [0,1]
        epoch_time = time.perf_counter() - t0
        eta = epoch_time * (epochs - epoch)
        print(f"epoch {epoch:>2}/{epochs}  train_loss {train_loss:.4f}  "
              f"val_acc {val_acc:.4f}  time {epoch_time:.1f}s  eta {eta/60:.1f}m")
        history.append({"epoch": epoch,
                        "train_loss": round(train_loss, 6),
                        "val_acc": round(val_acc, 6),
                        "time_s": round(epoch_time, 2)})
    return model, history


# --------------------------------------------------------------------------- #
# 5. CSV helpers
# --------------------------------------------------------------------------- #
def save_rows(rows, path, meta):
    """Write a list of dict rows to CSV, prefixing each with run metadata columns.
    Used for both the per-epoch curve and the per-subject breakdown so files from
    different experiments/runs concatenate and compare cleanly."""
    fieldnames = list(meta.keys()) + list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**meta, **row})


# --------------------------------------------------------------------------- #
# 6. Driver
# --------------------------------------------------------------------------- #
def run(subjects=DB5_SUBJECTS, exercises=(1, 2, 3), epochs=20, batch_size=256,
        lr=1e-4, save_model="full_subject_cnn.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    X, y_comp, reps, subj = load_all_subjects(subjects, exercises)
    y, class_ids = remap_labels(y_comp)                       # remap ONCE on the pool

    test_mask = np.isin(reps, TEST_REPS)                      # split by repetition
    Xtr, ytr = X[~test_mask], y[~test_mask]
    Xte, yte = X[test_mask], y[test_mask]
    subj_te, reps_te = subj[test_mask], reps[test_mask]       # kept for pooling

    e = X.shape[1]
    n_classes = len(class_ids)
    print(f"subjects {list(subjects)}  exercises {exercises}")
    print(f"  {X.shape[0]} windows, e={e}, {n_classes} gestures, "
          f"{len(np.unique(subj))} subjects")
    print(f"  train windows: {Xtr.shape[0]}   test windows: {Xte.shape[0]}")

    train_loader, val_loader = make_loaders(Xtr, ytr, Xte, yte, batch_size)
    model = EMGAdapt(e=e, n_classes=n_classes)
    model, history = train(model, train_loader, val_loader,
                           epochs=epochs, lr=lr, device=device)

    if save_model:
        torch.save(model.state_dict(), save_model)
        print(f"saved model -> {save_model}")

    # ---- final metrics ----
    win_acc = evaluate(model, val_loader, device)
    preds = predict_all(model, Xte, device)
    g_overall, g_per_subject, n_inst = majority_pooled(preds, yte, subj_te, reps_te)

    print(f"\nwindow-level  val acc                : {win_acc:.4f}")
    print(f"gesture-level (majority pooled) acc  : {g_overall:.4f}  "
          f"over {n_inst} gesture performances")
    print("  per-subject accuracy (window | gesture):")
    per_subject_rows = []
    for s in sorted(g_per_subject):
        m = subj_te == s
        w = accuracy_on(model, Xte[m], yte[m], device)
        p = g_per_subject[s]
        print(f"    subject {int(s):>2}:  {w:.4f}  |  {p:.4f}")
        per_subject_rows.append({"subject": int(s),
                                 "window_acc": round(w, 6),
                                 "pooled_acc": round(p, 6),
                                 "n_windows": int(m.sum())})

    # ---- save CSVs (per-epoch curve + per-subject breakdown) ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "experiment": "full_subject_pooled_crosssession",
        "n_subjects": len(np.unique(subj)),
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "n_classes": n_classes,
        "final_window_acc": round(win_acc, 6),
        "final_pooled_acc": round(g_overall, 6),
    }
    curve_path = f"results_full_{stamp}.csv"
    persubj_path = f"results_full_persubject_{stamp}.csv"
    save_rows(history, curve_path, meta)
    save_rows(per_subject_rows, persubj_path, meta)
    print(f"saved per-epoch curve   -> {curve_path}")
    print(f"saved per-subject table -> {persubj_path}")

    return model


if __name__ == "__main__":
    run(subjects=DB5_SUBJECTS, exercises=(1, 2, 3), epochs=20)