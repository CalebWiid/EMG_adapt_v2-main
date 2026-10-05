
"""
Train.py - step-1 supervised training of the EMG-Adapt CNN on DB5 CCA features.

This is the "Ours (w/o adaptation)" baseline: cross-entropy + Adam, NO prototypes
and NO meta-learning yet. Its job is to confirm the Preprocessing -> Model pipeline
trains, and to reproduce the paper's non-adaptation numbers as a Phase-0 checkpoint.

Data flow:
    Preprocessing.process_files(paths) -> (X, y, reps)
        X    : (n_windows, e)  CCA feature vectors
        y    : (n_windows,)    gesture ids (restimulus values, 1..N; rest excluded)
        reps : (n_windows,)    repetition ids  <-- used to split without leakage

Two things this file gets right and a naive script gets wrong:
  1. SPLIT BY REPETITION, never randomly over windows. Overlapping windows from the
     same repetition would otherwise leak across train/test and inflate accuracy.
     Cross-session protocol: test on repetitions {2, 5} (Atzori split for DB5).
  2. LABELS remapped to a contiguous 0..C-1 range for CrossEntropyLoss, with a
     per-exercise offset so E1/E2/E3 gesture numbering doesn't collide.
"""
import numpy as np
import torch
import torch.nn as nn
import csv 
import os 
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter

from Preprocessing import process_files, filename_for, FS
from CNN_model import EMGAdapt 



TEST_REPS = (2, 5)                          # cross-session test repetitions (DB5)


# --------------------------------------------------------------------------- #
# 1. Loading DB5 into (X, y, reps) with collision-free labels
# --------------------------------------------------------------------------- #
def load_db5_subject(subject, exercises=(1, 2, 3), fs=FS):
    """Load one subject's exercises and stack them, offsetting gesture ids so the
    same restimulus number in different exercises stays a distinct gesture."""
    X_all, y_all, rep_all = [], [], []
    offset = 0
    for ex in exercises:
        X, y, reps = process_files([filename_for(subject, ex)], fs=fs)
        if X.shape[0] == 0:
            continue
        y = y + offset                      # make ids unique across exercises
        offset = int(y.max()) + 1
        X_all.append(X); y_all.append(y); rep_all.append(reps)
    if not X_all:
        raise SystemExit(f"No data loaded for subject {subject}, exercises {exercises}.")
    return (np.concatenate(X_all, axis=0),
            np.concatenate(y_all, axis=0),
            np.concatenate(rep_all, axis=0))


def remap_labels(y):
    """Map arbitrary gesture ids to a contiguous 0..C-1 range.
    Returns (y_mapped, class_ids) where class_ids[i] is the original id of class i."""
    class_ids = np.unique(y)
    lookup = {c: i for i, c in enumerate(class_ids)}
    y_mapped = np.array([lookup[v] for v in y], dtype=np.int64)
    return y_mapped, class_ids


def split_by_rep(X, y, reps, test_reps=TEST_REPS):
    """Leakage-safe split: whole repetitions go to train or test, never both."""
    test_mask = np.isin(reps, test_reps)
    train = (X[~test_mask], y[~test_mask])
    test = (X[test_mask], y[test_mask], reps[test_mask])
    return train, test


# --------------------------------------------------------------------------- #
# 2. Dataset / DataLoader
# --------------------------------------------------------------------------- #
class CCADataset(Dataset):
    """Wraps (X, y) numpy arrays as float32 features / int64 labels."""
    def __init__(self, X, y):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)   # (N, e)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)      # (N,)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loaders(Xtr, ytr, Xte, yte, batch_size=256):
    train_loader = DataLoader(CCADataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CCADataset(Xte, yte), batch_size=512, shuffle=False)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# 3. Train / evaluate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, logits = model(xb)                 # forward returns (embedding, logits)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.numel()
    return correct / total if total else 0.0

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

def majority_pooled(preds, y_true, reps):
    """GESTURE-LEVEL accuracy via majority pooling (single subject).
 
    Windows of one performance -- keyed by (repetition, true label) -- are collapsed
    into one prediction by majority vote. Returns (accuracy, n_performances).
    """
    votes = defaultdict(list)
    truth = {}
    for i in range(len(preds)):
        key = (int(reps[i]), int(y_true[i]))
        votes[key].append(int(preds[i]))
        truth[key] = int(y_true[i])           # constant within a performance
    correct = sum(Counter(v).most_common(1)[0][0] == truth[k] for k, v in votes.items())
    return correct / len(votes), len(votes)


#fixed learnign rate of 0.0001
def train(model, train_loader, val_loader,  Xte, yte, reps_te, epochs=20, lr=0.0001, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
 
    history = []                                   # one row per epoch (for the CSV log)
    for epoch in range(1, epochs + 1):
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
        val_acc = evaluate(model, val_loader, device)
        preds = predict_all(model, Xte, device)                 # per-window preds
        pooled_acc, _ = majority_pooled(preds, yte, reps_te)    # gesture-level
        print(f"epoch {epoch:>2}/{epochs}  train_loss {train_loss:.4f}  "
              f"val_acc {val_acc:.4f}  pooled_acc {pooled_acc:.4f}")
        history.append({"epoch": epoch,
                        "train_loss": round(train_loss, 6),
                        "val_acc": round(val_acc, 6),
                        "pooled_acc": round(pooled_acc, 6)})
    return model, history

def save_history(history, path, meta):
    """Write per-epoch history to a CSV, prefixing each row with run metadata.
    meta columns (experiment name, subject, hyperparams, final accuracies) repeat
    on every row so files from different experiments concatenate and compare cleanly."""
    fieldnames = list(meta.keys()) + list(history[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({**meta, **row})


# --------------------------------------------------------------------------- #
# 4. Driver
# --------------------------------------------------------------------------- #
def run(subject=1, exercises=(1, 2, 3), epochs=20, batch_size=256, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    X, y, reps = load_db5_subject(subject, exercises)
    y, class_ids = remap_labels(y)                              # 0..C-1
    (Xtr, ytr), (Xte, yte, reps_te) = split_by_rep(X, y, reps)  # test on reps {2,5}
 
    e = X.shape[1]
    n_classes = len(class_ids)
    print(f"subject {subject}  exercises {exercises}: "
          f"{X.shape[0]} windows, e={e}, {n_classes} gestures")
    print(f"  train windows: {Xtr.shape[0]}   test windows: {Xte.shape[0]}")
 
    train_loader, val_loader = make_loaders(Xtr, ytr, Xte, yte, batch_size)
    model = EMGAdapt(e=e, n_classes=n_classes)
    model, history = train(model, train_loader, val_loader, Xte, yte, reps_te,
                  epochs=epochs, lr=lr, device=device)
 
    # ---- report BOTH metrics on the same test set ----
    win_acc = evaluate(model, val_loader, device)
    preds = predict_all(model, Xte, device)
    g_acc, n_inst = majority_pooled(preds, yte, reps_te)
    print(f"\nwindow-level  val acc               : {win_acc:.4f}")
    print(f"gesture-level (majority pooled) acc : {g_acc:.4f}  over {n_inst} gesture performances")
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results_single_subj{subject}_{stamp}.csv"
    meta = {
        "experiment": "single_subject_baseline",
        "subject": subject,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "n_classes": n_classes,
        "final_window_acc": round(win_acc, 6),
        "final_pooled_acc": round(g_acc, 6),
    }
    save_history(history, csv_path, meta)
    print(f"saved results -> {csv_path}")
    
    return model


if __name__ == "__main__":
    run(subject=1, exercises=(1, 2, 3), epochs=20)

    