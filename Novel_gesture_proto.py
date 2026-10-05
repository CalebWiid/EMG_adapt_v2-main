"""
This script is adds a Hybrid loss function to the CNN model and builds a prototype model for each gesture in the embedding space. This means that for each gesture there is a general/prototype area in the embedding space that is specific to a gesture. As a result this script gets rid of the softmax layer to allow for new/novel gestures to be given to the model and then classified based on the distance to the prototype of each gesture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from datetime import datetime
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
 
from Preprocessing import process_files, filename_for, FS
from CNN_model import EMGAdapt

# ---- configuration ----
SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)                     # development on subject 1; a parameter for scale-up
EXERCISES = (1, 2, 3)
EXP_BASE = 1000                     # composite gesture id = exercise*1000 + restimulus
N_NOVEL = 12                        # held-out novel gestures (40 base / 12 novel)
ENROLL_REPS = (1, 3, 4)            # build novel prototypes from these reps
TEST_REPS = (2, 5, 6)             # test novel recognition on these reps
LAMBDA = 0.0                     # hybrid-loss weight on the prototype term
LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 256

#------Load Data---------
def load_subject_gestures(subjects=SUBJECTS, exercises=EXERCISES, fs=FS): 
    X_all, y_all, rep_all, subj_all = [], [], [], []
    for s in subjects: 
        for ex in exercises:
            try:
                X, y, reps = process_files([filename_for(s, ex)], fs=fs)
            except FileNotFoundError: 
                print(f" [Skip] missing files: S{s}_E{ex}")
                continue 
            if X.shape[0] == 0: 
                continue
            X_all.append(X)
            y_all.append(y + ex*EXP_BASE)  # make ids unique across exercises
            rep_all.append(reps)
            subj_all.append(np.full(X.shape[0], s, dtype=np.int64))  # subject id
    if not X_all: 
        raise SystemExit("No data loaded - check DATA_DIR / filenames.")
    return (np.concatenate(X_all), np.concatenate(y_all),
            np.concatenate(rep_all), np.concatenate(subj_all))

#-------Gesture level splt-----------
def split_base_novel(y, n_novel=N_NOVEL, seed=0): 
    classes = np.unique(np.asarray(y))
    n = len(classes)
    if n_novel >= n:
        raise ValueError(f"n_novel={n_novel} but only {n} gesture classes present")
    rng = np.random.default_rng(seed)
    novel_idx = rng.choice(n, size=n_novel, replace=False)     # choose INDICES 0..n-1
    is_novel = np.zeros(n, dtype=bool)
    is_novel[novel_idx] = True
    novel = classes[is_novel]
    base = classes[~is_novel]
    assert len(np.intersect1d(base, novel)) == 0, "base/novel overlap!"
    return [int(c) for c in base], [int(c) for c in novel]

#-------Train on base gestures--------
def make_base_dataset(X, y, base_classes): 
    base_set = set(int(c) for c in base_classes)
    mask = np.array([v in base_set for v in y])
    Xb, yb = X[mask], y[mask]
    lookup = {int(c): i for i, c in enumerate(base_classes)}
    yb_remapped = np.array([lookup[int(v)] for v in yb], dtype=np.int64)
    assert set(np.unique(yb_remapped).tolist()) <= set(range(len(base_classes)))
    return Xb, yb_remapped, lookup 

class CCADataset(Dataset): 
    def __init__(self, X, y):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
 
    def __len__(self):
        return self.y.shape[0]
 
    def __getitem__(self, i):
        return self.X[i], self.y[i]

#------Embedding + prototype helpers------
@torch.no_grad()
def embed_all(model, X, device, batch=512): 
    """Frozen-model embeddings for every row of X -> (N, 128) cpu tensor."""
    model.eval()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    out = [model.embed(Xt[i:i + batch].to(device)).cpu() for i in range(0, len(Xt), batch)]
    return torch.cat(out)

def class_prototypes(embs, y, class_list): 
    """Mean embedding per class -> (len(class_list), 128) tensor aligned to class_list."""
    D = embs.shape[1]
    protos = torch.zeros(len(class_list), D)
    y = np.asarray(y)
    for i, c in enumerate(class_list):
        m = (y == c)
        if m.any():
            protos[i] = embs[torch.as_tensor(m)].mean(0)
    return protos

#------Training with hybrid loss------
def train_embedding(model, Xtr, ytr, n_base, epochs=EPOCHS, lr=LR,
                    lambda_=LAMBDA, batch_size=BATCH_SIZE, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(CCADataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    ce_fn = nn.CrossEntropyLoss()
    base_class_ids = list(range(n_base))
 
    history = []
    for epoch in range(1, epochs + 1):
        # prototypes for the loss: base-class centroids over the WHOLE train set,
        # recomputed each epoch with the current (frozen for this step) embedding.
        protos = class_prototypes(embed_all(model, Xtr, device), ytr, base_class_ids).to(device)
 
        model.train()
        tot = tot_ce = tot_pr = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            emb, logits = model(xb)
            ce = ce_fn(logits, yb)
            targets = protos[yb]                                   # (B,128) detached
            proto = ((emb - targets) ** 2).sum(1).clamp_min(1e-12).sqrt().mean()
            loss = (1 - lambda_) * ce + lambda_ * proto
            loss.backward()
            optimiser.step()
            n = yb.numel()
            tot += loss.item() * n; tot_ce += ce.item() * n; tot_pr += proto.item() * n
        N = len(loader.dataset)
        print(f"epoch {epoch:>2}/{epochs}  loss {tot/N:.4f}  ce {tot_ce/N:.4f}  proto {tot_pr/N:.4f}")
        history.append({"epoch": epoch, "loss": round(tot/N, 6),
                        "ce": round(tot_ce/N, 6), "proto": round(tot_pr/N, 6)})
    return model, history

#------Enrollment and nearest-protoypes evaluation------
def enroll(model, X, y, reps, novel_classes, enroll_reps, device):
    """Build one prototype per novel gesture from its enrollment repetitions.
    Returns (proto_matrix (12,128) tensor, novel_classes list) aligned by row."""
    mask = np.isin(reps, enroll_reps) & np.isin(y, novel_classes)
    protos = class_prototypes(embed_all(model, X[mask], device), y[mask], novel_classes)
    return protos.to(device), list(novel_classes)
 
 
def majority_pooled(preds, y_true, subj, reps):
    """Gesture-level accuracy: majority vote per (subject, rep, true label)."""
    votes = defaultdict(list); truth = {}
    for i in range(len(preds)):
        key = (int(subj[i]), int(reps[i]), int(y_true[i]))
        votes[key].append(int(preds[i])); truth[key] = int(y_true[i])
    correct = sum(Counter(v).most_common(1)[0][0] == truth[k] for k, v in votes.items())
    return correct / len(votes), len(votes)

@torch.no_grad()
def evaluate_novel(model, X, y, reps, subj, novel_classes, proto_matrix, test_reps, device):
    """Classify held-out novel-gesture windows by nearest prototype, then pool."""
    mask = np.isin(reps, test_reps) & np.isin(y, novel_classes)
    Xq, yq, rq, sq = X[mask], y[mask], reps[mask], subj[mask]
    embs = embed_all(model, Xq, device)                             # (Nq,128)
    d = torch.cdist(embs, proto_matrix.cpu())                       # (Nq,12)
    idx = d.argmin(1).numpy()
    class_arr = np.asarray(novel_classes)
    preds = class_arr[idx]                                          # predicted composite labels

    win_acc = float((preds == yq).mean())
    pooled_acc, n_perf = majority_pooled(preds, yq, sq, rq)

    per_subject = {}
    for s in np.unique(sq): 
        m = sq == s
        w =  float((preds[m] == yq[m]).mean())
        p, _ = majority_pooled(preds[m], yq[m], sq[m], rq[m])
        per_subject[int(s)] = {"window_acc": w, "pooled_acc": p}

    return {"window_acc": win_acc, "pooled_acc": pooled_acc, "n_performances": n_perf, "per_subject": per_subject}

#--------save to csv--------
def save_rows(rows, path, meta):
    fieldnames = list(meta.keys()) + list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({**meta, **r})

#-------run the whole pipeline--------
def run(subjects=SUBJECTS, exercises=EXERCISES, n_novel=N_NOVEL,
        enroll_reps=ENROLL_REPS, test_reps=TEST_REPS, lambda_=LAMBDA,
        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    assert len(set(enroll_reps) & set(test_reps)) == 0, "enroll/test reps overlap!"
 
    X, y, reps, subj = load_subject_gestures(subjects, exercises)
    base_classes, novel_classes = split_base_novel(y, n_novel=n_novel, seed=seed)
    # guardrail: no novel gesture appears in the base training set
    assert len(set(base_classes) & set(novel_classes)) == 0
 
    Xtr, ytr, _ = make_base_dataset(X, y, base_classes)
    n_base = len(base_classes)
    print(f"subjects {list(subjects)}  |  {n_base} base gestures, {len(novel_classes)} novel")
    print(f"  base train windows: {Xtr.shape[0]}  (all reps of base gestures)")
    print(f"  enroll reps {enroll_reps}  test reps {test_reps}")
 
    e = X.shape[1]
    model = EMGAdapt(e=e, n_classes=n_base)
    model, history = train_embedding(model, Xtr, ytr, n_base, epochs=epochs,
                                     lr=lr, lambda_=lambda_, batch_size=batch_size, device=device)
 
    # Piece 2: enroll novel gestures, then classify by nearest prototype
    proto_matrix, novel_list = enroll(model, X, y, reps, novel_classes, enroll_reps, device)
    result = evaluate_novel(model, X, y, reps, subj, novel_list, proto_matrix, test_reps, device)

    ps = result["per_subject"]
    subject_rows = []
    for s in sorted(ps):
        subject_rows.append({
            "subject_id": s,
            "window_accuracy": round(ps[s]["window_acc"], 6),
            "pooled_accuracy": round(ps[s]["pooled_acc"], 6),
        })
    # last row = average across subjects
    mean_win = np.mean([r["window_accuracy"] for r in subject_rows])
    mean_pool = np.mean([r["pooled_accuracy"] for r in subject_rows])
    subject_rows.append({
        "subject_id": "AVERAGE",
        "window_accuracy": round(float(mean_win), 6),
        "pooled_accuracy": round(float(mean_pool), 6),
    })
 
    chance = 1.0 / len(novel_classes)
    print(f"\nNOVEL-GESTURE recognition ({len(novel_classes)}-way, unseen gestures):")
    print(f"  window-level : {result['window_acc']:.4f}")
    print(f"  gesture-level (majority pooled): {result['pooled_acc']:.4f}  "
          f"over {result['n_performances']} performances   (chance {chance:.3f})")
 
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "lambda": lambda_, "lr": lr, "epochs": epochs, "seed": seed,
    }
    path = f"results_novel_proto_{stamp}.csv"
    save_rows(subject_rows, path, meta)
    print(f"saved results -> {path}")
    return model, result
 

#---------Main-------- 
if __name__ == "__main__":
    run()