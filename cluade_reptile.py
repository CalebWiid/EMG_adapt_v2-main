"""
Reptile_learning.py - Piece 3: Reptile meta-learning for novel-gesture recognition.

IDEA
  Instead of training the embedding once in bulk (Novel_gesture_proto.py), train it
  across many small EPISODES that each simulate deployment: "here are N gestures with
  K examples each - learn to separate them". Repeating this teaches an initialisation
  whose embedding places UNSEEN gestures well from few examples.

PROTOCOL
  * Meta-training tasks are sampled ONLY from the 40 BASE gestures (5-way, 5-shot).
  * The 12 NOVEL gestures are held out and used only for evaluation
    (12-way, 3-rep enrollment) - identical to the Piece-2 baseline, so the number
    is directly comparable to that ~0.78 pooled baseline.

REPTILE (first-order):
  each meta-iteration m:
    1. sample a task (5 base gestures, 5 support windows each)
    2. clone the model -> inner-adapt it with the hybrid loss for `inner_steps` steps
    3. move the global weights toward the adapted weights:  theta <- theta + beta*(theta' - theta)
       with step-size decay  beta = (1 - m/M) * s     (paper Eq. 11 / Eq. 12)
"""
import numpy as np
import torch
import torch.nn as nn
import csv
import copy
import os
import time
from datetime import datetime
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader

from Preprocessing import process_files, filename_for, FS
from CNN_model import EMGAdapt

# ---- configuration ----
SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
EXERCISES = (1, 2, 3)
EXP_BASE = 1000
N_NOVEL = 12                      # held-out novel gestures (40 base / 12 novel)

# meta-training (tasks sampled from base gestures)
N_WAY = 12                         # gestures per task
K_SHOT = 5                        # support PERFORMANCES per class per task (across subjects)
META_ITERS = 200                 # M: number of meta-iterations
INNER_STEPS = 3                  # gradient steps on the support set per task
INNER_LR = 1e-4                  # inner-loop Adam lr
STEP_SIZE = 0.1                  # s: initial Reptile step size (beta = (1-m/M)*s)
LAMBDA = 0.25                    # prototype-loss weight for META-training (the meta setting)
FREEZE_FEATURES = True

# Phase 1: supervised pre-training of the embedding (matches your Piece-2 model)
PRETRAIN_EPOCHS = 20
PRETRAIN_LR = 1e-4
PRETRAIN_LAMBDA = 0.0            # your finding: plain cross-entropy pre-trains the best embedding

# evaluation (novel gestures) - matches the Piece-2 baseline exactly
ENROLL_REPS = (1, 3, 4)
TEST_REPS = (2, 5, 6)
EVAL_EVERY = 50                  # run novel-gesture eval every N meta-iterations


# --------------------------------------------------------------------------- #
# Data (reused from the Piece-2 script)
# --------------------------------------------------------------------------- #
def load_subject_gestures(subjects=SUBJECTS, exercises=EXERCISES, fs=FS):
    X_all, y_all, rep_all, subj_all = [], [], [], []
    for s in subjects:
        for ex in exercises:
            try:
                X, y, reps = process_files([filename_for(s, ex)], fs=fs)
            except FileNotFoundError:
                print(f" [skip] missing files: S{s}_E{ex}")
                continue
            if X.shape[0] == 0:
                continue
            X_all.append(X)
            y_all.append(y + ex * EXP_BASE)
            rep_all.append(reps)
            subj_all.append(np.full(X.shape[0], s, dtype=np.int64))
    if not X_all:
        raise SystemExit("No data loaded - check DATA_DIR / filenames.")
    return (np.concatenate(X_all), np.concatenate(y_all),
            np.concatenate(rep_all), np.concatenate(subj_all))
 
 
def split_base_novel(y, n_novel=N_NOVEL, seed=0):
    classes = np.unique(np.asarray(y))
    n = len(classes)
    if n_novel >= n:
        raise ValueError(f"n_novel={n_novel} but only {n} classes present")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_novel, replace=False)
    is_novel = np.zeros(n, dtype=bool); is_novel[idx] = True
    return [int(c) for c in classes[~is_novel]], [int(c) for c in classes[is_novel]]
 
 
@torch.no_grad()
def embed_all(model, X, device, batch=512):
    model.eval()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    out = [model.embed(Xt[i:i + batch].to(device)).cpu() for i in range(0, len(Xt), batch)]
    return torch.cat(out)
 
 
def class_prototypes(embs, y, class_list):
    protos = torch.zeros(len(class_list), embs.shape[1])
    y = np.asarray(y)
    for i, c in enumerate(class_list):
        m = (y == c)
        if m.any():
            protos[i] = embs[torch.as_tensor(m)].mean(0)
    return protos
 
 
# --------------------------------------------------------------------------- #
# Phase 1: supervised pre-training of the embedding (40-way, matches Piece 2)
# --------------------------------------------------------------------------- #
class CCADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
 
    def __len__(self):
        return self.y.shape[0]
 
    def __getitem__(self, i):
        return self.X[i], self.y[i]
 
 
def make_base_dataset(X, y, base_classes):
    """Filter to the base gestures and remap their ids to a contiguous 0..n_base-1."""
    base_set = set(int(c) for c in base_classes)
    mask = np.array([v in base_set for v in y])
    lookup = {int(c): i for i, c in enumerate(base_classes)}
    yb = np.array([lookup[int(v)] for v in y[mask]], dtype=np.int64)
    return X[mask], yb
 
 
def pretrain_embedding(Xtr, ytr, n_base, e, epochs, lr, lambda_, batch_size, device):
    """Ordinary supervised training of the n_base-way model (hybrid loss). Its
    `features` + `embedding` layers are what get transferred into the meta-model."""
    model = EMGAdapt(e=e, n_classes=n_base).to(device)
    loader = DataLoader(CCADataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_fn = nn.CrossEntropyLoss()
    base_ids = list(range(n_base))
    for epoch in range(1, epochs + 1):
        protos = class_prototypes(embed_all(model, Xtr, device), ytr, base_ids).to(device)
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            emb, logits = model(xb)
            ce = ce_fn(logits, yb)
            proto = ((emb - protos[yb]) ** 2).sum(1).clamp_min(1e-12).sqrt().mean()
            loss = (1 - lambda_) * ce + lambda_ * proto
            loss.backward()
            opt.step()
            tot += loss.item() * yb.numel()
        print(f"  [pretrain] epoch {epoch:>2}/{epochs}  loss {tot/len(loader.dataset):.4f}")
    return model
 
 
# --------------------------------------------------------------------------- #
# Task sampler (base gestures only) - the heart of meta-learning
# --------------------------------------------------------------------------- #
class TaskSampler:
    """Samples N-way K-shot tasks from the BASE gestures, where a SHOT is a whole
    PERFORMANCE (all windows of one (subject, rep)), not a single window.
 
    A class's K shots are drawn from its performances ACROSS SUBJECTS, so each task
    forces the embedding to treat the same gesture done by different people/sessions
    as one class -- the person-invariance that makes a new user's enrollment work.
    Returns support windows + task-local labels 0..N-1 (fresh labels every task).
    """
    def __init__(self, Xbase, ybase, reps_base, subj_base, base_classes, n_way, k_shot, seed=0):
        self.X = Xbase
        self.n_way = n_way
        self.k_shot = k_shot
        self.rng = np.random.default_rng(seed)
 
        # group windows into performances: a unique key per (class, subject, rep)
        ybase = np.asarray(ybase); reps_base = np.asarray(reps_base); subj_base = np.asarray(subj_base)
        keys = (ybase.astype(np.int64) * 1_000_000
                + subj_base.astype(np.int64) * 1000
                + reps_base.astype(np.int64))
        order = np.argsort(keys, kind="stable")
        bounds = np.where(np.diff(keys[order]) != 0)[0] + 1
        groups = np.split(order, bounds)                 # each = window indices of one performance
 
        # class -> list of its performances (each an index array of that performance's windows)
        self.class_perfs = defaultdict(list)
        for g in groups:
            self.class_perfs[int(ybase[g[0]])].append(g)
 
        # keep only base classes with at least k_shot performances available
        self.classes = [c for c in base_classes if len(self.class_perfs[c]) >= k_shot]
        if len(self.classes) < n_way:
            raise ValueError("not enough base classes with >= k_shot performances")
 
    def sample(self):
        chosen = self.rng.choice(self.classes, size=self.n_way, replace=False)
        sX, sy = [], []
        for task_label, c in enumerate(chosen):
            perfs = self.class_perfs[c]
            pick = self.rng.choice(len(perfs), size=self.k_shot, replace=False)  # k PERFORMANCES
            for j in pick:
                wins = perfs[j]                          # all windows of that performance
                sX.append(self.X[wins])
                sy += [task_label] * len(wins)
        return np.concatenate(sX, axis=0), np.array(sy, dtype=np.int64)
 
 
def task_prototypes(emb, sy_t, n_way):
    """Per-task class centroids from support embeddings -> (n_way, 128)."""
    return torch.stack([emb[sy_t == c].mean(0) for c in range(n_way)])
 
 
# --------------------------------------------------------------------------- #
# Reptile meta-training
# --------------------------------------------------------------------------- #
def inner_adapt(model, sX, sy, inner_steps, inner_lr, n_way, lambda_, device, inner_batch=256,
                freeze_features=False):
    """Clone the model and adapt it to one task with the hybrid loss. The support set
    is now a large set of windows (whole performances), so each inner step uses a
    random mini-batch of it. Returns the adapted clone (global model untouched).
 
    If freeze_features, the conv features are locked: the optimiser skips their
    parameters and their BatchNorm runs in eval mode, so only the embedding head adapts.
    """
    fast = copy.deepcopy(model).to(device)
    trainable = [p for p in fast.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=inner_lr)
    Xt = torch.as_tensor(sX, dtype=torch.float32)
    yt = torch.as_tensor(sy, dtype=torch.long)
    N = len(yt)
    ce_fn = nn.CrossEntropyLoss()
    fast.train()
    if freeze_features:
        fast.features.eval()            # keep frozen BN running stats fixed
    for _ in range(inner_steps):
        idx = torch.randperm(N)[:inner_batch] if N > inner_batch else torch.arange(N)
        xb = Xt[idx].to(device)
        yb = yt[idx].to(device)
        opt.zero_grad()
        emb, logits = fast(xb)
        ce = ce_fn(logits, yb)
        protos = task_prototypes(emb.detach(), yb, n_way)        # centroids from this mini-batch
        proto = ((emb - protos[yb]) ** 2).sum(1).clamp_min(1e-12).sqrt().mean()
        loss = (1 - lambda_) * ce + lambda_ * proto
        loss.backward()
        opt.step()
    return fast
 
 
def reptile_step(model, fast, beta):
    """Move the global model toward the adapted clone: p <- p + beta*(p_fast - p)."""
    with torch.no_grad():
        for pg, pf in zip(model.parameters(), fast.parameters()):
            pg.add_(beta * (pf.detach() - pg))
        for bg, bf in zip(model.buffers(), fast.buffers()):   # BN running stats
            if bg.dtype.is_floating_point:
                bg.add_(beta * (bf.detach() - bg))
            else:
                bg.copy_(bf)
 
 
def reptile_train(model, sampler, eval_fn, meta_iters=META_ITERS, inner_steps=INNER_STEPS,
                  inner_lr=INNER_LR, step_size=STEP_SIZE, n_way=N_WAY, lambda_=LAMBDA,
                  eval_every=EVAL_EVERY, freeze_features=False, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = []
    for m in range(1, meta_iters + 1):
        sX, sy = sampler.sample()
        fast = inner_adapt(model, sX, sy, inner_steps, inner_lr, n_way, lambda_, device,
                           freeze_features=freeze_features)
        beta = (1 - m / meta_iters) * step_size          # step-size decay (Eq. 12)
        reptile_step(model, fast, beta)
 
        if m % eval_every == 0 or m == meta_iters:
            res = eval_fn(model)
            print(f"meta-iter {m:>4}/{meta_iters}  beta {beta:.4f}  "
                  f"novel window {res['window_acc']:.4f}  pooled {res['pooled_acc']:.4f}")
            history.append({"meta_iter": m, "beta": round(beta, 6),
                            "novel_window_acc": round(res['window_acc'], 6),
                            "novel_pooled_acc": round(res['pooled_acc'], 6)})
    return model, history
 
 
# --------------------------------------------------------------------------- #
# Novel-gesture evaluation (reused from Piece 2; per-subject breakdown)
# --------------------------------------------------------------------------- #
def enroll(model, X, y, reps, novel_classes, enroll_reps, device):
    mask = np.isin(reps, enroll_reps) & np.isin(y, novel_classes)
    protos = class_prototypes(embed_all(model, X[mask], device), y[mask], novel_classes)
    return protos.to(device), list(novel_classes)
 
 
def majority_pooled(preds, y_true, subj, reps):
    votes = defaultdict(list); truth = {}
    for i in range(len(preds)):
        key = (int(subj[i]), int(reps[i]), int(y_true[i]))
        votes[key].append(int(preds[i])); truth[key] = int(y_true[i])
    correct = sum(Counter(v).most_common(1)[0][0] == truth[k] for k, v in votes.items())
    return correct / len(votes), len(votes)
 
 
@torch.no_grad()
def evaluate_novel(model, X, y, reps, subj, novel_classes, proto_matrix, test_reps, device):
    mask = np.isin(reps, test_reps) & np.isin(y, novel_classes)
    Xq, yq, rq, sq = X[mask], y[mask], reps[mask], subj[mask]
    embs = embed_all(model, Xq, device)
    idx = torch.cdist(embs, proto_matrix.cpu()).argmin(1).numpy()
    preds = np.asarray(novel_classes)[idx]
    win_acc = float((preds == yq).mean())
    pooled_acc, n_perf = majority_pooled(preds, yq, sq, rq)
    per_subject = {}
    for s in np.unique(sq):
        m = sq == s
        w = float((preds[m] == yq[m]).mean())
        p, _ = majority_pooled(preds[m], yq[m], sq[m], rq[m])
        per_subject[int(s)] = {"window_acc": w, "pooled_acc": p}
    return {"window_acc": win_acc, "pooled_acc": pooled_acc,
            "n_performances": n_perf, "per_subject": per_subject}
 
 
def save_rows(rows, path, meta):
    fieldnames = list(meta.keys()) + list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({**meta, **r})
 
 
# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(subjects=SUBJECTS, exercises=EXERCISES, n_novel=N_NOVEL, n_way=N_WAY, k_shot=K_SHOT,
        meta_iters=META_ITERS, inner_steps=INNER_STEPS, inner_lr=INNER_LR, step_size=STEP_SIZE,
        lambda_=LAMBDA, enroll_reps=ENROLL_REPS, test_reps=TEST_REPS, eval_every=EVAL_EVERY,
        pretrained_path=None, pretrain_epochs=PRETRAIN_EPOCHS, pretrain_lr=PRETRAIN_LR,
        pretrain_lambda=PRETRAIN_LAMBDA, freeze_features=FREEZE_FEATURES, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(seed)
    assert len(set(enroll_reps) & set(test_reps)) == 0
 
    X, y, reps, subj = load_subject_gestures(subjects, exercises)
    base_classes, novel_classes = split_base_novel(y, n_novel=n_novel, seed=seed)
    assert set(base_classes).isdisjoint(novel_classes)
 
    # base pool for task sampling (all reps / subjects of base gestures)
    bmask = np.isin(y, base_classes)
    Xbase, ybase, reps_base, subj_base = X[bmask], y[bmask], reps[bmask], subj[bmask]
    sampler = TaskSampler(Xbase, ybase, reps_base, subj_base, base_classes, n_way, k_shot, seed=seed)
    print(f"subjects {list(subjects)} | {len(base_classes)} base, {len(novel_classes)} novel")
    print(f"  meta-train: {n_way}-way {k_shot}-shot (shots = performances, across subjects) | "
          f"eval: {len(novel_classes)}-way, enroll {enroll_reps} test {test_reps}")
 
    # model: n_way head (used only for meta-training CE; eval uses embed + prototypes)
    e = X.shape[1]
 
    # ---- Phase 1: supervised 40-way embedding (loaded from cache, else trained) ----
    # Cached per seed so the split matches; delete the file to force a fresh pre-train.
    pre_path = pretrained_path or f"pretrained_embedding_seed{seed}.pt"
    if os.path.exists(pre_path):
        pretrained = EMGAdapt(e=e, n_classes=len(base_classes))
        pretrained.load_state_dict(torch.load(pre_path, map_location=device))
        pretrained.to(device)
        print(f"loaded pretrained embedding <- {pre_path}")
    else:
        print(f"pre-training {len(base_classes)}-way embedding (lambda={pretrain_lambda}) ...")
        Xtr, ytr = make_base_dataset(X, y, base_classes)
        pretrained = pretrain_embedding(Xtr, ytr, len(base_classes), e, pretrain_epochs,
                                        pretrain_lr, pretrain_lambda, 256, device)
        torch.save(pretrained.state_dict(), pre_path)
        print(f"saved pretrained embedding -> {pre_path}")
 
    # ---- Phase 2: meta-model (n_way head) initialised FROM the pretrained embedding ----
    model = EMGAdapt(e=e, n_classes=n_way).to(device)
    model.features.load_state_dict(pretrained.features.state_dict())
    model.embedding.load_state_dict(pretrained.embedding.state_dict())
    print(f"transferred embedding into the {n_way}-way meta-model (classifier head is fresh)")
    if freeze_features:
        for p_ in model.features.parameters():
            p_.requires_grad_(False)
        print("froze conv features -> meta-learning adapts only the embedding head")
 
    # eval closure: enroll novel gestures, classify held-out reps by nearest prototype
    def eval_fn(mdl):
        proto_matrix, novel_list = enroll(mdl, X, y, reps, novel_classes, enroll_reps, device)
        return evaluate_novel(mdl, X, y, reps, subj, novel_list, proto_matrix, test_reps, device)
 
    t0 = time.perf_counter()
    model, history = reptile_train(model, sampler, eval_fn, meta_iters=meta_iters,
                                   inner_steps=inner_steps, inner_lr=inner_lr,
                                   step_size=step_size, n_way=n_way, lambda_=lambda_,
                                   eval_every=eval_every, freeze_features=freeze_features, device=device)
    print(f"meta-training done in {(time.perf_counter()-t0)/60:.1f} min")
 
    result = eval_fn(model)
    chance = 1.0 / len(novel_classes)
    print(f"\nFINAL novel-gesture recognition ({len(novel_classes)}-way):")
    print(f"  window-level : {result['window_acc']:.4f}")
    print(f"  gesture-level (majority pooled): {result['pooled_acc']:.4f} "
          f"over {result['n_performances']} performances  (chance {chance:.3f})")
 
    # ---- CSVs: meta-training curve + final per-subject table ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {"lambda": lambda_, "inner_lr": inner_lr, "step_size": step_size,
              "meta_iters": meta_iters, "inner_steps": inner_steps,
              "n_way": n_way, "k_shot": k_shot, "freeze_features": freeze_features, "seed": seed}
    save_rows(history, f"results_reptile_curve_{stamp}.csv", config)
 
    ps = result["per_subject"]
    subject_rows = [{"subject_id": s,
                     "window_accuracy": round(ps[s]["window_acc"], 6),
                     "pooled_accuracy": round(ps[s]["pooled_acc"], 6)} for s in sorted(ps)]
    subject_rows.append({"subject_id": "AVERAGE",
                         "window_accuracy": round(float(np.mean([r["window_accuracy"] for r in subject_rows])), 6),
                         "pooled_accuracy": round(float(np.mean([r["pooled_accuracy"] for r in subject_rows])), 6)})
    save_rows(subject_rows, f"results_reptile_persubject_{stamp}.csv", config)
    print(f"saved curve + per-subject CSVs ({stamp})")
    return model, result
 
 


# --------------------------------------------------------------------------- #
# Cross-user (leave-N-subjects-out) evaluation harness
# --------------------------------------------------------------------------- #
def subject_folds(subjects, n_holdout=2):
    """Yield (train_subjects, held_out_subjects) folds so each subject is held out once."""
    subs = list(subjects)
    for i in range(0, len(subs), n_holdout):
        held = subs[i:i + n_holdout]
        train = [s for s in subs if s not in held]
        yield train, held


@torch.no_grad()
def evaluate_crossuser(model, X, y, reps, subj, novel_classes, held_subjects,
                       enroll_reps, test_reps, device):
    """PER-USER enrollment on held-out subjects: for each held-out subject, enroll
    their novel gestures from THEIR enroll reps and test on THEIR test reps (the model
    has never seen this subject). Returns per-subject accuracies + the mean over them."""
    per = {}
    for s in held_subjects:
        sm = subj == s
        Xs, ys, rs = X[sm], y[sm], reps[sm]
        emask = np.isin(rs, enroll_reps) & np.isin(ys, novel_classes)
        protos = class_prototypes(embed_all(model, Xs[emask], device), ys[emask], novel_classes).to(device)
        tmask = np.isin(rs, test_reps) & np.isin(ys, novel_classes)
        Xq, yq, rq = Xs[tmask], ys[tmask], rs[tmask]
        embs = embed_all(model, Xq, device)
        preds = np.asarray(novel_classes)[torch.cdist(embs, protos.cpu()).argmin(1).numpy()]
        win = float((preds == yq).mean())
        pooled, _ = majority_pooled(preds, yq, np.full(len(yq), s), rq)
        per[int(s)] = {"window_acc": win, "pooled_acc": pooled}
    return {"per_subject": per,
            "mean_window": float(np.mean([v["window_acc"] for v in per.values()])),
            "mean_pooled": float(np.mean([v["pooled_acc"] for v in per.values()]))}


def run_crossval(subjects=SUBJECTS, exercises=EXERCISES, n_holdout=2, n_novel=N_NOVEL,
                 n_way=N_WAY, k_shot=K_SHOT, meta_iters=META_ITERS, inner_steps=INNER_STEPS,
                 inner_lr=INNER_LR, step_size=STEP_SIZE, lambda_=LAMBDA,
                 enroll_reps=ENROLL_REPS, test_reps=TEST_REPS, eval_every=EVAL_EVERY,
                 pretrain_epochs=PRETRAIN_EPOCHS, pretrain_lr=PRETRAIN_LR,
                 pretrain_lambda=PRETRAIN_LAMBDA, freeze_features=FREEZE_FEATURES, seed=0):
    """Leave-N-subjects-out cross-validation comparing SUPERVISED vs REPTILE on
    genuinely unseen users. For each fold: pre-train the embedding on the TRAINING
    subjects' base gestures, evaluate it on the HELD-OUT subjects (= supervised),
    then meta-train and evaluate again on the SAME held-out subjects (= reptile).
    Both numbers come from one run on identical data, so the only difference is meta."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(seed)
    assert len(set(enroll_reps) & set(test_reps)) == 0

    X, y, reps, subj = load_subject_gestures(subjects, exercises)
    base_classes, novel_classes = split_base_novel(y, n_novel=n_novel, seed=seed)
    assert set(base_classes).isdisjoint(novel_classes)
    e = X.shape[1]

    fold_rows = []
    for fold_i, (train_subs, held_subs) in enumerate(subject_folds(subjects, n_holdout), 1):
        assert set(train_subs).isdisjoint(held_subs)
        print(f"\n=== Fold {fold_i}: train {train_subs} | held-out {held_subs} ===")

        # training data = TRAINING subjects only (held-out subjects never seen)
        trm = np.isin(subj, train_subs)
        Xtr, ytr, reps_tr, subj_tr = X[trm], y[trm], reps[trm], subj[trm]
        assert set(np.unique(subj_tr)).isdisjoint(held_subs)   # guardrail: no leakage

        # ---- Phase 1: pre-train the embedding on TRAINING subjects' base gestures ----
        pre_path = f"pretrained_fold{fold_i}_seed{seed}.pt"
        if os.path.exists(pre_path):
            pretrained = EMGAdapt(e=e, n_classes=len(base_classes))
            pretrained.load_state_dict(torch.load(pre_path, map_location=device))
            pretrained.to(device)
            print(f"  loaded fold embedding <- {pre_path}")
        else:
            print(f"  pre-training embedding on {len(train_subs)} subjects (lambda={pretrain_lambda}) ...")
            Xb, yb = make_base_dataset(Xtr, ytr, base_classes)
            pretrained = pretrain_embedding(Xb, yb, len(base_classes), e, pretrain_epochs,
                                            pretrain_lr, pretrain_lambda, 256, device)
            torch.save(pretrained.state_dict(), pre_path)
            print(f"  saved fold embedding -> {pre_path}")

        # ---- SUPERVISED: evaluate the pretrained embedding on the held-out subjects ----
        sup = evaluate_crossuser(pretrained, X, y, reps, subj, novel_classes,
                                 held_subs, enroll_reps, test_reps, device)
        print(f"  [supervised] held-out pooled {sup['mean_pooled']:.4f}  window {sup['mean_window']:.4f}")

        # ---- Phase 2: meta-train from that embedding (tasks from TRAIN subjects only) ----
        model = EMGAdapt(e=e, n_classes=n_way).to(device)
        model.features.load_state_dict(pretrained.features.state_dict())
        model.embedding.load_state_dict(pretrained.embedding.state_dict())
        if freeze_features:
            for p_ in model.features.parameters():
                p_.requires_grad_(False)
        bm = np.isin(ytr, base_classes)
        sampler = TaskSampler(Xtr[bm], ytr[bm], reps_tr[bm], subj_tr[bm],
                              base_classes, n_way, k_shot, seed=seed)

        def eval_fn(mdl):
            r = evaluate_crossuser(mdl, X, y, reps, subj, novel_classes,
                                   held_subs, enroll_reps, test_reps, device)
            return {"window_acc": r["mean_window"], "pooled_acc": r["mean_pooled"]}

        model, _ = reptile_train(model, sampler, eval_fn, meta_iters=meta_iters,
                                 inner_steps=inner_steps, inner_lr=inner_lr, step_size=step_size,
                                 n_way=n_way, lambda_=lambda_, eval_every=eval_every,
                                 freeze_features=freeze_features, device=device)

        # ---- REPTILE: evaluate the meta-trained model on the SAME held-out subjects ----
        rep = evaluate_crossuser(model, X, y, reps, subj, novel_classes,
                                 held_subs, enroll_reps, test_reps, device)
        print(f"  [reptile]    held-out pooled {rep['mean_pooled']:.4f}  window {rep['mean_window']:.4f}")

        fold_rows.append({"fold": fold_i, "held_out": "-".join(map(str, held_subs)),
                          "supervised_pooled": round(sup["mean_pooled"], 6),
                          "reptile_pooled": round(rep["mean_pooled"], 6),
                          "supervised_window": round(sup["mean_window"], 6),
                          "reptile_window": round(rep["mean_window"], 6),
                          "delta_pooled": round(rep["mean_pooled"] - sup["mean_pooled"], 6)})

    # ---- average across folds ----
    def col(k): return float(np.mean([r[k] for r in fold_rows]))
    avg = {"fold": "AVERAGE", "held_out": "all",
           "supervised_pooled": round(col("supervised_pooled"), 6),
           "reptile_pooled": round(col("reptile_pooled"), 6),
           "supervised_window": round(col("supervised_window"), 6),
           "reptile_window": round(col("reptile_window"), 6),
           "delta_pooled": round(col("delta_pooled"), 6)}
    fold_rows.append(avg)

    print("\n===== CROSS-USER SUMMARY (mean over folds) =====")
    print(f"  supervised pooled : {avg['supervised_pooled']:.4f}")
    print(f"  reptile    pooled : {avg['reptile_pooled']:.4f}")
    print(f"  delta (reptile - supervised) : {avg['delta_pooled']:+.4f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {"experiment": "crossuser_cv", "n_holdout": n_holdout, "lambda": lambda_,
              "inner_lr": inner_lr, "step_size": step_size, "inner_steps": inner_steps,
              "n_way": n_way, "k_shot": k_shot, "freeze_features": freeze_features, "seed": seed}
    save_rows(fold_rows, f"results_crossuser_cv_{stamp}.csv", config)
    print(f"saved cross-user CV results -> results_crossuser_cv_{stamp}.csv")
    return fold_rows

#-----------------varies the seed numebr-----------------# 

def seed_sweep(seeds=(0, 1, 2, 3)):
    """Run the cross-user CV for several seeds (different novel-gesture sets) and
    summarise supervised vs reptile. Each seed re-draws which 12 gestures are novel."""
    summary = []
    for sd in seeds:
        print(f"\n########## SEED {sd} ##########")
        fold_rows = run_crossval(seed=sd)
        avg = fold_rows[-1]                      # the AVERAGE row of that seed's CV
        summary.append({
            "seed": sd,
            "supervised_pooled": avg["supervised_pooled"],
            "reptile_pooled": avg["reptile_pooled"],
            "delta_pooled": avg["delta_pooled"],
        })

    import numpy as np
    def stat(k):
        vals = [s[k] for s in summary]
        return float(np.mean(vals)), float(np.std(vals))

    print("\n===== MULTI-SEED SUMMARY =====")
    print(f"{'seed':>4}  {'supervised':>10}  {'reptile':>10}  {'delta':>8}")
    for s in summary:
        print(f"{s['seed']:>4}  {s['supervised_pooled']:>10.4f}  "
              f"{s['reptile_pooled']:>10.4f}  {s['delta_pooled']:>+8.4f}")
    sup_m, sup_s = stat("supervised_pooled")
    rep_m, rep_s = stat("reptile_pooled")
    dl_m, dl_s = stat("delta_pooled")
    print(f"mean  {sup_m:>10.4f}  {rep_m:>10.4f}  {dl_m:>+8.4f}")
    print(f" std  {sup_s:>10.4f}  {rep_s:>10.4f}  {dl_s:>8.4f}")

    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_rows(summary, f"results_seed_sweep_{stamp}.csv",
              {"experiment": "crossuser_seed_sweep"})
    print(f"saved seed sweep -> results_seed_sweep_{stamp}.csv")
    return summary

if __name__ == "__main__":
    seed_sweep(seeds=(0, 1, 2, 3))