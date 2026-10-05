"""
Reptile learning algorithem. This is a first order meta-learinign technique that help learn novel classes with few examples. 

TRAINING:
    - Split the data into 40 base gestures and 12 novel gesture 
    - Trained by using 5-way, 5-shot examples for meta-learning.

TESTING: 
    - 12-way, 3 shot testing for novel gesture classification.
"""
import numpy as np
import torch
import torch.nn as nn
import csv
import copy
import time
from datetime import datetime
from collections import defaultdict, Counter
 
from Preprocessing import process_files, filename_for, FS
from CNN_model import EMGAdapt

SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
EXERCISES = (1, 2, 3)
EXP_BASE = 1000
N_NOVEL = 12 

# Meta training parameters 
N_WAY = 5 
K_SHOT = 5
META_ITERS = 200
INNER_STEPS = 25 
INNER_LR = 1e-3 
STEP_SIZE = 0.5 
LAMBDA = 0.25 

# Evaluation parameters
ENROLL_REPS = (1, 3, 4)
TEST_REPS = (2, 5, 6)
EVAL_EVERY = 50 

#-----------load data-----------#

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

#-----------Task sampler-----------#

class TaskSampler:
    """Samples N-way K-shot tasks from the BASE gestures, where a SHOT is a whole
    PERFORMANCE.
 
    A class's K shots are drawn from its performances ACROSS SUBJECTS, so each task
    forces the embedding to treat the same gesture done by different people/sessions
    as one class.
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

#-----------Reptile meta-larning-----------#

def inner_adapt(model, sX, sy, inner_steps, inner_lr, n_way, lambda_, device, inner_batch=256):
    """Clone the model and adapt it to one task with the hybrid loss. The support set
    is now a large set of windows (whole performances), so each inner step uses a
    random mini-batch of it. Returns the adapted clone (global model untouched)."""
    fast = copy.deepcopy(model).to(device)
    opt = torch.optim.Adam(fast.parameters(), lr=inner_lr)
    Xt = torch.as_tensor(sX, dtype=torch.float32)
    yt = torch.as_tensor(sy, dtype=torch.long)
    N = len(yt)
    ce_fn = nn.CrossEntropyLoss()
    fast.train()
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
                  eval_every=EVAL_EVERY, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = []
    for m in range(1, meta_iters + 1):
        sX, sy = sampler.sample()
        fast = inner_adapt(model, sX, sy, inner_steps, inner_lr, n_way, lambda_, device)
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

#-----------Novel gesture evaluation-----------#

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

#-----------Main-----------#

def run(subjects=SUBJECTS, exercises=EXERCISES, n_novel=N_NOVEL, n_way=N_WAY, k_shot=K_SHOT,
        meta_iters=META_ITERS, inner_steps=INNER_STEPS, inner_lr=INNER_LR, step_size=STEP_SIZE,
        lambda_=LAMBDA, enroll_reps=ENROLL_REPS, test_reps=TEST_REPS, eval_every=EVAL_EVERY,
        seed=0):
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
    model = EMGAdapt(e=X.shape[1], n_classes=n_way)
 
    # eval closure: enroll novel gestures, classify held-out reps by nearest prototype
    def eval_fn(mdl):
        proto_matrix, novel_list = enroll(mdl, X, y, reps, novel_classes, enroll_reps, device)
        return evaluate_novel(mdl, X, y, reps, subj, novel_list, proto_matrix, test_reps, device)
 
    t0 = time.perf_counter()
    model, history = reptile_train(model, sampler, eval_fn, meta_iters=meta_iters,
                                   inner_steps=inner_steps, inner_lr=inner_lr,
                                   step_size=step_size, n_way=n_way, lambda_=lambda_,
                                   eval_every=eval_every, device=device)
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
              "n_way": n_way, "k_shot": k_shot, "seed": seed}
    save_rows(history, f"results_reptile_curve_{stamp}.csv", config)
 
    ps = result["per_subject"]
    subject_rows = [{"subject_id": s,
                     "window_accuracy": round(ps[s]["window_acc"], 6),
                     "pooled_accuracy": round(ps[s]["pooled_acc"], 6)} for s in sorted(ps)]
    subject_rows.append({"subject_id": "AVERAGE",
                         "window_accuracy": round(float(np.mean([r["window_accuracy"] for r in subject_rows])), 6),
                         "pooled_accuracy": round(float(np.mean([r["pooled_accuracy"] for r in subject_rows])), 6)})
    save_rows(subject_rows, f"results_reptile_{stamp}.csv", config)
    print(f"saved curve + per-subject CSVs ({stamp})")
    return model, result
 
 
if __name__ == "__main__":
    run()