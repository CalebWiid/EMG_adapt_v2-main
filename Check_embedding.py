"""
Check_embedding.py - run ONCE (needs the DB5 data, not the armband).

1. LOAD CHECK: confirms deployment_embedding.pt really loads its trained weights.
2. DB5 CHECK: the deployment embedding was trained on the 40 base gestures only,
   so the 12 novel gestures (same seed-0 split) are still unseen. For every
   subject it enrolls the novel gestures from ENROLL_REPS and tests on TEST_REPS
   (per-user prototypes, nearest prototype, majority pooling) - the same test as
   before. Expect roughly your cross-validation result or a bit higher (these
   subjects were seen in training, their novel gestures were not).
3. REFERENCE: saves the DB5 feature scale (median CCA value of gesture windows)
   to db5_reference.npz. Enroll.py uses it to align the MindRove amplitude.

Run:  python Check_embedding.py
"""
from collections import defaultdict, Counter

import numpy as np
import torch
import Common as lc
from Preprocessing import process_files, filename_for, FS

MODEL_PATH = "deployment_embedding.pt"
SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
EXERCISES = (1, 2, 3)
EXP_BASE = 1000
N_NOVEL = 12
SPLIT_SEED = 0                  # must match save_deployment_embedding(seed=0)
ENROLL_REPS = (1, 3, 4)
TEST_REPS = (2, 5, 6)


def load_db5(subjects=SUBJECTS, exercises=EXERCISES):
    X_all, y_all, rep_all, subj_all = [], [], [], []
    for s in subjects:
        for ex in exercises:
            try:
                X, y, reps = process_files([filename_for(s, ex)], fs=FS)
            except FileNotFoundError:
                print(f"  [skip] missing S{s}_E{ex}")
                continue
            if X.shape[0] == 0:
                continue
            X_all.append(X)
            y_all.append(y + ex * EXP_BASE)
            rep_all.append(reps)
            subj_all.append(np.full(X.shape[0], s, dtype=np.int64))
        print(f"  loaded subject {s}")
    if not X_all:
        raise SystemExit("No DB5 data loaded - check DATA_DIR in Preprocessing.py.")
    return (np.concatenate(X_all), np.concatenate(y_all),
            np.concatenate(rep_all), np.concatenate(subj_all))


def split_base_novel(y, n_novel=N_NOVEL, seed=SPLIT_SEED):
    """Identical to the split used for training (index-based, seeded)."""
    classes = np.unique(np.asarray(y))
    rng = np.random.default_rng(seed)
    novel_idx = rng.choice(len(classes), size=n_novel, replace=False)
    is_novel = np.zeros(len(classes), dtype=bool)
    is_novel[novel_idx] = True
    return [int(c) for c in classes[~is_novel]], [int(c) for c in classes[is_novel]]


def pooled_accuracy(preds, truth, reps):
    votes = defaultdict(list)
    for p, t, r in zip(preds, truth, reps):
        votes[(int(t), int(r))].append(int(p))
    correct = sum(Counter(v).most_common(1)[0][0] == k[0] for k, v in votes.items())
    return correct / max(1, len(votes))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. load check (raises if the weights don't load) ----
    model = lc.load_embedding(MODEL_PATH, device=device)
    print(f"[1] load check OK - trained weights loaded from {MODEL_PATH}\n")

    # ---- 2. DB5 novel-gesture check ----
    print("[2] loading DB5...")
    X, y, reps, subj = load_db5()
    base, novel = split_base_novel(y)
    print(f"novel (unseen) gestures: {novel}\n")
    print(f"{'subject':>8}{'window':>10}{'pooled':>10}")
    results = []
    for s in sorted(np.unique(subj)):
        m_s = (subj == s) & np.isin(y, novel)
        enr = m_s & np.isin(reps, ENROLL_REPS)
        tst = m_s & np.isin(reps, TEST_REPS)
        if not enr.any() or not tst.any():
            continue
        e_emb = lc.embed_features(model, X[enr], device)
        protos = torch.stack([e_emb[torch.as_tensor(y[enr] == c)].mean(0) for c in novel])
        idx = lc.nearest(lc.embed_features(model, X[tst], device), protos)
        preds = np.asarray(novel)[idx]
        win = float((preds == y[tst]).mean())
        pooled = pooled_accuracy(preds, y[tst], reps[tst])
        results.append((win, pooled))
        print(f"{s:>8}{win:>10.3f}{pooled:>10.3f}")
    w, p = np.mean(results, axis=0)
    print(f"{'mean':>8}{w:>10.3f}{p:>10.3f}")
    print("  -> pooled well below ~0.9 means the saved embedding is not the model you "
          "evaluated; retrain/re-save it.\n")

    # ---- 3. DB5 feature-scale reference ----
    med = float(np.median(X))
    np.savez(lc.REFERENCE_PATH, median=med, p5=np.percentile(X, 5),
             p95=np.percentile(X, 95), channel_median=np.median(X, axis=0))
    print(f"[3] DB5 feature median = {med:.4f} (5th-95th pct "
          f"{np.percentile(X, 5):.4f} to {np.percentile(X, 95):.4f})")
    print(f"    saved -> {lc.REFERENCE_PATH}")


if __name__ == "__main__":
    main()