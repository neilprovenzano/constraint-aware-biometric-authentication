#!/usr/bin/env python3
"""
CMU Keystroke Dynamics -- constraint-driven classifier evaluation (REVISED harness)
Rebuilt for the Electronics major revision (electronics-4507835).

This replaces the lost single-seed scripts. It produces the statistical layer the
paper needs and answers the reviewers' methodological objections BY CONSTRUCTION:

  * SUBJECT-level statistics (no seed pseudo-replication): one EER per subject,
    inference run across the ~51 subjects.
  * SESSION-DISJOINT split (train on sessions 1-6, test on 7-8) reported alongside
    a correctly-done RANDOM split, so temporal leakage is measured, not hidden.
  * FIXED held-out test set + NESTED N (10 subset of 50 subset of 200): the test
    set never changes with N.
  * CLASS-BALANCE control: 'fixed' impostor count vs 'balanced' (impostor == N),
    to separate sample-size from class-ratio effects.
  * VALIDATION-BASED tuning for tunable classifiers (k, C, MLP width/alpha) on an
    inner validation split -- never on test, using compact classifier-appropriate grids.
  * ROC-INTERPOLATED EER + operating points (FAR@FRR, FRR@FAR).
  * The THRESHOLD rule is a proper CONTINUOUS score (mean standardized deviation),
    swept via ROC like every other classifier -- resolving the 2-sigma-vs-grid
    contradiction and the "no continuous score" claim.

Run:
  python3 cmu_eval.py            # FULL config (use this on your Mac)
  python3 cmu_eval.py --quick    # fast validation pass (fewer seeds/grids)

Outputs (in ./out_cmu/):
  persubject.csv   raw per-subject EER + operating points (release artifact)
  summary.csv      subject-level means + 95% CIs
  stats.csv        paired subject-level tests (t, Wilcoxon, Cohen's d, Holm)
"""
import os, argparse, warnings, json
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from scipy import stats
warnings.filterwarnings('ignore')

# ----------------------------- CONFIG -----------------------------
CSV_PATH       = 'DSL-StrongPasswordData.csv'
N_LIST         = [10, 50, 200]
TRAIN_SESSIONS = [1, 2, 3, 4, 5, 6]
TEST_SESSIONS  = [7, 8]
GEN_TEST_SIZE  = 100        # fixed genuine-test count (both splits)
IMP_TRAIN_PER  = 2          # impostors drawn per other subject (fixed-imp condition)
IMP_TEST_PER   = 5          # impostors drawn per other subject for test
OP_FRR, OP_FAR = 0.01, 0.01 # operating points: FAR@FRR=1%, FRR@FAR=1%
OUTDIR         = 'out_cmu'

CLASSIFIERS = ['k-NN', 'LDA', 'Threshold', 'Linear SVM', 'MLP']

# hyperparameter grids (selected on inner validation, never on test)
GRID_FULL = {
    'k-NN':       [{'k': k} for k in (1, 3, 5, 7)],
    'LDA':        [{}],
    'Threshold':  [{}],
    'Linear SVM': [{'C': c} for c in (0.01, 0.1, 1, 10, 100)],
    'MLP':        [{'h': h, 'alpha': a} for h in (32, 64, 128)
                                        for a in (1e-4, 1e-3, 1e-2)],
}
GRID_QUICK = {
    'k-NN':       [{'k': k} for k in (1, 3, 5)],
    'LDA':        [{}],
    'Threshold':  [{}],
    'Linear SVM': [{'C': c} for c in (0.1, 1, 10)],
    'MLP':        [{'h': h, 'alpha': a} for h in (32, 64) for a in (1e-3,)],
}
MLP_INITS_FULL, MLP_INITS_QUICK = 3, 1

# ----------------------------- METRICS -----------------------------
def eer_and_ops(y, score):
    """ROC-interpolated EER + FAR@FRR and FRR@FAR. y: 1=genuine, higher score=genuine."""
    fpr, tpr, _ = roc_curve(y, score)           # fpr = FAR, tpr = genuine-accept
    fnr = 1 - tpr                                # frr
    diff = fpr - fnr
    sign_change = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change):
        i = sign_change[0]
        d0, d1 = diff[i], diff[i + 1]
        t = 0.0 if d1 == d0 else -d0 / (d1 - d0)
        eer = fpr[i] + t * (fpr[i + 1] - fpr[i])
    else:
        eer = fpr[np.argmin(np.abs(diff))]
    far_at_frr = fpr[fnr <= OP_FRR].min() if np.any(fnr <= OP_FRR) else 1.0
    frr_at_far = fnr[fpr <= OP_FAR].min() if np.any(fpr <= OP_FAR) else 1.0
    return float(eer), float(far_at_frr), float(frr_at_far)

# ----------------------------- CLASSIFIERS -----------------------------
def make_clf(name, p, seed=0):
    if name == 'k-NN':       return KNeighborsClassifier(n_neighbors=p['k'], metric='euclidean')
    if name == 'LDA':        return LinearDiscriminantAnalysis(solver='svd')
    if name == 'Linear SVM': return LinearSVC(C=p['C'], max_iter=5000)
    if name == 'MLP':        return MLPClassifier(hidden_layer_sizes=(p['h'],), alpha=p['alpha'],
                                                  activation='relu', solver='adam',
                                                  max_iter=1000, early_stopping=True,
                                                  n_iter_no_change=20, random_state=seed)
    raise ValueError(name)

def get_score(clf, X):
    if hasattr(clf, 'predict_proba'):    return clf.predict_proba(X)[:, 1]
    if hasattr(clf, 'decision_function'): return clf.decision_function(X)
    return clf.predict(X).astype(float)

def select_hyperparams(name, Xtr_raw, ytr, grid, seed):
    """
    Pick the grid entry with lowest EER on an inner validation split.

    IMPORTANT: preprocessing is fit on the INNER-TRAINING fold only. The inner
    validation fold never contributes to StandardScaler statistics. This keeps
    hyperparameter selection free of validation-fold preprocessing leakage.
    """
    if len(grid) == 1:
        return grid[0]
    try:
        Xi_raw, Xv_raw, yi, yv = train_test_split(
            Xtr_raw, ytr, test_size=0.3, stratify=ytr, random_state=seed
        )
    except ValueError:
        return grid[0]                      # too few of a class to split; use default
    if len(np.unique(yi)) < 2 or len(np.unique(yv)) < 2:
        return grid[0]

    # Fit preprocessing ONLY on the inner-training fold, then apply it to the
    # untouched validation fold. The same transformed Xi/Xv can be reused across
    # candidates because the preprocessing itself has no tuned hyperparameters.
    sc_inner = StandardScaler().fit(Xi_raw)
    Xi = sc_inner.transform(Xi_raw)
    Xv = sc_inner.transform(Xv_raw)

    best, best_eer = grid[0], 1.0
    for p in grid:
        try:
            c = make_clf(name, p, seed=seed)
            c.fit(Xi, yi)
            e, _, _ = eer_and_ops(yv, get_score(c, Xv))
        except Exception:
            continue
        if e < best_eer:
            best_eer, best = e, p
    return best

# ----------------------------- DATA / TASK -----------------------------
def load_cmu(path):
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c not in ('subject', 'sessionIndex', 'rep')]
    subjects = sorted(df['subject'].unique())
    S, idx = {}, {}
    for i, s in enumerate(subjects):
        sub = df[df['subject'] == s]
        S[s] = {'X': sub[feats].values.astype(float), 'sess': sub['sessionIndex'].values}
        idx[s] = i
    return S, subjects, idx, len(feats)

def build_task(S, subjects, idx, genuine, N, seed, split, balance):
    rng = np.random.RandomState(seed * 100003 + idx[genuine])
    g = S[genuine]
    if split == 'session':
        gtr_pool = g['X'][np.isin(g['sess'], TRAIN_SESSIONS)]
        gte_pool = g['X'][np.isin(g['sess'], TEST_SESSIONS)]
    else:  # random, but with a FIXED reserved test set per seed
        perm = rng.permutation(len(g['X']))
        gte_pool = g['X'][perm[:GEN_TEST_SIZE]]
        gtr_pool = g['X'][perm[GEN_TEST_SIZE:]]
    if len(gtr_pool) < N or len(gte_pool) == 0:
        return None
    g_train = gtr_pool[rng.permutation(len(gtr_pool))[:N]]     # nested subset
    g_test  = gte_pool[:GEN_TEST_SIZE]

    others = [s for s in subjects if s != genuine]
    def pool(which):
        out = []
        for s in others:
            o = S[s]
            if split == 'session':
                sess = TRAIN_SESSIONS if which == 'train' else TEST_SESSIONS
                ids = np.where(np.isin(o['sess'], sess))[0]
            else:
                pr = np.random.RandomState(seed * 911 + idx[s]).permutation(len(o['X']))
                h = len(pr) // 2
                ids = pr[:h] if which == 'train' else pr[h:]
            out.append((o['X'], ids))
        return out

    if balance == 'fixed':
        imp_train = np.vstack([X[rng.choice(ids, min(IMP_TRAIN_PER, len(ids)), replace=False)]
                               for X, ids in pool('train')])
    else:  # balanced: total impostor-train == N
        allp = np.vstack([X[ids] for X, ids in pool('train')])
        sel = rng.choice(len(allp), min(N, len(allp)), replace=False)
        imp_train = allp[sel]
    imp_test = np.vstack([X[rng.choice(ids, min(IMP_TEST_PER, len(ids)), replace=False)]
                          for X, ids in pool('test')])
    return g_train, imp_train, g_test, imp_test

# ----------------------------- EVALUATE ONE CELL -----------------------------
def evaluate(name, task, grid, mlp_inits, seed):
    g_train, imp_train, g_test, imp_test = task
    if name == 'Threshold':
        mu, sig = g_train.mean(0), g_train.std(0) + 1e-9
        yte = np.r_[np.ones(len(g_test)), np.zeros(len(imp_test))]
        Xte = np.vstack([g_test, imp_test])
        score = -np.mean(np.abs((Xte - mu) / sig), axis=1)   # continuous, higher=genuine
        return eer_and_ops(yte, score)

    Xtr_raw = np.vstack([g_train, imp_train]); ytr = np.r_[np.ones(len(g_train)), np.zeros(len(imp_train))]
    Xte_raw = np.vstack([g_test, imp_test]);  yte = np.r_[np.ones(len(g_test)), np.zeros(len(imp_test))]

    # Hyperparameter selection receives RAW outer-training data and performs its
    # own inner-training-only scaling. After selection, preprocessing is refit on
    # the complete outer-training set for the final model, then applied to test.
    best = select_hyperparams(name, Xtr_raw, ytr, grid, seed)
    sc = StandardScaler().fit(Xtr_raw)
    Xtr = sc.transform(Xtr_raw)
    Xte = sc.transform(Xte_raw)

    inits = mlp_inits if name == 'MLP' else 1
    scores = []
    for it in range(inits):
        try:
            c = make_clf(name, best, seed=it); c.fit(Xtr, ytr)
            scores.append(get_score(c, Xte))
        except Exception:
            continue
    if not scores:
        return None
    return eer_and_ops(yte, np.mean(scores, axis=0))

# ----------------------------- SUBJECT-LEVEL STATS -----------------------------
def paired_stats(a, b):
    """a, b: per-subject EER arrays (paired). Returns dict of t/Wilcoxon/Cohen d."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    t_p = stats.ttest_rel(a, b).pvalue
    try:
        w_p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w_p = np.nan
    cohen = d.mean() / (d.std(ddof=1) + 1e-12)
    return {'mean_delta': d.mean() * 100, 't_p': t_p, 'wilcoxon_p': w_p, 'cohen_d': cohen, 'n': len(a)}

def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        run = max(run, val)
        adj[i] = min(run, 1.0)
    return adj

# ----------------------------- MAIN -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--csv', default=CSV_PATH)
    args = ap.parse_args()

    quick   = args.quick
    grid    = GRID_QUICK if quick else GRID_FULL
    inits   = MLP_INITS_QUICK if quick else MLP_INITS_FULL
    seeds   = [1, 2] if quick else [1, 2, 3, 4, 5]
    splits  = ['session'] if quick else ['session', 'random']
    bals    = ['fixed']   if quick else ['fixed', 'balanced']

    os.makedirs(OUTDIR, exist_ok=True)
    S, subjects, idx, nfeat = load_cmu(args.csv)
    print(f"Loaded CMU: {len(subjects)} subjects, {nfeat} features "
          f"| mode={'QUICK' if quick else 'FULL'} | seeds={seeds} | splits={splits} | balance={bals}")

    rows = []
    for split in splits:
        for balance in bals:
            for N in N_LIST:
                for name in CLASSIFIERS:
                    for genuine in subjects:
                        vals = []
                        for seed in seeds:
                            task = build_task(S, subjects, idx, genuine, N, seed, split, balance)
                            if task is None:
                                continue
                            r = evaluate(name, task, grid[name], inits, seed)
                            if r is not None:
                                vals.append(r)
                        if not vals:
                            continue
                        vals = np.array(vals)  # (seeds, 3)
                        rows.append(dict(split=split, balance=balance, N=N, classifier=name,
                                         subject=genuine,
                                         eer=vals[:, 0].mean(),
                                         far_at_frr=vals[:, 1].mean(),
                                         frr_at_far=vals[:, 2].mean()))
                    print(f"  done: split={split} balance={balance} N={N:>3} {name}", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(f'{OUTDIR}/persubject.csv', index=False)

    # subject-level summary with 95% CI (t-based, across subjects)
    def ci95(x):
        x = np.asarray(x); n = len(x)
        if n < 2: return (np.nan, np.nan)
        h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
        return (x.mean() - h, x.mean() + h)
    summ = []
    for (split, balance, N, name), grp in per.groupby(['split', 'balance', 'N', 'classifier']):
        lo, hi = ci95(grp['eer'].values * 100)
        summ.append(dict(split=split, balance=balance, N=N, classifier=name,
                         eer_mean=grp['eer'].mean() * 100, eer_sd=grp['eer'].std(ddof=1) * 100,
                         ci95_lo=lo, ci95_hi=hi,
                         far_at_frr=grp['far_at_frr'].mean() * 100,
                         frr_at_far=grp['frr_at_far'].mean() * 100,
                         n_subjects=len(grp)))
    summ = pd.DataFrame(summ).sort_values(['split', 'balance', 'N', 'eer_mean'])
    summ.to_csv(f'{OUTDIR}/summary.csv', index=False)

    # ---- print the headline table (session / fixed) ----
    print("\n" + "=" * 82)
    print("SUBJECT-LEVEL EER  (session-disjoint split, fixed-impostor)  mean [95% CI] over subjects")
    print("=" * 82)
    view = summ[(summ.split == 'session') & (summ.balance == 'fixed')]
    print(f"{'Classifier':<12}" + "".join(f"{('N='+str(n)):>22}" for n in N_LIST))
    for name in CLASSIFIERS:
        line = f"{name:<12}"
        for N in N_LIST:
            r = view[(view.classifier == name) & (view.N == N)]
            if len(r):
                r = r.iloc[0]
                line += f"{r.eer_mean:>7.1f} [{r.ci95_lo:4.1f},{r.ci95_hi:4.1f}]  "
            else:
                line += f"{'--':>22}"
        print(line)

    # ---- subject-level paired stats (session / fixed) for the key pairs ----
    def eer_vec(name, N, split='session', balance='fixed'):
        d = per[(per.classifier == name) & (per.N == N) &
                (per.split == split) & (per.balance == balance)].sort_values('subject')
        return d.set_index('subject')['eer']
    pairs = [('k-NN', 'LDA', 200), ('k-NN', 'Linear SVM', 200), ('LDA', 'Linear SVM', 200),
             ('k-NN', 'MLP', 200), ('k-NN', 'MLP', 10), ('LDA', 'MLP', 10), ('k-NN', 'LDA', 10)]
    stat_rows, pvals = [], []
    for a, b, N in pairs:
        va, vb = eer_vec(a, N), eer_vec(b, N)
        common = va.index.intersection(vb.index)
        st = paired_stats(va.loc[common].values, vb.loc[common].values)
        st.update(dict(A=a, B=b, N=N)); stat_rows.append(st); pvals.append(st['t_p'])
    if pvals:
        adj = holm(np.array(pvals))
        for r, a in zip(stat_rows, adj): r['holm_p'] = a
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(f'{OUTDIR}/stats.csv', index=False)

    print("\n" + "=" * 82)
    print("SUBJECT-LEVEL PAIRED TESTS  (session-disjoint / fixed)  -- delta<0 favours A")
    print("=" * 82)
    print(f"{'A vs B':<22}{'N':>5}{'dEER%':>8}{'paired t p':>13}{'Wilcoxon p':>13}{'Cohen d':>9}{'Holm p':>10}")
    for r in stat_rows:
        print(f"{r['A']+' vs '+r['B']:<22}{r['N']:>5}{r['mean_delta']:>8.2f}"
              f"{r['t_p']:>13.2e}{r['wilcoxon_p']:>13.2e}{r['cohen_d']:>9.2f}{r.get('holm_p', np.nan):>10.2e}")
    print(f"\nWrote {OUTDIR}/persubject.csv, summary.csv, stats.csv")

if __name__ == '__main__':
    main()
