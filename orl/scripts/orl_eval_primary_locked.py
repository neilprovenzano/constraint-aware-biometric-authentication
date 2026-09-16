#!/usr/bin/env python3
"""
Revised ORL face-verification experiment for the manuscript.

Primary analysis:
  - N=5 genuine enrollment images per subject
  - 5 held-out genuine test images per subject
  - repeated random splits
  - one-vs-rest verification for all 40 ORL subjects
  - k-NN and LDA
  - raw standardized pixels and PCA
  - PCA fit on training data only
  - ROC-interpolated EER
  - subject-level aggregation and 95% CIs

N=8 is intentionally excluded from the primary analysis because it leaves
only two genuine test images per subject. The manuscript should remove the
old 0.4% N=8 headline result rather than treat it as robust evidence.

Run:
    python3 orl_eval.py

Optional:
    python3 orl_eval.py --orl-path /path/to/orl_faces
    python3 orl_eval.py --seeds 1 2 3 4 5 6 7 8 9 10

Outputs:
    out_orl/persubject.csv
    out_orl/summary.csv
    out_orl/paired_tests.csv
    out_orl/environment.json
"""

import argparse
import json
import os
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DEFAULT_ORL_PATH = "orl_faces"
N_TRAIN = 5
KNN_K = 3
PCA_COMPONENTS = 40


def read_pgm(path):
    with open(path, "rb") as f:
        if f.readline() != b"P5\n":
            raise ValueError(f"Unexpected PGM format: {path}")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        _ = int(f.readline())
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w).astype(np.float64)


def load_orl(root):
    X, y, image_num = [], [], []
    for subj in range(1, 41):
        folder = os.path.join(root, f"s{subj}")
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing ORL subject folder: {folder}")
        for img in range(1, 11):
            path = os.path.join(folder, f"{img}.pgm")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing ORL image: {path}")
            X.append(read_pgm(path).ravel())
            y.append(subj - 1)
            image_num.append(img)
    return np.asarray(X), np.asarray(y), np.asarray(image_num)


def eer_roc(genuine_scores, impostor_scores):
    """ROC-interpolated EER; higher score means more genuine."""
    y_true = np.r_[np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))]
    scores = np.r_[genuine_scores, impostor_scores]
    fpr, tpr, _ = roc_curve(y_true, scores, drop_intermediate=False)
    fnr = 1.0 - tpr
    d = fpr - fnr

    exact = np.where(np.isclose(d, 0.0))[0]
    if len(exact):
        return float((fpr[exact[0]] + fnr[exact[0]]) / 2.0)

    crossings = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(crossings):
        i = crossings[0]
        denom = d[i + 1] - d[i]
        a = 0.0 if denom == 0 else -d[i] / denom
        far = fpr[i] + a * (fpr[i + 1] - fpr[i])
        frr = fnr[i] + a * (fnr[i + 1] - fnr[i])
        return float((far + frr) / 2.0)

    i = int(np.argmin(np.abs(d)))
    return float((fpr[i] + fnr[i]) / 2.0)


def score_classifier(clf, X):
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        # class 1 is genuine
        col = list(clf.classes_).index(1)
        return p[:, col]
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(X)).ravel()
    raise RuntimeError("Classifier does not provide a continuous score.")


def make_classifier(name):
    if name == "k-NN":
        return KNeighborsClassifier(n_neighbors=KNN_K, metric="euclidean")
    if name == "LDA":
        return LinearDiscriminantAnalysis()
    raise ValueError(name)


def split_indices(y, genuine, seed):
    """
    Repeated per-identity split.

    For the genuine identity: 5 train / 5 test.
    For every impostor identity: 5 train / 5 test.

    This keeps train/test image sets disjoint within every identity and avoids
    the old fixed image-1/2 vs image-3..10 impostor partition.
    """
    rng = np.random.RandomState(seed)
    g_idx = np.where(y == genuine)[0]
    g_perm = rng.permutation(g_idx)
    g_train = g_perm[:N_TRAIN]
    g_test = g_perm[N_TRAIN:]

    imp_train, imp_test = [], []
    for s in np.unique(y):
        if s == genuine:
            continue
        idx = np.where(y == s)[0]
        # subject-specific deterministic mixing for this repeated split
        perm = rng.permutation(idx)
        imp_train.extend(perm[:N_TRAIN])
        imp_test.extend(perm[N_TRAIN:])

    return (
        np.asarray(g_train, dtype=int),
        np.asarray(g_test, dtype=int),
        np.asarray(imp_train, dtype=int),
        np.asarray(imp_test, dtype=int),
    )


def fit_and_score(X, y, genuine, seed, classifier, use_pca):
    gtr, gte, itr, ite = split_indices(y, genuine, seed)

    Xtr = np.vstack([X[gtr], X[itr]])
    ytr = np.r_[np.ones(len(gtr), dtype=int),
                np.zeros(len(itr), dtype=int)]
    Xgte = X[gte]
    Xite = X[ite]

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xgte_s = scaler.transform(Xgte)
    Xite_s = scaler.transform(Xite)

    n_components = None
    if use_pca:
        # Predefined dimensionality; PCA is fit ONLY on training data.
        # 40 components is fixed before looking at test performance.
        n_components = min(PCA_COMPONENTS, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
        pca = PCA(n_components=n_components, svd_solver="full")
        Xtr_s = pca.fit_transform(Xtr_s)
        Xgte_s = pca.transform(Xgte_s)
        Xite_s = pca.transform(Xite_s)

    clf = make_classifier(classifier)
    clf.fit(Xtr_s, ytr)

    gs = score_classifier(clf, Xgte_s)
    ims = score_classifier(clf, Xite_s)
    eer = eer_roc(gs, ims)

    return {
        "eer": eer,
        "n_genuine_train": len(gtr),
        "n_genuine_test": len(gte),
        "n_impostor_train": len(itr),
        "n_impostor_test": len(ite),
        "pca_components": n_components,
    }


def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    if n < 2:
        return mean, np.nan, np.nan
    se = stats.sem(x)
    q = stats.t.ppf(0.975, n - 1)
    return mean, float(mean - q * se), float(mean + q * se)


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        out[idx] = running
    return out


def paired_test(a, b):
    """Subject-level paired inference; a-b < 0 favors A."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b, zero_method="wilcox")
        wp = float(w.pvalue)
    except Exception:
        wp = np.nan
    sd = np.std(d, ddof=1)
    dz = float(np.mean(d) / sd) if sd > 0 else 0.0
    return float(np.mean(d)), float(t.pvalue), wp, dz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orl-path", default=DEFAULT_ORL_PATH)
    ap.add_argument("--outdir", default="out_orl")
    ap.add_argument(
        "--seeds", nargs="+", type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Repeated split seeds."
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, image_num = load_orl(args.orl_path)
    subjects = np.unique(y)

    print(
        f"Loaded ORL: {len(X)} images, {len(subjects)} subjects, "
        f"{X.shape[1]} pixels/image"
    )
    print(
        f"Primary protocol: N={N_TRAIN}, {10-N_TRAIN} genuine test images/subject, "
        f"{len(args.seeds)} repeated splits"
    )
    print(
        f"Classifiers: k-NN (k={KNN_K}) and LDA | "
        f"representations: raw standardized pixels and PCA-{PCA_COMPONENTS}"
    )
    print("EER: ROC interpolation | inference unit: subject\n")

    rows = []

    for seed in args.seeds:
        print(f"Split seed {seed}/{args.seeds[-1]} ...", flush=True)
        for genuine in subjects:
            for classifier in ["k-NN", "LDA"]:
                for use_pca in [False, True]:
                    try:
                        r = fit_and_score(
                            X, y, genuine, seed, classifier, use_pca
                        )
                    except Exception as e:
                        print(
                            f"WARNING: seed={seed}, subject={genuine+1}, "
                            f"{classifier}, PCA={use_pca}: {e}"
                        )
                        continue
                    rows.append({
                        "seed": seed,
                        "subject": int(genuine + 1),
                        "classifier": classifier,
                        "representation": (
                            f"PCA-{r['pca_components']}" if use_pca else "Raw"
                        ),
                        "eer": r["eer"],
                        "eer_pct": 100.0 * r["eer"],
                        "n_genuine_train": r["n_genuine_train"],
                        "n_genuine_test": r["n_genuine_test"],
                        "n_impostor_train": r["n_impostor_train"],
                        "n_impostor_test": r["n_impostor_test"],
                        "pca_components": r["pca_components"],
                    })

    raw = pd.DataFrame(rows)

    # Repeated seeds are NOT treated as independent statistical units.
    # First average each condition over seeds for each subject.
    per_subject = (
        raw.groupby(["subject", "classifier", "representation"], as_index=False)
        .agg(
            eer_pct=("eer_pct", "mean"),
            split_sd_pct=("eer_pct", "std"),
            n_splits=("seed", "nunique"),
            n_genuine_test=("n_genuine_test", "first"),
            n_impostor_test=("n_impostor_test", "first"),
        )
    )
    per_subject.to_csv(outdir / "persubject.csv", index=False)

    summary_rows = []
    for (clf, rep), grp in per_subject.groupby(
        ["classifier", "representation"], sort=False
    ):
        mean, lo, hi = mean_ci95(grp["eer_pct"])
        summary_rows.append({
            "classifier": clf,
            "representation": rep,
            "n_subjects": len(grp),
            "eer_mean_pct": mean,
            "ci95_low_pct": lo,
            "ci95_high_pct": hi,
            "between_subject_sd_pct": float(grp["eer_pct"].std(ddof=1)),
            "mean_within_subject_split_sd_pct": float(
                grp["split_sd_pct"].mean()
            ),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "summary.csv", index=False)

    # Predefined subject-level paired comparisons.
    comparisons = [
        ("k-NN Raw vs PCA", ("k-NN", "Raw"), ("k-NN", f"PCA-{PCA_COMPONENTS}")),
        ("LDA Raw vs PCA", ("LDA", "Raw"), ("LDA", f"PCA-{PCA_COMPONENTS}")),
        ("k-NN vs LDA Raw", ("k-NN", "Raw"), ("LDA", "Raw")),
        ("k-NN vs LDA PCA", ("k-NN", f"PCA-{PCA_COMPONENTS}"),
                             ("LDA", f"PCA-{PCA_COMPONENTS}")),
    ]

    test_rows = []
    for label, A, B in comparisons:
        a = per_subject[
            (per_subject["classifier"] == A[0]) &
            (per_subject["representation"] == A[1])
        ].sort_values("subject")
        b = per_subject[
            (per_subject["classifier"] == B[0]) &
            (per_subject["representation"] == B[1])
        ].sort_values("subject")

        merged = a[["subject", "eer_pct"]].merge(
            b[["subject", "eer_pct"]], on="subject", suffixes=("_a", "_b")
        )
        delta, tp, wp, dz = paired_test(
            merged["eer_pct_a"], merged["eer_pct_b"]
        )
        test_rows.append({
            "comparison": label,
            "A": f"{A[0]} / {A[1]}",
            "B": f"{B[0]} / {B[1]}",
            "n_subjects": len(merged),
            "mean_delta_A_minus_B_pct_points": delta,
            "paired_t_p": tp,
            "wilcoxon_p": wp,
            "cohen_dz": dz,
        })

    tests = pd.DataFrame(test_rows)
    tests["holm_p_paired_t"] = holm_adjust(tests["paired_t_p"].values)
    tests.to_csv(outdir / "paired_tests.csv", index=False)

    env = {
        "dataset": "ORL face dataset",
        "orl_path": args.orl_path,
        "images": int(len(X)),
        "subjects": int(len(subjects)),
        "images_per_subject": 10,
        "N_train_genuine": N_TRAIN,
        "N_test_genuine": 10 - N_TRAIN,
        "impostor_train_per_identity": N_TRAIN,
        "impostor_test_per_identity": 10 - N_TRAIN,
        "seeds": args.seeds,
        "knn_k": KNN_K,
        "pca_components_predefined": PCA_COMPONENTS,
        "eer_method": "ROC interpolation",
        "statistical_unit": "subject after averaging repeated splits",
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
        "platform": platform.platform(),
        "note": (
            "N=8 intentionally excluded from primary revised analysis because "
            "it leaves only two genuine test images per subject."
        ),
    }
    (outdir / "environment.json").write_text(json.dumps(env, indent=2))

    print("\n" + "=" * 86)
    print("ORL REVISED SUBJECT-LEVEL RESULTS — N=5")
    print("=" * 86)
    print(
        f"{'Classifier':<10} {'Representation':<15} "
        f"{'Mean EER %':>12} {'95% CI':>24} {'Subjects':>10}"
    )
    for _, r in summary.iterrows():
        ci = f"[{r['ci95_low_pct']:.2f}, {r['ci95_high_pct']:.2f}]"
        print(
            f"{r['classifier']:<10} {r['representation']:<15} "
            f"{r['eer_mean_pct']:>12.2f} {ci:>24} "
            f"{int(r['n_subjects']):>10}"
        )

    print("\nPAIRED SUBJECT-LEVEL COMPARISONS")
    print("=" * 86)
    for _, r in tests.iterrows():
        print(
            f"{r['comparison']:<22} "
            f"delta={r['mean_delta_A_minus_B_pct_points']:+.2f} pp | "
            f"t p={r['paired_t_p']:.4g} | "
            f"W p={r['wilcoxon_p']:.4g} | "
            f"dz={r['cohen_dz']:+.2f} | "
            f"Holm={r['holm_p_paired_t']:.4g}"
        )

    print("\nRepeated split seeds are averaged within subject before inference.")
    print("PCA is fitted on training data only; test images never fit PCA/scaling.")
    print("N=8 is excluded from the revised primary analysis.")
    print("\nWrote:")
    print(f"  {outdir}/persubject.csv")
    print(f"  {outdir}/summary.csv")
    print(f"  {outdir}/paired_tests.csv")
    print(f"  {outdir}/environment.json")


if __name__ == "__main__":
    main()
