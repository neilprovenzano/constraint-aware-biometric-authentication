#!/usr/bin/env python3

"""
ORL PCA dimensionality sensitivity analysis.

Supplementary analysis for the revised manuscript.
This does NOT replace the locked primary ORL analysis.

Purpose:
  1. Report cumulative PCA explained variance.
  2. Quantify stability of explained variance across repeated splits.
  3. Evaluate sensitivity of EER to PCA dimensionality.

Protocol:
  - ORL: 40 subjects x 10 images
  - N=5 genuine enrollment / 5 genuine test
  - 5 impostor train / 5 impostor test images per other identity
  - 10 repeated split seeds
  - PCA dimensions: 10, 20, 40, 80, 150
  - StandardScaler and PCA fit on training data only
  - k-NN (k=3) and LDA
  - ROC-interpolated EER
  - subject is the statistical unit
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


N_TRAIN = 5
KNN_K = 3
PCA_DIMS = [10, 20, 40, 80, 150]
DEFAULT_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


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

    return np.frombuffer(
        data, dtype=np.uint8
    ).reshape(h, w).astype(np.float64)


def load_orl(root):
    X, y = [], []

    for subj in range(1, 41):
        folder = os.path.join(root, f"s{subj}")

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Missing ORL subject folder: {folder}"
            )

        for img in range(1, 11):
            path = os.path.join(folder, f"{img}.pgm")

            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing ORL image: {path}"
                )

            X.append(read_pgm(path).ravel())
            y.append(subj - 1)

    return np.asarray(X), np.asarray(y)


def split_indices(y, genuine, seed):
    """
    Same split construction as the locked primary ORL analysis.
    """

    rng = np.random.RandomState(seed)

    g_idx = np.where(y == genuine)[0]
    g_perm = rng.permutation(g_idx)

    g_train = g_perm[:N_TRAIN]
    g_test = g_perm[N_TRAIN:]

    imp_train = []
    imp_test = []

    for subject in np.unique(y):

        if subject == genuine:
            continue

        idx = np.where(y == subject)[0]
        perm = rng.permutation(idx)

        imp_train.extend(perm[:N_TRAIN])
        imp_test.extend(perm[N_TRAIN:])

    return (
        np.asarray(g_train, dtype=int),
        np.asarray(g_test, dtype=int),
        np.asarray(imp_train, dtype=int),
        np.asarray(imp_test, dtype=int),
    )


def eer_roc(genuine_scores, impostor_scores):

    y_true = np.r_[
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ]

    scores = np.r_[
        genuine_scores,
        impostor_scores
    ]

    fpr, tpr, _ = roc_curve(
        y_true,
        scores,
        drop_intermediate=False
    )

    fnr = 1.0 - tpr
    d = fpr - fnr

    exact = np.where(np.isclose(d, 0.0))[0]

    if len(exact):
        i = exact[0]
        return float((fpr[i] + fnr[i]) / 2.0)

    crossings = np.where(
        np.sign(d[:-1]) != np.sign(d[1:])
    )[0]

    if len(crossings):

        i = crossings[0]

        denom = d[i + 1] - d[i]

        alpha = (
            0.0
            if denom == 0
            else -d[i] / denom
        )

        far = (
            fpr[i]
            + alpha * (fpr[i + 1] - fpr[i])
        )

        frr = (
            fnr[i]
            + alpha * (fnr[i + 1] - fnr[i])
        )

        return float((far + frr) / 2.0)

    i = int(np.argmin(np.abs(d)))

    return float((fpr[i] + fnr[i]) / 2.0)


def classifier_score(clf, X):

    if hasattr(clf, "predict_proba"):

        p = clf.predict_proba(X)

        genuine_col = list(clf.classes_).index(1)

        return p[:, genuine_col]

    if hasattr(clf, "decision_function"):

        return np.asarray(
            clf.decision_function(X)
        ).ravel()

    raise RuntimeError(
        "Classifier has no continuous score."
    )


def mean_ci95(values):

    values = np.asarray(values, dtype=float)

    mean = float(np.mean(values))

    if len(values) < 2:
        return mean, np.nan, np.nan

    se = stats.sem(values)

    q = stats.t.ppf(
        0.975,
        len(values) - 1
    )

    return (
        mean,
        float(mean - q * se),
        float(mean + q * se),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--orl-path",
        default="."
    )

    parser.add_argument(
        "--outdir",
        default="out_orl_pca_sensitivity"
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    X, y = load_orl(args.orl_path)

    subjects = np.unique(y)

    print(
        f"Loaded ORL: {len(X)} images, "
        f"{len(subjects)} subjects, "
        f"{X.shape[1]} pixels/image"
    )

    print(
        f"PCA sensitivity dimensions: {PCA_DIMS}"
    )

    print(
        f"Repeated split seeds: {args.seeds}"
    )

    print(
        "StandardScaler and PCA are fitted "
        "on training data only.\n"
    )

    eer_rows = []
    variance_rows = []

    for seed in args.seeds:

        print(
            f"Split seed {seed}/{args.seeds[-1]} ...",
            flush=True
        )

        for genuine in subjects:

            gtr, gte, itr, ite = split_indices(
                y,
                genuine,
                seed
            )

            Xtr = np.vstack([
                X[gtr],
                X[itr]
            ])

            ytr = np.r_[
                np.ones(len(gtr), dtype=int),
                np.zeros(len(itr), dtype=int)
            ]

            Xgte = X[gte]
            Xite = X[ite]

            # Training-only standardization.
            scaler = StandardScaler()

            Xtr_s = scaler.fit_transform(Xtr)
            Xgte_s = scaler.transform(Xgte)
            Xite_s = scaler.transform(Xite)

            for dim in PCA_DIMS:

                n_components = min(
                    dim,
                    Xtr_s.shape[0] - 1,
                    Xtr_s.shape[1]
                )

                # Training-only PCA.
                pca = PCA(
                    n_components=n_components,
                    svd_solver="full"
                )

                Xtr_p = pca.fit_transform(Xtr_s)
                Xgte_p = pca.transform(Xgte_s)
                Xite_p = pca.transform(Xite_s)

                explained = float(
                    np.sum(
                        pca.explained_variance_ratio_
                    )
                )

                variance_rows.append({
                    "seed": seed,
                    "subject": int(genuine + 1),
                    "requested_components": dim,
                    "actual_components": n_components,
                    "explained_variance": explained,
                    "explained_variance_pct":
                        100.0 * explained,
                })

                for classifier in [
                    "k-NN",
                    "LDA"
                ]:

                    if classifier == "k-NN":

                        clf = KNeighborsClassifier(
                            n_neighbors=KNN_K,
                            metric="euclidean"
                        )

                    else:

                        clf = (
                            LinearDiscriminantAnalysis()
                        )

                    clf.fit(
                        Xtr_p,
                        ytr
                    )

                    genuine_scores = (
                        classifier_score(
                            clf,
                            Xgte_p
                        )
                    )

                    impostor_scores = (
                        classifier_score(
                            clf,
                            Xite_p
                        )
                    )

                    eer = eer_roc(
                        genuine_scores,
                        impostor_scores
                    )

                    eer_rows.append({
                        "seed": seed,
                        "subject":
                            int(genuine + 1),
                        "classifier":
                            classifier,
                        "pca_components":
                            n_components,
                        "eer_pct":
                            100.0 * eer,
                    })

    eer_raw = pd.DataFrame(eer_rows)

    variance_raw = pd.DataFrame(
        variance_rows
    )

    eer_raw.to_csv(
        outdir / "eer_by_split_subject.csv",
        index=False
    )

    variance_raw.to_csv(
        outdir / "explained_variance_by_split_subject.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Subject-level EER aggregation
    # ---------------------------------------------------------

    per_subject = (
        eer_raw
        .groupby(
            [
                "subject",
                "classifier",
                "pca_components"
            ],
            as_index=False
        )
        .agg(
            eer_pct=("eer_pct", "mean"),
            split_sd_pct=("eer_pct", "std"),
            n_splits=("seed", "nunique")
        )
    )

    per_subject.to_csv(
        outdir / "eer_per_subject.csv",
        index=False
    )

    eer_summary_rows = []

    for (classifier, dim), group in (
        per_subject.groupby(
            [
                "classifier",
                "pca_components"
            ]
        )
    ):

        mean, lo, hi = mean_ci95(
            group["eer_pct"]
        )

        eer_summary_rows.append({
            "classifier": classifier,
            "pca_components": dim,
            "mean_eer_pct": mean,
            "ci95_low_pct": lo,
            "ci95_high_pct": hi,
            "n_subjects": len(group),
        })

    eer_summary = pd.DataFrame(
        eer_summary_rows
    ).sort_values(
        [
            "classifier",
            "pca_components"
        ]
    )

    eer_summary.to_csv(
        outdir / "eer_sensitivity_summary.csv",
        index=False
    )

    # ---------------------------------------------------------
    # PCA explained-variance stability
    # ---------------------------------------------------------

    variance_summary = (
        variance_raw
        .groupby(
            "actual_components",
            as_index=False
        )
        .agg(
            mean_explained_variance_pct=(
                "explained_variance_pct",
                "mean"
            ),
            sd_explained_variance_pct=(
                "explained_variance_pct",
                "std"
            ),
            min_explained_variance_pct=(
                "explained_variance_pct",
                "min"
            ),
            max_explained_variance_pct=(
                "explained_variance_pct",
                "max"
            ),
            n_fits=(
                "explained_variance_pct",
                "size"
            )
        )
    )

    variance_summary.to_csv(
        outdir / "pca_variance_summary.csv",
        index=False
    )

    # Split-level PCA-40 stability.
    pca40 = variance_raw[
        variance_raw[
            "actual_components"
        ] == 40
    ]

    pca40_split = (
        pca40
        .groupby(
            "seed",
            as_index=False
        )
        .agg(
            mean_explained_variance_pct=(
                "explained_variance_pct",
                "mean"
            )
        )
    )

    pca40_split.to_csv(
        outdir / "pca40_split_stability.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Terminal report
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 88
    )

    print(
        "PCA EXPLAINED-VARIANCE SENSITIVITY"
    )

    print(
        "=" * 88
    )

    print(
        f"{'Components':>12} "
        f"{'Mean variance %':>18} "
        f"{'SD %':>10} "
        f"{'Min %':>10} "
        f"{'Max %':>10}"
    )

    for _, row in (
        variance_summary.iterrows()
    ):

        print(
            f"{int(row['actual_components']):>12} "
            f"{row['mean_explained_variance_pct']:>18.2f} "
            f"{row['sd_explained_variance_pct']:>10.2f} "
            f"{row['min_explained_variance_pct']:>10.2f} "
            f"{row['max_explained_variance_pct']:>10.2f}"
        )

    print(
        "\nPCA-40 SPLIT-LEVEL STABILITY"
    )

    print(
        "=" * 88
    )

    for _, row in (
        pca40_split.iterrows()
    ):

        print(
            f"Seed {int(row['seed']):>2}: "
            f"{row['mean_explained_variance_pct']:.2f}%"
        )

    print(
        "\nAcross split means:"
    )

    print(
        f"Mean = "
        f"{pca40_split['mean_explained_variance_pct'].mean():.2f}%"
    )

    print(
        f"SD   = "
        f"{pca40_split['mean_explained_variance_pct'].std(ddof=1):.3f} "
        "percentage points"
    )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "PCA DIMENSIONALITY SENSITIVITY — SUBJECT-LEVEL EER"
    )

    print(
        "=" * 88
    )

    print(
        f"{'Classifier':<12} "
        f"{'Components':>12} "
        f"{'Mean EER %':>14} "
        f"{'95% CI':>22}"
    )

    for _, row in (
        eer_summary.iterrows()
    ):

        ci = (
            f"[{row['ci95_low_pct']:.2f}, "
            f"{row['ci95_high_pct']:.2f}]"
        )

        print(
            f"{row['classifier']:<12} "
            f"{int(row['pca_components']):>12} "
            f"{row['mean_eer_pct']:>14.2f} "
            f"{ci:>22}"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a sensitivity analysis. "
        "PCA dimensionality is NOT selected "
        "from test performance."
    )

    print(
        "The predefined PCA-40 condition "
        "remains the primary PCA comparison."
    )

    print(
        "\nWrote:"
    )

    for filename in [
        "eer_by_split_subject.csv",
        "explained_variance_by_split_subject.csv",
        "eer_per_subject.csv",
        "eer_sensitivity_summary.csv",
        "pca_variance_summary.csv",
        "pca40_split_stability.csv",
    ]:

        print(
            f"  {outdir}/{filename}"
        )


if __name__ == "__main__":
    main()
