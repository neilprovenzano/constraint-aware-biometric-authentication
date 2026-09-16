#!/usr/bin/env python3
"""
Latency characterization for the revised Electronics manuscript.

This script intentionally does NOT claim "hard real-time" behavior.
It characterizes single-query CPU latency on the machine on which it is run.

It reuses the exact CMU task construction and hyperparameter-selection logic
from cmu_eval.py so the latency models match the revised accuracy experiment.

Primary protocol:
  - session-disjoint CMU split
  - fixed-impostor training condition
  - N = 10, 50, 200
  - all five classifier families
  - validation-selected hyperparameters
  - MLP uses the same 3-initialization score ensemble as the FULL accuracy run

Reported timing:
  - preprocessing-only latency
  - model-only latency
  - end-to-end latency (preprocessing + score)
  - mean, SD, P50, P95, P99, maximum
  - single-query batch size = 1
  - fixed warm-up before timing
  - fixed number of timed queries per subject/model/N

Also reported:
  - serialized deployment-bundle size
  - tuning time
  - final fit time
  - selected hyperparameters
  - machine/software metadata

Run from the same directory as cmu_eval.py and DSL-StrongPasswordData.csv:

    python3 latency_eval.py

Optional:
    python3 latency_eval.py --reps 500 --warmup 100
    python3 latency_eval.py --quick

Outputs:
    out_latency/latency_persubject.csv
    out_latency/latency_summary.csv
    out_latency/selected_hyperparams.csv
    out_latency/environment.json
    out_latency/raw_timings.csv.gz   (only with --save-raw)
"""

import argparse
import gc
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

import cmu_eval as ce


N_LIST = [10, 50, 200]
CLASSIFIERS = ce.CLASSIFIERS
DEFAULT_OUTDIR = "out_latency"


def machine_info():
    """Collect reproducibility metadata without requiring extra packages."""
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "cpu_count_logical": os.cpu_count(),
        "timer": "time.perf_counter_ns",
        "timing_batch_size": 1,
        "cpu_affinity_controlled": False,
        "cpu_frequency_controlled": False,
        "note": (
            "Desktop CPU characterization on the host OS; "
            "not a hard-real-time or embedded-hardware guarantee."
        ),
    }

    # Helpful on macOS / Apple Silicon; harmless if unavailable.
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if brand:
            info["cpu_brand"] = brand
    except Exception:
        pass

    try:
        model_name = subprocess.check_output(
            ["sysctl", "-n", "hw.model"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if model_name:
            info["hardware_model"] = model_name
    except Exception:
        pass

    return info


def percentile_summary(values_ns):
    """Return latency statistics in milliseconds."""
    x_ms = np.asarray(values_ns, dtype=np.float64) / 1e6
    return {
        "mean_ms": float(np.mean(x_ms)),
        "sd_ms": float(np.std(x_ms, ddof=1)) if len(x_ms) > 1 else 0.0,
        "p50_ms": float(np.percentile(x_ms, 50)),
        "p95_ms": float(np.percentile(x_ms, 95)),
        "p99_ms": float(np.percentile(x_ms, 99)),
        "max_ms": float(np.max(x_ms)),
        "n_queries": int(len(x_ms)),
    }


def build_deployment_bundle(name, task, grid, mlp_inits, seed):
    """
    Train one deployment-equivalent verifier.

    Returns:
      bundle      dict containing scaler/model(s) or threshold parameters
      best_params validation-selected hyperparameters
      tune_ms     hyperparameter-selection time
      fit_ms      final model-fit time
    """
    g_train, imp_train, g_test, imp_test = task

    if name == "Threshold":
        t0 = time.perf_counter_ns()
        mu = g_train.mean(axis=0)
        sig = g_train.std(axis=0) + 1e-9
        fit_ms = (time.perf_counter_ns() - t0) / 1e6
        bundle = {
            "name": name,
            "mu": mu,
            "sig": sig,
            "scaler": None,
            "models": None,
        }
        return bundle, {}, 0.0, fit_ms

    Xtr = np.vstack([g_train, imp_train])
    ytr = np.r_[np.ones(len(g_train)), np.zeros(len(imp_train))]

    # Scaling is part of deployment preprocessing.
    scaler = ce.StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)

    t0 = time.perf_counter_ns()
    best = ce.select_hyperparams(name, Xtr_s, ytr, grid, seed)
    tune_ms = (time.perf_counter_ns() - t0) / 1e6

    models = []
    n_inits = mlp_inits if name == "MLP" else 1

    t0 = time.perf_counter_ns()
    for init in range(n_inits):
        clf = ce.make_clf(name, best, seed=init)
        clf.fit(Xtr_s, ytr)
        models.append(clf)
    fit_ms = (time.perf_counter_ns() - t0) / 1e6

    bundle = {
        "name": name,
        "scaler": scaler,
        "models": models,
        "mu": None,
        "sig": None,
    }
    return bundle, best, tune_ms, fit_ms


def threshold_score(bundle, x_raw):
    z = np.abs((x_raw - bundle["mu"]) / bundle["sig"])
    return -float(np.mean(z))


def model_score(bundle, x_scaled):
    """Score one already-preprocessed query."""
    scores = []
    for clf in bundle["models"]:
        s = ce.get_score(clf, x_scaled.reshape(1, -1))
        scores.append(float(np.asarray(s).ravel()[0]))
    return float(np.mean(scores))


def warm_up(bundle, X_queries, order):
    """Untimed warm-up using the same end-to-end path as the benchmark."""
    name = bundle["name"]
    for j in order:
        x = X_queries[j]
        if name == "Threshold":
            _ = threshold_score(bundle, x)
        else:
            xs = bundle["scaler"].transform(x.reshape(1, -1))[0]
            _ = model_score(bundle, xs)


def benchmark_bundle(bundle, X_queries, order):
    """
    Time each query three ways:
      1) preprocessing only
      2) model score only, on already-preprocessed input
      3) full end-to-end preprocessing + score

    Random/query-selection overhead is excluded from timed regions because
    'order' is generated before timing.
    """
    name = bundle["name"]

    pre_ns = np.zeros(len(order), dtype=np.int64)
    model_ns = np.zeros(len(order), dtype=np.int64)
    e2e_ns = np.zeros(len(order), dtype=np.int64)

    for k, j in enumerate(order):
        x = X_queries[j]

        if name == "Threshold":
            # There is no separate StandardScaler stage for the threshold rule.
            t0 = time.perf_counter_ns()
            _ = threshold_score(bundle, x)
            t1 = time.perf_counter_ns()
            dt = t1 - t0
            pre_ns[k] = 0
            model_ns[k] = dt
            e2e_ns[k] = dt
            continue

        # Preprocessing-only
        t0 = time.perf_counter_ns()
        xs = bundle["scaler"].transform(x.reshape(1, -1))[0]
        t1 = time.perf_counter_ns()
        pre_ns[k] = t1 - t0

        # Model-only
        t0 = time.perf_counter_ns()
        _ = model_score(bundle, xs)
        t1 = time.perf_counter_ns()
        model_ns[k] = t1 - t0

        # End-to-end: preprocessing + model score in one contiguous timed region
        t0 = time.perf_counter_ns()
        xs2 = bundle["scaler"].transform(x.reshape(1, -1))[0]
        _ = model_score(bundle, xs2)
        t1 = time.perf_counter_ns()
        e2e_ns[k] = t1 - t0

    return pre_ns, model_ns, e2e_ns


def mode_json_strings(series):
    vals = [str(v) for v in series if pd.notna(v)]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=ce.CSV_PATH)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=1000,
                    help="Timed single-query repetitions per subject/model/N.")
    ap.add_argument("--warmup", type=int, default=100,
                    help="Untimed warm-up queries per subject/model/N.")
    ap.add_argument("--quick", action="store_true",
                    help="Smaller grids, 200 reps, 25 warm-up queries.")
    ap.add_argument("--save-raw", action="store_true",
                    help="Also save every query timing to raw_timings.csv.gz.")
    args = ap.parse_args()

    if args.quick:
        grid = ce.GRID_QUICK
        mlp_inits = ce.MLP_INITS_QUICK
        reps = min(args.reps, 200)
        warmup_n = min(args.warmup, 25)
    else:
        grid = ce.GRID_FULL
        mlp_inits = ce.MLP_INITS_FULL
        reps = args.reps
        warmup_n = args.warmup

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = machine_info()
    env.update({
        "dataset": args.csv,
        "split": "session",
        "balance": "fixed",
        "seed": args.seed,
        "N_values": N_LIST,
        "classifiers": CLASSIFIERS,
        "warmup_queries_per_subject_model_N": warmup_n,
        "timed_queries_per_subject_model_N": reps,
        "mlp_initializations": mlp_inits,
        "preprocessing_in_end_to_end": True,
        "raw_query_timing_saved": bool(args.save_raw),
    })
    (outdir / "environment.json").write_text(json.dumps(env, indent=2))

    S, subjects, idx, nfeat = ce.load_cmu(args.csv)

    print(
        f"Loaded CMU: {len(subjects)} subjects, {nfeat} features | "
        f"session-disjoint / fixed-impostor | seed={args.seed}"
    )
    print(
        f"Warm-up={warmup_n} queries/model/subject | "
        f"Timed={reps} queries/model/subject | batch size=1"
    )
    print("Timing: preprocessing-only, model-only, and end-to-end")
    print("NOTE: desktop CPU characterization; not a hard-real-time guarantee.\n")

    per_subject_rows = []
    hp_rows = []
    raw_rows = []

    for N in N_LIST:
        for name in CLASSIFIERS:
            print(f"Benchmarking N={N:>3}  {name} ...", flush=True)

            for genuine in subjects:
                task = ce.build_task(
                    S, subjects, idx, genuine, N, args.seed,
                    split="session", balance="fixed"
                )
                if task is None:
                    continue

                bundle, best, tune_ms, fit_ms = build_deployment_bundle(
                    name, task, grid[name], mlp_inits, args.seed
                )

                _, _, g_test, imp_test = task
                Xq = np.vstack([g_test, imp_test])

                # Deterministic query orders generated outside timed regions.
                rng = np.random.RandomState(
                    args.seed * 1000003 + idx[genuine] * 97 + N
                )
                warm_order = rng.randint(0, len(Xq), size=warmup_n)
                timed_order = rng.randint(0, len(Xq), size=reps)

                gc.collect()
                warm_up(bundle, Xq, warm_order)

                # Serialized deployment bundle size. This is a storage-size proxy,
                # not a claim of exact resident-memory consumption.
                bundle_bytes = len(
                    pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
                )

                pre_ns, model_ns, e2e_ns = benchmark_bundle(
                    bundle, Xq, timed_order
                )

                row = {
                    "N": N,
                    "classifier": name,
                    "subject": genuine,
                    "selected_params": json.dumps(best, sort_keys=True),
                    "tune_ms": tune_ms,
                    "fit_ms": fit_ms,
                    "bundle_kb": bundle_bytes / 1024.0,
                    "warmup_queries": warmup_n,
                    "timed_queries": reps,
                }

                for prefix, arr in [
                    ("pre", pre_ns),
                    ("model", model_ns),
                    ("e2e", e2e_ns),
                ]:
                    st = percentile_summary(arr)
                    for key, value in st.items():
                        row[f"{prefix}_{key}"] = value

                per_subject_rows.append(row)

                hp_rows.append({
                    "N": N,
                    "classifier": name,
                    "subject": genuine,
                    "selected_params": json.dumps(best, sort_keys=True),
                })

                if args.save_raw:
                    for q_i in range(reps):
                        raw_rows.append({
                            "N": N,
                            "classifier": name,
                            "subject": genuine,
                            "query_index": q_i,
                            "pre_ms": pre_ns[q_i] / 1e6,
                            "model_ms": model_ns[q_i] / 1e6,
                            "e2e_ms": e2e_ns[q_i] / 1e6,
                        })

    per = pd.DataFrame(per_subject_rows)
    per.to_csv(outdir / "latency_persubject.csv", index=False)

    hp = pd.DataFrame(hp_rows)
    hp.to_csv(outdir / "selected_hyperparams.csv", index=False)

    if args.save_raw:
        pd.DataFrame(raw_rows).to_csv(
            outdir / "raw_timings.csv.gz",
            index=False,
            compression="gzip"
        )

    # Publication-facing summary:
    # Each subject contributes one subject-level latency summary.
    # We summarize those subject-level statistics, preventing subjects with
    # repeated query timings from being treated as independent experimental units.
    summary_rows = []
    for (N, name), grp in per.groupby(["N", "classifier"], sort=False):
        row = {
            "N": N,
            "classifier": name,
            "n_subjects": int(len(grp)),
            "selected_params_mode": mode_json_strings(grp["selected_params"]),
            "bundle_kb_mean": float(grp["bundle_kb"].mean()),
            "bundle_kb_max": float(grp["bundle_kb"].max()),
            "tune_ms_mean": float(grp["tune_ms"].mean()),
            "fit_ms_mean": float(grp["fit_ms"].mean()),
        }

        # Average subject-level percentile estimates.
        for timing in ["pre", "model", "e2e"]:
            for stat in ["mean_ms", "sd_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"]:
                col = f"{timing}_{stat}"
                row[col] = float(grp[col].mean())

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "latency_summary.csv", index=False)

    # Compact terminal table focused on the manuscript-facing end-to-end numbers.
    print("\n" + "=" * 110)
    print("SINGLE-QUERY END-TO-END CPU LATENCY (session-disjoint / fixed-impostor)")
    print("Values below are means of the per-subject latency summaries.")
    print("=" * 110)
    header = (
        f"{'Classifier':<12}{'N':>6}{'P50 ms':>12}{'P95 ms':>12}"
        f"{'P99 ms':>12}{'Mean ms':>12}{'Max ms':>12}{'Bundle KB':>13}"
    )
    print(header)

    for N in N_LIST:
        for name in CLASSIFIERS:
            r = summary[(summary["N"] == N) &
                        (summary["classifier"] == name)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(
                f"{name:<12}{N:>6}"
                f"{r['e2e_p50_ms']:>12.4f}"
                f"{r['e2e_p95_ms']:>12.4f}"
                f"{r['e2e_p99_ms']:>12.4f}"
                f"{r['e2e_mean_ms']:>12.4f}"
                f"{r['e2e_max_ms']:>12.4f}"
                f"{r['bundle_kb_mean']:>13.1f}"
            )

    print("\nPreprocessing is INCLUDED in the end-to-end numbers.")
    print("Timer: time.perf_counter_ns(); batch size: 1.")
    print(
        "No CPU affinity or frequency locking is applied, so tail latency "
        "includes normal macOS scheduling/DVFS effects."
    )
    print("This is a desktop CPU characterization, NOT a hard-real-time guarantee.")
    print(f"\nWrote:")
    print(f"  {outdir}/latency_persubject.csv")
    print(f"  {outdir}/latency_summary.csv")
    print(f"  {outdir}/selected_hyperparams.csv")
    print(f"  {outdir}/environment.json")
    if args.save_raw:
        print(f"  {outdir}/raw_timings.csv.gz")


if __name__ == "__main__":
    main()
