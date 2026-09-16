# Constraint-Aware Evaluation of Lightweight Classifiers for Biometric Authentication

This repository contains the evaluation code, experimental configurations, and derived result files supporting the manuscript:

**“Constraint-Aware Evaluation of Lightweight Classifiers for Biometric Authentication”**

The study evaluates lightweight biometric authentication classifiers under deployment-oriented constraints, including training-set size, session-disjoint evaluation, impostor sampling strategy, dimensionality reduction, computational latency, and model storage requirements.

## Repository Structure

```text
constraint-aware-biometric-authentication/
├── README.md
├── requirements.txt
├── environment.json
├── .gitignore
├── cmu/
│   ├── scripts/
│   └── results/
├── orl/
│   ├── scripts/
│   └── results/
├── latency/
│   ├── scripts/
│   └── results/
└── figures/
```

### CMU Keystroke Experiments

`cmu/scripts/` contains the finalized evaluation code for the CMU Keystroke Dynamics Benchmark.

`cmu/results/` contains:

- `persubject.csv` — per-subject evaluation results
- `summary.csv` — aggregate classifier results
- `stats.csv` — statistical output from the primary evaluation script
- `stats_11_comparisons.csv` — finalized predefined 11-comparison statistical family reported in the manuscript

The primary classifiers evaluated are:

- k-nearest neighbors (k-NN)
- linear discriminant analysis (LDA)
- threshold-based matching
- linear support vector machine (SVM)
- multilayer perceptron (MLP)

Experiments include training-set sizes of **N = 10, 50, and 200**, session-disjoint and random-split protocols, and fixed versus balanced impostor sampling.

The primary evaluation script contains an earlier seven-comparison statistical block in `stats.csv`. The finalized manuscript reports the predefined expanded family of 11 comparisons preserved in `stats_11_comparisons.csv`.

### ORL Face Experiments

`orl/scripts/` contains the finalized ORL face evaluation and PCA sensitivity analysis code.

`orl/results/` contains the primary subject-level results, paired statistical tests, and PCA sensitivity outputs.

The primary ORL protocol uses:

- 40 subjects
- 5 training images per subject
- 5 genuine test images per subject
- 10 predefined splits
- k-NN and LDA classifiers
- raw standardized features and PCA representations

PCA sensitivity was evaluated at **10, 20, 40, 80, and 150 components**. PCA-40 was the predefined primary representation; the additional dimensionalities are reported as sensitivity analyses rather than being used for post hoc model selection.

## Latency Evaluation

`latency/scripts/` contains the finalized latency evaluation code.

`latency/results/` contains:

- `latency_persubject.csv`
- `latency_summary.csv`
- `selected_hyperparams.csv`
- `environment.json`

Latency measurements were performed using batch size 1 with 100 warm-up queries followed by 1,000 timed queries per model/subject.

Reported latency statistics include the mean, median (P50), P95, P99, and observed maximum.

These measurements characterize the tested hardware/software configuration and should not be interpreted as hard real-time guarantees.

## Datasets

This repository does **not** redistribute biometric datasets.

The experiments use two publicly available datasets:

1. **CMU Keystroke Dynamics Benchmark**  
   Users should obtain the dataset from its original public source and follow the corresponding experiment script requirements.

2. **ORL (AT&T) Database of Faces**  
   Users should obtain the dataset from its original public source and preserve the standard subject-directory organization (`s1` through `s40`).

Raw biometric samples and dataset archives are intentionally excluded from this repository.

## Software Environment

The finalized experiments were recorded using:

```text
Python:       3.12.4
NumPy:        2.5.3
SciPy:        1.18.1
pandas:       3.0.5
scikit-learn: 1.9.1
```

Install the primary Python dependencies with:

```bash
pip install -r requirements.txt
```

Experiment-specific environment information is also retained with the corresponding result artifacts where applicable.

## Hardware

Finalized experiments were executed on an **Apple M3 Pro MacBook Pro** under macOS.

Latency measurements were CPU-only. GPU acceleration was not used. CPU affinity and processor frequency were not explicitly controlled.

Accordingly, latency values should be interpreted as measurements from the documented experimental platform rather than universal timing guarantees.

## Reproducing the Analyses

Download the CMU and ORL datasets separately from their original public sources before running the evaluation scripts.

The finalized scripts are organized by experiment:

```text
cmu/scripts/
orl/scripts/
latency/scripts/
```

Derived outputs from the finalized study are preserved under the corresponding `results/` directories so that reported values can be inspected without rerunning the experiments.

Because dataset locations may differ between systems, users should verify dataset paths expected by each script before execution.

## Result Mapping

The repository artifacts support the manuscript analyses as follows:

- **CMU primary results:** `cmu/results/summary.csv` and `cmu/results/persubject.csv`
- **CMU statistical comparisons:** `cmu/results/stats_11_comparisons.csv`
- **ORL primary results:** `orl/results/summary.csv` and `orl/results/persubject.csv`
- **ORL paired tests:** `orl/results/paired_tests.csv`
- **ORL PCA sensitivity:** PCA sensitivity CSV files in `orl/results/`
- **Latency analysis:** `latency/results/latency_summary.csv` and `latency/results/latency_persubject.csv`
- **Selected latency configurations:** `latency/results/selected_hyperparams.csv`

## Reproducibility Scope

The repository is intended to support **computational repeatability** of the reported analyses: re-executing the same evaluation procedures and source code against the same public datasets.

This should be distinguished from independent reproducibility, which would require an independent implementation of the experimental methodology.

The preserved artifacts include evaluation code, experiment configurations, split seeds where applicable, per-subject outputs, summary statistics, paired-test results, PCA sensitivity results, latency measurements, and software-environment information.

## Data Availability

No new primary biometric data were collected for this study. The CMU Keystroke Dynamics Benchmark and ORL (AT&T) Database of Faces should be obtained from their respective original sources.

This repository provides the evaluation code and derived experimental results but does not redistribute the underlying biometric datasets.

## Citation

If you use this repository, please cite the associated manuscript:

> Neil Provenzano, Ilaisaane Tilisa Fonua, and Shahram Latifi, “Constraint-Aware Evaluation of Lightweight Classifiers for Biometric Authentication.”

Publication information and DOI will be added following publication.

## License

Licensing information for the repository source code will be provided in the repository `LICENSE` file.

The original biometric datasets are subject to the terms established by their respective providers and are not covered by this repository's software license.# constraint-aware-biometric-authentication
