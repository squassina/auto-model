#!/usr/bin/env python3
"""
auto_model_selection.py
-----------------------
Automated machine-learning model selection simulation.

Evaluates a fixed set of scikit-learn estimators on a dataset using
cross-validation and reports the best performing model.

Usage examples
--------------
# Built-in sklearn toy dataset
python auto_model_selection.py --dataset iris --task classification

# Local CSV file
python auto_model_selection.py \
    --dataset path/to/data.csv \
    --target label_column \
    --task classification \
    --cv 10 \
    --scoring f1_macro \
    --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as sk_datasets
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Candidate model pools
# ---------------------------------------------------------------------------
CLASSIFIERS = [
    ("LogisticRegression", LogisticRegression(max_iter=1000)),
    ("RandomForestClassifier", RandomForestClassifier()),
    ("GradientBoostingClassifier", GradientBoostingClassifier()),
    ("SVC", SVC()),
    ("KNeighborsClassifier", KNeighborsClassifier()),
    ("DecisionTreeClassifier", DecisionTreeClassifier()),
    ("GaussianNB", GaussianNB()),
]

REGRESSORS = [
    ("LinearRegression", LinearRegression()),
    ("Ridge", Ridge()),
    ("RandomForestRegressor", RandomForestRegressor()),
    ("GradientBoostingRegressor", GradientBoostingRegressor()),
    ("SVR", SVR()),
    ("KNeighborsRegressor", KNeighborsRegressor()),
    ("DecisionTreeRegressor", DecisionTreeRegressor()),
]

# ---------------------------------------------------------------------------
# Built-in dataset loaders
# ---------------------------------------------------------------------------
BUILTIN_DATASETS = {
    "iris": sk_datasets.load_iris,
    "wine": sk_datasets.load_wine,
    "breast_cancer": sk_datasets.load_breast_cancer,
    "digits": sk_datasets.load_digits,
    "diabetes": sk_datasets.load_diabetes,
    "california_housing": sk_datasets.fetch_california_housing,
}


def load_builtin(name: str):
    """Load a built-in sklearn dataset and return (X, y, description)."""
    loader = BUILTIN_DATASETS[name]
    data = loader()
    return data.data, data.target, f"{name} ({data.data.shape[0]} samples, {data.data.shape[1]} features)"


def load_csv(path: str, target_col: str):
    """Load a CSV file and return (X, y, description)."""
    df = pd.read_csv(path)
    if target_col not in df.columns:
        sys.exit(f"[error] Column '{target_col}' not found in {path}.\n"
                 f"  Available columns: {list(df.columns)}")
    y_raw = df[target_col].values
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    # Encode string labels for classification
    if y_raw.dtype == object or y_raw.dtype.kind in ("U", "S"):
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw.astype(float)
    desc = f"{Path(path).name} ({len(df)} samples, {X.shape[1]} numeric features)"
    return X, y, desc


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_simulation(X, y, task: str, cv: int, scoring: str, seed: int) -> list[dict]:
    """
    Cross-validate every candidate model and return a list of result dicts
    sorted by mean CV score (descending).

    ``seed`` is forwarded to each estimator that exposes a ``random_state``
    parameter, ensuring reproducible results across runs.
    """
    candidates = CLASSIFIERS if task == "classification" else REGRESSORS
    results = []

    for name, model in candidates:
        # Inject random_state only for estimators that support it
        if "random_state" in model.get_params():
            model.set_params(random_state=seed)

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        results.append(
            {
                "model": name,
                "mean": float(round(scores.mean(), 4)),
                "std": float(round(scores.std(), 4)),
                "scores": [float(round(s, 4)) for s in scores],
            }
        )

    results.sort(key=lambda r: r["mean"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def print_table(results: list[dict], scoring: str) -> None:
    col_w = max(len(r["model"]) for r in results) + 2
    header = f"  {'Model':<{col_w}}  {'Mean CV Score':<16}  Std Dev"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(f"  {r['model']:<{col_w}}  {r['mean']:<16.4f}  ± {r['std']:.4f}")


def print_report(results: list[dict], task: str, dataset_desc: str, cv: int, scoring: str) -> None:
    sep = "=" * 40
    print(f"\nAuto Model Selection Simulation")
    print(sep)
    print(f"Task       : {task}")
    print(f"Dataset    : {dataset_desc}")
    print(f"CV folds   : {cv}")
    print(f"Scoring    : {scoring}\n")
    print_table(results, scoring)
    best = results[0]
    print(f"\nBest model : {best['model']}  ({scoring} = {best['mean']:.4f})\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Auto model selection simulation using cross-validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="iris",
        help=f"Built-in dataset name ({', '.join(BUILTIN_DATASETS)}) or path to a CSV file.",
    )
    parser.add_argument(
        "--target",
        default="target",
        help="Target column name (used only when --dataset is a CSV file).",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Learning task type.",
    )
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument(
        "--scoring",
        default="accuracy",
        help="sklearn scoring metric (e.g. accuracy, f1_macro, r2, neg_mean_squared_error).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write results as JSON.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # --- Load data ---
    if args.dataset in BUILTIN_DATASETS:
        X, y, dataset_desc = load_builtin(args.dataset)
    elif Path(args.dataset).is_file():
        X, y, dataset_desc = load_csv(args.dataset, args.target)
    else:
        sys.exit(
            f"[error] '{args.dataset}' is not a recognized built-in dataset and the file does not exist.\n"
            f"  Built-in options: {', '.join(BUILTIN_DATASETS)}"
        )

    np.random.seed(args.seed)

    # --- Run simulation ---
    results = run_simulation(X, y, args.task, args.cv, args.scoring, args.seed)

    # --- Print report ---
    print_report(results, args.task, dataset_desc, args.cv, args.scoring)

    # --- Optional JSON output ---
    if args.output:
        payload = {
            "dataset": dataset_desc,
            "task": args.task,
            "cv": args.cv,
            "scoring": args.scoring,
            "seed": args.seed,
            "results": results,
            "best_model": results[0]["model"],
        }
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"Results written to {args.output}")

    return results[0]["model"]


if __name__ == "__main__":
    main()
