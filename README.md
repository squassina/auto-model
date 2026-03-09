# auto-model

A template for **automated machine learning model selection simulation** runnable from the GitHub command line (GitHub CLI / `gh`).

## Overview

`auto-model` evaluates multiple scikit-learn classifiers (or regressors) against your dataset, scores each one with cross-validation, and reports the best performing model — all from a single command.

## Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.9 |
| pip | latest |
| [GitHub CLI (`gh`)](https://cli.github.com/) | ≥ 2.0 (optional, for workflow dispatch) |

### 1 — Clone and install dependencies

```bash
gh repo clone squassina/auto-model
cd auto-model
pip install -r requirements.txt
```

### 2 — Run the simulation

```bash
python auto_model_selection.py \
  --dataset iris \
  --task classification \
  --cv 5 \
  --scoring accuracy
```

The script accepts a built-in sklearn toy dataset (`iris`, `wine`, `breast_cancer`, `digits`) **or** a path to a local CSV file:

```bash
python auto_model_selection.py \
  --dataset path/to/my_data.csv \
  --target my_label_column \
  --task classification
```

### 3 — Trigger via GitHub Actions (optional)

```bash
gh workflow run auto-model.yml \
  -f dataset=iris \
  -f task=classification \
  -f cv=5 \
  -f scoring=accuracy
```

Then watch the run:

```bash
gh run watch
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `iris` | Built-in dataset name **or** path to a CSV file |
| `--target` | `target` | Column name to use as the label (CSV only) |
| `--task` | `classification` | `classification` or `regression` |
| `--cv` | `5` | Number of cross-validation folds |
| `--scoring` | `accuracy` | sklearn scoring metric (e.g. `f1_macro`, `r2`) |
| `--seed` | `42` | Random seed for reproducibility |
| `--output` | _(stdout)_ | Path to write the JSON results file |

## Example Output

```
Auto Model Selection Simulation
================================
Task       : classification
Dataset    : iris (150 samples, 4 features)
CV folds   : 5
Scoring    : accuracy

  Model                        Mean CV Score    Std Dev
  ---------------------------  ---------------  ---------
  LogisticRegression           0.9733           ± 0.0249
  RandomForestClassifier       0.9667           ± 0.0211
  GradientBoostingClassifier   0.9600           ± 0.0327
  SVC                          0.9733           ± 0.0249
  KNeighborsClassifier         0.9600           ± 0.0327
  DecisionTreeClassifier       0.9400           ± 0.0490
  GaussianNB                   0.9533           ± 0.0422

Best model : LogisticRegression  (accuracy = 0.9733)
```

## Project Structure

```
auto-model/
├── README.md                    ← this file
├── requirements.txt             ← Python dependencies
├── auto_model_selection.py      ← main simulation script
└── .github/
    └── workflows/
        └── auto-model.yml       ← GitHub Actions workflow
```

## GitHub Actions Workflow

The included workflow (`.github/workflows/auto-model.yml`) lets you run the simulation directly from the GitHub UI or via `gh workflow run`.

Supported inputs mirror the CLI flags above (`dataset`, `task`, `cv`, `scoring`, `seed`).

## License

[MIT](LICENSE)
