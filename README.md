# Loan Approval Prediction

A small machine learning project that predicts whether a loan application gets approved or rejected, and — more usefully — compares ten different classifiers against each other on the same data so you can see which one actually earns its keep.

## Why this exists

Most loan-approval tutorials online train one model and call it done. I wanted to actually see how Logistic Regression stacks up against XGBoost, Random Forest, SVM, and the rest on the same train/test split and the same cross-validation folds, instead of taking any single model's word for it. This script does that comparison end to end: load the data, clean it up, train all ten models, and spit out a full report — metrics, ROC curves, confusion matrices, the works.

It's also been through a real debugging session (Windows multiprocessing decided to fight back at one point), so the fix for that is documented below in case it saves someone else an afternoon.

## What it does

The pipeline takes the raw CSV and runs it through the usual stages: it strips whitespace and encodes the categorical columns (education, self-employment status, the approve/reject label), splits the data 80/20 with stratification, and scales the numeric features for the models that need it. From there it trains all ten classifiers, scores each one on accuracy, precision, recall, F1, and ROC-AUC, and runs a 5-fold stratified cross-validation on top of that so you're not just looking at one lucky train/test split.

Once everything's trained, it builds out the visuals — a sorted accuracy comparison, ROC and precision-recall curves with every model overlaid, confusion matrices for the four best performers, a box plot of the cross-validation spread, and a feature-importance chart pulled from whichever tree-based model came out on top. Everything gets saved as PNGs, and the full comparison table gets written out to CSV.

## Project layout

```
loan-approval-prediction/
├── loan_approval_pipeline.py      # the pipeline itself
├── loan_approval_dataset.csv      # your dataset (not included)
├── requirements.txt
├── README.md
└── outputs/                       # generated each run
    ├── eda_distributions.png
    ├── correlation_heatmap.png
    ├── model_accuracy_comparison.png
    ├── roc_curves.png
    ├── precision_recall_curves.png
    ├── confusion_matrices.png
    ├── cv_boxplot.png
    ├── feature_importance.png
    └── model_comparison_results.csv
```

Right now the script saves everything into the working directory rather than into `outputs/` directly — see [Tweaking it](#tweaking-it) below if you'd rather have it organized that way from the start.

## The dataset

At minimum, the CSV needs these columns:

| Column | Type | What it is |
|---|---|---|
| `loan_id` | identifier | dropped before training, not predictive |
| `no_of_dependents` | numeric | number of dependents |
| `education` | categorical | Graduate / Not Graduate |
| `self_employed` | categorical | Yes / No |
| `income_annum` | numeric | annual income |
| `loan_amount` | numeric | amount requested |
| `loan_term` | numeric | term in years |
| `cibil_score` | numeric | credit score |
| `residential_assets_value` | numeric | value of residential assets |
| `loan_status` | categorical (target) | Approved / Rejected |

If your version of the dataset has extra numeric columns (other asset values, for instance), they'll get picked up automatically — the feature set is just "everything except `loan_id` and `loan_status`," so nothing needs to be hardcoded.

Point the script at your file by updating this line:

```python
df = pd.read_csv(r"D:\DATASETS\loan_approval_dataset.csv")
```

## Setup

```bash
git clone https://github.com/<your-username>/loan-approval-prediction.git
cd loan-approval-prediction
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

## Running it

Update the dataset path, then just run:

```bash
python loan_approval_pipeline.py
```

It'll print its progress through each stage as it goes, and once it's done you'll have a folder full of `.png` plots plus a `model_comparison_results.csv` sitting next to the script.

## How it's organized internally

The script runs through eight steps in order — load the data, preprocess it, do some EDA, train and evaluate all ten models, build the comparison table, generate the plots, pull out feature importance, and finally print a detailed report on whichever model had the best ROC-AUC. Each step prints a header to the console so it's easy to follow along while it's running, and nothing later in the pipeline depends on anything you'd need to tweak by hand mid-run.

## Models in the lineup

| Model | Family | Notes |
|---|---|---|
| Logistic Regression | Linear | baseline, trained on scaled features |
| Decision Tree | Tree | single tree, no scaling needed |
| Random Forest | Bagging | 200 trees |
| Gradient Boosting | Boosting | 200 stages |
| XGBoost | Boosting | 200 rounds, log-loss objective |
| AdaBoost | Boosting | 200 estimators |
| Extra Trees | Bagging | 200 trees, extra randomization |
| K-Nearest Neighbors | Distance-based | k=5, scaled features |
| Support Vector Machine | Kernel-based | RBF kernel, scaled features |
| Naive Bayes | Probabilistic | Gaussian, scaled features |

## On the metrics

Accuracy is the blunt instrument — overall percent correct, nothing more. Precision tells you how often a predicted approval was actually a good call, while recall tells you how many real approvals the model managed to catch. F1 just balances those two when neither one alone tells the whole story. ROC-AUC is probably the most useful number here since it measures how well a model ranks approvals above rejections across every possible threshold, not just the default one. The cross-validation mean and standard deviation round things out by showing whether a model's performance holds up across different folds or whether it just got lucky once.

## Seeing the output

Once you've actually run it and have plots sitting in your repo, drop them into the README like this:

```markdown
![Model Accuracy Comparison](outputs/model_accuracy_comparison.png)
![ROC Curves](outputs/roc_curves.png)
![Feature Importance](outputs/feature_importance.png)
```

## A Windows gotcha worth knowing about

If you run this on Windows and see something like:

```
joblib.externals.loky.process_executor.TerminatedWorkerError:
A worker process managed by the executor was unexpectedly terminated.
```

it's not your data or your code — it's nested parallelism. Models like `RandomForestClassifier` with `n_jobs=-1` spawn their own pool of worker processes, and then `cross_val_score(n_jobs=-1)` tries to spawn a second pool on top of that. Between the two, you end up oversubscribing CPU and memory until Windows just kills a worker outright.

Two things fix it: wrap the script's execution in `if __name__ == "__main__":` (multiprocessing on Windows needs this to avoid re-importing the module in every child process), and keep the models themselves single-threaded (`n_jobs=1`) so only the cross-validation step is parallelizing, never both layers at once.

## Tweaking it

A few things worth knowing if you want to adjust how it runs: the dataset path is set in the `pd.read_csv(...)` call near the top, the train/test split ratio is the `test_size` argument in `train_test_split`, and the number of cross-validation folds is `n_splits` in `StratifiedKFold`. If you're working with a much larger dataset or limited RAM, dropping `n_estimators` from 200 down to 100 on the ensemble models will speed things up noticeably without changing much else. And if you want the plots and CSV landing in an `outputs/` folder instead of cluttering the project root, just update the paths in the `plt.savefig(...)` and `to_csv(...)` calls.

## Worth keeping in mind

This assumes the dataset doesn't have missing values beyond stray whitespace — nothing here handles real imputation. `SVC` with `probability=True` can also get slow on bigger datasets since it runs its own internal cross-validation for probability calibration. And model selection here is based purely on test-set ROC-AUC; there's no hyperparameter tuning happening, so there's real room to push these numbers higher if you want to take it further.


## Author

**Biplob Kumar Dutta**
**Intern ID - CTIS9255**
**Duration - 8 week**
