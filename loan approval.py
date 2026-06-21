

# ──────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def main():
    # ──────────────────────────────────────────────────────────────────
    # 2. LOAD & EXPLORE
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1 — LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(r"D:\DATASETS\loan_approval_dataset.csv")

    # Strip whitespace from column names (the CSV has leading spaces)
    df.columns = df.columns.str.strip()

    print(f"Shape : {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    print(df.head())
    print("\n--- Data types ---")
    print(df.dtypes)
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    print("\n--- Statistical summary ---")
    print(df.describe())
    print("\n--- Target distribution ---")
    print(df["loan_status"].value_counts())
    print()

    # ──────────────────────────────────────────────────────────────────
    # 3. PREPROCESSING
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 2 — PREPROCESSING")
    print("=" * 70)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Encode categorical features
    label_encoders = {}
    categorical_cols = ["education", "self_employed", "loan_status"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Drop loan_id — not a predictor
    df.drop(columns=["loan_id"], inplace=True)

    # Feature / target split
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    feature_names = list(X.columns)
    print(f"\nFeatures ({len(feature_names)}): {feature_names}")
    print(f"Target classes : {np.unique(y)}  (0 = Rejected, 1 = Approved)")

    # Train / Test split  (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}   Test size: {X_test.shape[0]}")

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("Scaling complete.\n")

    # ──────────────────────────────────────────────────────────────────
    # 4. EXPLORATORY DATA ANALYSIS (EDA) PLOTS
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 3 — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("EDA — Feature Distributions by Loan Status", fontsize=16, y=1.02)

    plot_features = [
        "income_annum", "loan_amount", "cibil_score",
        "loan_term", "residential_assets_value", "no_of_dependents"
    ]

    for ax, feat in zip(axes.ravel(), plot_features):
        for label, color in [(0, "#e74c3c"), (1, "#2ecc71")]:
            subset = df[df["loan_status"] == label][feat]
            ax.hist(subset, bins=40, alpha=0.55, label="Rejected" if label == 0 else "Approved", color=color)
        ax.set_title(feat)
        ax.legend()

    plt.tight_layout()
    plt.savefig("eda_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → eda_distributions.png")

    # Correlation heatmap
    plt.figure(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → correlation_heatmap.png\n")

    # ──────────────────────────────────────────────────────────────────
    # 5. MODEL TRAINING & EVALUATION
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 4 — MODEL TRAINING & EVALUATION")
    print("=" * 70)

    models = {
        "Logistic Regression"    : LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"          : DecisionTreeClassifier(random_state=42),
        "Random Forest"          : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
        "Gradient Boosting"      : GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost"                : XGBClassifier(n_estimators=200, eval_metric="logloss",
                                                 random_state=42, verbosity=0, n_jobs=1),
        "AdaBoost"               : AdaBoostClassifier(n_estimators=200, random_state=42),
        "Extra Trees"            : ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=1),
        "K-Nearest Neighbors"    : KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine" : SVC(kernel="rbf", probability=True, random_state=42),
        "Naive Bayes"            : GaussianNB(),
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # Choose scaled data for distance/linear models, raw for tree models
        use_scaled = name in [
            "Logistic Regression", "K-Nearest Neighbors",
            "Support Vector Machine", "Naive Bayes"
        ]
        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled  if use_scaled else X_test

        # Fit
        model.fit(Xtr, y_train)

        # Predict
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        # Cross-validation accuracy. The model itself is single-threaded
        # (n_jobs=1 set above), so parallelizing across folds here does
        # NOT create nested worker pools.
        cv_scores = cross_val_score(
            model, Xtr, y_train, cv=cv, scoring="accuracy", n_jobs=1
        )

        results.append({
            "Model"         : name,
            "Accuracy"      : acc,
            "Precision"     : prec,
            "Recall"        : rec,
            "F1 Score"      : f1,
            "ROC AUC"       : auc,
            "CV Mean Acc"   : cv_scores.mean(),
            "CV Std"        : cv_scores.std(),
            "CV Scores"     : cv_scores,
            "y_pred"        : y_pred,
            "y_proba"       : y_proba,
            "model_obj"     : model,
            "use_scaled"    : use_scaled,
        })

        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  ROC AUC   : {auc:.4f}" if not np.isnan(auc) else "  ROC AUC   : N/A")
        print(f"  CV Acc     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ──────────────────────────────────────────────────────────────────
    # 6. RESULTS COMPARISON TABLE
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5 — MODEL COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(results).drop(
        columns=["y_pred", "y_proba", "model_obj", "use_scaled", "CV Scores"]
    )
    results_df = results_df.sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    print(results_df.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────
    # 7. VISUALISATIONS
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6 — PLOTS")
    print("=" * 70)

    # 7a. Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_res = sorted(results, key=lambda r: r["Accuracy"], reverse=True)
    names = [r["Model"] for r in sorted_res]
    accs  = [r["Accuracy"] for r in sorted_res]
    colors = sns.color_palette("viridis", len(names))
    bars = ax.barh(names, accs, color=colors)
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Accuracy Comparison", fontsize=14)
    ax.set_xlim(min(accs) - 0.02, 1.0)
    for bar, v in zip(bars, accs):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → model_accuracy_comparison.png")

    # 7b. ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for r in results:
        if r["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
            ax.plot(fpr, tpr, label=f'{r["Model"]} (AUC={r["ROC AUC"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → roc_curves.png")

    # 7c. Precision-Recall curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for r in results:
        if r["y_proba"] is not None:
            prec_vals, rec_vals, _ = precision_recall_curve(y_test, r["y_proba"])
            ap = average_precision_score(y_test, r["y_proba"])
            ax.plot(rec_vals, prec_vals, label=f'{r["Model"]} (AP={ap:.3f})')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig("precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → precision_recall_curves.png")

    # 7d. Confusion matrices for top-4 models
    top4 = sorted(results, key=lambda r: r["ROC AUC"] if not np.isnan(r["ROC AUC"]) else 0, reverse=True)[:4]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, r in zip(axes, top4):
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Rejected", "Approved"],
                    yticklabels=["Rejected", "Approved"])
        ax.set_title(f'{r["Model"]}\nAcc={r["Accuracy"]:.4f}', fontsize=11)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.suptitle("Confusion Matrices — Top 4 Models", fontsize=14, y=1.04)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → confusion_matrices.png")

    # 7e. Cross-validation box plot
    # Reuses the CV scores already computed in Step 4 instead of
    # re-running cross_val_score for every model a second time.
    fig, ax = plt.subplots(figsize=(14, 6))
    cv_data = [r["CV Scores"] for r in results]
    cv_labels = [r["Model"] for r in results]
    ax.boxplot(cv_data, labels=cv_labels, vert=True, patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_ylabel("Accuracy")
    ax.set_title("5-Fold Cross-Validation Accuracy Distribution", fontsize=14)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig("cv_boxplot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → cv_boxplot.png")

    # ──────────────────────────────────────────────────────────────────
    # 8. FEATURE IMPORTANCE (Best Tree Model)
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7 — FEATURE IMPORTANCE")
    print("=" * 70)

    # Pick the best tree-based model by AUC
    tree_models = [r for r in results if hasattr(r["model_obj"], "feature_importances_")]
    best_tree = max(tree_models, key=lambda r: r["ROC AUC"])
    importances = best_tree["model_obj"].feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\nBest tree-based model: {best_tree['Model']}  (AUC = {best_tree['ROC AUC']:.4f})")
    print("\nFeature ranking:")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]:30s} — {importances[idx]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_idx = np.argsort(importances)
    ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx], color="teal")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {best_tree['Model']}", fontsize=14)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → feature_importance.png")

    # ──────────────────────────────────────────────────────────────────
    # 9. DETAILED CLASSIFICATION REPORT — BEST MODEL
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 8 — BEST MODEL DETAILED REPORT")
    print("=" * 70)

    best = max(results, key=lambda r: r["ROC AUC"] if not np.isnan(r["ROC AUC"]) else 0)
    print(f"\n*  Best overall model: {best['Model']}")
    print(f"   Accuracy  : {best['Accuracy']:.4f}")
    print(f"   Precision : {best['Precision']:.4f}")
    print(f"   Recall    : {best['Recall']:.4f}")
    print(f"   F1 Score  : {best['F1 Score']:.4f}")
    print(f"   ROC AUC   : {best['ROC AUC']:.4f}")
    print(f"   CV Acc     : {best['CV Mean Acc']:.4f} ± {best['CV Std']:.4f}")
    print("\nClassification Report:")
    target_names = ["Rejected", "Approved"]
    print(classification_report(y_test, best["y_pred"], target_names=target_names))

    # ──────────────────────────────────────────────────────────────────
    # 10. SAVE RESULTS
    # ──────────────────────────────────────────────────────────────────
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("Results table saved → model_comparison_results.csv")
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
