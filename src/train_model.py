"""
train_model.py
---------------
Trains and compares a RandomForest and an XGBoost classifier to
predict a team's next-match outcome: Win / Draw / Loss.

Key design choices:
  - CHRONOLOGICAL train/test split, not a random shuffle. This is
    match data over time; a random split would leak future form into
    the training set (e.g. training on a team's April match while
    testing on its March match). We train on everything before a
    cutoff date and test on everything after it.
  - Class labels are label-encoded (L=0, D=1, W=2) for XGBoost, which
    requires integer targets.
  - Both models are evaluated with accuracy, a full classification
    report, and a confusion matrix plotted with matplotlib.

Usage:
    python src/train_model.py
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from features import FEATURE_COLS, TARGET_COL, build_features

TEST_FRACTION = 0.2  # last 20% of matches held out for testing


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    df = df.sort_values("Date").reset_index(drop=True)
    cutoff_idx = int(len(df) * (1 - test_fraction))
    cutoff_date = df.loc[cutoff_idx, "Date"]
    train = df[df["Date"] < cutoff_date]
    test = df[df["Date"] >= cutoff_date]
    return train, test, cutoff_date


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",   # draws are the minority class; don't ignore them
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train_enc, num_classes=3):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_enc)
    return model


def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {out_path}")


def main():
    raw = pd.read_csv("data/raw_matches.csv")
    feat_df = build_features(raw)
    feat_df["Date"] = pd.to_datetime(feat_df["Date"])

    train_df, test_df, cutoff = chronological_split(feat_df)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows | "
          f"chronological cutoff: {cutoff.date()}")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    label_order = ["L", "D", "W"]  # fixed order for consistent reporting/plots
    encoder = LabelEncoder()
    encoder.fit(label_order)
    y_train_enc = encoder.transform(y_train)
    y_test_enc = encoder.transform(y_test)

    baseline_acc = (y_test.value_counts(normalize=True).max())
    print(f"\nBaseline (always predict majority class) accuracy: {baseline_acc:.3f}\n")

    # ---------- Random Forest ----------
    rf = train_random_forest(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print("=== Random Forest ===")
    print(f"Accuracy: {rf_acc:.3f}")
    print(classification_report(y_test, rf_pred, labels=label_order))
    plot_confusion_matrix(y_test, rf_pred, label_order,
                           "Random Forest: Confusion Matrix",
                           "outputs/confusion_matrix_rf.png")

    # ---------- XGBoost ----------
    xgb = train_xgboost(X_train, y_train_enc, num_classes=len(label_order))
    xgb_pred_enc = xgb.predict(X_test)
    xgb_pred = encoder.inverse_transform(xgb_pred_enc)
    xgb_acc = accuracy_score(y_test_enc, xgb_pred_enc)
    print("\n=== XGBoost ===")
    print(f"Accuracy: {xgb_acc:.3f}")
    print(classification_report(y_test, xgb_pred, labels=label_order))
    plot_confusion_matrix(y_test, xgb_pred, label_order,
                           "XGBoost: Confusion Matrix",
                           "outputs/confusion_matrix_xgb.png")

    # ---------- Feature importance (XGBoost) ----------
    importances = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot(kind="barh", ax=ax, color="#2b6cb0")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig("outputs/feature_importance_xgb.png", dpi=150)
    plt.close(fig)
    print("Saved feature importance plot -> outputs/feature_importance_xgb.png")

    # ---------- Save the better model ----------
    best_name, best_model, best_acc = (
        ("xgboost", xgb, xgb_acc) if xgb_acc >= rf_acc else ("random_forest", rf, rf_acc)
    )
    joblib.dump({"model": best_model, "encoder": encoder, "features": FEATURE_COLS,
                 "model_type": best_name}, "models/best_model.joblib")
    print(f"\nBest model: {best_name} (accuracy {best_acc:.3f}) saved to models/best_model.joblib")

    summary = pd.DataFrame({
        "model": ["baseline_majority", "random_forest", "xgboost"],
        "accuracy": [baseline_acc, rf_acc, xgb_acc],
    })
    summary.to_csv("outputs/model_comparison.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
