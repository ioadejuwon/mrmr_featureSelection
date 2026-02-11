"""
MRMR Feature Selection Coursework Project
Author: Your Name
"""

# ==========================
# Imports
# ==========================

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from mrmr import mrmr_classif
print("MRMR is working...")


# ==========================
# Configuration
# ==========================

DATA_PATH = "diabetes.csv"   # change to your dataset
TARGET_COLUMN = "target"     # change if needed
TEST_SIZE = 0.2
RANDOM_STATE = 42
K_FEATURES = 5               # number of features to select


# ==========================
# Load Data
# ==========================

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Scale features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)


# ==========================
# Baseline Model (All Features)
# ==========================

print("\nRunning Baseline Model (All Features)...")

start_time = time.time()

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_scaled, y_train)

baseline_train_time = time.time() - start_time

y_pred_base = baseline_model.predict(X_test_scaled)
y_proba_base = baseline_model.predict_proba(X_test_scaled)[:, 1]

baseline_results = {
    "Accuracy": accuracy_score(y_test, y_pred_base),
    "F1 Score": f1_score(y_test, y_pred_base),
    "ROC-AUC": roc_auc_score(y_test, y_proba_base),
    "Train Time (s)": baseline_train_time
}


# ==========================
# MRMR Feature Selection
# ==========================

print("\nRunning MRMR Feature Selection...")

selected_features = mrmr_classif(
    X=X_train_scaled,
    y=y_train,
    K=K_FEATURES
)

print(f"Selected Features: {selected_features}")

X_train_mrmr = X_train_scaled[selected_features]
X_test_mrmr = X_test_scaled[selected_features]


# ==========================
# Model with MRMR Features
# ==========================

print("\nTraining Model with MRMR Features...")

start_time = time.time()

mrmr_model = LogisticRegression(max_iter=1000)
mrmr_model.fit(X_train_mrmr, y_train)

mrmr_train_time = time.time() - start_time

y_pred_mrmr = mrmr_model.predict(X_test_mrmr)
y_proba_mrmr = mrmr_model.predict_proba(X_test_mrmr)[:, 1]

mrmr_results = {
    "Accuracy": accuracy_score(y_test, y_pred_mrmr),
    "F1 Score": f1_score(y_test, y_pred_mrmr),
    "ROC-AUC": roc_auc_score(y_test, y_proba_mrmr),
    "Train Time (s)": mrmr_train_time
}


# ==========================
# Performance Comparison
# ==========================

print("\n==============================")
print("Performance Comparison")
print("==============================")

print("\nBaseline (All Features):")
for k, v in baseline_results.items():
    print(f"{k}: {v:.4f}")

print("\nMRMR (Selected Features):")
for k, v in mrmr_results.items():
    print(f"{k}: {v:.4f}")

print("\nFeature Reduction:")
print(f"Original Features: {X.shape[1]}")
print(f"Selected Features: {len(selected_features)}")
