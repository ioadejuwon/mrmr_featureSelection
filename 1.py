# 1. Import libraries
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

print("Libraries imported successfully.")


# 2. Configuration
DATA_PATH = "data/1/train.csv"
OUTER_SPLITS = 10
INNER_SPLITS_PREFERRED = 3
RANDOM_STATE = 42

# Debug switch:
# For the official baseline experiment, it is recommended to keep this as False
# If you only want to verify whether the perfect score comes from suspicious features, set it to True
DEBUG_REMOVE_SUSPICIOUS_FEATURES = False


# =========================
# 3. Helper functions
# =========================
def find_duplicate_feature_columns(X: pd.DataFrame):
    """
    Find completely duplicated feature columns.
    Return in the form [(colA, colB), ...]
    """
    duplicate_pairs = []
    cols = X.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            if X[c1].equals(X[c2]):
                duplicate_pairs.append((c1, c2))

    return duplicate_pairs


def get_duplicate_drop_list(duplicate_pairs):
    """
    For each duplicate feature pair, keep the first one and drop the second one.
    """
    drop_cols = []
    kept = set()

    for c1, c2 in duplicate_pairs:
        if c1 not in kept:
            kept.add(c1)
        if c2 not in kept:
            drop_cols.append(c2)
            kept.add(c2)

    # Remove duplicates
    drop_cols = list(dict.fromkeys(drop_cols))
    return drop_cols


def find_label_like_features(X: pd.DataFrame, y: pd.Series):
    """
    Check which features have each value associated with only one class.
    Such features are highly suspicious and often cause tree models to achieve near-perfect classification.
    Note: this is only for diagnosis and does not necessarily mean they should be removed.
    """
    suspicious = []
    target_name = "__target__"

    for col in X.columns:
        tmp = pd.concat([X[col], y.rename(target_name)], axis=1).dropna()

        # If a feature value always corresponds to only one label, it strongly resembles a target encoding / leakage feature
        label_count_per_value = tmp.groupby(col)[target_name].nunique()

        if len(label_count_per_value) > 0 and (label_count_per_value <= 1).all():
            suspicious.append(col)

    return suspicious


def build_pipeline(model_name, estimator):
    """
    Build a pipeline for different models.
    Tree-based models do not need standardization;
    SVM / KNN / MLP require standardization.
    """
    if model_name in ["DecisionTree", "RandomForest"]:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator)
        ])
    else:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator)
        ])

    return pipe


def build_inner_cv(y_train, preferred_splits=3, random_state=42):
    """
    Dynamically decide inner CV n_splits based on the smallest class count in the current outer-train set.
    This avoids inner CV errors caused by extremely small classes.
    """
    min_class_count = int(y_train.value_counts().min())
    inner_splits = max(2, min(preferred_splits, min_class_count))

    return StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state
    )


# =========================
# 4. Load data
# =========================
df = pd.read_csv(DATA_PATH)

X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1].copy()

# If the label is originally float but is essentially categorical, converting it to int is safer
try:
    y = y.astype(int)
except Exception:
    pass

print("\n===== Basic data diagnostics =====")
print(f"Data shape: {df.shape}")
print(f"Feature shape: {X.shape}")
print(f"Label name: {df.columns[-1]}")
print(f"Feature columns: {X.columns.tolist()}")

print("\nMissing values per feature:")
print(X.isnull().sum())

print("\nNumber of duplicated full rows:", df.duplicated().sum())
print("Number of duplicated feature rows:", X.duplicated().sum())

print("\nClass distribution:")
print(y.value_counts().sort_index())

min_class_count = int(y.value_counts().min())
if min_class_count < OUTER_SPLITS:
    print(
        f"\n[Warning] The smallest class has only {min_class_count} samples, "
        f"which is smaller than OUTER_SPLITS={OUTER_SPLITS}. "
        f"Some outer folds may not contain all classes."
    )

duplicate_feature_pairs = find_duplicate_feature_columns(X)
print("\nDuplicate feature columns:")
if len(duplicate_feature_pairs) == 0:
    print("None")
else:
    for p in duplicate_feature_pairs:
        print(p)

suspicious_features = find_label_like_features(X, y)
print("\nSuspicious label-like features:")
if len(suspicious_features) == 0:
    print("None")
else:
    print(suspicious_features)


# =========================
# 5. Optional debug-only removal
# =========================
# For the official baseline experiment, it is recommended not to remove anything and keep the original features
# This is only a debugging option to help verify whether a perfect score is caused by suspicious features
if DEBUG_REMOVE_SUSPICIOUS_FEATURES:
    duplicate_drop_cols = get_duplicate_drop_list(duplicate_feature_pairs)
    drop_cols = list(dict.fromkeys(duplicate_drop_cols + suspicious_features))

    print("\n[DEBUG MODE] Removing suspicious / duplicate features:")
    print(drop_cols)

    X = X.drop(columns=drop_cols, errors="ignore")
    print("New feature shape after debug-only removal:", X.shape)


# 6. Models and parameter grids
models = {
    "SVM": (
        SVC(),
        {
            "model__C": [1, 10],
            "model__kernel": ["rbf"],
            "model__gamma": ["scale"]
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "model__n_neighbors": [5, 7]
        }
    ),

    "DecisionTree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "model__max_depth": [None, 10],
            "model__min_samples_split": [2, 5]
        }
    ),

    "RandomForest": (
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        {
            "model__n_estimators": [100],
            "model__max_depth": [None, 10]
        }
    ),

    "MLP": (
        MLPClassifier(
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE
        ),
        {
            "model__hidden_layer_sizes": [(100,)],
            "model__learning_rate_init": [0.001]
        }
    )
}


# 7. Outer 10-fold CV
outer_cv = StratifiedKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = {
    name: {
        "acc": [],
        "f1": [],
        "best_params": []
    }
    for name in models.keys()
}

fold_records = []

print("\n===== Start nested CV =====")

for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
    print(f"\n----- Fold {fold} -----")

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()

    # inner CV: performed only on outer-train
    inner_cv = build_inner_cv(
        y_train=y_train,
        preferred_splits=INNER_SPLITS_PREFERRED,
        random_state=RANDOM_STATE
    )

    for name, (estimator, param_grid) in models.items():
        pipe = build_pipeline(name, estimator)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True
        )

        # When grid.fit(X_train, y_train) runs here,
        # missing-value imputation and standardization are both re-fitted inside the inner CV
        # This prevents the overall validation information from outer-train from leaking into inner CV
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        results[name]["acc"].append(acc)
        results[name]["f1"].append(f1)
        results[name]["best_params"].append(grid.best_params_)

        fold_records.append({
            "fold": fold,
            "model": name,
            "accuracy": acc,
            "f1_macro": f1,
            "best_params": str(grid.best_params_)
        })

        print(
            f"{name} | "
            f"Acc: {acc:.4f} | "
            f"F1: {f1:.4f} | "
            f"Best Params: {grid.best_params_}"
        )


# 8. Summary
print("\n===== RESULTS (Mean ± Std) =====")

summary_rows = []

for name in models.keys():
    acc_mean = np.mean(results[name]["acc"])
    acc_std = np.std(results[name]["acc"], ddof=1)

    f1_mean = np.mean(results[name]["f1"])
    f1_std = np.std(results[name]["f1"], ddof=1)

    summary_rows.append({
        "Model": name,
        "Accuracy Mean": acc_mean,
        "Accuracy Std": acc_std,
        "F1 Mean": f1_mean,
        "F1 Std": f1_std
    })

    print(f"{name}:")
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")

summary_df = pd.DataFrame(summary_rows)
fold_df = pd.DataFrame(fold_records)

print("\n===== Summary table =====")
print(summary_df)

print("\n===== Fold-level results =====")
print(fold_df)