import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/1/test.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------

# OPTION 1: Fill with global mean (simple)
X = X.fillna(X.mean())

# OPTION 2 (Better): Fill using class-wise mean
# Uncomment this instead if you prefer
"""
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col] = X.groupby(y)[col].transform(lambda x: x.fillna(x.mean()))
"""


# -----------------------------
# MODELS + PARAMS
# -----------------------------
models = {
    "SVM": (
        SVC(),
        {
            "C": [1, 10],
            "kernel": ["rbf"],
            "gamma": ["scale"]
        }
    ),
    
    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [5, 7]
        }
    ),
    
    "DecisionTree": (
        DecisionTreeClassifier(),
        {
            "max_depth": [None, 10]
        }
    ),
    
    "RandomForest": (
        RandomForestClassifier(),
        {
            "n_estimators": [100],
            "max_depth": [None, 10]
        }
    ),
    
    "MLP": (
        MLPClassifier(max_iter=500),
        {
            "hidden_layer_sizes": [(100,)],
            "learning_rate_init": [0.001]
        }
    )
}


# -----------------------------
# 10-FOLD CV
# -----------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = {name: {"acc": [], "f1": []} for name in models.keys()}


for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Scale inside fold
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    for name, (model, params) in models.items():
        
        grid = GridSearchCV(model, params, cv=3, scoring='f1_macro')
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')

        results[name]["acc"].append(acc)
        results[name]["f1"].append(f1)

        print(f"{name} | Acc: {acc:.4f} | F1: {f1:.4f}")


# -----------------------------
# FINAL RESULTS
# -----------------------------
print("\n\n=== FINAL RESULTS (Mean ± Std) ===")

for name in models.keys():
    acc_mean = np.mean(results[name]["acc"])
    acc_std = np.std(results[name]["acc"])
    
    f1_mean = np.mean(results[name]["f1"])
    f1_std = np.std(results[name]["f1"])

    print(f"{name}:")
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")