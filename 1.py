#Import Libraries
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
print("Libraries imported successfully.")

# LOAD DATA
df = pd.read_csv("data/1/train.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# CHECK MISSING VALUES
print("Missing values per column:\n")
print(X.isnull().sum())

print("Columns:" + str(X.columns.tolist()))
print("Number of duplicated rows: " + str(df.duplicated().sum()))


# MODELS + PARAMS
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



# 10-FOLD CV
# -----------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = {name: {"acc": [], "f1": []} for name in models.keys()}

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\nFold {fold}")
    
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # MISSING VALUE IMPUTATION (inside fold, no leakage)
    for col in X_train.columns:
        # Use global mean (simpler & safe)
        mean_val = X_train[col].mean()
        X_train[col] = X_train[col].fillna(mean_val)
        X_val[col] = X_val[col].fillna(mean_val)

    # SCALE DATA (inside fold)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # TRAIN MODELS WITH NESTED CV
    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=3, scoring='f1_macro')
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')

        results[name]["acc"].append(acc)
        results[name]["f1"].append(f1)

        print(f"{name} | Acc: {acc:.4f} | F1: {f1:.4f} | Best Params: {grid.best_params_}")



print("RESULTS Main (Mean ± Std)")

for name in models.keys():
    acc_mean = np.mean(results[name]["acc"])
    acc_std = np.std(results[name]["acc"])
    
    f1_mean = np.mean(results[name]["f1"])
    f1_std = np.std(results[name]["f1"])

    print(f"{name}:")
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")