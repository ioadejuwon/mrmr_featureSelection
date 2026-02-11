import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from mrmr import mrmr_classif

# Load data
df = pd.read_csv("diabetes.csv")
X, y = df.drop("target", axis=1), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler(); X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# MRMR features
sel_feats = mrmr_classif(pd.DataFrame(X_train, columns=X.columns), y_train, K=5)
X_train_mrmr = X_train[:, [X.columns.get_loc(f) for f in sel_feats]]
X_test_mrmr = X_test[:, [X.columns.get_loc(f) for f in sel_feats]]

# Classifiers
classifiers = {
    "SVM": SVC(probability=True),
    "k-NN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "MLP": MLPClassifier(max_iter=1000)
}

# Compare results
results = []
for name, clf in classifiers.items():
    for feat_set, X_tr, X_te in [("All Features", X_train, X_test), ("MRMR", X_train_mrmr, X_test_mrmr)]:
        clf.fit(X_tr, y_train)
        y_pred, y_proba = clf.predict(X_te), clf.predict_proba(X_te)[:,1]
        results.append([name, feat_set, accuracy_score(y_test, y_pred),
                        f1_score(y_test, y_pred), roc_auc_score(y_test, y_proba)])

# Display results
res_df = pd.DataFrame(results, columns=["Classifier", "Features", "Accuracy", "F1 Score", "ROC-AUC"])
print(res_df)
print("\nSelected MRMR Features:", sel_feats)
