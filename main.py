import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from mrmr import mrmr_classif

# Load data
df = pd.read_csv("data/diabetes.csv")
X, y = df.drop("diabetes", axis=1), df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# X_train_df = pd.DataFrame(X_train, columns=X_train.columns)
# sel_feats = mrmr_classif(X_train_df, y_train, K=5, method="MID")







X_train = pd.get_dummies(X_train, columns=['gender'])
X_test = pd.get_dummies(X_test, columns=['gender'])

X_train = pd.get_dummies(X_train, columns=['smoking_history'])
X_test = pd.get_dummies(X_test, columns=['smoking_history'])

# Align test columns with train columns (important!)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# After encoding and aligning
X_train_df = pd.DataFrame(X_train, columns=X_train.columns)
sel_feats = mrmr_classif(X_train_df, y_train, K=5, method="MID")


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
