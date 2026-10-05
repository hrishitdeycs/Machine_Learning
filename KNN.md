```python
# ============================================================
# KNN CLASSIFICATION - CREDIT-G DATASET
# ============================================================

import pandas as pd
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, f1_score, roc_auc_score
)

# 1. Load Dataset
data, meta = arff.loadarff("dataset_31_credit-g.arff")
df = pd.DataFrame(data)

# Decode byte values
for col in df.select_dtypes(["object"]).columns:
    df[col] = df[col].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )


# 2. Separate Features and Target
X = df.drop("class", axis=1)
y = df["class"].map({"good": 0, "bad": 1})


# 3. Identify Numerical and Categorical Features
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

# 4. Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

# 5. KNN Model
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)

model = Pipeline([
    ("preprocessing", preprocessor),
    ("knn", knn)
])

# 6. Split Dataset: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# 7. Train Model
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 9. Confusion Matrix
tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

print("\nConfusion Matrix:")
print(pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Actual Good", "Actual Bad"],
    columns=["Predicted Good", "Predicted Bad"]
))

# 10. Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
error = 1 - accuracy

# 11. Display Results
print("\nKNN PERFORMANCE METRICS")
print("-" * 40)
print("True Positive :", tp)
print("True Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)

print(f"\nAccuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"AUC         : {auc:.4f}")
print(f"Error       : {error:.4f}")

```
<img width="418" height="309" alt="image" src="https://github.com/user-attachments/assets/9791a26f-5bc2-44b3-907d-bad4471e159b" />
<img width="405" height="194" alt="image" src="https://github.com/user-attachments/assets/28e96225-900f-48fc-8dd4-201d2a2b997b" />
