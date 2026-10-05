```python
# ANN - German Credit Dataset
# Good = 0, Bad = 1
# Train = 80%, Test = 20%

import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

# Load ARFF
data, meta = arff.loadarff("dataset_31_credit-g.arff")
df = pd.DataFrame(data)

# Decode byte values
for col in df.select_dtypes("object"):
    df[col] = df[col].str.decode("utf-8")

# Features and target
target = df.columns[-1]
X = pd.get_dummies(df.drop(columns=target), drop_first=True)
y = df[target].map({"good": 0, "bad": 1})

# Check target
print("Classes:", df[target].unique())
print("Good = 0, Bad = 1")

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ANN
ann = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    random_state=42
)

# Train and predict
ann.fit(X_train, y_train)
y_pred = ann.predict(X_test)
y_prob = ann.predict_proba(X_test)[:, 1]

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
error = 1 - accuracy

# Results
print("\n===== CONFUSION MATRIX =====")
print("                 Predicted")
print("                 Good  Bad")
print(f"Actual Good     {tn:4d} {fp:4d}")
print(f"Actual Bad      {fn:4d} {tp:4d}")

print("\n===== ANN RESULTS =====")
print(f"Accuracy    : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision   : {precision:.4f}")
print(f"TP          : {tp}")
print(f"TN          : {tn}")
print(f"FP          : {fp}")
print(f"FN          : {fn}")
print(f"Error       : {error:.4f} ({error*100:.2f}%)")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"AUC         : {auc:.4f}")
```
<img width="266" height="178" alt="image" src="https://github.com/user-attachments/assets/24a1e941-07f7-423f-9848-38bce645d3bf" />
<img width="419" height="288" alt="image" src="https://github.com/user-attachments/assets/c847585d-6c8f-44de-bd4f-d2e300cf1f89" />
