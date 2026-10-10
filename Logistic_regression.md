```python
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

# Load ARFF dataset
data, meta = arff.loadarff("php0iVrYT.arff")

# Convert to DataFrame
df = pd.DataFrame(data)

# Decode byte-string columns, if any
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

# 2. Separate features and target (last column)
X = df.iloc[:, :-1]
y_original = df.iloc[:, -1]

# Original labels: 1 = did not donate, 2 = donated
# Convert to binary: 0 = did not donate, 1 = donated
y = (y_original == '2').astype(int)

# 3. Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Build and train Logistic Regression model
# Standardization is fitted only on the training data.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# 5. Predict probabilities and apply 50% threshold
y_probability = model.predict_proba(X_test)[:, 1]

threshold = 0.50
y_pred = (y_probability >= threshold).astype(int)

# 6. Calculate confusion matrix
# Positive class = donated (original label 2)
TN, FP, FN, TP = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

# 7. Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
error_rate = 1 - accuracy

f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_probability)

# 8. Display results
results = {
    "Training samples": len(X_train),
    "Testing samples": len(X_test),
    "Threshold": threshold,
    "TP": TP,
    "TN": TN,
    "FP": FP,
    "FN": FN,
    "Accuracy": accuracy,
    "Precision": precision,
    "Error Rate": error_rate,
    "Recall (Sensitivity)": recall,
    "Specificity": specificity,
    "F1-score": f1,
    "AUC": auc
}

print("\n--- Logistic Regression Results ---")
for metric, value in results.items():
    if isinstance(value, (float, np.floating)):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

# 9. display confusion matrix
print("\nConfusion Matrix:")
print(pd.DataFrame(
    [[TN, FP], [FN, TP]],
    index=["Actual Not Donated", "Actual Donated"],
    columns=["Predicted Not Donated", "Predicted Donated"]
))
```
<img width="357" height="333" alt="image" src="https://github.com/user-attachments/assets/6f196ca4-afc3-4d8c-a0e0-296338d3fa63" />
<img width="579" height="118" alt="image" src="https://github.com/user-attachments/assets/6624d858-9e2d-4aba-ba20-d9be64f3339b" />
