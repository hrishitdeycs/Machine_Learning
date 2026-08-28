```python
# ============================================================
# AGARICUS-LEPIOTA MUSHROOM CLASSIFICATION
# Naive Bayes | 80% Training / 20% Testing
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, roc_auc_score, roc_curve


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "agaricus-lepiota.data"
df = pd.read_csv(file_path, header=None)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

# 0 = Edible
# 1 = Poisonous

y = df.iloc[:, 0].map({'e': 0, 'p': 1})
X = df.iloc[:, 1:]


# ============================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X).astype(int)


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

print("\n============================================")
print("TRAIN-TEST SPLIT")
print("============================================")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. CREATE AND TRAIN NAIVE BAYES MODEL
# ============================================================

model = CategoricalNB()
model.fit(X_train, y_train)


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("\n============================================")
print("CONFUSION MATRIX")
print("============================================")
print("\n                  Predicted")
print("                  Edible  Poisonous")
print("--------------------------------------------")
print(f"Actual Edible    {TN:7d}  {FP:9d}")
print(f"Actual Poisonous {FN:7d}  {TP:9d}")


# ============================================================
# 8. PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
error = 1 - accuracy
recall = recall_score(y_test, y_pred, pos_label=1)
specificity = TN / (TN + FP)
f1 = f1_score(y_test, y_pred, pos_label=1)
auc = roc_auc_score(y_test, y_prob)


# ============================================================
# 9. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", linewidth=2, label=f"Naive Bayes (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.5, label="Random Classifier (AUC = 0.50)")
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Recall)", fontsize=11)
plt.title("ROC Curve - Naive Bayes Classifier", fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()


# ============================================================
# 10. FINAL PERFORMANCE RESULTS
# ============================================================

print("\n============================================")
print("NAIVE BAYES PERFORMANCE")
print("============================================")

total = len(y_test)

results = pd.DataFrame({
    "Metric": ["Accuracy", "TP", "TN", "FP", "FN", "Error", "Recall", "Specificity", "F1-Score", "AUC"],
    "Value": [accuracy, TP, TN, FP, FN, error, recall, specificity, f1, auc],
    "Percentage": [
        accuracy * 100,
        TP / total * 100,
        TN / total * 100,
        FP / total * 100,
        FN / total * 100,
        error * 100,
        recall * 100,
        specificity * 100,
        f1 * 100,
        auc * 100
    ]
})

results["Value"] = results["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
results["Percentage"] = results["Percentage"].apply(lambda x: f"{x:.2f}%")

print(results.to_string(index=False))


# ============================================================
# 11. CLASS INFORMATION
# ============================================================

print("\n============================================")
print("CLASS INFORMATION")
print("============================================")
print("0 = Edible")
print("1 = Poisonous")
print("Positive class = Poisonous")
print("============================================")

```
<img width="573" height="406" alt="image" src="https://github.com/user-attachments/assets/28867a78-f924-4b34-94b1-9b071152f318" />
<img width="553" height="426" alt="image" src="https://github.com/user-attachments/assets/d1e6d3a1-9503-47ee-9e1c-45e598b03d9a" />
<img width="800" height="600" alt="naive_bayes_" src="https://github.com/user-attachments/assets/d07be8f5-4f5e-4899-9170-d1356726eb66" />

