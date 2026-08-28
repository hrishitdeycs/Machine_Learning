import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color',
    'bruises', 'odor', 'gill-attachment', 'gill-spacing',
    'gill-size', 'gill-color', 'stalk-shape',
    'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color',
    'population', 'habitat'
]

df = pd.read_csv(
    "agaricus-lepiota.data",
    header=None,
    names=columns
)


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

# '?' means missing value
df = df.replace('?', np.nan)

# Fill missing values with the most frequent category
for column in df.columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop('class', axis=1)
y = df['class']


# ============================================================
# 4. ENCODE FEATURES
# ============================================================

# DecisionTreeClassifier requires numerical input.
# The encoded numbers are only used internally by the model.
# We will report the final classes as English labels.

for column in X.columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])


# ============================================================
# 5. ENCODE TARGET
# ============================================================

# Explicitly define:
# e = 0 = Edible
# p = 1 = Poisonous

y = y.map({
    'e': 0,
    'p': 1
})


# ============================================================
# 6. TRAIN-TEST SPLIT: 80% / 20%
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 7. DECISION TREE USING ENTROPY
# ============================================================

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# 8. PREDICTION
# ============================================================

y_pred = model.predict(X_test)

# Probability of Poisonous class
y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

TN, FP, FN, TP = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
).ravel()


# ============================================================
# 10. METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

error = 1 - accuracy

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)

specificity = TN / (TN + FP)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1
)

auc = roc_auc_score(
    y_test,
    y_prob
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("MUSHROOM CLASSIFICATION RESULTS")
print("========================================")

print("Positive Class = Poisonous")
print("Negative Class = Edible")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nTP =", TP, "(Poisonous predicted as Poisonous)")
print("TN =", TN, "(Edible predicted as Edible)")
print("FP =", FP, "(Edible predicted as Poisonous)")
print("FN =", FN, "(Poisonous predicted as Edible)")

print("\n----------------------------------------")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Error        : {error:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"AUC          : {auc:.4f}")
print("----------------------------------------")


# ============================================================
# 12. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Decision Tree (AUC = {auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    'r--',
    label='Random Classifier'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Mushroom Dataset")

plt.legend()
plt.grid()

plt.show()
