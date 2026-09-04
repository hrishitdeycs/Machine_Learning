# ============================================================
# KNN Classification - Mushroom Edibility Dataset
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score
)


# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

file_name = "11_mushroom_edibility.csv"

df = pd.read_csv(file_name)


# ------------------------------------------------------------
# 2. Define Features (X) and Target (y)
# ------------------------------------------------------------

# SampleID is an identifier and should not be used for prediction
X = df.drop(columns=["SampleID", "Class"])

y = df["Class"].astype(str).str.strip().str.lower()


# ------------------------------------------------------------
# 3. Explicitly Encode Target
# ------------------------------------------------------------
# Edible    = 0 (Negative Class)
# Poisonous = 1 (Positive Class)

y = y.map({
    "edible": 0,
    "poisonous": 1
})



print("\nClass Mapping:")
print("Edible    = 0")
print("Poisonous = 1")


# ------------------------------------------------------------
# 4. Identify Categorical Features
# ------------------------------------------------------------

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# ------------------------------------------------------------
# 5. One-Hot Encoding
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ------------------------------------------------------------
# 6. Create KNN Model
# ------------------------------------------------------------

knn = KNeighborsClassifier(
    n_neighbors=20
)


# ------------------------------------------------------------
# 7. Create Pipeline
# ------------------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("knn", knn)
])


# ------------------------------------------------------------
# 8. Split Dataset
# ------------------------------------------------------------
# 80% Training
# 20% Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ------------------------------------------------------------
# 9. Train KNN Model
# ------------------------------------------------------------

model.fit(X_train, y_train)


# ------------------------------------------------------------
# 10. Make Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# Probability of poisonous class
y_prob = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 11. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

TN, FP, FN, TP = cm.ravel()


# ------------------------------------------------------------
# 12. Calculate Performance Metrics
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 13. Display Confusion Matrix
# ------------------------------------------------------------

print("\n==============================================")
print("              CONFUSION MATRIX")
print("==============================================")

print("\n                 Predicted")
print("              Edible  Poisonous")
print("Actual Edible    ", TN, "     ", FP)
print("Actual Poisonous ", FN, "     ", TP)


# ------------------------------------------------------------
# 14. Display Metrics
# ------------------------------------------------------------

print("\n==============================================")
print("             KNN RESULTS")
print("==============================================")

print(f"Accuracy    : {accuracy:.4f}")
print(f"TP          : {TP}")
print(f"TN          : {TN}")
print(f"FP          : {FP}")
print(f"FN          : {FN}")
print(f"Error       : {error:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"AUC         : {auc:.4f}")


# ------------------------------------------------------------
# 15. Display Percentage Metrics
# ------------------------------------------------------------

print("\n==============================================")
print("          METRICS IN PERCENTAGE")
print("==============================================")

print(f"Accuracy    : {accuracy * 100:.2f}%")
print(f"Error       : {error * 100:.2f}%")
print(f"Recall      : {recall * 100:.2f}%")
print(f"Specificity : {specificity * 100:.2f}%")
print(f"F1-score    : {f1 * 100:.2f}%")
print(f"AUC         : {auc:.4f}")


# ------------------------------------------------------------
# 16. Interpretation of TP/TN/FP/FN
# ------------------------------------------------------------

print("\n==============================================")
print("          CONFUSION MATRIX MEANING")
print("==============================================")

print("TP = Poisonous mushrooms correctly predicted as Poisonous")
print("TN = Edible mushrooms correctly predicted as Edible")
print("FP = Edible mushrooms incorrectly predicted as Poisonous")
print("FN = Poisonous mushrooms incorrectly predicted as Edible")
