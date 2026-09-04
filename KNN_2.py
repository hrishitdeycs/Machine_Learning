
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("07_loan_approval.csv")


# --------------------------------------------------
# 2. Convert Target Variable
# --------------------------------------------------

# Y = Approved = 1
# N = Not Approved = 0

df["LoanApproved"] = df["LoanApproved"].map({
    "Y": 1,
    "N": 0
})

# Check for invalid/missing target values
if df["LoanApproved"].isnull().any():
    raise ValueError("LoanApproved contains values other than Y or N.")

# --------------------------------------------------
# 3. Define X and Y
# --------------------------------------------------

# Remove ApplicantID because it is only an ID
X = df.drop(columns=["LoanApproved", "ApplicantID"])
y = df["LoanApproved"]

# --------------------------------------------------
# 4. Define Numerical and Categorical Columns
# --------------------------------------------------

numeric_features = [
    "Age",
    "AnnualIncome",
    "LoanAmount",
    "CreditScore",
    "EmploymentYears"
]

categorical_features = [
    "Education",
    "MaritalStatus",
    "PropertyArea",
    "SelfEmployed"
]

# --------------------------------------------------
# 5. 80% Training and 20% Testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

# --------------------------------------------------
# 7. KNN Model
# --------------------------------------------------

knn = KNeighborsClassifier(
    n_neighbors=5
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("knn", knn)
])

# --------------------------------------------------
# 8. Train the Model
# --------------------------------------------------

model.fit(X_train, y_train)

# --------------------------------------------------
# 9. Prediction
# --------------------------------------------------

y_pred = model.predict(X_test)

# Probability needed for AUC
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 10. Confusion Matrix
# --------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
).ravel()

# --------------------------------------------------
# 11. Calculate Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

error = 1 - accuracy

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

specificity = tn / (tn + fp)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_prob
)

# --------------------------------------------------
# 12. Display Results
# --------------------------------------------------

print("\n========================================")
print("          KNN CLASSIFICATION")
print("========================================")

print("Target Classes:")
print("Y = 1 (Loan Approved)")
print("N = 0 (Loan Not Approved)")

print("\nTrain-Test Split:")
print("Training = 80%")
print("Testing  = 20%")

print("\nK Value:", 5)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n========================================")
print("           PERFORMANCE METRICS")
print("========================================")

print(f"Accuracy    : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"TP          : {tp}")
print(f"TN          : {tn}")
print(f"FP          : {fp}")
print(f"FN          : {fn}")
print(f"Error       : {error:.4f} ({error*100:.2f}%)")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"AUC         : {auc:.4f}")

print("========================================")
