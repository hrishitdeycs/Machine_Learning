```python
# Naive Bayes Classification
# Dataset: loan.csv
# Train-Test Split: 80% Training / 20% Testing

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

df = pd.read_csv("07_loan_approval.csv")

# ---------------------------------------------------------
# 2. Convert LoanApproved Y/N to 1/0
# ---------------------------------------------------------

# Y = Approved = 1
# N = Not Approved = 0

df["LoanApproved"] = df["LoanApproved"].map({
    "Y": 1,
    "N": 0
})

print("\nLoanApproved values:")
print(df["LoanApproved"].value_counts())

# ---------------------------------------------------------
# 3. Define Features and Target
# ---------------------------------------------------------

# ApplicantID is an identifier, so remove it
X = df.drop(columns=["ApplicantID", "LoanApproved"])

y = df["LoanApproved"]

# ---------------------------------------------------------
# 4. Convert Categorical Variables to Numerical
# ---------------------------------------------------------

X = pd.get_dummies(X, drop_first=True)

# ---------------------------------------------------------
# 5. Handle Missing Values
# ---------------------------------------------------------

for column in X.columns:
    if X[column].isnull().any():
        X[column] = X[column].fillna(X[column].median())

# ---------------------------------------------------------
# 6. Split Dataset
#    80% Training
#    20% Testing
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ---------------------------------------------------------
# 7. Create Naive Bayes Model
# ---------------------------------------------------------

model = GaussianNB()

# ---------------------------------------------------------
# 8. Train Model
# ---------------------------------------------------------

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 9. Make Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)

# Probability of LoanApproved = Y (1)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 10. Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

# ---------------------------------------------------------
# 11. Calculate Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

error = 1 - accuracy

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_prob
)

# ---------------------------------------------------------
# 12. Display Confusion Matrix
# ---------------------------------------------------------

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print("                Predicted")
print("              N        Y")
print(f"Actual N    {TN:5d}    {FP:5d}")
print(f"Actual Y    {FN:5d}    {TP:5d}")

# ---------------------------------------------------------
# 13. Display Results
# ---------------------------------------------------------

print("\n==========================================")
print("NAIVE BAYES RESULTS")
print("==========================================")

print(f"Accuracy    : {accuracy:.4f}")
print(f"TP          : {TP}")
print(f"TN          : {TN}")
print(f"FP          : {FP}")
print(f"FN          : {FN}")
print(f"Error       : {error:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"AUC         : {auc:.4f}")

print("==========================================")


```
<img width="432" height="360" alt="image" src="https://github.com/user-attachments/assets/6aa455b0-97ac-4871-b081-d9e615ca9a5b" />
<img width="459" height="323" alt="image" src="https://github.com/user-attachments/assets/638ba139-d344-4c31-bd24-1b9332c65108" />
