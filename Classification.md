```python
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


# Load Iris dataset
iris = load_iris()


# Convert to DataFrame
X = pd.DataFrame(
    iris.data,
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)

y = pd.Series(iris.target)


# Convert numeric labels to flower names
flower_names = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

y_names = y.map(flower_names)


# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_names,
    test_size=0.2,
    random_state=42,
    stratify=y_names
)


# Save original test values for display
X_test_original = X_test.copy()


# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)


# Train model
knn.fit(X_train, y_train)


# Predict
y_pred = knn.predict(X_test)


# Display values with predictions
results = X_test_original.copy()
results["Actual Class"] = y_test.values
results["Predicted Class"] = y_pred

print(results)


# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nAccuracy:", accuracy)
print("F1 Score:", f1)
```
<img width="860" height="541" alt="Screenshot 2026-08-05 200720" src="https://github.com/user-attachments/assets/2a25e44b-bf34-4206-9a0d-20502c420fc0" />
<img width="854" height="314" alt="Screenshot 2026-08-05 200745" src="https://github.com/user-attachments/assets/b8b6ed5d-3761-43f0-9799-5e12d372b63f" />

This code performs a **flower species classification task using the K-Nearest Neighbors (KNN) machine learning algorithm** on the Iris dataset. It loads and prepares the dataset by converting flower measurements into features and mapping numerical labels to flower names, then splits the data into training and testing sets. The features are standardized to improve model performance, and a KNN classifier is trained to learn the relationship between flower measurements (sepal and petal dimensions) and species classes. Finally, the model predicts the flower species for test samples and evaluates its performance using **accuracy** and **F1 score** metrics.

## Accuracy Formula

Accuracy measures the proportion of correctly predicted samples out of all test samples.

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$

or

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$


## F1 Score Formula

The F1 score is the harmonic mean of **Precision** and **Recall**. It balances both false positives and false negatives.

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}
{\text{Precision} + \text{Recall}}
$$

where:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### Terms:

- **TP (True Positive):** Correctly predicted positive cases
- **TN (True Negative):** Correctly predicted negative cases
- **FP (False Positive):** Incorrectly predicted positive cases
- **FN (False Negative):** Incorrectly predicted negative cases
