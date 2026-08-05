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
