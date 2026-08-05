import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------------
# Load CSV Dataset
# -----------------------------------
df = pd.read_csv("FAOSTAT_data_en_8-4-2026.csv")

# Keep only Year and Value columns
df = df[["Year", "Value"]]

# Remove missing values
df = df.dropna()

# Sort data by Year
df = df.sort_values("Year")

# -----------------------------------
# Features (X) and Target (y)
# -----------------------------------
X = df["Year"].values
y = df["Value"].values

# -----------------------------------
# Split into Training (80%) and Testing (20%)
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Calculate weight (m) and bias (c) manually
# -----------------------------------

x_mean = np.mean(X_train)
y_mean = np.mean(y_train)

# Weight (Slope)
weight = np.sum((X_train - x_mean) * (y_train - y_mean)) / \
         np.sum((X_train - x_mean) ** 2)

# Bias (Intercept)
bias = y_mean - (weight * x_mean)

# -----------------------------------
# Predict Test Data
# -----------------------------------
y_pred = weight * X_test + bias

# -----------------------------------
# Mean Squared Error
# -----------------------------------
mse = mean_squared_error(y_test, y_pred)

# -----------------------------------
# Print Results
# -----------------------------------
print("Weight (m):", weight)
print("Bias (c):", bias)
print("Mean Squared Error (MSE):", mse)

# -----------------------------------
# Predict Future Years
# -----------------------------------
future_years = np.array([2027, 2028, 2029, 2030])
future_predictions = weight * future_years + bias

print("\nPredicted Yield (2027-2030)")
for year, prediction in zip(future_years, future_predictions):
    print(f"{year}: {prediction:.4f}")

# -----------------------------------
# Plot
# -----------------------------------
plt.figure(figsize=(10,6))

# Actual Data
plt.scatter(X, y, color='blue', label='Actual Data')

# Regression Line
line_years = np.arange(min(X), 2031)
line_predictions = weight * line_years + bias
plt.plot(line_years, line_predictions, color='red', linewidth=2, label='Regression Line')

# Future Predictions
plt.scatter(future_years, future_predictions,
            color='green', marker='*', s=180,
            label='Predicted (2027-2030)')

plt.title("Wheat Yield Prediction using Simple Linear Regression")
plt.xlabel("Year")
plt.ylabel("Yield (Value)")
plt.legend()
plt.grid(True)
plt.show()
