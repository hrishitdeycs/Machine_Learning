```python
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
```
<img width="685" height="326" alt="Screenshot 2026-08-05 193916" src="https://github.com/user-attachments/assets/2b988f0c-fc3b-4955-9164-4f968fa517fa" />
<img width="1000" height="600" alt="yield_prediction" src="https://github.com/user-attachments/assets/b94441cf-2113-4d8f-a53c-0488902b04ad" />

This code performs a **wheat yield prediction task using Simple Linear Regression**. It loads a CSV dataset containing yearly values, cleans and sorts the data, then uses **Year as the input feature** and **Yield/Value as the target** to build a regression model by manually calculating the slope (weight) and intercept (bias). The dataset is divided into training and testing sets to evaluate the model using Mean Squared Error (MSE). Finally, the model predicts future wheat yields for the years 2027–2030 and visualizes the actual data, regression trend line, and future predictions in a graph.
The code uses the **Simple Linear Regression equation**:

$$
y = mx + c
$$

where:

- **\(y\)** = Dependent variable (the value being predicted) → **Yield / Value**
- **\(x\)** = Independent variable (the input used for prediction) → **Year**
- **\(m\)** = Weight or slope (shows how much the yield changes with each year)
- **\(c\)** = Bias or intercept (the predicted value when Year = 0)

In this code:

- **Independent variable (X):** `Year`
- **Dependent variable (y):** `Value` (wheat yield)

The model learns the relationship:

$$
\text{Yield} = (\text{weight} \times \text{Year}) + \text{bias}
$$

and uses this equation to predict future yields for the years **2027–2030**.
