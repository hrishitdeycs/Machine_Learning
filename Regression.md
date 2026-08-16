```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------------
# Load CSV Dataset
# -----------------------------------
df = pd.read_csv("FAOSTAT_data_en_8-16-2026.csv")

# Keep only Year and Value columns
df = df[["Year", "Value"]]

# Remove missing values
df = df.dropna()

# Sort data by Year
df = df.sort_values("Year")

# -----------------------------------
# Reshape Dataset
# First Value of each year = Area Harvested
# Second Value of each year = Yield
# -----------------------------------
df["Type"] = df.groupby("Year").cumcount()

area_df = df[df["Type"] == 0][["Year", "Value"]].rename(
    columns={"Value": "Area Harvested"}
)

yield_df = df[df["Type"] == 1][["Year", "Value"]].rename(
    columns={"Value": "Yield"}
)

df = pd.merge(area_df, yield_df, on="Year")

# -----------------------------------
# Features (X) and Target (y)
# -----------------------------------
X = df["Year"].values
y = df["Yield"].values
area = df["Area Harvested"].values

# -----------------------------------
# Split into Training (80%) and Testing (20%)
# -----------------------------------
X_train, X_test, y_train, y_test, area_train, area_test = train_test_split(
    X,
    y,
    area,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# 1. LINEAR REGRESSION
# Year -> Yield
# -----------------------------------

x_mean = np.mean(X_train)
y_mean = np.mean(y_train)

# Weight (Slope)
weight = np.sum((X_train - x_mean) * (y_train - y_mean)) / \
         np.sum((X_train - x_mean) ** 2)

# Bias (Intercept)
bias = y_mean - (weight * x_mean)

# Predict Test Data
y_pred_linear = weight * X_test + bias

# Metrics
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# -----------------------------------
# 2. POLYNOMIAL REGRESSION
# Year -> Yield
# -----------------------------------

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
X_test_poly = poly.transform(X_test.reshape(-1, 1))

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

# Metrics
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# -----------------------------------
# 3. MULTIVARIATE LINEAR REGRESSION
# Year + Area Harvested -> Yield
# -----------------------------------

X_train_multi = np.column_stack((X_train, area_train))
X_test_multi = np.column_stack((X_test, area_test))

multi_model = LinearRegression()
multi_model.fit(X_train_multi, y_train)

y_pred_multi = multi_model.predict(X_test_multi)

# Metrics
mse_multi = mean_squared_error(y_test, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
mae_multi = mean_absolute_error(y_test, y_pred_multi)
r2_multi = r2_score(y_test, y_pred_multi)

# -----------------------------------
# Compare Models
# -----------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Multivariate Regression"
    ],
    "MSE": [
        mse_linear,
        mse_poly,
        mse_multi
    ],
    "RMSE": [
        rmse_linear,
        rmse_poly,
        rmse_multi
    ],
    "MAE": [
        mae_linear,
        mae_poly,
        mae_multi
    ],
    "R²": [
        r2_linear,
        r2_poly,
        r2_multi
    ]
})

print("\nModel Comparison (20% Test Data)")
print(comparison.to_string(index=False))

# -----------------------------------
# Determine Best Model
# Lower RMSE = Better
# -----------------------------------

best_model = comparison.loc[comparison["RMSE"].idxmin(), "Model"]

print("\nBest Fit Model:", best_model)

# -----------------------------------
# Predict Future Years
# 2027 - 2030
# -----------------------------------

future_years = np.array([2027, 2028, 2029, 2030])

# -----------------------------------
# Linear Regression Predictions
# -----------------------------------

linear_future_predictions = weight * future_years + bias

# -----------------------------------
# Polynomial Regression Predictions
# -----------------------------------

future_years_poly = poly.transform(future_years.reshape(-1, 1))
poly_future_predictions = poly_model.predict(future_years_poly)

# -----------------------------------
# Estimate Future Area Harvested
# for Multivariate Model
# -----------------------------------

area_model = LinearRegression()
area_model.fit(df[["Year"]], df["Area Harvested"])

future_area = area_model.predict(
    future_years.reshape(-1, 1)
)

# -----------------------------------
# Multivariate Yield Predictions
# -----------------------------------

future_multi_X = np.column_stack((
    future_years,
    future_area
))

multi_future_predictions = multi_model.predict(future_multi_X)

# -----------------------------------
# Print Future Predictions
# -----------------------------------

print("\nPredicted Yield (2027-2030)")

print("\nLinear Regression:")
for year, prediction in zip(
    future_years,
    linear_future_predictions
):
    print(f"{year}: {prediction:.4f}")

print("\nPolynomial Regression:")
for year, prediction in zip(
    future_years,
    poly_future_predictions
):
    print(f"{year}: {prediction:.4f}")

print("\nMultivariate Regression:")
for year, prediction in zip(
    future_years,
    multi_future_predictions
):
    print(f"{year}: {prediction:.4f}")

# -----------------------------------
# Plot
# -----------------------------------

plt.figure(figsize=(10, 6))

# Actual Data
plt.scatter(
    X,
    y,
    color="blue",
    label="Actual Yield"
)

# Linear Regression Line
line_years = np.arange(min(X), 2031)

line_predictions_linear = (
    weight * line_years + bias
)

plt.plot(
    line_years,
    line_predictions_linear,
    color="red",
    linewidth=2,
    label="Linear Regression"
)

# Polynomial Regression Curve
line_years_poly = poly.transform(
    line_years.reshape(-1, 1)
)

line_predictions_poly = poly_model.predict(
    line_years_poly
)

plt.plot(
    line_years,
    line_predictions_poly,
    color="orange",
    linewidth=2,
    label="Polynomial Regression"
)

# Future Linear Predictions
plt.scatter(
    future_years,
    linear_future_predictions,
    color="red",
    marker="*",
    s=180,
    label="Linear Future Prediction"
)

# Future Polynomial Predictions
plt.scatter(
    future_years,
    poly_future_predictions,
    color="orange",
    marker="*",
    s=180,
    label="Polynomial Future Prediction"
)

# Future Multivariate Predictions
plt.scatter(
    future_years,
    multi_future_predictions,
    color="green",
    marker="*",
    s=180,
    label="Multivariate Future Prediction"
)

plt.title("Wheat Yield Prediction using Three Regression Models")
plt.xlabel("Year")
plt.ylabel("Yield")
plt.legend()
plt.grid(True)
plt.show()
```
<img width="904" height="423" alt="Screenshot 2026-08-16 162649" src="https://github.com/user-attachments/assets/eafa6b42-35ca-4f2f-b75b-fad94623637a" />
<img width="329" height="246" alt="Screenshot 2026-08-16 162713" src="https://github.com/user-attachments/assets/5d6ef109-a53e-4707-9fc3-b02ee14f85d3" />
<img width="1000" height="600" alt="multi_yield" src="https://github.com/user-attachments/assets/53834963-137e-4639-95a5-e9af6fb04cc6" />

# Regression-Based Yield Prediction

This Python code analyzes yearly agricultural data by separating **area harvested** and **yield** values, then compares **linear, polynomial, and multivariate regression** models. It evaluates the models using **MSE, RMSE, MAE, and R²**, identifies the best-performing model, predicts yield for **2027–2030**, and visualizes the actual values, regression trends, and future predictions using a graph.
