```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ============================================================
# 1. LOAD CSV DATASET
# ============================================================

df = pd.read_csv("FAOSTAT_data_en_8-16-2026.csv")

# Keep only Year and Value columns
df = df[["Year", "Value"]]

# Remove missing values
df = df.dropna()

# Sort data by Year
df = df.sort_values("Year").reset_index(drop=True)


# ============================================================
# 2. RESHAPE DATASET
# First Value of each year = Area Harvested
# Second Value of each year = Yield
# ============================================================

df["Type"] = df.groupby("Year").cumcount()

area_df = df[df["Type"] == 0][["Year", "Value"]].rename(
    columns={"Value": "Area Harvested"}
)

yield_df = df[df["Type"] == 1][["Year", "Value"]].rename(
    columns={"Value": "Yield"}
)

df = pd.merge(
    area_df,
    yield_df,
    on="Year"
)

# Sort again after merge
df = df.sort_values("Year").reset_index(drop=True)


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df["Year"].values
y = df["Yield"].values
area = df["Area Harvested"].values


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test, area_train, area_test = train_test_split(
    X,
    y,
    area,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 5. Z-SCORE NORMALIZATION
#
# Formula:
#
# Z = (X - Mean) / Standard Deviation
#
# IMPORTANT:
# Mean and standard deviation are calculated ONLY
# from training data to avoid data leakage.
# ============================================================

# -------------------------------
# Year
# -------------------------------

year_mean = np.mean(X_train)
year_std = np.std(X_train)

X_train_norm = (X_train - year_mean) / year_std
X_test_norm = (X_test - year_mean) / year_std


# -------------------------------
# Yield
# -------------------------------

yield_mean = np.mean(y_train)
yield_std = np.std(y_train)

y_train_norm = (y_train - yield_mean) / yield_std
y_test_norm = (y_test - yield_mean) / yield_std


# -------------------------------
# Area Harvested
# -------------------------------

area_mean = np.mean(area_train)
area_std = np.std(area_train)

area_train_norm = (
    (area_train - area_mean) / area_std
)

area_test_norm = (
    (area_test - area_mean) / area_std
)

# ============================================================
# 6. PRINT NORMALIZED VALUES
# ============================================================

normalized_df = pd.DataFrame({
    "Year": df["Year"],
    "Area Harvested": df["Area Harvested"],
    "Yield": df["Yield"],
})

# Normalize complete dataset ONLY for displaying values.
# This is not used for model training.
normalized_df["Year_Normalized"] = (
    (normalized_df["Year"] - year_mean) / year_std
)

normalized_df["Area_Normalized"] = (
    (normalized_df["Area Harvested"] - area_mean) / area_std
)

normalized_df["Yield_Normalized"] = (
    (normalized_df["Yield"] - yield_mean) / yield_std
)


print("\n")
print("=" * 80)
print("Z-SCORE NORMALIZED DATA")
print("=" * 80)

# Print ONLY normalized columns
print(
    normalized_df[
        [
            "Year_Normalized",
            "Area_Normalized",
            "Yield_Normalized"
        ]
    ].to_string(index=False)
)
# ============================================================
# 7. PRINT NORMALIZATION PARAMETERS
# ============================================================

print("\n")
print("=" * 80)
print("Z-SCORE NORMALIZATION PARAMETERS")
print("=" * 80)

print(f"Year Mean              : {year_mean:.4f}")
print(f"Year Standard Deviation: {year_std:.4f}")

print(f"\nArea Mean              : {area_mean:.4f}")
print(f"Area Standard Deviation: {area_std:.4f}")

print(f"\nYield Mean             : {yield_mean:.4f}")
print(f"Yield Standard Deviation: {yield_std:.4f}")


# ============================================================
# 8. 1. LINEAR REGRESSION
# Normalized Year -> Normalized Yield
# ============================================================

# Calculate mean of normalized training data
x_mean_norm = np.mean(X_train_norm)
y_mean_norm = np.mean(y_train_norm)

# Weight / Slope
weight = np.sum(
    (X_train_norm - x_mean_norm) *
    (y_train_norm - y_mean_norm)
) / np.sum(
    (X_train_norm - x_mean_norm) ** 2
)

# Bias / Intercept
bias = y_mean_norm - (weight * x_mean_norm)


# Predict normalized test data
y_pred_linear_norm = (
    weight * X_test_norm + bias
)


# Convert predictions back to original Yield
y_pred_linear = (
    y_pred_linear_norm * yield_std
    + yield_mean
)


# ============================================================
# 9. LINEAR REGRESSION METRICS
# ============================================================

mse_linear = mean_squared_error(
    y_test,
    y_pred_linear
)

rmse_linear = np.sqrt(mse_linear)

mae_linear = mean_absolute_error(
    y_test,
    y_pred_linear
)

r2_linear = r2_score(
    y_test,
    y_pred_linear
)


# ============================================================
# 10. 2. POLYNOMIAL REGRESSION
# Normalized Year -> Normalized Yield
# ============================================================

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(
    X_train_norm.reshape(-1, 1)
)

X_test_poly = poly.transform(
    X_test_norm.reshape(-1, 1)
)


poly_model = LinearRegression()

poly_model.fit(
    X_train_poly,
    y_train_norm
)


# Predict normalized Yield
y_pred_poly_norm = poly_model.predict(
    X_test_poly
)


# Convert predictions back to original Yield
y_pred_poly = (
    y_pred_poly_norm * yield_std
    + yield_mean
)


# ============================================================
# 11. POLYNOMIAL REGRESSION METRICS
# ============================================================

mse_poly = mean_squared_error(
    y_test,
    y_pred_poly
)

rmse_poly = np.sqrt(mse_poly)

mae_poly = mean_absolute_error(
    y_test,
    y_pred_poly
)

r2_poly = r2_score(
    y_test,
    y_pred_poly
)


# ============================================================
# 12. 3. MULTIVARIATE LINEAR REGRESSION
#
# Normalized Year +
# Normalized Area Harvested
# ->
# Normalized Yield
# ============================================================

X_train_multi = np.column_stack(
    (
        X_train_norm,
        area_train_norm
    )
)

X_test_multi = np.column_stack(
    (
        X_test_norm,
        area_test_norm
    )
)


multi_model = LinearRegression()

multi_model.fit(
    X_train_multi,
    y_train_norm
)


# Predict normalized Yield
y_pred_multi_norm = multi_model.predict(
    X_test_multi
)


# Convert predictions back to original Yield
y_pred_multi = (
    y_pred_multi_norm * yield_std
    + yield_mean
)


# ============================================================
# 13. MULTIVARIATE REGRESSION METRICS
# ============================================================

mse_multi = mean_squared_error(
    y_test,
    y_pred_multi
)

rmse_multi = np.sqrt(mse_multi)

mae_multi = mean_absolute_error(
    y_test,
    y_pred_multi
)

r2_multi = r2_score(
    y_test,
    y_pred_multi
)


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

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


print("\n")
print("=" * 80)
print("MODEL COMPARISON (20% TEST DATA)")
print("=" * 80)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 15. DETERMINE BEST MODEL
# Lower RMSE = Better
# ============================================================

best_model = comparison.loc[
    comparison["RMSE"].idxmin(),
    "Model"
]

print("\nBest Fit Model:", best_model)


# ============================================================
# 16. FUTURE YEARS
# 2027 - 2030
# ============================================================

future_years = np.array([
    2027,
    2028,
    2029,
    2030
])


# ============================================================
# 17. NORMALIZE FUTURE YEARS
# ============================================================

future_years_norm = (
    future_years - year_mean
) / year_std


# ============================================================
# 18. LINEAR REGRESSION FUTURE PREDICTIONS
# ============================================================

linear_future_predictions_norm = (
    weight * future_years_norm
    + bias
)


# Convert normalized predictions to original Yield
linear_future_predictions = (
    linear_future_predictions_norm
    * yield_std
    + yield_mean
)


# ============================================================
# 19. POLYNOMIAL REGRESSION FUTURE PREDICTIONS
# ============================================================

future_years_poly = poly.transform(
    future_years_norm.reshape(-1, 1)
)

poly_future_predictions_norm = (
    poly_model.predict(
        future_years_poly
    )
)


# Convert to original Yield
poly_future_predictions = (
    poly_future_predictions_norm
    * yield_std
    + yield_mean
)


# ============================================================
# 20. ESTIMATE FUTURE AREA HARVESTED
# ============================================================

area_model = LinearRegression()

area_model.fit(
    df[["Year"]],
    df["Area Harvested"]
)


future_area = area_model.predict(
    future_years.reshape(-1, 1)
)


# Normalize future Area using training statistics
future_area_norm = (
    future_area - area_mean
) / area_std


# ============================================================
# 21. MULTIVARIATE FUTURE PREDICTIONS
# ============================================================

future_multi_X = np.column_stack(
    (
        future_years_norm,
        future_area_norm
    )
)


multi_future_predictions_norm = (
    multi_model.predict(
        future_multi_X
    )
)


# Convert normalized prediction
# back to original Yield
multi_future_predictions = (
    multi_future_predictions_norm
    * yield_std
    + yield_mean
)



# ============================================================
# 22. PRINT FUTURE YIELD PREDICTIONS
# ============================================================

print("\n")
print("=" * 80)
print("PREDICTED YIELD (2027-2030)")
print("=" * 80)


print("\nLinear Regression:")

for year, prediction in zip(
    future_years,
    linear_future_predictions
):

    print(
        f"{year}: {prediction:.4f}"
    )


print("\nPolynomial Regression:")

for year, prediction in zip(
    future_years,
    poly_future_predictions
):

    print(
        f"{year}: {prediction:.4f}"
    )


print("\nMultivariate Regression:")

for year, prediction in zip(
    future_years,
    multi_future_predictions
):

    print(
        f"{year}: {prediction:.4f}"
    )


# ============================================================
# 23. FUTURE PREDICTION TABLE
# ============================================================

future_comparison = pd.DataFrame({

    "Year": future_years,

    "Estimated Area Harvested": future_area,

    "Linear Yield": linear_future_predictions,

    "Polynomial Yield": poly_future_predictions,

    "Multivariate Yield": multi_future_predictions

})


print("\n")
print("=" * 80)
print("FUTURE YIELD PREDICTION COMPARISON")
print("=" * 80)

print(
    future_comparison.to_string(
        index=False
    )
)


# ============================================================
# 24. PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)


# ------------------------------------------------------------
# Actual Data
# ------------------------------------------------------------

plt.scatter(
    X,
    y,
    color="blue",
    s=60,
    label="Actual Yield"
)


# ------------------------------------------------------------
# Years for Regression Lines
# ------------------------------------------------------------

line_years = np.arange(
    min(X),
    2031
)


# Normalize line years
line_years_norm = (
    line_years - year_mean
) / year_std


# ------------------------------------------------------------
# Linear Regression Line
# ------------------------------------------------------------

line_predictions_linear_norm = (
    weight * line_years_norm
    + bias
)


# Convert back to original Yield
line_predictions_linear = (
    line_predictions_linear_norm
    * yield_std
    + yield_mean
)


plt.plot(
    line_years,
    line_predictions_linear,
    color="red",
    linewidth=2,
    label="Linear Regression"
)


# ------------------------------------------------------------
# Polynomial Regression Curve
# ------------------------------------------------------------

line_years_poly = poly.transform(
    line_years_norm.reshape(-1, 1)
)


line_predictions_poly_norm = (
    poly_model.predict(
        line_years_poly
    )
)


# Convert to original Yield
line_predictions_poly = (
    line_predictions_poly_norm
    * yield_std
    + yield_mean
)


plt.plot(
    line_years,
    line_predictions_poly,
    color="orange",
    linewidth=2,
    label="Polynomial Regression"
)


# ------------------------------------------------------------
# Future Linear Predictions
# ------------------------------------------------------------

plt.scatter(
    future_years,
    linear_future_predictions,
    color="red",
    marker="*",
    s=180,
    label="Linear Future Prediction"
)


# ------------------------------------------------------------
# Future Polynomial Predictions
# ------------------------------------------------------------

plt.scatter(
    future_years,
    poly_future_predictions,
    color="orange",
    marker="*",
    s=180,
    label="Polynomial Future Prediction"
)
# ------------------------------------------------------------
# Future Multivariate Predictions
# ------------------------------------------------------------
plt.scatter(
    future_years,
    multi_future_predictions,
    color="green",
    marker="*",
    s=180,
    label="Multivariate Future Prediction"
)

# ------------------------------------------------------------
# Plot Formatting
# ------------------------------------------------------------

plt.title(
    "Wheat Yield Prediction Using Three Regression Models"
)
plt.xlabel("Year")
plt.ylabel("Yield")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<img width="608" height="557" alt="image" src="https://github.com/user-attachments/assets/74892bbb-edff-4088-978f-04bbf816e6c1" />
<img width="410" height="478" alt="image" src="https://github.com/user-attachments/assets/1ad63cbd-878d-4431-84ba-1f4fb43e9f2b" />
<img width="675" height="436" alt="image" src="https://github.com/user-attachments/assets/7f702a51-b533-4dc2-bcec-65c453a47b86" />
<img width="970" height="370" alt="image" src="https://github.com/user-attachments/assets/031055a3-8aef-4e43-9d7f-a416c0f26fc7" />
<img width="681" height="521" alt="image" src="https://github.com/user-attachments/assets/de18d2db-7fd2-4baa-8b81-c97dfee259c9" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/fd2302d7-b6c1-42af-a02b-9e5e3cab5b63" />

