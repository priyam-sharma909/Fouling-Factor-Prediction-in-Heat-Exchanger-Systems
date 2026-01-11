import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def compute_sse_aic_bic(residuals: np.ndarray, n_params: int):

    residuals = residuals.reshape(-1, 1)  # ensure 2D (N,1)
    n_samples, n_outputs = residuals.shape
    n_eff = n_samples * n_outputs 

    sse = np.sum(residuals ** 2)
    sigma2_hat = sse / n_eff
    sigma2_hat = max(sigma2_hat, 1e-12)  # avoid log(0)


    ll = -0.5 * n_eff * (np.log(2 * np.pi * sigma2_hat) + 1.0)

    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n_eff) - 2 * ll

    return sse, aic, bic


def compute_acf(residuals_1d: np.ndarray, max_lag: int = 40):

    res = residuals_1d - np.mean(residuals_1d)
    denom = np.sum(res ** 2)
    if denom < 1e-12:
        return np.zeros(max_lag + 1)

    acf_vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            num = np.sum(res[lag:] * res[:-lag])
            acf_vals.append(num / denom)
    return np.array(acf_vals)



data_path = r"C:\Users\Hp\OneDrive\Documents\CH-512 Project\predictive models_fouling factor\All collected data.xlsx"
df = pd.read_excel(data_path)

df = df.drop(columns=["Density (Kg/m3)", "Time (hr)", "Reference"])

# Target column
target_col = "Fouling factor (m2 K/kW)"

# Split into X, y
X_full = df.drop(columns=[target_col])
y_full = df[target_col].values.astype(np.float64)  # 1D array

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.3, random_state=42, shuffle=True
)

X_train_vals = X_train.values.astype(np.float64)
feat_names = X_train.columns.tolist()

corrs = []

y = y_train
y_mean = y.mean()
y_d = y - y_mean
y_den = np.sqrt(np.sum(y_d**2)) + 1e-12  

for j in range(X_train_vals.shape[1]):
    x = X_train_vals[:, j]
    x_mean = x.mean()
    x_d = x - x_mean
    x_den = np.sqrt(np.sum(x_d**2)) + 1e-12

    num = np.sum(x_d * y_d)
    corr = num / (x_den * y_den)

    corrs.append(corr)

corr_df = pd.DataFrame({
    "feature": feat_names,
    "corr_y": corrs
})
corr_df["abs_corr_y"] = corr_df["corr_y"].abs()
corr_df = corr_df.sort_values("abs_corr_y", ascending=False).reset_index(drop=True)

print("Top correlations with fouling factor:")
print(corr_df)

k = 5
top_k_features = corr_df["feature"].iloc[:k].tolist()
print(f"\nUsing top {k} features:", top_k_features)

X_train_k = X_train[top_k_features]
X_test_k  = X_test[top_k_features]

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train_k)
X_test_scaled  = scaler_x.transform(X_test_k)

def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:35s}  R² = {r2:7.4f},  RMSE = {rmse:10.6f}")


X_train_lin = X_train_scaled
X_test_lin  = X_test_scaled

X_train_lin = np.c_[np.ones((X_train_lin.shape[0], 1)), X_train_lin]
X_test_lin  = np.c_[np.ones((X_test_lin.shape[0], 1)),  X_test_lin]

# Theta = (X^T X)^(-1) X^T y
Theta_lin = np.linalg.inv(X_train_lin.T @ X_train_lin) @ (X_train_lin.T @ y_train)

y_train_pred_lin = X_train_lin @ Theta_lin
y_test_pred_lin  = X_test_lin  @ Theta_lin

print_metrics("Degree-1 LR TRAIN", y_train, y_train_pred_lin)
print_metrics("Degree-1 LR TEST ", y_test,  y_test_pred_lin)


X_train_poly = np.hstack([X_train_scaled, X_train_scaled**2])
X_test_poly  = np.hstack([X_test_scaled,  X_test_scaled**2])

X_train_poly_design = np.c_[np.ones((X_train_poly.shape[0], 1)), X_train_poly]
X_test_poly_design  = np.c_[np.ones((X_test_poly.shape[0], 1)),  X_test_poly]

Theta_poly = np.linalg.inv(X_train_poly_design.T @ X_train_poly_design) @ (X_train_poly_design.T @ y_train)

y_train_pred_poly = X_train_poly_design @ Theta_poly
y_test_pred_poly  = X_test_poly_design  @ Theta_poly

print_metrics("Degree-2 LR TRAIN", y_train, y_train_pred_poly)
print_metrics("Degree-2 LR TEST ", y_test,  y_test_pred_poly)

plt.figure(figsize=(12,5))
plt.plot(y_test,             label="Actual fouling factor", linewidth=2)
plt.plot(y_test_pred_lin,    label="Predicted (deg1)",      linewidth=2, alpha=0.7)
plt.title(f"Degree-1 LR (top {k} features) — Fouling factor (Test Set)")
plt.xlabel("Test Sample Index")
plt.ylabel("Fouling factor (m²·K/kW)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(y_test,             label="Actual fouling factor", linewidth=2)
plt.plot(y_test_pred_poly,   label="Predicted (deg2)",      linewidth=2, alpha=0.7)
plt.title(f"Degree-2 LR (top {k} features, [X,X^2]) — Fouling factor (Test Set)")
plt.xlabel("Test Sample Index")
plt.ylabel("Fouling factor (m²·K/kW)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

res_train_lin = y_train - y_train_pred_lin
res_test_lin  = y_test  - y_test_pred_lin

n_params_lin = X_train_lin.shape[1]  

sse_train_lin, aic_train_lin, bic_train_lin = compute_sse_aic_bic(res_train_lin, n_params_lin)
sse_test_lin,  aic_test_lin,  bic_test_lin  = compute_sse_aic_bic(res_test_lin,  n_params_lin)

print("\n[Degree-1 LR] Metrics:")
print(f"  n_params: {n_params_lin}")
print(f"  TRAIN SSE: {sse_train_lin:.6f}, AIC: {aic_train_lin:.2f}, BIC: {bic_train_lin:.2f}")
print(f"  TEST  SSE: {sse_test_lin:.6f}, AIC: {aic_test_lin:.2f}, BIC: {bic_test_lin:.2f}")


res_train_poly = y_train - y_train_pred_poly
res_test_poly  = y_test  - y_test_pred_poly

n_params_poly = X_train_poly_design.shape[1]  

sse_train_poly, aic_train_poly, bic_train_poly = compute_sse_aic_bic(res_train_poly, n_params_poly)
sse_test_poly,  aic_test_poly,  bic_test_poly  = compute_sse_aic_bic(res_test_poly,  n_params_poly)

print("\n[Degree-2 LR] Metrics:")
print(f"  n_params: {n_params_poly}")
print(f"  TRAIN SSE: {sse_train_poly:.6f}, AIC: {aic_train_poly:.2f}, BIC: {bic_train_poly:.2f}")
print(f"  TEST  SSE: {sse_test_poly:.6f}, AIC: {aic_test_poly:.2f}, BIC: {bic_test_poly:.2f}")


plt.figure(figsize=(12,4))
plt.plot(res_test_lin, linewidth=1.5)
plt.title("Degree-1 LR — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual (deg1)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(res_test_poly, linewidth=1.5)
plt.title("Degree-2 LR — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual (deg2)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


max_lag = 40
lags = np.arange(max_lag + 1)

acf_lin  = compute_acf(res_test_lin,  max_lag=max_lag)
acf_poly = compute_acf(res_test_poly, max_lag=max_lag)

print("\nResidual ACF (first few lags):")
print("Degree-1:", acf_lin[:5])
print("Degree-2:", acf_poly[:5])

# ACF plots
plt.figure(figsize=(8,4))
plt.stem(lags, acf_lin)
plt.axhline(0, linewidth=1)
plt.title("Degree-1 LR — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.stem(lags, acf_poly)
plt.axhline(0, linewidth=1)
plt.title("Degree-2 LR — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
