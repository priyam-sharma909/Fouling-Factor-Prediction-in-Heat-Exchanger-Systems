import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


def compute_sse_aic_bic(residuals: np.ndarray, n_params: int):
    
    residuals = residuals.reshape(-1, 1)  # ensure 2D (N,1)
    n_samples, n_outputs = residuals.shape
    n_eff = n_samples * n_outputs  # here = N

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


path = r"C:\Users\Hp\OneDrive\Documents\CH-512 Project\predictive models_fouling factor\All collected data.xlsx"
df = pd.read_excel(path)

# Drop unnecessary columns
df = df.drop(columns=["Density (Kg/m3)", "Time (hr)", "Reference"])

# Target column
target_col = "Fouling factor (m2 K/kW)"

# Split into X, Y
X_full = df.drop(columns=[target_col])
Y_full = df[target_col].values.astype(np.float64)  # 1D array

X_train, X_test, Y_train, Y_test = train_test_split(
    X_full.values, Y_full, test_size=0.3, random_state=42, shuffle=True
)

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled  = scaler_x.transform(X_test)


def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:25s}  R² = {r2:7.4f},  RMSE = {rmse:10.6f}")


X_train_lin = X_train_scaled
X_test_lin  = X_test_scaled

X_train_lin = np.c_[np.ones((X_train_lin.shape[0], 1)), X_train_lin]
X_test_lin  = np.c_[np.ones((X_test_lin.shape[0], 1)),  X_test_lin]

# Theta = (X^T X)^(-1) X^T Y
Theta_lin = np.linalg.inv(X_train_lin.T @ X_train_lin) @ (X_train_lin.T @ Y_train)

Y_train_pred_lin = X_train_lin @ Theta_lin
Y_test_pred_lin  = X_test_lin  @ Theta_lin

print_metrics("Degree-1 LR (1,x) TRAIN", Y_train, Y_train_pred_lin)
print_metrics("Degree-1 LR (1,x) TEST ", Y_test,  Y_test_pred_lin)

X_train_poly = np.hstack([X_train_scaled, X_train_scaled**2])
X_test_poly  = np.hstack([X_test_scaled,  X_test_scaled**2])

X_train_poly = np.c_[np.ones((X_train_poly.shape[0], 1)), X_train_poly]
X_test_poly  = np.c_[np.ones((X_test_poly.shape[0], 1)),  X_test_poly]

Theta_poly = np.linalg.inv(X_train_poly.T @ X_train_poly) @ (X_train_poly.T @ Y_train)

Y_train_pred_poly = X_train_poly @ Theta_poly
Y_test_pred_poly  = X_test_poly  @ Theta_poly

print_metrics("Degree-2 LR (1,x,x^2) TRAIN", Y_train, Y_train_pred_poly)
print_metrics("Degree-2 LR (1,x,x^2) TEST ", Y_test,  Y_test_pred_poly)


alpha_deg1 = 0.001 

lasso_deg1 = Lasso(alpha=alpha_deg1, max_iter=10000)
lasso_deg1.fit(X_train_scaled, Y_train)

Y_train_pred_lasso1 = lasso_deg1.predict(X_train_scaled)
Y_test_pred_lasso1  = lasso_deg1.predict(X_test_scaled)

print(f"alpha = {alpha_deg1}")
print_metrics("LASSO deg1 TRAIN", Y_train, Y_train_pred_lasso1)
print_metrics("LASSO deg1 TEST ", Y_test,  Y_test_pred_lasso1)
print(f"Non-zero coeffs (deg1): {np.sum(lasso_deg1.coef_ != 0)} / {lasso_deg1.coef_.size}")


alpha_deg2 = 0.005

lasso_deg2 = Lasso(alpha=alpha_deg2, max_iter=20000)
lasso_deg2.fit(X_train_poly, Y_train)

Y_train_pred_lasso2 = lasso_deg2.predict(X_train_poly)
Y_test_pred_lasso2  = lasso_deg2.predict(X_test_poly)

print(f"alpha = {alpha_deg2}")
print_metrics("LASSO deg2 TRAIN", Y_train, Y_train_pred_lasso2)
print_metrics("LASSO deg2 TEST ", Y_test,  Y_test_pred_lasso2)
print(f"Non-zero coeffs (deg2): {np.sum(lasso_deg2.coef_ != 0)} / {lasso_deg2.coef_.size}")


plt.figure(figsize=(12,5))
plt.plot(Y_test,              label='Actual fouling factor', linewidth=2, color='black')
plt.plot(Y_test_pred_lin,     label='LR (deg1)',             linewidth=2, alpha=0.7, color='red')
plt.plot(Y_test_pred_lasso1,  label='LASSO (deg1)',          linewidth=2, alpha=0.7, color='blue')
plt.title("Degree-1 — Fouling Factor (Test Set)")
plt.xlabel("Test Sample Index")
plt.ylabel("Fouling factor (m²·K/kW)")
plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(Y_test,              label='Actual fouling factor', linewidth=2, color='black')
plt.plot(Y_test_pred_poly,    label='LR (deg2)',             linewidth=2, alpha=0.7, color='red')
plt.plot(Y_test_pred_lasso2,  label='LASSO (deg2)',          linewidth=2, alpha=0.7, color='blue')
plt.title("Degree-2 — Fouling Factor (Test Set)")
plt.xlabel("Test Sample Index")
plt.ylabel("Fouling factor (m²·K/kW)")
plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


res_train_lin = Y_train - Y_train_pred_lin
res_test_lin  = Y_test  - Y_test_pred_lin

n_params_lin = X_train_lin.shape[1]  # bias + all features

sse_train_lin, aic_train_lin, bic_train_lin = compute_sse_aic_bic(res_train_lin, n_params_lin)
sse_test_lin,  aic_test_lin,  bic_test_lin  = compute_sse_aic_bic(res_test_lin,  n_params_lin)

print("\n[LR Degree-1] Metrics:")
print(f"  n_params: {n_params_lin}")
print(f"  TRAIN SSE: {sse_train_lin:.6f}, AIC: {aic_train_lin:.2f}, BIC: {bic_train_lin:.2f}")
print(f"  TEST  SSE: {sse_test_lin:.6f}, AIC: {aic_test_lin:.2f}, BIC: {bic_test_lin:.2f}")

res_train_poly = Y_train - Y_train_pred_poly
res_test_poly  = Y_test  - Y_test_pred_poly

n_params_poly = X_train_poly.shape[1]  # bias + linear + squared terms

sse_train_poly, aic_train_poly, bic_train_poly = compute_sse_aic_bic(res_train_poly, n_params_poly)
sse_test_poly,  aic_test_poly,  bic_test_poly  = compute_sse_aic_bic(res_test_poly,  n_params_poly)

print("\n[LR Degree-2] Metrics:")
print(f"  n_params: {n_params_poly}")
print(f"  TRAIN SSE: {sse_train_poly:.6f}, AIC: {aic_train_poly:.2f}, BIC: {bic_train_poly:.2f}")
print(f"  TEST  SSE: {sse_test_poly:.6f}, AIC: {aic_test_poly:.2f}, BIC: {bic_test_poly:.2f}")


res_train_lasso1 = Y_train - Y_train_pred_lasso1
res_test_lasso1  = Y_test  - Y_test_pred_lasso1

# effective param count = non-zero coefficients + intercept
n_params_lasso1 = np.count_nonzero(lasso_deg1.coef_) + 1

sse_train_lasso1, aic_train_lasso1, bic_train_lasso1 = compute_sse_aic_bic(res_train_lasso1, n_params_lasso1)
sse_test_lasso1,  aic_test_lasso1,  bic_test_lasso1  = compute_sse_aic_bic(res_test_lasso1,  n_params_lasso1)

print("\n[LASSO Degree-1] Metrics:")
print(f"  effective n_params: {n_params_lasso1}")
print(f"  TRAIN SSE: {sse_train_lasso1:.6f}, AIC: {aic_train_lasso1:.2f}, BIC: {bic_train_lasso1:.2f}")
print(f"  TEST  SSE: {sse_test_lasso1:.6f}, AIC: {aic_test_lasso1:.2f}, BIC: {bic_test_lasso1:.2f}")


res_train_lasso2 = Y_train - Y_train_pred_lasso2
res_test_lasso2  = Y_test  - Y_test_pred_lasso2

n_params_lasso2 = np.count_nonzero(lasso_deg2.coef_) + 1

sse_train_lasso2, aic_train_lasso2, bic_train_lasso2 = compute_sse_aic_bic(res_train_lasso2, n_params_lasso2)
sse_test_lasso2,  aic_test_lasso2,  bic_test_lasso2  = compute_sse_aic_bic(res_test_lasso2,  n_params_lasso2)

print("\n[LASSO Degree-2] Metrics:")
print(f"  effective n_params: {n_params_lasso2}")
print(f"  TRAIN SSE: {sse_train_lasso2:.6f}, AIC: {aic_train_lasso2:.2f}, BIC: {bic_train_lasso2:.2f}")
print(f"  TEST  SSE: {sse_test_lasso2:.6f}, AIC: {aic_test_lasso2:.2f}, BIC: {bic_test_lasso2:.2f}")


plt.figure(figsize=(12,4))
plt.plot(res_test_lin, linewidth=1.5)
plt.title("LR Degree-1 — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(res_test_poly, linewidth=1.5)
plt.title("LR Degree-2 — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(res_test_lasso1, linewidth=1.5)
plt.title("LASSO Degree-1 — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(res_test_lasso2, linewidth=1.5)
plt.title("LASSO Degree-2 — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


max_lag = 40
lags = np.arange(max_lag + 1)

acf_lin      = compute_acf(res_test_lin,      max_lag=max_lag)
acf_poly     = compute_acf(res_test_poly,     max_lag=max_lag)
acf_lasso1   = compute_acf(res_test_lasso1,   max_lag=max_lag)
acf_lasso2   = compute_acf(res_test_lasso2,   max_lag=max_lag)

print("\nResidual ACF (first few lags):")
print("LR Deg1   :", acf_lin[:5])
print("LR Deg2   :", acf_poly[:5])
print("LASSO Deg1:", acf_lasso1[:5])
print("LASSO Deg2:", acf_lasso2[:5])

# ACF plots
plt.figure(figsize=(8,4))
plt.stem(lags, acf_lin)
plt.axhline(0, linewidth=1)
plt.title("LR Degree-1 — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.stem(lags, acf_poly)
plt.axhline(0, linewidth=1)
plt.title("LR Degree-2 — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.stem(lags, acf_lasso1)
plt.axhline(0, linewidth=1)
plt.title("LASSO Degree-1 — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.stem(lags, acf_lasso2)
plt.axhline(0, linewidth=1)
plt.title("LASSO Degree-2 — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
