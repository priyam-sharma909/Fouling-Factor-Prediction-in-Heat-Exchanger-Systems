import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


def compute_sse_aic_bic(residuals: np.ndarray, n_params: int):
  
    resid = residuals.reshape(-1, 1)
    n_samples, n_outputs = resid.shape
    n_eff = n_samples * n_outputs  

    sse = np.sum(resid ** 2)
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

# Drop unnecessary columns
df = df.drop(columns=["Density (Kg/m3)", "Time (hr)", "Reference"])

# Target column
target_col = "Fouling factor (m2 K/kW)"

# Split into X, Y
X_full = df.drop(columns=[target_col]).values.astype(np.float64)
Y_full = df[target_col].values.astype(np.float64)  # 1D array

X_train, X_test, Y_train, Y_test = train_test_split(
    X_full, Y_full, test_size=0.3, random_state=42, shuffle=True
)

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled  = scaler_x.transform(X_test)

def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:25s}  R² = {r2:7.4f},  RMSE = {rmse:10.6f}")



lambda_deg1 = 5.0  # ridge strength

X_train_deg1 = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_deg1  = np.c_[np.ones((X_test_scaled.shape[0], 1)),  X_test_scaled]

# ridge: Theta = (X^T X + lam I)^(-1) X^T y
I_deg1 = np.eye(X_train_deg1.shape[1])
A_deg1 = X_train_deg1.T @ X_train_deg1 + lambda_deg1 * I_deg1
Theta_deg1 = np.linalg.inv(A_deg1) @ (X_train_deg1.T @ Y_train)

Y_train_pred_deg1 = X_train_deg1 @ Theta_deg1
Y_test_pred_deg1  = X_test_deg1  @ Theta_deg1

print(f"lambda = {lambda_deg1}")
print_metrics("Ridge deg1 TRAIN", Y_train, Y_train_pred_deg1)
print_metrics("Ridge deg1 TEST ", Y_test,  Y_test_pred_deg1)

lambda_deg2 = 12.0  

# Polynomial features: [x, x^2]
X_train_poly = np.hstack([X_train_scaled, X_train_scaled**2])
X_test_poly  = np.hstack([X_test_scaled,  X_test_scaled**2])

X_train_deg2 = np.c_[np.ones((X_train_poly.shape[0], 1)), X_train_poly]
X_test_deg2  = np.c_[np.ones((X_test_poly.shape[0], 1)),  X_test_poly]

I_deg2 = np.eye(X_train_deg2.shape[1])
A_deg2 = X_train_deg2.T @ X_train_deg2 + lambda_deg2 * I_deg2
Theta_deg2 = np.linalg.inv(A_deg2) @ (X_train_deg2.T @ Y_train)

Y_train_pred_deg2 = X_train_deg2 @ Theta_deg2
Y_test_pred_deg2  = X_test_deg2  @ Theta_deg2

print(f"lambda = {lambda_deg2}")
print_metrics("Ridge deg2 TRAIN", Y_train, Y_train_pred_deg2)
print_metrics("Ridge deg2 TEST ", Y_test,  Y_test_pred_deg2)


plt.figure(figsize=(12,5))
plt.plot(Y_test,             label='Actual fouling factor', linewidth=2, color='black')
plt.plot(Y_test_pred_deg1,   label='Ridge predicted (deg1)', linewidth=2, alpha=0.7, color='red')
plt.title(f"Ridge Regression — Fouling Factor (Degree 1, lambda={lambda_deg1})")
plt.xlabel("Test Sample Index")
plt.ylabel("Fouling factor (m²·K/kW)")
plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(Y_test,             label='Actual fouling factor', linewidth=2, color='black')
plt.plot(Y_test_pred_deg2,   label='Ridge predicted (deg2)', linewidth=2, alpha=0.7, color='red')
plt.title(f"Ridge Regression — Fouling Factor (Degree 2, lambda={lambda_deg2})")
plt.xlabel("Test Sample Index")
plt.ylabel("Fouling factor (m²·K/kW)")
plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


res_train_deg1 = Y_train - Y_train_pred_deg1
res_test_deg1  = Y_test  - Y_test_pred_deg1

n_params_deg1 = X_train_deg1.shape[1] 

sse_tr1, aic_tr1, bic_tr1 = compute_sse_aic_bic(res_train_deg1, n_params_deg1)
sse_te1, aic_te1, bic_te1 = compute_sse_aic_bic(res_test_deg1,  n_params_deg1)

print("\n[Ridge Degree-1] Metrics:")
print(f"  n_params: {n_params_deg1}")
print(f"  TRAIN SSE: {sse_tr1:.6f}, AIC: {aic_tr1:.2f}, BIC: {bic_tr1:.2f}")
print(f"  TEST  SSE: {sse_te1:.6f}, AIC: {aic_te1:.2f}, BIC: {bic_te1:.2f}")


res_train_deg2 = Y_train - Y_train_pred_deg2
res_test_deg2  = Y_test  - Y_test_pred_deg2

n_params_deg2 = X_train_deg2.shape[1]  

sse_tr2, aic_tr2, bic_tr2 = compute_sse_aic_bic(res_train_deg2, n_params_deg2)
sse_te2, aic_te2, bic_te2 = compute_sse_aic_bic(res_test_deg2,  n_params_deg2)

print("\n[Ridge Degree-2] Metrics:")
print(f"  n_params: {n_params_deg2}")
print(f"  TRAIN SSE: {sse_tr2:.6f}, AIC: {aic_tr2:.2f}, BIC: {bic_tr2:.2f}")
print(f"  TEST  SSE: {sse_te2:.6f}, AIC: {aic_te2:.2f}, BIC: {bic_te2:.2f}")


plt.figure(figsize=(12,4))
plt.plot(res_test_deg1, linewidth=1.5)
plt.title("Ridge Degree-1 — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual (deg1)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(res_test_deg2, linewidth=1.5)
plt.title("Ridge Degree-2 — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual (deg2)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


max_lag = 40
lags = np.arange(max_lag + 1)

acf_deg1 = compute_acf(res_test_deg1, max_lag=max_lag)
acf_deg2 = compute_acf(res_test_deg2, max_lag=max_lag)

print("\nResidual ACF (first few lags):")
print("Ridge deg1:", acf_deg1[:5])
print("Ridge deg2:", acf_deg2[:5])

plt.figure(figsize=(8,4))
plt.stem(lags, acf_deg1)
plt.axhline(0, linewidth=1)
plt.title("Ridge Degree-1 — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.stem(lags, acf_deg2)
plt.axhline(0, linewidth=1)
plt.title("Ridge Degree-2 — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
