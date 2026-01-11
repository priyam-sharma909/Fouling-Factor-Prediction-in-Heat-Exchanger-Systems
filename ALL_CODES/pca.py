import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


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

df = df.drop(columns=["Density (Kg/m3)", "Time (hr)", "Reference"])

target_col = "Fouling factor (m2 K/kW)"

X_full = df.drop(columns=[target_col]).values.astype(np.float64)
Y_full = df[target_col].values.astype(np.float64).reshape(-1, 1)

# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_full, Y_full, test_size=0.3, random_state=42, shuffle=True
)

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled  = scaler_x.transform(X_test)


cov = (1.0 / X_train_scaled.shape[0]) * (X_train_scaled.T @ X_train_scaled)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Scree Plot
plt.figure(figsize=(6,4))
plt.plot(range(1,len(eigvals)+1), eigvals, "o-")
plt.title("Scree Plot — PCA on Fouling Inputs")
plt.xlabel("PC Index"); plt.ylabel("Eigenvalue")
plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# Number of principal components 
k = min(10, X_train_scaled.shape[1])
print("Using k =", k)

# Projection
P = eigvecs[:, :k]
X_train_pca = X_train_scaled @ P
X_test_pca  = X_test_scaled  @ P

Xtr1 = np.c_[np.ones((X_train_pca.shape[0],1)), X_train_pca]
Xte1 = np.c_[np.ones((X_test_pca.shape[0],1)),  X_test_pca]

Theta1 = np.linalg.inv(Xtr1.T @ Xtr1) @ (Xtr1.T @ Y_train)

Y_train_pred1 = Xtr1 @ Theta1
Y_test_pred1  = Xte1 @ Theta1

r2_tr1 = r2_score(Y_train, Y_train_pred1)
r2_te1 = r2_score(Y_test,  Y_test_pred1)

print(f"\nDegree-1 PCA+LR  R² TRAIN: {r2_tr1:.4f}")
print(f"Degree-1 PCA+LR  R² TEST : {r2_te1:.4f}")

X_train_poly = np.hstack([X_train_pca, X_train_pca**2])
X_test_poly  = np.hstack([X_test_pca,  X_test_pca**2])

Xtr2 = np.c_[np.ones((X_train_poly.shape[0],1)), X_train_poly]
Xte2 = np.c_[np.ones((X_test_poly.shape[0],1)),  X_test_poly]

Theta2 = np.linalg.inv(Xtr2.T @ Xtr2) @ (Xtr2.T @ Y_train)

Y_train_pred2 = Xtr2 @ Theta2
Y_test_pred2  = Xte2 @ Theta2

r2_tr2 = r2_score(Y_train, Y_train_pred2)
r2_te2 = r2_score(Y_test,  Y_test_pred2)

print(f"\nDegree-2 PCA+LR  R² TRAIN: {r2_tr2:.4f}")
print(f"Degree-2 PCA+LR  R² TEST : {r2_te2:.4f}")


plt.figure(figsize=(12,5))
plt.plot(Y_test, label="Actual fouling", linewidth=2, color="black")
plt.plot(Y_test_pred1, label="Pred (PCA+LR deg1)", linewidth=2, alpha=0.7, color="red")
plt.title("Degree-1 PCA + Linear Regression — Fouling Factor")
plt.xlabel("Test Sample Index"); plt.ylabel("Fouling factor")
plt.grid(alpha=0.4); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(12,5))
plt.plot(Y_test, label="Actual fouling", linewidth=2, color="black")
plt.plot(Y_test_pred2, label="Pred (PCA+LR deg2)", linewidth=2, alpha=0.7, color="red")
plt.title("Degree-2 PCA + Linear Regression — Fouling Factor")
plt.xlabel("Test Sample Index"); plt.ylabel("Fouling factor")
plt.grid(alpha=0.4); plt.legend(); plt.tight_layout(); plt.show()


res_train_1 = (Y_train[:, 0] - Y_train_pred1[:, 0])
res_test_1  = (Y_test[:, 0]  - Y_test_pred1[:, 0])

n_params_1 = Xtr1.shape[1]  

sse_tr1, aic_tr1, bic_tr1 = compute_sse_aic_bic(res_train_1, n_params_1)
sse_te1, aic_te1, bic_te1 = compute_sse_aic_bic(res_test_1,  n_params_1)

print("\n[Degree-1 PCA+LR] Metrics:")
print(f"  n_params: {n_params_1}")
print(f"  TRAIN SSE: {sse_tr1:.6f}, AIC: {aic_tr1:.2f}, BIC: {bic_tr1:.2f}")
print(f"  TEST  SSE: {sse_te1:.6f}, AIC: {aic_te1:.2f}, BIC: {bic_te1:.2f}")

res_train_2 = (Y_train[:, 0] - Y_train_pred2[:, 0])
res_test_2  = (Y_test[:, 0]  - Y_test_pred2[:, 0])

n_params_2 = Xtr2.shape[1]  

sse_tr2, aic_tr2, bic_tr2 = compute_sse_aic_bic(res_train_2, n_params_2)
sse_te2, aic_te2, bic_te2 = compute_sse_aic_bic(res_test_2,  n_params_2)

print("\n[Degree-2 PCA+LR] Metrics:")
print(f"  n_params: {n_params_2}")
print(f"  TRAIN SSE: {sse_tr2:.6f}, AIC: {aic_tr2:.2f}, BIC: {bic_tr2:.2f}")
print(f"  TEST  SSE: {sse_te2:.6f}, AIC: {aic_te2:.2f}, BIC: {bic_te2:.2f}")


plt.figure(figsize=(12,4))
plt.plot(res_test_1, linewidth=1.5)
plt.title("Degree-1 PCA+LR — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual (deg1)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(res_test_2, linewidth=1.5)
plt.title("Degree-2 PCA+LR — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual (deg2)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


max_lag = 40
lags = np.arange(max_lag + 1)

acf_1 = compute_acf(res_test_1, max_lag=max_lag)
acf_2 = compute_acf(res_test_2, max_lag=max_lag)

print("\nResidual ACF (first few lags):")
print("Degree-1 PCA+LR:", acf_1[:5])
print("Degree-2 PCA+LR:", acf_2[:5])

plt.figure(figsize=(8,4))
plt.stem(lags, acf_1)
plt.axhline(0, linewidth=1)
plt.title("Degree-1 PCA+LR — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.stem(lags, acf_2)
plt.axhline(0, linewidth=1)
plt.title("Degree-2 PCA+LR — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
