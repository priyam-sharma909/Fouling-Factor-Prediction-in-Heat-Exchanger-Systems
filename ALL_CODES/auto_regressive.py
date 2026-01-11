import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
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


data_path = r"C:\Users\Hp\OneDrive\Documents\CH-512 Project\predictive models_fouling factor\All collected data.xlsx"
df = pd.read_excel(data_path)

print("Columns:", df.columns.tolist())

# sort by time so t-1, t is meaningful 
if "Time (hr)" in df.columns:
    df = df.sort_values(by="Time (hr)").reset_index(drop=True)

# keep only base features + fouling, drop density, time, reference
df_feat = df.drop(columns=["Density (Kg/m3)", "Time (hr)", "Reference"])

target_col = "Fouling factor (m2 K/kW)"
feat_names = [c for c in df_feat.columns if c != target_col]

X_full = df_feat[feat_names].values.astype(np.float64)          # (N, n_features)
Y_full = df_feat[[target_col]].values.astype(np.float64)        # (N, 1)

print("Base features:", feat_names)
print("Target:", target_col)
print("X_full:", X_full.shape, "Y_full:", Y_full.shape)

# build autoregressive dataset
X_ar_list = []
Y_ar_list = []

for t in range(1, len(X_full)):
    x_t = X_full[t]          # current base features
    y_prev = Y_full[t-1, 0]  # previous fouling scalar
    x_ext = np.concatenate([x_t, [y_prev]]) 
    X_ar_list.append(x_ext)
    Y_ar_list.append(Y_full[t, 0])          

X_ar = np.array(X_ar_list)   
Y_ar = np.array(Y_ar_list)   


ext_feat_names = feat_names + ["fouling_prev"]

X_ar_df = pd.DataFrame(X_ar, columns=ext_feat_names)
Y_ar_df = pd.Series(Y_ar, name=target_col)

print("AR X shape:", X_ar_df.shape, "AR Y shape:", Y_ar_df.shape)


X_train, X_test, Y_train, Y_test = train_test_split(
    X_ar_df, Y_ar_df, test_size=0.3, random_state=42, shuffle=True
)

# correlation on base inputs (not including previous fouling)
X_train_base = X_train[feat_names].values
Y_train_vals = Y_train.values

y = Y_train_vals
y_mean = y.mean()
y_d = y - y_mean
y_den = np.sqrt(np.sum(y_d**2)) + 1e-12

corrs_y = []

for j in range(X_train_base.shape[1]):
    x = X_train_base[:, j]
    x_mean = x.mean()
    x_d = x - x_mean
    x_den = np.sqrt(np.sum(x_d**2)) + 1e-12

    num = np.sum(x_d * y_d)
    corr = num / (x_den * y_den)
    corrs_y.append(corr)

corr_df = pd.DataFrame({
    "feature": feat_names,
    "corr_y": corrs_y
})
corr_df["abs_corr"] = corr_df["corr_y"].abs()
corr_df = corr_df.sort_values("abs_corr", ascending=False).reset_index(drop=True)

print("\nTop correlated base features (no lags):")
print(corr_df)


k = 5
top_k_features = corr_df["feature"].iloc[:k].tolist()
print(f"\nUsing top {k} base features:", top_k_features)


final_features = top_k_features + ["fouling_prev"]
print("Final AR feature set:", final_features)

X_train_k = X_train[final_features]
X_test_k  = X_test[final_features]

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_x.fit_transform(X_train_k.values)
X_test_scaled  = scaler_x.transform(X_test_k.values)

Y_train_scaled = scaler_y.fit_transform(Y_train.values.reshape(-1, 1))
Y_test_scaled  = scaler_y.transform(Y_test.values.reshape(-1, 1))

X_train_lin = X_train_scaled
X_test_lin  = X_test_scaled

# design matrices with bias term
X_train_lin_design = np.c_[np.ones((X_train_lin.shape[0], 1)), X_train_lin]
X_test_lin_design  = np.c_[np.ones((X_test_lin.shape[0], 1)),  X_test_lin]


Theta_lin = np.linalg.inv(X_train_lin_design.T @ X_train_lin_design) @ (X_train_lin_design.T @ Y_train_scaled)

Y_train_pred_scaled_lin = X_train_lin_design @ Theta_lin
Y_test_pred_scaled_lin  = X_test_lin_design  @ Theta_lin

Y_train_pred_lin = scaler_y.inverse_transform(Y_train_pred_scaled_lin)
Y_test_pred_lin  = scaler_y.inverse_transform(Y_test_pred_scaled_lin)

r2_train_lin = r2_score(Y_train.values, Y_train_pred_lin[:, 0])
r2_test_lin  = r2_score(Y_test.values,  Y_test_pred_lin[:, 0])

print(f"\nAR Degree-1 LR (top {k} + prev fouling)  R² TRAIN: {r2_train_lin:.4f}")
print(f"AR Degree-1 LR (top {k} + prev fouling)  R² TEST : {r2_test_lin:.4f}")


X_train_poly = np.hstack([X_train_scaled, X_train_scaled**2])
X_test_poly  = np.hstack([X_test_scaled,  X_test_scaled**2])

X_train_poly_design = np.c_[np.ones((X_train_poly.shape[0], 1)), X_train_poly]
X_test_poly_design  = np.c_[np.ones((X_test_poly.shape[0], 1)),  X_test_poly]

Theta_poly = np.linalg.inv(X_train_poly_design.T @ X_train_poly_design) @ (X_train_poly_design.T @ Y_train_scaled)

Y_train_pred_scaled_poly = X_train_poly_design @ Theta_poly
Y_test_pred_scaled_poly  = X_test_poly_design  @ Theta_poly

Y_train_pred_poly = scaler_y.inverse_transform(Y_train_pred_scaled_poly)
Y_test_pred_poly  = scaler_y.inverse_transform(Y_test_pred_scaled_poly)

r2_train_poly = r2_score(Y_train.values, Y_train_pred_poly[:, 0])
r2_test_poly  = r2_score(Y_test.values,  Y_test_pred_poly[:, 0])

print(f"\nAR Degree-2 LR (top {k} + prev fouling)  R² TRAIN: {r2_train_poly:.4f}")
print(f"AR Degree-2 LR (top {k} + prev fouling)    R² TEST : {r2_test_poly:.4f}")


plt.figure(figsize=(12,5))
plt.plot(Y_test.values,          label="Actual fouling factor", linewidth=2, color="black")
plt.plot(Y_test_pred_lin[:,0],   label="AR Pred fouling (deg1)", linewidth=2, alpha=0.7, color="red")
plt.title(f"AR Degree-1 LR — Fouling factor (top {k} + prev fouling)")
plt.xlabel("Test Sample Index")
plt.ylabel("Fouling factor (m²·K/kW)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(Y_test.values,           label="Actual fouling factor", linewidth=2, color="black")
plt.plot(Y_test_pred_poly[:,0],   label="AR Pred fouling (deg2)", linewidth=2, alpha=0.7, color="red")
plt.title(f"AR Degree-2 LR — Fouling factor (top {k} + prev fouling, [X,X^2])")
plt.xlabel("Test Sample Index")
plt.ylabel("Fouling factor (m²·K/kW)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

res_train_lin = Y_train.values - Y_train_pred_lin[:, 0]
res_test_lin  = Y_test.values  - Y_test_pred_lin[:, 0]

n_params_lin = X_train_lin_design.shape[1]  

sse_train_lin, aic_train_lin, bic_train_lin = compute_sse_aic_bic(res_train_lin, n_params_lin)
sse_test_lin,  aic_test_lin,  bic_test_lin  = compute_sse_aic_bic(res_test_lin,  n_params_lin)

print("\n[Degree-1 AR] Metrics:")
print(f"  n_params: {n_params_lin}")
print(f"  TRAIN SSE: {sse_train_lin:.6f}, AIC: {aic_train_lin:.2f}, BIC: {bic_train_lin:.2f}")
print(f"  TEST  SSE: {sse_test_lin:.6f}, AIC: {aic_test_lin:.2f}, BIC: {bic_test_lin:.2f}")

res_train_poly = Y_train.values - Y_train_pred_poly[:, 0]
res_test_poly  = Y_test.values  - Y_test_pred_poly[:, 0]

n_params_poly = X_train_poly_design.shape[1]  

sse_train_poly, aic_train_poly, bic_train_poly = compute_sse_aic_bic(res_train_poly, n_params_poly)
sse_test_poly,  aic_test_poly,  bic_test_poly  = compute_sse_aic_bic(res_test_poly,  n_params_poly)

print("\n[Degree-2 AR] Metrics:")
print(f"  n_params: {n_params_poly}")
print(f"  TRAIN SSE: {sse_train_poly:.6f}, AIC: {aic_train_poly:.2f}, BIC: {bic_train_poly:.2f}")
print(f"  TEST  SSE: {sse_test_poly:.6f}, AIC: {aic_test_poly:.2f}, BIC: {bic_test_poly:.2f}")


plt.figure(figsize=(12,4))
plt.plot(res_test_lin, linewidth=1.5)
plt.title("AR Degree-1 LR — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual (deg1)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(res_test_poly, linewidth=1.5)
plt.title("AR Degree-2 LR — Residuals vs Sample (TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual (deg2)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

max_lag = 40
lags = np.arange(max_lag + 1)

acf_lin = compute_acf(res_test_lin, max_lag=max_lag)
acf_poly = compute_acf(res_test_poly, max_lag=max_lag)

print("\nResidual ACF (first few lags):")
print("Degree-1:", acf_lin[:5])
print("Degree-2:", acf_poly[:5])

# ACF plots
plt.figure(figsize=(8,4))
plt.stem(lags, acf_lin)
plt.axhline(0, linewidth=1)
plt.title("AR Degree-1 LR — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.stem(lags, acf_poly)
plt.axhline(0, linewidth=1)
plt.title("AR Degree-2 LR — Residual ACF (TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
