import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def compute_sse_aic_bic(err, n_params):
    r = err.reshape(-1, 1)
    n_samples, n_outputs = r.shape
    n_eff = n_samples * n_outputs
    sse = np.sum(r ** 2)
    sigma2 = sse / n_eff
    sigma2 = max(sigma2, 1e-12)
    ll = -0.5 * n_eff * (np.log(2 * np.pi * sigma2) + 1.0)
    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n_eff) - 2 * ll
    return sse, aic, bic

def compute_acf(e, max_lag=40):
    r = e - np.mean(e)
    denom = np.sum(r ** 2)
    if denom < 1e-12:
        return np.zeros(max_lag + 1)
    acf_vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            num = np.sum(r[lag:] * r[:-lag])
            acf_vals.append(num / denom)
    return np.array(acf_vals)

path = r"C:\Users\Hp\OneDrive\Documents\CH-512 Project\predictive models_fouling factor\All collected data.xlsx"
data = pd.read_excel(path)
data = data.drop(columns=["Density (Kg/m3)", "Time (hr)", "Reference"])

y_col = "Fouling factor (m2 K/kW)"
x_cols = [c for c in data.columns if c != y_col]

X_all = data[x_cols].values.astype(np.float64)
y_all = data[y_col].values.astype(np.float64)

X_tr_raw, X_va_raw, y_tr_raw, y_va_raw = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, shuffle=True
)

sx = StandardScaler()
sy = StandardScaler()

X_tr = sx.fit_transform(X_tr_raw)
X_va = sx.transform(X_va_raw)

y_tr_s = sy.fit_transform(y_tr_raw.reshape(-1, 1))

k = 20
h = 8

nnbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
nnbrs.fit(X_tr)

y_va_pred = np.zeros_like(y_va_raw, dtype=float)

for i in range(len(X_va)):
    q = X_va[i].reshape(1, -1)
    _, idx = nnbrs.kneighbors(q, return_distance=True)
    idx = idx.flatten()
    X_loc = X_tr[idx]
    y_loc = y_tr_s[idx].ravel()
    mlp = MLPRegressor(
        hidden_layer_sizes=(h,),
        activation="relu",
        solver="adam",
        alpha=1e-2,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_loc, y_loc)
    y_pred_s = mlp.predict(q).reshape(1, 1)
    y_pred_orig = sy.inverse_transform(y_pred_s)
    y_va_pred[i] = y_pred_orig[0, 0]

mse = mean_squared_error(y_va_raw, y_va_pred)
r2 = r2_score(y_va_raw, y_va_pred)

print("\n==== JIT + Local 1-HL NN (Adam) — Fouling ====")
print(f"MSE  : {mse:.6e}")
print(f"R²   : {r2:.4f}")

res = y_va_raw - y_va_pred

n_feat = X_all.shape[1]
n_out = 1
n_params = n_feat * h + h + h * n_out + n_out

sse, aic, bic = compute_sse_aic_bic(res, n_params)
print("\n[JIT Local 1-HL NN] Information criteria (validation):")
print(f"  n_params: {n_params}")
print(f"  SSE: {sse:.6f}, AIC: {aic:.2f}, BIC: {bic:.2f}")

plt.figure(figsize=(12,5))
plt.plot(y_va_raw, label="Actual fouling factor", linewidth=2)
plt.plot(y_va_pred, label="Predicted fouling factor (JIT+NN)", linewidth=2, alpha=0.7)
plt.title("JIT + Local 1-HL NN (Adam) — Fouling Factor")
plt.xlabel("Validation sample index")
plt.ylabel("Fouling factor (m²·K/kW)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(res, linewidth=1.5)
plt.title("JIT + Local 1-HL NN — Residuals vs Sample (Validation)")
plt.xlabel("Validation sample index")
plt.ylabel("Residual (true - pred)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

max_lag = 40
lags = np.arange(max_lag + 1)
acf_vals = compute_acf(res, max_lag=max_lag)

print("\n[JIT + Local 1-HL NN] Residual autocorrelation (first few lags):")
print( acf_vals[:5])

plt.figure(figsize=(8,4))
plt.stem(lags, acf_vals)
plt.axhline(0, linewidth=1)
plt.title("JIT + Local 1-HL NN — Residual ACF (Validation)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()