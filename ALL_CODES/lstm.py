import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


def compute_sse_aic_bic(residuals: np.ndarray, n_params: int):
    
    resid = residuals.reshape(-1, 1)     # (N,1)
    n_samples, n_outputs = resid.shape
    n_eff = n_samples * n_outputs        # here = N

    sse = np.sum(resid ** 2)
    sigma2_hat = sse / n_eff
    sigma2_hat = max(sigma2_hat, 1e-12)  # avoid log(0)

    # Gaussian log-likelihood (up to constants)
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



seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = "mps" if torch.mps.is_available() else "cpu"
print("Device:", device)

df = pd.read_excel("C:\\Users\\Hp\\OneDrive\\Documents\\CH-512 Project\\predictive models_fouling factor\\All collected data.xlsx")

# If there is a time column, sorting it
time_col_candidates = ["Time (hr)", "Time (year)", "Time"]
for c in time_col_candidates:
    if c in df.columns:
        df = df.sort_values(by=c).reset_index(drop=True)
        print(f"Sorted by time column: {c}")
        break

# Drop columns not used as features/targets
cols_to_drop = [c for c in ["Density (Kg/m3)", "Time (hr)", "Time (year)", "Reference"] if c in df.columns]
df = df.drop(columns=cols_to_drop)

FOULING_COL = "Fouling factor (m2 K/kW)"
if FOULING_COL not in df.columns:
    raise ValueError(f"Column '{FOULING_COL}' not found in df.columns: {list(df.columns)}")

# Inputs: all columns except fouling factor
input_cols = [c for c in df.columns if c != FOULING_COL]

X_raw = df[input_cols].values.astype(np.float32)          # (N, n_features)
Y_raw = df[[FOULING_COL]].values.astype(np.float32)       # (N, 1)

print("Input columns:", input_cols)
print("Target column:", FOULING_COL)
print("Raw shapes (X, Y):", X_raw.shape, Y_raw.shape)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_x.fit_transform(X_raw)
Y_scaled = scaler_y.fit_transform(Y_raw)


def create_sequences(X, Y, seq_len=20, horizon=1):
    
    X_seq, Y_seq = [], []
    T = len(X)
    start = seq_len
    end = T - (horizon - 1)
    for t in range(start, end):
        x_window = X[t-seq_len:t]          # (seq_len, n_features)
        y_target = Y[t + (horizon - 1)]    # (1,)
        X_seq.append(x_window)
        Y_seq.append(y_target)
    return np.stack(X_seq), np.stack(Y_seq)

seq_len = 20
horizon = 1   # 1-step ahead

X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, seq_len, horizon)
print("All sequences shape (X_seq, Y_seq):", X_seq.shape, Y_seq.shape)  

N_seq = X_seq.shape[0]
n_train = int(0.7 * N_seq)
n_val   = int(0.15 * N_seq)
n_test  = N_seq - n_train - n_val

X_train_seq = X_seq[:n_train]
Y_train_seq = Y_seq[:n_train]

X_val_seq   = X_seq[n_train:n_train+n_val]
Y_val_seq   = Y_seq[n_train:n_train+n_val]

X_test_seq  = X_seq[n_train+n_val:]
Y_test_seq  = Y_seq[n_train+n_val:]

print("Train seq:", X_train_seq.shape, Y_train_seq.shape)
print("Val   seq:", X_val_seq.shape,   Y_val_seq.shape)
print("Test  seq:", X_test_seq.shape,  Y_test_seq.shape)

X_train_t = torch.from_numpy(X_train_seq.astype(np.float32))
Y_train_t = torch.from_numpy(Y_train_seq.astype(np.float32))
X_val_t   = torch.from_numpy(X_val_seq.astype(np.float32))
Y_val_t   = torch.from_numpy(Y_val_seq.astype(np.float32))
X_test_t  = torch.from_numpy(X_test_seq.astype(np.float32))
Y_test_t  = torch.from_numpy(Y_test_seq.astype(np.float32))


class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

batch_size = 128

train_dataset = SeqDataset(X_train_t, Y_train_t)
val_dataset   = SeqDataset(X_val_t,   Y_val_t)
test_dataset  = SeqDataset(X_test_t,  Y_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# LSTM Model

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x) 
        last = out[:, -1, :]            # last time step
        last = self.dropout(last)
        return self.fc(last)            # (batch, num_outputs)

input_size  = X_train_seq.shape[2]
num_outputs = Y_train_seq.shape[1]  # should be 1
hidden_size = 64

model = LSTMRegressor(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=1,
    num_outputs=num_outputs,
    dropout=0.2
).to(device)

# Training setup (MSE loss)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

epochs   = 200
patience = 20
best_val_loss = float("inf")
best_state = None
patience_counter = 0

# Training loop with early stopping

for epoch in range(epochs):
    # Train 
    model.train()
    running_train = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)

        optimizer.step()
        running_train += loss.item()

    avg_train_loss = running_train / len(train_loader)

    # Validation
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            running_val += loss.item()

    avg_val_loss = running_val / len(val_loader)

    # Track best model
    if avg_val_loss < best_val_loss - 1e-5:
        best_val_loss = avg_val_loss
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"Epoch {epoch+1:03d} | train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | patience={patience_counter}")

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Load best weights
if best_state is not None:
    model.load_state_dict(best_state)


def evaluate(loader, name=""):
    model.eval()
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_pred = model(xb)
            preds_scaled.append(y_pred.cpu().numpy())
            trues_scaled.append(yb.cpu().numpy())

    preds_scaled = np.vstack(preds_scaled)  # (N, 1)
    trues_scaled = np.vstack(trues_scaled)  # (N, 1)

    # R² in scaled space
    r2_scaled = r2_score(trues_scaled, preds_scaled)
    mse_scaled = mean_squared_error(trues_scaled, preds_scaled)


    preds_orig = scaler_y.inverse_transform(preds_scaled)
    trues_orig = scaler_y.inverse_transform(trues_scaled)

    r2_orig = r2_score(trues_orig, preds_orig)
    mse_orig = mean_squared_error(trues_orig, preds_orig)

    print(f"{name} — R² (scaled) = {r2_scaled:.4f}, MSE (scaled) = {mse_scaled:.6f}")
    print(f"{name} — R² (orig)   = {r2_orig:.4f}, MSE (orig)   = {mse_orig:.6f}")

    return trues_orig, preds_orig

print("\n=== Final evaluation ===")
train_true_orig, train_pred_orig = evaluate(train_loader, "Train")
val_true_orig,   val_pred_orig   = evaluate(val_loader,   "Val")
test_true_orig,  test_pred_orig  = evaluate(test_loader,  "Test")

plt.figure(figsize=(12,5))
plt.plot(test_true_orig[:,0], label="Actual fouling factor",   linewidth=2, color="black")
plt.plot(test_pred_orig[:,0], label="Predicted fouling factor", linewidth=2, alpha=0.7, color="red")
plt.title("Test Set Prediction vs Actual — Fouling Factor (LSTM)")
plt.xlabel("Sequence index (test)")
plt.ylabel("Fouling factor (m²·K/kW)")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()


n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


res_train = (train_true_orig[:, 0] - train_pred_orig[:, 0])
res_val   = (val_true_orig[:, 0]   - val_pred_orig[:, 0])
res_test  = (test_true_orig[:, 0]  - test_pred_orig[:, 0])

sse_train, aic_train, bic_train = compute_sse_aic_bic(res_train, n_params)
sse_val,   aic_val,   bic_val   = compute_sse_aic_bic(res_val,   n_params)
sse_test,  aic_test,  bic_test  = compute_sse_aic_bic(res_test,  n_params)

print("\n[LSTM Fouling Model] Metrics (original units):")
print(f"  Train: SSE={sse_train:.6f}, AIC={aic_train:.2f}, BIC={bic_train:.2f}")
print(f"  Val  : SSE={sse_val:.6f},   AIC={aic_val:.2f},   BIC={bic_val:.2f}")
print(f"  Test : SSE={sse_test:.6f},  AIC={aic_test:.2f},  BIC={bic_test:.2f}")

plt.figure(figsize=(12,4))
plt.plot(res_test, linewidth=1.5)
plt.title("LSTM — Residuals vs Sample (Fouling, TEST)")
plt.xlabel("Test sequence index")
plt.ylabel("Residual (true - pred)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ACF of TEST residuals
max_lag = 40
lags = np.arange(max_lag + 1)

acf_test = compute_acf(res_test, max_lag=max_lag)
print("\nLSTM Fouling — residual ACF (first few lags):")
print(acf_test[:5])

plt.figure(figsize=(8,4))
plt.stem(lags, acf_test)
plt.axhline(0, linewidth=1)
plt.title("LSTM — Residual ACF (Fouling, TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
