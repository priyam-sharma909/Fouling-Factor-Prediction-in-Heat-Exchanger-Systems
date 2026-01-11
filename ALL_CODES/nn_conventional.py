import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def compute_sse_aic_bic(residuals: np.ndarray, n_params: int):

    resid = residuals.reshape(-1, 1)
    n_samples, n_outputs = resid.shape
    n_eff = n_samples * n_outputs  # here = N

    sse = np.sum(resid ** 2)
    sigma2_hat = sse / n_eff
    sigma2_hat = max(sigma2_hat, 1e-12)  # avoid log(0)

    # Gaussian log-likelihood
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


device = "mps" if torch.mps.is_available() else "cpu"
print("Device:", device)

df = pd.read_excel(r"C:\Users\Hp\OneDrive\Documents\CH-512 Project\predictive models_fouling factor\All collected data.xlsx")

# Drop non-feature columns
df = df.drop(columns=["Density (Kg/m3)", "Time (hr)", "Reference"])

input_data  = df.iloc[:, 0:4].values.astype(np.float32)
output_data = df.iloc[:, 5].values.astype(np.float32)

X_train, X_test, Y_train, Y_test = train_test_split(
    input_data,
    output_data,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

scalerx = StandardScaler()
scalery = StandardScaler()

X_train = scalerx.fit_transform(X_train)
X_test  = scalerx.transform(X_test)

Y_train = scalery.fit_transform(Y_train.reshape(-1, 1))
Y_test  = scalery.transform(Y_test.reshape(-1, 1))

X_train = torch.from_numpy(X_train.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
X_test  = torch.from_numpy(X_test.astype(np.float32))
Y_test  = torch.from_numpy(Y_test.astype(np.float32))

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

X_tr, X_val, Y_tr, Y_val = train_test_split(
    X_train.numpy(),  
    Y_train.numpy(),
    test_size=0.2,
    random_state=123,
    shuffle=True
)

X_tr  = torch.from_numpy(X_tr)
Y_tr  = torch.from_numpy(Y_tr)
X_val = torch.from_numpy(X_val)
Y_val = torch.from_numpy(Y_val)

train_dataset = CustomDataset(X_tr, Y_tr)
val_dataset   = CustomDataset(X_val, Y_val)
test_dataset  = CustomDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

class MyNN(nn.Module):
    def __init__(self, num_features, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels)
        )
    def forward(self, x):
        return self.net(x)

model = MyNN(X_train.shape[1], Y_train.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
epochs = 128
patience = 20  # early stopping patience

best_val_loss = float("inf")
best_state = None
patience_counter = 0

# Training loop with early stopping

for epoch in range(epochs):
    # ---- train ----
    model.train()
    running_train = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        loss.backward()
        optimizer.step()

        running_train += loss.item()

    avg_train_loss = running_train / len(train_loader)

    # validation 
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

    # early stopping check
    if avg_val_loss < best_val_loss - 1e-5:
        best_val_loss = avg_val_loss
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    print(
        f"Epoch {epoch+1:03d} | "
        f"train_loss={avg_train_loss:.4f} | "
        f"val_loss={avg_val_loss:.4f} | "
        f"patience={patience_counter}"
    )

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Load best model weights
if best_state is not None:
    model.load_state_dict(best_state)

# R² computation

def compute_r2(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu())
            trues.append(yb.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    return r2_score(trues, preds)

r2_train = compute_r2(train_loader)
r2_val   = compute_r2(val_loader)
r2_test  = compute_r2(test_loader)

print(f"R² on train data: {r2_train:.4f}")
print(f"R² on val   data: {r2_val:.4f}")
print(f"R² on test  data: {r2_test:.4f}")


model.eval()
with torch.no_grad():
    Y_pred_test = model(X_test.to(device)).cpu().numpy()

Y_test_orig = scalery.inverse_transform(Y_test.cpu().numpy())
Y_pred_orig = scalery.inverse_transform(Y_pred_test)

plt.figure(figsize=(12,5))
plt.plot(Y_test_orig, label='Actual Fouling Factor', linewidth=2, color='blue')
plt.plot(Y_pred_orig, label='Predicted Fouling Factor', linewidth=2, alpha=0.8, color='orange')
plt.title('Prediction vs Actual — Fouling Factor (Test Set)')
plt.xlabel('Test Sample Index')
plt.ylabel('Fouling Factor')
plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

def get_orig(loader):
    model.eval()
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_pred = model(xb)
            preds_scaled.append(y_pred.cpu().numpy())
            trues_scaled.append(yb.cpu().numpy())
    preds_scaled = np.vstack(preds_scaled)   # (N,1)
    trues_scaled = np.vstack(trues_scaled)   # (N,1)
    preds_orig = scalery.inverse_transform(preds_scaled)
    trues_orig = scalery.inverse_transform(trues_scaled)
    return trues_orig[:, 0], preds_orig[:, 0]


train_true_orig, train_pred_orig = get_orig(train_loader)
val_true_orig,   val_pred_orig   = get_orig(val_loader)

test_true_orig = Y_test_orig[:, 0]
test_pred_orig = Y_pred_orig[:, 0]

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# residuals
res_train = train_true_orig - train_pred_orig
res_val   = val_true_orig   - val_pred_orig
res_test  = test_true_orig  - test_pred_orig

sse_train, aic_train, bic_train = compute_sse_aic_bic(res_train, n_params)
sse_val,   aic_val,   bic_val   = compute_sse_aic_bic(res_val,   n_params)
sse_test,  aic_test,  bic_test  = compute_sse_aic_bic(res_test,  n_params)

print("\n[Feedforward NN Fouling Model] Metrics (original units):")
print(f"  Train: SSE={sse_train:.6f}, AIC={aic_train:.2f}, BIC={bic_train:.2f}")
print(f"  Val  : SSE={sse_val:.6f},   AIC={aic_val:.2f},   BIC={bic_val:.2f}")
print(f"  Test : SSE={sse_test:.6f},  AIC={aic_test:.2f},  BIC={bic_test:.2f}")

# Residual vs sample (TEST)
plt.figure(figsize=(12,4))
plt.plot(res_test, linewidth=1.5)
plt.title("Feedforward NN — Residuals vs Sample (Fouling, TEST)")
plt.xlabel("Test Sample Index")
plt.ylabel("Residual (true - pred)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ACF of TEST residuals
max_lag = 40
lags = np.arange(max_lag + 1)

acf_test = compute_acf(res_test, max_lag=max_lag)
print("\nFeedforward NN Fouling — residual ACF (first few lags):")
print(acf_test[:5])

plt.figure(figsize=(8,4))
plt.stem(lags, acf_test)
plt.axhline(0, linewidth=1)
plt.title("Feedforward NN — Residual ACF (Fouling, TEST)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
