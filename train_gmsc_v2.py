# train_gmsc_v2.py
import warnings, os, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------
# 2. Data
# --------------------------------------------------
TRAIN_CSV = "cs-training.csv"
TEST_CSV  = "cs-test.csv"      # -> dùng inference nếu cần

df = pd.read_csv(TRAIN_CSV, index_col=0)

FEATS = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]
LABEL = "SeriousDlqin2yrs"

# --------------------------------------------------
# 3. Pre-processing
#   median-impute → Yeo-Johnson power-transform → StandardScaler
#   PowerTransformer giảm skew (RevolvingUtilization, DebtRatio, MonthlyIncome …)
# --------------------------------------------------
pipe = Pipeline([
    ("imputer",  SimpleImputer(strategy="median")),
    ("yjnorm",   PowerTransformer(method="yeo-johnson")),
    ("scaler",   StandardScaler())
])
X = pipe.fit_transform(df[FEATS])
y = df[LABEL].astype(np.float32).values

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                  torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)),
    batch_size=2048, shuffle=True, drop_last=False
)

# --------------------------------------------------
# 4. Model
# --------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  nn.ReLU(),

            nn.Linear(64, 1)          # logits
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MLP(len(FEATS)).to(device)

# --------------------------------------------------
# 5. Focal Loss
# --------------------------------------------------
class FocalLoss(nn.Module):
    """Binary Focal Loss with logits."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()

criterion = FocalLoss(alpha=0.25, gamma=2.0)

# --------------------------------------------------
# 6. Optimizer & Cosine-Annealing Warm Restarts
# --------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2)

# --------------------------------------------------
# 7. Early-Stopping
# --------------------------------------------------
PATIENCE   = 6
best_auc   = 0.0
epochs_no_improve = 0
BEST_PATH  = "best_model_v2.pt"

MAX_EPOCH = 100
print("{:>5} | {:>8} | {:>8} | {:>8} | {:>8}".format(
    "Epoch", "Loss", "AUC", "ACC", "BAC"))
for epoch in range(1, MAX_EPOCH + 1):
    # ---- Train ----
    model.train()
    run_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * xb.size(0)

    epoch_loss = run_loss / len(train_loader.dataset)
    scheduler.step(epoch - 1 + len(train_loader) / len(train_loader))  # update on epoch boundary

    # ---- Validation ----
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32).to(device))
        val_prob   = torch.sigmoid(val_logits).cpu().numpy().squeeze()

    val_auc = roc_auc_score(y_val, val_prob)
    val_acc = accuracy_score(y_val, (val_prob > 0.5).astype(int))
    val_bac = balanced_accuracy_score(y_val, (val_prob > 0.5).astype(int))

    print(f"{epoch:5d} | {epoch_loss:8.4f} | {val_auc:8.4f} | "
          f"{val_acc:8.4f} | {val_bac:8.4f}")

    # ---- Early-Stopping ----
    if val_auc > best_auc + 1e-4:
        best_auc = val_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_PATH)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early-stopping at epoch {epoch}")
            break

print(f"\nBest val AUC: {best_auc:.4f}  ->  saved {BEST_PATH}")

# --------------------------------------------------
# 8. Inference (optional)
# --------------------------------------------------
if os.path.exists(TEST_CSV):
    test_df = pd.read_csv(TEST_CSV, index_col=0)
    X_test  = pipe.transform(test_df[FEATS])

    model.load_state_dict(torch.load(BEST_PATH, weights_only=True))
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_prob   = torch.sigmoid(test_logits).cpu().numpy().squeeze()

    # pd.DataFrame({"Id": test_df.index, "Probability": test_prob}).to_csv("submission.csv", index=False)

