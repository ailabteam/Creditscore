# train_gmsc_full.py
import os, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ---------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------
# 2. Đọc dữ liệu
# ---------------------------------------------------------------------
TRAIN_CSV = "cs-training.csv"   # upload cùng thư mục
TEST_CSV  = "cs-test.csv"       # chưa dùng, để inference sau

train_df = pd.read_csv(TRAIN_CSV, index_col=0)

FEATS = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]
LABEL = "SeriousDlqin2yrs"

# ---------------------------------------------------------------------
# 3. Tiền xử lý
# ---------------------------------------------------------------------
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
X_full = pipe.fit_transform(train_df[FEATS])
y_full = train_df[LABEL].astype(np.float32).values

# ---------------------------------------------------------------------
# 4. Train / Validation split
# ---------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=SEED
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                  torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)),
    batch_size=1024, shuffle=True, drop_last=False
)

# ---------------------------------------------------------------------
# 5. MLP + BN + Dropout
# ---------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  nn.ReLU(),

            nn.Linear(32, 1)      # logits
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MLP(len(FEATS)).to(device)

# ---------------------------------------------------------------------
# 6. Loss, Optimizer, Scheduler, Early-Stopping
# ---------------------------------------------------------------------
pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, verbose=True)

PATIENCE  = 5          # epochs không cải thiện → dừng
best_auc  = 0.0
epochs_no_improve = 0
BEST_PATH = "best_model.pt"

# ---------------------------------------------------------------------
# 7. Training loop
# ---------------------------------------------------------------------
MAX_EPOCH = 100
for epoch in range(1, MAX_EPOCH + 1):
    # --- Train ---
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32).to(device))
        val_probs  = torch.sigmoid(val_logits).cpu().numpy().squeeze()
        val_auc    = roc_auc_score(y_val, val_probs)
        val_acc    = accuracy_score(y_val, (val_probs > 0.5).astype(int))

    # Scheduler & Early-Stopping
    scheduler.step(val_auc)
    if val_auc > best_auc + 1e-4:
        best_auc = val_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_PATH)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early-stopping at epoch {epoch}")
            break

    # --- Logs ---
    print(f"Epoch {epoch:3d} | loss {epoch_loss:.4f} | "
          f"val AUC {val_auc:.4f} | val ACC {val_acc:.4f} | "
          f"lr {optimizer.param_groups[0]['lr']:.1e}")

print(f"Best val AUC: {best_auc:.4f} → model saved to {BEST_PATH}")

# ---------------------------------------------------------------------
# 8. (Optional) Inference on test set
# ---------------------------------------------------------------------
if os.path.exists(TEST_CSV):
    test_df = pd.read_csv(TEST_CSV, index_col=0)
    X_test  = pipe.transform(test_df[FEATS])
    model.load_state_dict(torch.load(BEST_PATH))
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_probs  = torch.sigmoid(test_logits).cpu().numpy().squeeze()
    # pd.DataFrame({"Id": test_df.index, "Probability": test_probs})\
    #     .to_csv("submission.csv", index=False)

