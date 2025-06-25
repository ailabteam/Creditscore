# train_gmsc_v3.py  ─  PyTorch + Tversky + Threshold search
import os, random, warnings, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             balanced_accuracy_score, f1_score,
                             roc_curve, confusion_matrix,
                             classification_report)

# 1 ───────────────────────────── reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# 2 ───────────────────────────── data
FEATS = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]
LABEL = "SeriousDlqin2yrs"
TRAIN_CSV, TEST_CSV = "cs-training.csv", "cs-test.csv"

df = pd.read_csv(TRAIN_CSV, index_col=0)

# 3 ───────────────────────────── preprocessing
pipe = Pipeline([
    ("imputer",  SimpleImputer(strategy="median")),
    ("yjnorm",   PowerTransformer(method="yeo-johnson")),
    ("scaler",   StandardScaler())
])
X_all = pipe.fit_transform(df[FEATS])
y_all = df[LABEL].astype(np.float32).values

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                  torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)),
    batch_size=2048, shuffle=True)

# 4 ───────────────────────────── model
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
            nn.Linear(64, 1)              # logits
        )
    def forward(self, x):  return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MLP(len(FEATS)).to(device)

# 5 ───────────────────────────── Tversky-Loss
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, eps=1e-7):
        super().__init__(); self.a, self.b, self.eps = alpha, beta, eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p * targets).sum(); fp = ((1-targets) * p).sum(); fn = (targets * (1-p)).sum()
        tversky = (tp + self.eps) / (tp + self.a*fp + self.b*fn + self.eps)
        return 1 - tversky

criterion = TverskyLoss(alpha=0.7, beta=0.3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# 6 ───────────────────────────── training
MAX_EPOCH, PATIENCE = 100, 6
best_auc = best_bac = 0.0
BEST_AUC_PTH, BEST_BAC_PTH = "best_auc.pt", "best_bac.pt"
epochs_no_imp = 0

print("{:>5} | {:>8} | {:>8} | {:>8} | {:>8} | {:>6} | {:>4}".format(
    "Ep", "Loss", "AUC", "ACC", "BAC", "F1", "thr"))

for epoch in range(1, MAX_EPOCH+1):
    # ---- train
    model.train(); run_loss=0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        run_loss += loss.item()*xb.size(0)
    epoch_loss = run_loss / len(train_loader.dataset)
    scheduler.step(epoch + 0.0)          # update per epoch
    
    # ---- validation
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32).to(device))
        val_prob   = torch.sigmoid(val_logits).cpu().numpy().squeeze()

    # threshold search (ROC-Youden)
    fpr,tpr,thr = roc_curve(y_val, val_prob)
    best_idx = np.argmax(tpr - fpr)
    thr_opt  = float(thr[best_idx])
    preds    = (val_prob > thr_opt).astype(int)

    val_auc  = roc_auc_score(y_val, val_prob)
    val_acc  = accuracy_score(y_val, preds)
    val_bac  = balanced_accuracy_score(y_val, preds)
    val_f1   = f1_score(y_val, preds)

    print(f"{epoch:5d} | {epoch_loss:8.4f} | {val_auc:8.4f} | "
          f"{val_acc:8.4f} | {val_bac:8.4f} | {val_f1:6.4f} | {thr_opt:4.2f}")

    # save checkpoints
    if val_auc > best_auc + 1e-4:
        best_auc = val_auc; torch.save(model.state_dict(), BEST_AUC_PTH)
    if val_bac > best_bac + 1e-4:
        best_bac = val_bac; torch.save(model.state_dict(), BEST_BAC_PTH)

    # early-stopping by AUC
    if val_auc >= best_auc: epochs_no_imp = 0
    else:                   epochs_no_imp += 1
    if epochs_no_imp >= PATIENCE:
        print(f"Early-stopping at epoch {epoch}"); break

print(f"\nBest AUC {best_auc:.4f} (model → {BEST_AUC_PTH})")
print(f"Best BAC {best_bac:.4f} (model → {BEST_BAC_PTH})")

# 7 ───────────────────────────── final report
cm = confusion_matrix(y_val, preds)
print("\nConfusion matrix (thr {:.2f}):\n{}".format(thr_opt, cm))
print("\n", classification_report(y_val, preds, digits=4))

# 8 ───────────────────────────── inference (optional)
if os.path.exists(TEST_CSV):
    test_df = pd.read_csv(TEST_CSV, index_col=0)
    X_test  = pipe.transform(test_df[FEATS])
    # choose whichever metric you prefer ↓
    model.load_state_dict(torch.load(BEST_BAC_PTH, weights_only=True))
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_prob   = torch.sigmoid(test_logits).cpu().numpy().squeeze()
    # pd.DataFrame({"Id": test_df.index, "Probability": test_prob}).to_csv("submission.csv", index=False)

