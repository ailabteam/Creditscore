# train_gmsc_ensemble.py  –  MLP (PyTorch) + LightGBM ensemble
import os, random, numpy as np, pandas as pd, torch, lightgbm as lgb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             balanced_accuracy_score, f1_score,
                             confusion_matrix, classification_report)

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ────────────────────────── data
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

# ────────────────────────── preprocessing pipeline
pipe = Pipeline([
    ("imputer",  SimpleImputer(strategy="median")),
    ("yjnorm",   PowerTransformer(method="yeo-johnson")),
    ("scaler",   StandardScaler())
])
X_all = pipe.fit_transform(df[FEATS])
y_all = df[LABEL].astype(np.float32).values

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)

# ────────────────────────── 1) MLP model
class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 1)                # logits
        )
    def forward(self,x): return self.net(x)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = MLP(len(FEATS)).to(device)
pos_w   = torch.tensor([(y_tr==0).sum()/(y_tr==1).sum()]).to(device)
crit    = nn.BCEWithLogitsLoss(pos_weight=pos_w)
optim   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loader  = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                   torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)),
                     batch_size=2048, shuffle=True)
best_auc, best_state, patience, no_imp = 0.0, None, 8, 0

for epoch in range(1, 80+1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward(); optim.step()
        epoch_loss += loss.item()*xb.size(0)
    epoch_loss /= len(loader.dataset)

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32).to(device))
        val_prob   = torch.sigmoid(val_logits).cpu().numpy().squeeze()
    auc = roc_auc_score(y_val, val_prob)
    if auc > best_auc + 1e-4:
        best_auc, best_state, no_imp = auc, model.state_dict(), 0
    else:
        no_imp += 1
    if no_imp >= patience: break

model.load_state_dict(best_state)
torch.save(model.state_dict(), "best_mlp.pt")
print(f"Finished MLP – best AUC {best_auc:.4f}  (epoch {epoch})")

# ────────────────────────── 2) LightGBM model
neg, pos = (y_tr==0).sum(), (y_tr==1).sum()
scale_pos = neg/pos
params = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=31,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    scale_pos_weight=scale_pos,  # handle imbalance
    seed=SEED, verbose=-1
)
train_ds = lgb.Dataset(X_tr, label=y_tr)
valid_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
lgb_model = lgb.train(params, train_ds, num_boost_round=800,
                      valid_sets=[valid_ds], valid_names=["val"],
                      early_stopping_rounds=100, verbose_eval=False)
lgb_model.save_model("lgb_model.txt")
print(f"Finished LightGBM – best iteration {lgb_model.best_iteration}")

# ────────────────────────── 3) Ensemble & threshold sweep (maximize ACC)
with torch.no_grad():
    mlp_prob = torch.sigmoid(
        model(torch.tensor(X_val, dtype=torch.float32).to(device))
    ).cpu().numpy().squeeze()
lgb_prob  = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
ens_prob  = (mlp_prob + lgb_prob) / 2  # average probabilities

best_thr, best_acc, best_metrics = 0.5, 0.0, None
for thr in np.linspace(0.05, 0.50, 46):            # 5% … 50%
    pred = (ens_prob > thr).astype(int)
    acc  = accuracy_score(y_val, pred)
    if acc > best_acc:
        best_acc = acc; best_thr = thr
        best_metrics = (
            roc_auc_score(y_val, ens_prob),
            balanced_accuracy_score(y_val, pred),
            f1_score(y_val, pred),
            confusion_matrix(y_val, pred)
        )
auc_ens, bac_ens, f1_ens, cm_ens = best_metrics
np.savetxt("best_threshold.txt", np.array([best_thr]))
print("\n===== ENSEMBLE VALIDATION METRICS =====")
print(f"AUC      : {auc_ens:.4f}")
print(f"ACC (thr={best_thr:.2f}) : {best_acc:.4f}")
print(f"BAC      : {bac_ens:.4f}")
print(f"F1       : {f1_ens:.4f}")
print("Confusion matrix:\n", cm_ens)
print("========================================")

# ────────────────────────── 4) optional inference on test-set
if os.path.exists(TEST_CSV):
    test_df = pd.read_csv(TEST_CSV, index_col=0)
    X_test  = pipe.transform(test_df[FEATS])
    with torch.no_grad():
        mlp_test = torch.sigmoid(
            model(torch.tensor(X_test, dtype=torch.float32).to(device))
        ).cpu().numpy().squeeze()
    lgb_test  = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    ens_test  = (mlp_test + lgb_test) / 2
    # pd.DataFrame({"Id": test_df.index, "Probability": ens_test})\
    #   .to_csv("submission.csv", index=False)

