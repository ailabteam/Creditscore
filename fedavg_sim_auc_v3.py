#!/usr/bin/env python
# fedavg_sim_auc_v3.py  –  Federated Learning pure PyTorch
# 목표: AUC ≥ 0.86 on GiveMeSomeCredit

import warnings, random, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 0 ───────────── Config
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

N_CLIENTS     = 10
NUM_ROUNDS    = 60
LOCAL_EPOCHS  = 3
BATCH_SIZE    = 512
LR            = 1e-3
MOMENTUM_BETA = 0.9          # FedOpt momentum
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1 ───────────── Load & preprocess data
FEATS = [
    "RevolvingUtilizationOfUnsecuredLines","age",
    "NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]
LABEL = "SeriousDlqin2yrs"

df = pd.read_csv("cs-training.csv", index_col=0)

pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler())
])
X_all = pipe.fit_transform(df[FEATS]).astype(np.float32)
y_all = df[LABEL].values.astype(np.float32)

def split_noniid(X, y, n=10, minority=(0.08,0.18)):
    pos, neg = np.where(y==1)[0], np.where(y==0)[0]
    np.random.shuffle(pos); np.random.shuffle(neg)
    pos_chunks = np.array_split(pos, n); neg_chunks = np.array_split(neg, n)
    out=[]
    for i in range(n):
        pct  = np.random.uniform(*minority)
        need = int(pct*(len(pos_chunks[i])+len(neg_chunks[i])))
        idx  = np.concatenate([pos_chunks[i][:need], neg_chunks[i]])
        Xtr,Xte,ytr,yte = train_test_split(
            X[idx], y[idx], test_size=0.2, stratify=y[idx], random_state=SEED)
        out.append(((Xtr,ytr),(Xte,yte)))
    return out

clients = split_noniid(X_all, y_all, N_CLIENTS)

# 2 ───────────── Model
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

global_model = MLP(len(FEATS)).to(DEVICE)

def get_params(model):
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]

def set_params(model, params):
    model.load_state_dict({k: torch.tensor(v) 
                           for k,v in zip(model.state_dict().keys(), params)}, strict=True)

# 3 ───────────── Focal-Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5, reduction="mean"):
        super().__init__(); self.a, self.g, self.red = alpha, gamma, reduction
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets,
                                                             reduction="none")
        p_t = torch.exp(-bce)
        loss = self.a * (1-p_t)**self.g * bce
        return loss.mean() if self.red=="mean" else loss.sum()

# FedOpt momentum buffer (server-side)
momentum_buffer = None

# 4 ───────────── Federated Training Loop
print(f"Device {DEVICE} | Rounds {NUM_ROUNDS} | Clients {N_CLIENTS}\n")

for rnd in range(1, NUM_ROUNDS+1):
    new_weights, samples = [], []

    # ---------- Local training per client ----------
    for (Xt,yt),_ in clients:
        local = MLP(len(FEATS)).to(DEVICE)
        set_params(local, get_params(global_model))
        opt = torch.optim.Adam(local.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=LOCAL_EPOCHS, T_mult=1)

        dataset = TensorDataset(torch.tensor(Xt), torch.tensor(yt).unsqueeze(1))
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        local.train()
        crit = FocalLoss(alpha=0.25, gamma=1.5)
        for _ in range(LOCAL_EPOCHS):
            for xb, yb in loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(local(xb), yb)
                loss.backward(); opt.step()
            sched.step()

        new_weights.append(get_params(local))
        samples.append(len(yt))

    # ---------- FedOpt momentum aggregation ----------
    avg=[]
    total = sum(samples)
    for layer in zip(*new_weights):
        avg.append(sum(w*n for w,n in zip(layer,samples)) / total)

    # momentum update
    if momentum_buffer is None:
        momentum_buffer = [np.zeros_like(p) for p in avg]

    current = get_params(global_model)
    updated=[]
    for i,(g,new) in enumerate(zip(current, avg)):
        momentum_buffer[i] = MOMENTUM_BETA*momentum_buffer[i] + (1-MOMENTUM_BETA)*(new - g)
        updated.append(g + momentum_buffer[i])
    set_params(global_model, updated)

    # ---------- Evaluation ----------
    print(f"Round {rnd:2d}")
    print("Client |  loss |  acc |  auc |   n")
    gl_loss=gl_acc=gl_auc=0.0; tot=0
    for cid, ((Xt,yt),(Xv,yv)) in enumerate(clients):
        Xv_t = torch.tensor(Xv, device=DEVICE)
        yv_t = torch.tensor(yv, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            logits = global_model(Xv_t)
            prob   = torch.sigmoid(logits).cpu().numpy().squeeze()
        loss = nn.BCEWithLogitsLoss()(logits, yv_t).item()
        acc  = ((prob>0.5)==yv).mean()
        try:    auc = roc_auc_score(yv, prob)
        except ValueError: auc = float("nan")
        n=len(yv)
        print(f"  {cid:2d}   | {loss:.3f} | {acc:.3f} | {auc:.3f} | {n}")
        gl_loss+=loss*n; gl_acc+=acc*n
        if not np.isnan(auc): gl_auc+=auc*n
        tot+=n
    print(f"GLOBAL | {gl_loss/tot:.3f} | {gl_acc/tot:.3f} | {gl_auc/tot:.3f} | {tot}")
    print("-"*60)

print("Finished.")

