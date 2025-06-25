#!/usr/bin/env python
# fedavg_ae_mlp.py
# --------------------------------------------------------------
# Federated Autoencoder + Classifier (AE-MLP) – Give-Me-Some-Credit
# FedAvg + momentum, 10 client non-IID, 60 rounds × 3 local epoch
# * Loss = α·MSE(recon) + BCE(class)   (α = 0.5)
# * Log Loss_rec | Loss_cls | ACC | AUC   mỗi round
# --------------------------------------------------------------

import warnings, random, sys, logging
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ────────── logger (console + file) ──────────
logging.basicConfig(level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("ae_run.log", mode="w")])
log = logging.getLogger()

# ────────── reproducibility & config ─────────
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLIENTS   = 10
ROUNDS      = 60
LOCAL_EPOCH = 3
BATCH_SIZE  = 1024
LR          = 2e-3
MOM_BETA    = 0.9
ALPHA       = 0.5          # weight for reconstruction loss in total loss

# ────────── load & preprocess data ───────────
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
    ("sc",  StandardScaler()),
])
X_all = pipe.fit_transform(df[FEATS]).astype(np.float32)
y_all = df[LABEL].values.astype(np.float32)

def split_noniid(X, y, n=10, minority=(0.08, 0.18)):
    pos, neg = np.where(y==1)[0], np.where(y==0)[0]
    np.random.shuffle(pos); np.random.shuffle(neg)
    pos_chunks = np.array_split(pos, n)
    neg_chunks = np.array_split(neg, n)
    out=[]
    for pc,nc in zip(pos_chunks, neg_chunks):
        pct  = np.random.uniform(*minority)
        need = int(pct*(len(pc)+len(nc)))
        idx  = np.concatenate([pc[:need], nc])
        Xtr,Xte,ytr,yte = train_test_split(X[idx], y[idx],
                                           test_size=0.2,
                                           stratify=y[idx],
                                           random_state=SEED)
        out.append(((Xtr,ytr),(Xte,yte)))
    return out

clients = split_noniid(X_all, y_all, N_CLIENTS)

# ────────── model ────────────────────────────
class AE_MLP(nn.Module):
    def __init__(self, d_in, bottleneck=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),  nn.ReLU(),
            nn.Linear(64, 128),         nn.ReLU(),
            nn.Linear(128, d_in)
        )
        self.classifier = nn.Linear(bottleneck, 1)
    def forward(self, x):
        z   = self.encoder(x)
        x_r = self.decoder(z)
        logit = self.classifier(z)
        return x_r, logit

global_model = AE_MLP(len(FEATS)).to(DEVICE)

# helpers
def get_params(model):
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]
def set_params(model, params):
    model.load_state_dict({k: torch.tensor(v)
                           for k,v in zip(model.state_dict().keys(),params)},
                          strict=True)

# ────────── training ─────────────────────────
mse_global = nn.MSELoss()
momentum = None
log.info(f"Device {DEVICE} | AE-MLP FedAvg | {ROUNDS} rounds × {LOCAL_EPOCH} epoch\n")

for rd in range(1, ROUNDS+1):
    # local weights & sizes
    new_w, sizes = [], []

    for (Xt, yt), _ in clients:
        local = AE_MLP(len(FEATS)).to(DEVICE)
        set_params(local, get_params(global_model))

        # pos_weight for the **current client**
        pos_w = torch.tensor([(yt==0).sum()/(yt==1).sum()],
                             dtype=torch.float32, device=DEVICE)
        bce_local = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        mse_local = mse_global

        opt = torch.optim.AdamW(local.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=LOCAL_EPOCH)

        dl = DataLoader(
            TensorDataset(torch.tensor(Xt), torch.tensor(yt).unsqueeze(1)),
            batch_size=BATCH_SIZE, shuffle=True)

        local.train()
        for _ in range(LOCAL_EPOCH):
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                x_r, logit = local(xb)
                loss_rec = mse_local(x_r, xb)
                loss_cls = bce_local(logit, yb)
                (ALPHA*loss_rec + loss_cls).backward()
                opt.step()
            sch.step()

        new_w.append(get_params(local)); sizes.append(len(yt))

    # FedAvg + momentum
    total = sum(sizes)
    avg = [sum(w*n for w,n in zip(layer, sizes))/total for layer in zip(*new_w)]
    if momentum is None:
        momentum = [np.zeros_like(p) for p in avg]
    cur = get_params(global_model)
    updated=[]
    for i,(g,n) in enumerate(zip(cur, avg)):
        momentum[i] = MOM_BETA*momentum[i] + (1-MOM_BETA)*(n-g)
        updated.append(g + momentum[i])
    set_params(global_model, updated)

    # evaluation
    gl_rec=gl_cls=gl_acc=gl_auc=0.0; tot=0
    for cid,((Xt,yt),(Xv,yv)) in enumerate(clients):
        Xv_t=torch.tensor(Xv,device=DEVICE)
        yv_t=torch.tensor(yv,device=DEVICE).unsqueeze(1)
        global_model.eval()
        with torch.no_grad():
            x_r, logit = global_model(Xv_t)
            prob = torch.sigmoid(logit).cpu().numpy().squeeze()
            loss_rec = mse_global(x_r, Xv_t).item()
            loss_cls = bce_local(logit, yv_t).item()  # use same pos_w
        acc = ((prob>0.5)==yv).mean()
        try: auc = roc_auc_score(yv, prob)
        except ValueError: auc = float("nan")
        n=len(yv)
        gl_rec+=loss_rec*n; gl_cls+=loss_cls*n
        gl_acc+=acc*n;      gl_auc+=0 if np.isnan(auc) else auc*n
        tot+=n

    log.info(f"Round {rd:02d} | rec {gl_rec/tot:.4f} | cls {gl_cls/tot:.4f} "
             f"| acc {gl_acc/tot:.3f} | auc {gl_auc/tot:.3f}")

log.info("Finished.")

