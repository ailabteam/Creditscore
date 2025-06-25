#!/usr/bin/env python
# fedavg_ae_mlp_PN_stable.py
# --------------------------------------------------------------
# FedAvg + FedProx – AutoEncoder + Classifier với PopulationNorm
# • Grad-clip 5, var-clamp để tránh NaN
# • 10 client non-IID, 60 round × 3 epoch
# • Logs ra console + ae_PN_run.log
# --------------------------------------------------------------

import warnings, random, sys, logging, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ─── logger ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("ae_PN_run.log", mode="w")])
log = logging.getLogger()

# ─── config ─────────────────────────────────────────────
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLIENTS   = 10
ROUNDS      = 60
LOCAL_EPOCH = 3
BATCH_SIZE  = 512
LR          = 2e-3
MOM_BETA    = 0.9
MU_PROX     = 0.01
ALPHA       = 0.5
CLIP_NORM   = 5.0

# ─── Population Normalization 1-D ─────────────────────
class PN1d(nn.Module):
    def __init__(self, n, eps=1e-4, momentum=0.1, affine=True):
        super().__init__(); self.eps, self.m = eps, momentum
        self.register_buffer("pop_mean", torch.zeros(n))
        self.register_buffer("pop_var",  torch.ones(n))
        if affine:
            self.weight = nn.Parameter(torch.ones(n))
            self.bias   = nn.Parameter(torch.zeros(n))
        else:
            self.weight = self.bias = None
    def forward(self, x):
        if self.training:
            m_b = x.mean(0).detach()
            v_b = x.var(0, unbiased=False).detach() + self.eps
            with torch.no_grad():
                self.pop_mean.mul_(1-self.m).add_(self.m*m_b)
                self.pop_var .mul_(1-self.m).add_(self.m*v_b).clamp_(min=self.eps)
        out = (x - self.pop_mean) / torch.sqrt(self.pop_var + self.eps)
        if self.weight is not None:
            out = out * self.weight + self.bias
        return out

# ─── dataset ───────────────────────────────────────────
FEATS = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]
LABEL = "SeriousDlqin2yrs"

df = pd.read_csv("cs-training.csv", index_col=0)
X = Pipeline([("imp", SimpleImputer(strategy="median")),
              ("sc",  StandardScaler())]).fit_transform(df[FEATS]).astype(np.float32)
y = df[LABEL].astype(np.float32).values

def split_noniid(X, y, k=10, minority=(0.08, 0.18)):
    pos, neg = np.where(y==1)[0], np.where(y==0)[0]
    np.random.shuffle(pos); np.random.shuffle(neg)
    out = []
    for p, n in zip(np.array_split(pos, k), np.array_split(neg, k)):
        need = int(np.random.uniform(*minority)*(len(p)+len(n)))
        idx  = np.concatenate([p[:need], n])
        Xtr,Xte,ytr,yte = train_test_split(X[idx], y[idx], test_size=0.2,
                                           stratify=y[idx], random_state=SEED)
        out.append(((Xtr,ytr),(Xte,yte)))
    return out
clients = split_noniid(X, y, N_CLIENTS)

# ─── model ─────────────────────────────────────────────
class AE_MLP(nn.Module):
    def __init__(self, d_in, bottleneck=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, 256), PN1d(256), nn.ReLU(),
            nn.Linear(256, 128), PN1d(128), nn.ReLU(),
            nn.Linear(128, bottleneck)
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, d_in)
        )
        self.cls = nn.Linear(bottleneck, 1)
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), self.cls(z)

global_model = AE_MLP(len(FEATS)).to(DEVICE)
getP = lambda m: [p.detach().cpu().numpy() for p in m.state_dict().values()]
def setP(m, ps): m.load_state_dict({k: torch.tensor(v) for k,v in zip(m.state_dict(), ps)}, strict=True)

mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
momentum = None
log.info(f"Device {DEVICE} | AE-MLP + PN | {ROUNDS} rounds × {LOCAL_EPOCH} epoch\n")

# ─── federated training loop ─────────────────────────
for rd in range(1, ROUNDS+1):
    new_w, sizes = [], []
    for (Xt, yt), _ in clients:
        local = AE_MLP(len(FEATS)).to(DEVICE)
        setP(local, getP(global_model))

        opt = torch.optim.AdamW(local.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=LOCAL_EPOCH)

        sampler = WeightedRandomSampler(np.where(yt==1, 0.7, 0.3),
                                        len(yt), replacement=True)
        dl = DataLoader(TensorDataset(torch.tensor(Xt), torch.tensor(yt).unsqueeze(1)),
                        batch_size=BATCH_SIZE, sampler=sampler)

        local.train()
        for _ in range(LOCAL_EPOCH):
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                xr, logit = local(xb)
                loss = ALPHA*mse(xr, xb) + bce(logit, yb)
                prox = sum((w - w0.detach().to(DEVICE)).pow(2).sum()
                           for w, w0 in zip(local.parameters(), global_model.parameters()))
                (loss + MU_PROX/2 * prox).backward()
                nn.utils.clip_grad_norm_(local.parameters(), CLIP_NORM)
                opt.step()
            sch.step()

        new_w.append(getP(local)); sizes.append(len(yt))

    # FedAvg + momentum
    total = sum(sizes)
    avg = [sum(w*n for w,n in zip(layer, sizes))/total for layer in zip(*new_w)]
    if momentum is None:
        momentum = [np.zeros_like(p) for p in avg]
    cur = getP(global_model); upd = []
    for i,(g,n) in enumerate(zip(cur, avg)):
        momentum[i] = MOM_BETA*momentum[i] + (1-MOM_BETA)*(n-g)
        upd.append(g + momentum[i])
    setP(global_model, upd)

    # evaluation
    rec = cls = acc = auc = 0.0; tot = 0
    for (Xt, yt), (Xv, yv) in clients:
        Xv_t = torch.tensor(Xv, device=DEVICE)
        yv_t = torch.tensor(yv, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            xr, logit = global_model(Xv_t)
            prob = torch.sigmoid(logit).cpu().numpy().squeeze()
            rec += mse(xr, Xv_t).item() * len(yv)
            cls += bce(logit, yv_t).item() * len(yv)
        acc += ((prob > 0.5) == yv).mean() * len(yv)
        try: auc_i = roc_auc_score(yv, prob)
        except ValueError: auc_i = float("nan")
        if not np.isnan(auc_i): auc += auc_i * len(yv)
        tot += len(yv)

    log.info(f"Round {rd:02d} | rec {rec/tot:.4f} | cls {cls/tot:.4f} "
             f"| acc {acc/tot:.3f} | auc {auc/tot:.3f}")

log.info("Finished.")

