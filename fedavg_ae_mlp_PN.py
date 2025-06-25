#!/usr/bin/env python
# fedavg_ae_mlp_PN.py  –  FIXED (detach buffers)

import warnings, random, sys, logging
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ─── logger ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("ae_PN_run.log", mode="w")])
log = logging.getLogger()

# ─── config ───────────────────────────────────────────────────
warnings.filterwarnings("ignore")
SEED=42
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
GAMMA_FOCAL = 2

# ─── Population Normalization (DETACH) ───────────────────────
class PN1d(nn.Module):
    def __init__(self, n, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.eps, self.m = eps, momentum
        self.register_buffer("pop_mean", torch.zeros(n))
        self.register_buffer("pop_var",  torch.ones(n))
        if affine:
            self.weight = nn.Parameter(torch.ones(n))
            self.bias   = nn.Parameter(torch.zeros(n))
        else:
            self.weight = self.bias = None

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(0).detach()
            batch_var  = x.var(0, unbiased=False).detach()
            # update buffers w/o grad
            with torch.no_grad():
                self.pop_mean.mul_(1-self.m).add_(self.m*batch_mean)
                self.pop_var .mul_(1-self.m).add_(self.m*batch_var)
        out = (x - self.pop_mean) / torch.sqrt(self.pop_var + self.eps)
        if self.weight is not None:
            out = out * self.weight + self.bias
        return out

# ─── Focal BCE ───────────────────────────────────────────────
class FocalBCE(nn.Module):
    def __init__(self, gamma=2, pos_weight=None):
        super().__init__(); self.g,self.w = gamma,pos_weight
    def forward(self, l, t):
        ce = nn.functional.binary_cross_entropy_with_logits(l, t,
                weight=self.w, reduction="none")
        pt = torch.exp(-ce)
        return ((1-pt)**self.g * ce).mean()

# ─── data ────────────────────────────────────────────────────
FEATS=[ "RevolvingUtilizationOfUnsecuredLines","age",
        "NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents"]; LABEL="SeriousDlqin2yrs"

df=pd.read_csv("cs-training.csv",index_col=0)
X=Pipeline([("imp",SimpleImputer(strategy="median")),
            ("sc",StandardScaler())]).fit_transform(df[FEATS]).astype(np.float32)
y=df[LABEL].astype(np.float32).values

def split_noniid(X,y,k=10,minor=(0.08,0.18)):
    pos,neg=np.where(y==1)[0],np.where(y==0)[0]
    np.random.shuffle(pos); np.random.shuffle(neg)
    clients=[]
    for p,ne in zip(np.array_split(pos,k),np.array_split(neg,k)):
        need=int(np.random.uniform(*minor)*(len(p)+len(ne)))
        idx=np.concatenate([p[:need], ne])
        Xtr,Xte,ytr,yte=train_test_split(X[idx],y[idx],test_size=0.2,
                                         stratify=y[idx],random_state=SEED)
        clients.append(((Xtr,ytr),(Xte,yte)))
    return clients
clients=split_noniid(X,y,N_CLIENTS)

# ─── model ───────────────────────────────────────────────────
class AE_MLP(nn.Module):
    def __init__(self,d,bottle=64):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Linear(d,256), PN1d(256), nn.ReLU(),
            nn.Linear(256,128), PN1d(128), nn.ReLU(),
            nn.Linear(128,bottle)
        )
        self.dec=nn.Sequential(
            nn.Linear(bottle,128), nn.ReLU(),
            nn.Linear(128,256),    nn.ReLU(),
            nn.Linear(256,d)
        )
        self.cls=nn.Linear(bottle,1)
    def forward(self,x):
        z=self.enc(x)
        return self.dec(z), self.cls(z)

global_model=AE_MLP(len(FEATS)).to(DEVICE)
getP=lambda m:[p.detach().cpu().numpy() for p in m.state_dict().values()]
def setP(m,ps): m.load_state_dict(
    {k:torch.tensor(v) for k,v in zip(m.state_dict(),ps)}, strict=True)

mse = nn.MSELoss()
momentum=None
log.info(f"Device {DEVICE} | AE-MLP + PN (fix) | {ROUNDS}r × {LOCAL_EPOCH}e\n")

# ─── Federated loop ──────────────────────────────────────────
for rd in range(1,ROUNDS+1):
    new,sizes=[],[]
    for (Xt,yt),_ in clients:
        local=AE_MLP(len(FEATS)).to(DEVICE); setP(local,getP(global_model))
        pos_w=torch.tensor([(yt==0).sum()/(yt==1).sum()],device=DEVICE)
        focal=FocalBCE(GAMMA_FOCAL,pos_w)

        opt=torch.optim.AdamW(local.parameters(),lr=LR,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=LOCAL_EPOCH)

        sampler=WeightedRandomSampler(np.where(yt==1,0.7,0.3),len(yt),True)
        dl=DataLoader(TensorDataset(torch.tensor(Xt),
                                    torch.tensor(yt).unsqueeze(1)),
                      batch_size=BATCH_SIZE,sampler=sampler)

        local.train()
        for _ in range(LOCAL_EPOCH):
            for xb,yb in dl:
                xb,yb=xb.to(DEVICE),yb.to(DEVICE)
                opt.zero_grad()
                xr,logit=local(xb)
                loss = ALPHA*mse(xr,xb)+focal(logit,yb)
                # FedProx (detach global params)
                prox=sum((w - w0.detach().to(DEVICE)).pow(2).sum()
                         for w,w0 in zip(local.parameters(),global_model.parameters()))
                (loss + MU_PROX/2*prox).backward(); opt.step()
            sch.step()

        new.append(getP(local)); sizes.append(len(yt))

    # FedAvg + momentum
    tot=sum(sizes)
    avg=[sum(w*n for w,n in zip(layer,sizes))/tot for layer in zip(*new)]
    if momentum is None: momentum=[np.zeros_like(p) for p in avg]
    upd=[g + (MOM_BETA*mb + (1-MOM_BETA)*(n-g))
         for g,n,mb in zip(getP(global_model),avg,momentum)]
    momentum=[MOM_BETA*mb + (1-MOM_BETA)*(n-g)
              for mb,g,n in zip(momentum,getP(global_model),avg)]
    setP(global_model,upd)

    # evaluation
    rec=cls=acc=auc=0.0; total=0
    for (Xt,yt),(Xv,yv) in clients:
        Xv_t=torch.tensor(Xv,device=DEVICE); yv_t=torch.tensor(yv,device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            xr,logit=global_model(Xv_t)
            prob=torch.sigmoid(logit).cpu().numpy().squeeze()
            rec+=mse(xr,Xv_t).item()*len(yv)
            cls+=focal(logit,yv_t).item()*len(yv)
        acc+=((prob>0.5)==yv).mean()*len(yv)
        try: auc_i=roc_auc_score(yv,prob)
        except ValueError: auc_i=float("nan")
        if not np.isnan(auc_i): auc+=auc_i*len(yv)
        total+=len(yv)

    log.info(f"Round {rd:02d} | rec {rec/total:.4f} | cls {cls/total:.4f} "
             f"| acc {acc/total:.3f} | auc {auc/total:.3f}")

log.info("Finished.")

