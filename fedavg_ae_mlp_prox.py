#!/usr/bin/env python
# fedavg_ae_mlp_prox.py
# --------------------------------------------------------------
# FedAvg (momentum) – AutoEncoder + Classifier with improvements
# • BatchNorm + Dropout in encoder
# • Focal BCE + FedProx (μ = 0.01)
# • Class-balance via WeightedRandomSampler per-client
# • 10 clients non-IID, 60 round × 3 local-epoch
# • Logs (console + ae_prox_run.log): rec loss | cls loss | ACC | AUC
# --------------------------------------------------------------

import warnings, random, sys, logging
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ────────── logger ──────────
logging.basicConfig(level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("ae_prox_run.log", mode="w")])
log = logging.getLogger()

# ────────── cfg ──────────
warnings.filterwarnings("ignore")
SEED=42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLIENTS=10; ROUNDS=60; LOCAL_EPOCH=3
BATCH=512; LR=2e-3;
MOM_BETA=0.9; MU_PROX=0.01
ALPHA=0.5                 # weight MSE in total loss
GAMMA_FOCAL=2

# ────────── data ──────────
FEATS=["RevolvingUtilizationOfUnsecuredLines","age",
       "NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome",
       "NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
       "NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse",
       "NumberOfDependents"]; LABEL="SeriousDlqin2yrs"
df=pd.read_csv("cs-training.csv",index_col=0)

pipe=Pipeline([("imp",SimpleImputer(strategy="median")),
               ("sc",StandardScaler())])
X=pipe.fit_transform(df[FEATS]).astype(np.float32)
y=df[LABEL].astype(np.float32).values

def split_noniid(X,y,n=10,minor=(0.08,0.18)):
    pos,neg=np.where(y==1)[0],np.where(y==0)[0]
    np.random.shuffle(pos); np.random.shuffle(neg)
    sl=[]
    for p,ne in zip(np.array_split(pos,n),np.array_split(neg,n)):
        need=int(np.random.uniform(*minor)*(len(p)+len(ne)))
        idx=np.concatenate([p[:need], ne])
        Xtr,Xte,ytr,yte=train_test_split(X[idx],y[idx],test_size=0.2,
                                         stratify=y[idx],random_state=SEED)
        sl.append(((Xtr,ytr),(Xte,yte)))
    return sl
clients=split_noniid(X,y,N_CLIENTS)

# ────────── model ──────────
class AE_MLP(nn.Module):
    def __init__(self,d,bottleneck=64):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Linear(d,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128,bottleneck)
        )
        self.dec=nn.Sequential(
            nn.Linear(bottleneck,128), nn.ReLU(),
            nn.Linear(128,256), nn.ReLU(),
            nn.Linear(256,d)
        )
        self.cls=nn.Linear(bottleneck,1)
    def forward(self,x):
        z=self.enc(x)
        return self.dec(z), self.cls(z)

global_model=AE_MLP(len(FEATS)).to(DEVICE)
gp=lambda m:[p.detach().cpu().numpy() for p in m.state_dict().values()]
def sp(m,ps): m.load_state_dict({k:torch.tensor(v) for k,v in zip(m.state_dict(),ps)},strict=True)

# ────────── losses ──────────
class FocalBCE(nn.Module):
    def __init__(self,gamma=2,pos_weight=None):
        super().__init__(); self.g=gamma; self.w=pos_weight
    def forward(self,logit,target):
        ce=nn.functional.binary_cross_entropy_with_logits(
            logit,target,weight=self.w,reduction="none")
        pt=torch.exp(-ce)
        return ((1-pt)**self.g * ce).mean()

mse=nn.MSELoss()
momentum=None
log.info(f"Device {DEVICE} | AE-MLP+FedProx | {ROUNDS}r × {LOCAL_EPOCH}e\n")

# ────────── federated loop ──────────
for rd in range(1,ROUNDS+1):
    new,sizes=[],[]
    for (Xt,yt),_ in clients:
        local=AE_MLP(len(FEATS)).to(DEVICE); sp(local,gp(global_model))
        pos_w=torch.tensor([(yt==0).sum()/(yt==1).sum()],
                           dtype=torch.float32,device=DEVICE)
        focal=FocalBCE(GAMMA_FOCAL,pos_weight=pos_w)
        opt=torch.optim.AdamW(local.parameters(),lr=LR,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=LOCAL_EPOCH)

        # class-balanced sampler
        weights=np.where(yt==1,0.7,0.3)
        sampler=WeightedRandomSampler(weights,len(yt),replacement=True)
        dl=DataLoader(TensorDataset(torch.tensor(Xt),torch.tensor(yt).unsqueeze(1)),
                      batch_size=BATCH,sampler=sampler)

        local.train()
        for _ in range(LOCAL_EPOCH):
            for xb,yb in dl:
                xb,yb=xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                xr,logit=local(xb)
                loss=ALPHA*mse(xr,xb)+focal(logit,yb)
                # FedProx
                prox=0.0
                for w,w0 in zip(local.parameters(),global_model.parameters()):
                    prox+=(w-w0.to(DEVICE)).pow(2).sum()
                loss=loss + MU_PROX/2*prox
                loss.backward(); opt.step()
            sch.step()

        new.append(gp(local)); sizes.append(len(yt))

    # FedAvg + momentum
    tot=sum(sizes)
    avg=[sum(w*n for w,n in zip(layer,sizes))/tot for layer in zip(*new)]
    if momentum is None:
        momentum=[np.zeros_like(p) for p in avg]
    cur=gp(global_model); upd=[]
    for i,(g,n) in enumerate(zip(cur,avg)):
        momentum[i]=MOM_BETA*momentum[i]+(1-MOM_BETA)*(n-g)
        upd.append(g+momentum[i])
    sp(global_model,upd)

    # evaluation
    rec=cls=acc=auc=0.0; tot=0
    for (Xt,yt),(Xv,yv) in clients:
        Xv_t=torch.tensor(Xv,device=DEVICE)
        yv_t=torch.tensor(yv,device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            xr,logit=global_model(Xv_t)
            prob=torch.sigmoid(logit).cpu().numpy().squeeze()
            rec_l=mse(xr,Xv_t).item()
            cls_l=focal(logit,yv_t).item()
        a=((prob>0.5)==yv).mean()
        try: au=roc_auc_score(yv,prob)
        except ValueError: au=float("nan")
        n=len(yv)
        rec+=rec_l*n; cls+=cls_l*n; acc+=a*n
        auc+=0 if np.isnan(au) else au*n; tot+=n

    log.info(f"Round {rd:02d} | rec {rec/tot:.4f} | cls {cls/tot:.4f} "
             f"| acc {acc/tot:.3f} | auc {auc/tot:.3f}")

log.info("Finished.")

