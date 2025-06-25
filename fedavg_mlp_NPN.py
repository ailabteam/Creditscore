#!/usr/bin/env python
# fedavg_mlp_NPN.py
# --------------------------------------------------------------
# FedAvg + FedProx  –  MLP classifier với Noisy Population Norm
# • 10 clients non-IID (Give-Me-Some-Credit) – 60 round × 3 epoch
# • NPN: noise_beta = 0.05, var-clamp, grad-clip 5
# • Log → console + mlp_NPN_run.log
# --------------------------------------------------------------

import warnings, random, sys, logging, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ───────────────────── logger ─────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("mlp_NPN_run.log", mode="w")])
log = logging.getLogger()

# ───────────────────── config ─────────────────────
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
CLIP_NORM   = 5.0
NOISE_BETA  = 0.05         # σ = β·√var

# ────────── Population & Noisy PN ──────────
class PN1d(nn.Module):
    def __init__(self,n,eps=1e-4,momentum=0.1,affine=True):
        super().__init__(); self.eps,self.m=eps,momentum
        self.register_buffer("mean",torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        if affine:
            self.weight=nn.Parameter(torch.ones(n)); self.bias=nn.Parameter(torch.zeros(n))
        else: self.weight=self.bias=None
    def _upd(self,x):
        m_b=x.mean(0).detach()
        v_b=x.var(0,unbiased=False).detach()+self.eps
        with torch.no_grad():
            self.mean.mul_(1-self.m).add_(self.m*m_b)
            self.var .mul_(1-self.m).add_(self.m*v_b).clamp_(min=self.eps)
    def _norm(self,x): return (x-self.mean)/torch.sqrt(self.var+self.eps)
    def forward(self,x):
        if self.training: self._upd(x)
        y=self._norm(x)
        if self.weight is not None: y=y*self.weight+self.bias
        return y

class NPN1d(PN1d):
    def __init__(self,n,noise_beta=0.05,**kw):
        super().__init__(n,**kw); self.beta=noise_beta
    def forward(self,x):
        y=super().forward(x)
        if self.training and self.beta>0:
            std=torch.sqrt(y.var(0,unbiased=False)+self.eps)
            y = y + torch.randn_like(y)*(self.beta*std)
        return y

# ────────── data prep ──────────
FEATS=[
 "RevolvingUtilizationOfUnsecuredLines","age",
 "NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome",
 "NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
 "NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse",
 "NumberOfDependents"
]; LABEL="SeriousDlqin2yrs"

df=pd.read_csv("cs-training.csv",index_col=0)
X=Pipeline([("imp",SimpleImputer(strategy="median")),
            ("sc",StandardScaler())]).fit_transform(df[FEATS]).astype(np.float32)
y=df[LABEL].astype(np.float32).values

def split_noniid(X,y,k=10,minor=(0.08,0.18)):
    pos,neg=np.where(y==1)[0],np.where(y==0)[0]
    np.random.shuffle(pos); np.random.shuffle(neg)
    out=[]
    for p,ne in zip(np.array_split(pos,k),np.array_split(neg,k)):
        need=int(np.random.uniform(*minor)*(len(p)+len(ne)))
        idx=np.concatenate([p[:need], ne])
        Xtr,Xte,ytr,yte=train_test_split(X[idx],y[idx],test_size=0.2,
                                         stratify=y[idx],random_state=SEED)
        out.append(((Xtr,ytr),(Xte,yte)))
    return out
clients=split_noniid(X,y,N_CLIENTS)

# ────────── MLP with NPN ──────────
class MLP_NPN(nn.Module):
    def __init__(self,d_in, hidden=(256,128)):
        super().__init__(); layers=[]; cur=d_in
        for h in hidden:
            layers += [nn.Linear(cur,h), NPN1d(h,noise_beta=NOISE_BETA), nn.ReLU()]
            cur=h
        layers.append(nn.Linear(cur,1))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

global_model=MLP_NPN(len(FEATS)).to(DEVICE)
getP=lambda m:[p.detach().cpu().numpy() for p in m.state_dict().values()]
def setP(m,ps): m.load_state_dict({k:torch.tensor(v) for k,v in zip(m.state_dict(),ps)},strict=True)

bce=nn.BCEWithLogitsLoss()
momentum=None
log.info(f"Device {DEVICE} | MLP + NPN β={NOISE_BETA} | 60r×3e\n")

# ────────── Fed loop ──────────
for rd in range(1,ROUNDS+1):
    new,sizes=[],[]
    # --- local ---
    for (Xt,yt),_ in clients:
        local=MLP_NPN(len(FEATS)).to(DEVICE); setP(local,getP(global_model))
        opt=torch.optim.AdamW(local.parameters(),lr=LR,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=LOCAL_EPOCH)

        sampler=WeightedRandomSampler(np.where(yt==1,0.7,0.3),len(yt),True)
        dl=DataLoader(TensorDataset(torch.tensor(Xt),torch.tensor(yt).unsqueeze(1)),
                      batch_size=BATCH_SIZE,sampler=sampler)

        local.train()
        for _ in range(LOCAL_EPOCH):
            for xb,yb in dl:
                xb,yb=xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logit=local(xb)
                loss=bce(logit,yb)
                prox=sum((w-w0.detach().to(DEVICE)).pow(2).sum()
                         for w,w0 in zip(local.parameters(),global_model.parameters()))
                (loss+MU_PROX/2*prox).backward()
                nn.utils.clip_grad_norm_(local.parameters(),CLIP_NORM)
                opt.step()
            sch.step()
        new.append(getP(local)); sizes.append(len(yt))

    # --- FedAvg + momentum ---
    tot=sum(sizes)
    avg=[sum(w*n for w,n in zip(layer,sizes))/tot for layer in zip(*new)]
    if momentum is None: momentum=[np.zeros_like(p) for p in avg]
    cur=getP(global_model); upd=[]
    for i,(g,n) in enumerate(zip(cur,avg)):
        momentum[i]=MOM_BETA*momentum[i]+(1-MOM_BETA)*(n-g)
        upd.append(g+momentum[i])
    setP(global_model,upd)

    # --- eval ---
    loss_sum=acc_sum=auc_sum=0.0; total=0
    for (Xt,yt),(Xv,yv) in clients:
        Xv_t=torch.tensor(Xv,device=DEVICE); yv_t=torch.tensor(yv,device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            logit=global_model(Xv_t)
            prob=torch.sigmoid(logit).cpu().numpy().squeeze()
            loss_sum += bce(logit,yv_t).item()*len(yv)
        acc_sum += ((prob>0.5)==yv).mean()*len(yv)
        try: auc_i=roc_auc_score(yv,prob)
        except ValueError: auc_i=float("nan")
        if not np.isnan(auc_i): auc_sum += auc_i*len(yv)
        total += len(yv)

    log.info(f"Round {rd:02d} | loss {loss_sum/total:.4f} "
             f"| acc {acc_sum/total:.3f} | auc {auc_sum/total:.3f}")

log.info("Finished.")

