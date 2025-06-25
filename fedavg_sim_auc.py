#!/usr/bin/env python
# fedavg_sim_auc.py
# -------------------------------------------------------------
# Pure-PyTorch Federated Learning (FedAvg) demo for
# Give-Me-Some-Credit (Kaggle) – log AUC, ACC, Loss per client.
#
# No Flower, no third-party FL libraries.
# -------------------------------------------------------------

import warnings, random, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.pipeline   import Pipeline
from sklearn.impute     import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 0 ───────────── Config
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

N_CLIENTS     = 10
NUM_ROUNDS    = 20
LOCAL_EPOCHS  = 2
LR            = 1e-3
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

def split_noniid(X,y,n=10, minority=(0.05,0.25)):
    pos, neg = np.where(y==1)[0], np.where(y==0)[0]
    np.random.shuffle(pos); np.random.shuffle(neg)
    pos_chunks = np.array_split(pos,n); neg_chunks = np.array_split(neg,n)
    out=[]
    for i in range(n):
        pct = np.random.uniform(*minority)
        need = int(pct*(len(pos_chunks[i])+len(neg_chunks[i])))
        idx  = np.concatenate([pos_chunks[i][:need], neg_chunks[i]])
        Xtr,Xte,ytr,yte = train_test_split(
            X[idx],y[idx],test_size=0.2,stratify=y[idx],random_state=SEED)
        out.append(((Xtr,ytr),(Xte,yte)))
    return out

clients = split_noniid(X_all,y_all,N_CLIENTS)

# 2 ───────────── Model
class MLP(nn.Module):
    def __init__(self,d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64),   nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,x): return self.net(x)

global_model = MLP(len(FEATS)).to(DEVICE)

def get_params(model):
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]

def set_params(model, params):
    state = {k: torch.tensor(v) for k,v in zip(model.state_dict().keys(), params)}
    model.load_state_dict(state, strict=True)

# 3 ───────────── Federated loop
print(f"Device {DEVICE} | Clients {N_CLIENTS} | Rounds {NUM_ROUNDS}\n")

for rnd in range(1, NUM_ROUNDS+1):
    new_params, samples = [], []
    # --- local training
    for cid, ((Xt,yt),(Xv,yv)) in enumerate(clients):
        local = MLP(len(FEATS)).to(DEVICE)
        set_params(local, get_params(global_model))
        opt = torch.optim.Adam(local.parameters(), lr=LR)
        pos_w = torch.tensor([(yt==0).sum()/(yt==1).sum()]).to(DEVICE)
        crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        X_tensor = torch.tensor(Xt, device=DEVICE)
        y_tensor = torch.tensor(yt, device=DEVICE).unsqueeze(1)

        local.train()
        for _ in range(LOCAL_EPOCHS):
            opt.zero_grad()
            loss = crit(local(X_tensor), y_tensor)
            loss.backward(); opt.step()

        new_params.append(get_params(local))
        samples.append(len(yt))

    # --- FedAvg aggregation
    total = sum(samples)
    agg_params=[]
    for layer in zip(*new_params):
        agg_params.append(sum(w*n for w,n in zip(layer,samples)) / total)
    set_params(global_model, agg_params)

    # --- evaluation
    print(f"\nRound {rnd:2d}")
    print("Client |   loss |  acc |  auc |   n")
    gl_loss=gl_acc=gl_auc=0.0; total_eval=0
    for cid, ((Xt,yt),(Xv,yv)) in enumerate(clients):
        Xv_t=torch.tensor(Xv,device=DEVICE)
        yv_t=torch.tensor(yv,device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            logits=global_model(Xv_t)
            prob=torch.sigmoid(logits).cpu().numpy().squeeze()
        loss=nn.BCEWithLogitsLoss()(logits,yv_t).item()
        acc=((prob>0.5)==yv).mean()
        try:
            auc=roc_auc_score(yv,prob)
        except ValueError:                # single-class edge case
            auc=np.nan
        n=len(yv)
        print(f"  {cid:2d}   | {loss:.4f} | {acc:.3f} | {auc:.3f} | {n}")
        gl_loss+=loss*n; gl_acc+=acc*n
        if not np.isnan(auc): gl_auc+=auc*n
        total_eval+=n

    print(f"GLOBAL | {gl_loss/total_eval:.4f} | {gl_acc/total_eval:.3f} | {gl_auc/total_eval:.3f} | {total_eval}")
    print("-"*60)

print("Done.")

