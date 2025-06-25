# fedavg_sim.py ────────────────────────────────────────────────────────
"""
Pure-PyTorch Federated Learning demo (FedAvg) – GiveMeSomeCredit dataset
No Flower, no third-party FL libs.
"""

import warnings, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0. Settings ─────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

N_CLIENTS      = 10
NUM_ROUNDS     = 20
LOCAL_EPOCHS   = 2
BATCH_SIZE     = 1024           # toàn batch (dữ liệu nhỏ)
LR             = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load & preprocess GMSC ───────────────────────────────────────────
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
    ("imp",   SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
X_all = pipe.fit_transform(df[FEATS]).astype(np.float32)
y_all = df[LABEL].values.astype(np.float32)

# helper: split non-IID label skew
def split_noniid(X,y,n=10, minority=(0.05,0.25)):
    pos_idx, neg_idx = np.where(y==1)[0], np.where(y==0)[0]
    np.random.shuffle(pos_idx); np.random.shuffle(neg_idx)
    pos_chunks = np.array_split(pos_idx, n)
    neg_chunks = np.array_split(neg_idx, n)
    clients=[]
    for i in range(n):
        pct = np.random.uniform(*minority)
        need = int(pct*(len(pos_chunks[i])+len(neg_chunks[i])))
        idx  = np.concatenate([pos_chunks[i][:need], neg_chunks[i]])
        Xt,Xv,yt,yv = train_test_split(X[idx], y[idx],
                                       test_size=0.2, stratify=y[idx],
                                       random_state=SEED)
        clients.append(((Xt,yt),(Xv,yv)))
    return clients

clients = split_noniid(X_all, y_all, N_CLIENTS)

# 2. Model ────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self,d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64),   nn.ReLU(),
            nn.Linear(64,1)              # logits
        )
    def forward(self,x): return self.net(x)

global_model = MLP(len(FEATS)).to(DEVICE)

# util: get / set params as list[np.ndarray]
def get_params(model):  return [p.detach().cpu().numpy() for p in model.state_dict().values()]
def set_params(model, params):
    state = {k: torch.tensor(v) for k,v in zip(model.state_dict().keys(), params)}
    model.load_state_dict(state, strict=True)

# 3. Federated Training Loop ──────────────────────────────────────────
print(f"Device: {DEVICE} | Rounds {NUM_ROUNDS} | Clients {N_CLIENTS}\n")

for rnd in range(1, NUM_ROUNDS+1):
    new_params, num_examples = [], []
    # ----- client-side -----
    for cid, ((Xt,yt),(Xv,yv)) in enumerate(clients):
        # copy global → local
        local_model = MLP(len(FEATS)).to(DEVICE)
        set_params(local_model, get_params(global_model))
        opt = torch.optim.Adam(local_model.parameters(), lr=LR)
        pos_w = torch.tensor([(yt==0).sum()/(yt==1).sum()]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        Xt_t = torch.tensor(Xt, device=DEVICE)
        yt_t = torch.tensor(yt, device=DEVICE).unsqueeze(1)

        local_model.train()
        for _ in range(LOCAL_EPOCHS):
            opt.zero_grad()
            loss = criterion(local_model(Xt_t), yt_t)
            loss.backward(); opt.step()

        # store updated weights
        new_params.append(get_params(local_model))
        num_examples.append(len(yt))

    # ----- server aggregation FedAvg -----
    total = sum(num_examples)
    averaged = []
    for layer in zip(*new_params):                # iterate layer wise
        agg = sum(w*n for w,n in zip(layer,num_examples)) / total
        averaged.append(agg)
    set_params(global_model, averaged)

    # ----- evaluation per client -----
    global_model.eval()
    losses, accs = [], []
    for cid, ((Xt,yt),(Xv,yv)) in enumerate(clients):
        Xv_t = torch.tensor(Xv, device=DEVICE)
        yv_t = torch.tensor(yv, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            logits = global_model(Xv_t)
            loss   = nn.BCEWithLogitsLoss()(logits, yv_t).item()
            acc    = ((torch.sigmoid(logits)>0.5)==yv_t).float().mean().item()
        losses.append(loss); accs.append(acc)
        print(f"Rnd {rnd:2d} | Client {cid:2d} | loss {loss:.4f} | acc {acc:.4f}")

    print(f"Rnd {rnd:2d} | Global   | loss {np.mean(losses):.4f} | acc {np.mean(accs):.4f}")
    print("-"*60)

print("Finished.")

