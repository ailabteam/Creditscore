#!/usr/bin/env python
# fl_mlp_gpu_clean.py – FL + MLP (PyTorch GPU) | log per-client, per-round
# Tested on Flower ≥ 1.8

import os, warnings, random, numpy as np, pandas as pd, torch, torch.nn as nn
import flwr as fl
from flwr.common import Context, EvaluateRes
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ─────────── 0.  GLOBAL SETTINGS ───────────
warnings.filterwarnings("ignore", category=UserWarning)   # ẩn warning không quan trọng
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ─────────── 1. DATA UTILS ───────────
FEATS = [
    "RevolvingUtilizationOfUnsecuredLines","age",
    "NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]
LABEL = "SeriousDlqin2yrs"

def load_gmsc(csv="cs-training.csv"):
    df = pd.read_csv(csv, index_col=0)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    X = pipe.fit_transform(df[FEATS])
    y = df[LABEL].astype(np.float32).values
    return X, y, pipe

def split_non_iid(X, y, n_clients=10, minority=(0.05,0.25)):
    idx_pos, idx_neg = np.where(y==1)[0], np.where(y==0)[0]
    np.random.shuffle(idx_pos); np.random.shuffle(idx_neg)
    pos_chunks = np.array_split(idx_pos, n_clients)
    neg_chunks = np.array_split(idx_neg, n_clients)
    slices=[]
    for i in range(n_clients):
        pct = np.random.uniform(*minority)
        need_pos = int(pct * (len(pos_chunks[i])+len(neg_chunks[i])))
        idx_i = np.concatenate([pos_chunks[i][:need_pos], neg_chunks[i]])
        X_i, y_i = X[idx_i], y[idx_i]
        X_tr,X_te,y_tr,y_te = train_test_split(
            X_i,y_i,test_size=0.2,stratify=y_i,random_state=SEED)
        slices.append(((X_tr,y_tr),(X_te,y_te)))
    return slices

# ─────────── 2.  MODEL ───────────
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64),   nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,x): return self.net(x)

# ─────────── 3.  FLOWER CLIENT ───────────
class CreditClient(fl.client.Client):          # dùng Client (API mới)
    def __init__(self, cid, train, test, d_in):
        self.cid = cid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = MLP(d_in).to(self.device)

        self.Xtr, self.ytr = [torch.tensor(a, dtype=torch.float32).to(self.device)
                              for a in train]
        self.Xte, self.yte = [torch.tensor(a, dtype=torch.float32).to(self.device)
                              for a in test]

        pos_weight = torch.tensor([(self.ytr==0).sum() / (self.ytr==1).sum()]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    # • get_parameters / set_parameters follow new Client API
    def get_parameters(self, config):
        return [p.cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, params):
        keys = list(self.model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, params)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, params, config):
        self.set_parameters(params)
        self.model.train()
        for _ in range(config["local_epochs"]):
            self.optimizer.zero_grad()
            logits = self.model(self.Xtr)
            loss   = self.criterion(logits, self.ytr.unsqueeze(1))
            loss.backward(); self.optimizer.step()
        return self.get_parameters(config), len(self.ytr), {}

    def evaluate(self, params, config):
        self.set_parameters(params)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.Xte)
            loss   = self.criterion(logits, self.yte.unsqueeze(1)).item()
            acc    = ((torch.sigmoid(logits)>0.5)==self.yte.unsqueeze(1)).float().mean().item()
        return float(loss), len(self.yte), {"accuracy": acc}

# ─────────── 4.  CUSTOM STRATEGY (log per-client) ───────────
class FedAvgPrint(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        print(f"\n─── Round {rnd} – client metrics ───")
        for res in results:                # res is EvaluateRes
            cid      = res.client.cid
            loss     = res.loss
            acc      = res.metrics["accuracy"]
            n        = res.num_examples
            print(f"Client {cid:>2}: loss={loss:.4f}  acc={acc:.4f}  n={n}")
        agg = super().aggregate_evaluate(rnd, results, failures)
        if agg:  # tuple(loss, metrics)
            loss, metrics = agg
            print(f" ⇒ Aggregated: loss={loss:.4f}  acc={metrics['accuracy']:.4f}")
        return agg

# ─────────── 5.  MAIN ───────────
def main():
    X, y, _ = load_gmsc()
    slices  = split_non_iid(X, y, n_clients=10)
    d_in    = X.shape[1]

    def client_fn(ctx: Context):
        cid_int = int(ctx.client_id)
        (Xtr,ytr),(Xte,yte) = slices[cid_int]
        return CreditClient(cid_int, (Xtr,ytr), (Xte,yte), d_in)

    strategy = FedAvgPrint(
        fraction_fit=1.0,
        min_fit_clients=10,
        min_available_clients=10,          # tránh warning
        on_fit_config_fn=lambda r: {"local_epochs":2},
        evaluate_metrics_aggregation_fn=fl.server.strategy.aggregate
    )

    fl.simulation.start_simulation(        # API vẫn dùng được (sẽ bỏ ở bản sau >=2.0)
        client_fn=client_fn,
        num_clients=len(slices),
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )

if __name__ == "__main__":
    # nếu muốn ép dùng GPU 0:  os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()

