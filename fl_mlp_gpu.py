#!/usr/bin/env python
# fl_mlp_gpu.py – Federated Learning (Flower) với MLP, log từng client
# --------------------------------------------------------------------
import os, random, numpy as np, pandas as pd, torch, torch.nn as nn
import flwr as fl
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ────────────────────────── 1. Load & preprocess GMSC
FEATS = [
    "RevolvingUtilizationOfUnsecuredLines","age",
    "NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]
LABEL = "SeriousDlqin2yrs"

def load_gmsc(path="cs-training.csv"):
    df = pd.read_csv(path, index_col=0)
    pipe = Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
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
        need_pos = int(pct*(len(pos_chunks[i])+len(neg_chunks[i])))
        idx_i = np.concatenate([pos_chunks[i][:need_pos], neg_chunks[i]])
        X_i, y_i = X[idx_i], y[idx_i]
        X_tr,X_te,y_tr,y_te = train_test_split(
            X_i,y_i,test_size=0.2,stratify=y_i,random_state=SEED)
        slices.append(((X_tr,y_tr),(X_te,y_te)))
    return slices

# ────────────────────────── 2. MLP model
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64),   nn.ReLU(),
            nn.Linear(64,1)               # logits
        )
    def forward(self,x): return self.net(x)

# ────────────────────────── 3. Flower client
class CreditClient(fl.client.NumPyClient):
    def __init__(self, cid, train, test, d_in):
        self.cid   = cid
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(d_in).to(self.device)
        self.X_train, self.y_train = [torch.tensor(a, dtype=torch.float32).to(self.device)
                                      for a in train]
        self.X_test,  self.y_test  = [torch.tensor(a, dtype=torch.float32).to(self.device)
                                      for a in test]

        pos_w = torch.tensor([(self.y_train==0).sum()/(self.y_train==1).sum()]).to(self.device)
        self.crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        self.opt  = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    # Flower API
    def get_parameters(self, config):       # -> List[np.ndarray]
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, params):       # List[np.ndarray] -> model
        state = dict(zip(self.model.state_dict().keys(),
                         [torch.tensor(p) for p in params]))
        self.model.load_state_dict(state, strict=True)

    def fit(self, params, config):
        self.set_parameters(params)
        self.model.train()
        for _ in range(config["local_epochs"]):
            self.opt.zero_grad()
            logits = self.model(self.X_train)
            loss   = self.crit(logits, self.y_train.unsqueeze(1))
            loss.backward(); self.opt.step()
        return self.get_parameters(config), len(self.y_train), {}

    def evaluate(self, params, config):
        self.set_parameters(params)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_test)
            loss   = self.crit(logits, self.y_test.unsqueeze(1)).item()
            prob   = torch.sigmoid(logits).cpu().numpy()
            acc    = ((prob>0.5)==self.y_test.cpu().numpy()).mean()
        # metrics dict
        return loss, len(self.y_test), {"accuracy": acc}

# ────────────────────────── 4. Custom FedAvg to print per-client
class FedAvgPrint(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        print(f"\n─── Round {rnd} – client metrics ───")
        for client, res in results:
            loss, num, metrics = res
            print(f"Client {client.cid:>2}: "
                  f"loss={loss:.4f}  acc={metrics['accuracy']:.4f}  n={num}")
        # call parent to get aggregated loss/metrics
        agg = super().aggregate_evaluate(rnd, results, failures)
        if agg is not None:
            loss, metrics = agg
            print(f" ⇒ Aggregated: loss={loss:.4f}  acc={metrics['accuracy']:.4f}")
        return agg

# ────────────────────────── 5. Run simulation
def main():
    X, y, _ = load_gmsc()
    slices  = split_non_iid(X, y, n_clients=10)

    def client_fn(cid: str):
        cid_int = int(cid)
        (Xtr,ytr),(Xte,yte) = slices[cid_int]
        return CreditClient(cid_int, (Xtr,ytr), (Xte,yte), d_in=Xtr.shape[1])

    strategy = FedAvgPrint(
        fraction_fit=1.0, min_fit_clients=10,
        on_fit_config_fn=lambda rnd: {"local_epochs":2},
        evaluate_metrics_aggregation_fn=fl.server.strategy.aggregate
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(slices),
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # chọn GPU 0; xoá dòng này nếu không cần
    main()

