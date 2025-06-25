"""
AI‑Driven Credit Scoring Using Federated Learning
-------------------------------------------------
PyTorch + Flower implementation of the experimental setup described in the
paper "AI‑Driven Credit Scoring Using Federated Learning for Privacy‑Preserving
Microfinance".

Key features
~~~~~~~~~~~~
* Supports German Credit (UCI) and Give‑Me‑Some‑Credit (Kaggle) datasets.
* Simulates heterogeneous clients (non‑IID partitions) with configurable size.
* Implements the three deep‑learning models evaluated in the paper:
    - Multi‑Layer Perceptron (MLP)
    - TabNet (via ``pytorch‑tabnet``)
    - Autoencoder + MLP classifier
* Provides an XGBoost baseline (centralised) for quick comparison.
* Integrates differential privacy (Gaussian mechanism) by adding noise to model
  parameters before they are returned to the server.
* Uses Flower (>=1.8) as the FL orchestration layer, but can be swapped for
  PySyft with minimal effort.

Installation
~~~~~~~~~~~~
```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio flwr scikit‑learn pandas numpy pytorch‑tabnet xgboost tqdm
```

Usage examples
~~~~~~~~~~~~~~
```bash
# Federated MLP on German Credit (5 clients, 50 rounds)
python federated_credit_scoring.py --model mlp --dataset german --n_clients 5 --rounds 50

# TabNet with differential privacy on Give‑Me‑Some‑Credit (10 clients)
python federated_credit_scoring.py --model tabnet --dataset gmsc --n_clients 10 \
                                  --rounds 50 --local_epochs 3 --dp_sigma 0.05

# Centralised XGBoost baseline on German Credit
python federated_credit_scoring.py --model xgb --dataset german --centralised
```
"""
from __future__ import annotations

import argparse
import copy
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Flower imports (server & client)
import flwr as fl

# Optional import for TabNet (install with pip install pytorch‑tabnet)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None  # handled later

# ----------------------------------------------------------------------------------
# 1. Model definitions
# ----------------------------------------------------------------------------------
class MLP(nn.Module):
    """3‑layer MLP used in the paper."""

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class AutoencoderClassifier(nn.Module):
    """Encoder part followed by a classifier as described in the paper."""

    def __init__(self, in_features: int):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Decoder (unused for classification but helps representation learning)
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, in_features),
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        out = self.classifier(z)
        return out, recon


# ----------------------------------------------------------------------------------
# 2. Dataset helpers
# ----------------------------------------------------------------------------------

def load_german_credit() -> Tuple[np.ndarray, np.ndarray]:
    """Load and return (X, y) for the German Credit dataset (openml id=31)."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    X = pd.get_dummies(data.data, drop_first=True).values.astype(np.float32)
    y = (data.target == "bad").astype(np.float32).values  # 1=default/bad credit
    return X, y


def load_gmsc(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Give‑Me‑Some‑Credit dataset from a local CSV (download from Kaggle)."""
    df = pd.read_csv(csv_path)
    # The label column in GMSC is "SeriousDlqin2yrs": 1 indicates default within 2 yrs
    y = df["SeriousDlqin2yrs"].values.astype(np.float32)
    X = df.drop(columns=["SeriousDlqin2yrs"]).values.astype(np.float32)
    # Replace NaNs with column median
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    return X, y


def partition_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition the dataset into *non‑IID* shards by sorting by label then snaking."""
    if shuffle:
        rng_state = np.random.get_state()
        perm = np.random.permutation(len(y))
        X, y = X[perm], y[perm]
        np.random.set_state(rng_state)

    # Sort by label to create label‑skew, then split in order
    order = np.argsort(y)
    X, y = X[order], y[order]
    shard_size = len(y) // n_clients
    shards = [
        (X[i * shard_size : (i + 1) * shard_size], y[i * shard_size : (i + 1) * shard_size])
        for i in range(n_clients)
    ]
    return shards


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    train: bool = True,
) -> DataLoader:
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y).unsqueeze(1)
    ds = TensorDataset(X_tensor, y_tensor)
    shuffle = train
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ----------------------------------------------------------------------------------
# 3. Differential‑privacy helper
# ----------------------------------------------------------------------------------

def add_gaussian_noise(parameters: List[np.ndarray], sigma: float) -> List[np.ndarray]:
    """Return parameters + N(0, sigma^2)."""
    noisy = [p + np.random.normal(loc=0.0, scale=sigma, size=p.shape) for p in parameters]
    return noisy


# ----------------------------------------------------------------------------------
# 4. Flower client implementation
# ----------------------------------------------------------------------------------
class CreditClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        dp_sigma: float | None = None,
        ae_loss_weight: float = 0.3,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dp_sigma = dp_sigma
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.ae_loss_weight = ae_loss_weight  # only used for autoencoder model

    # --- Flower required methods --------------------------------------------------
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = self.model.state_dict()
        for key, np_val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(np_val)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(config.get("local_epochs", 1)):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                if isinstance(self.model, AutoencoderClassifier):
                    preds, recon = self.model(X_batch)
                    loss_cls = self.criterion(preds, y_batch)
                    loss_recon = nn.functional.mse_loss(recon, X_batch)
                    loss = loss_cls + self.ae_loss_weight * loss_recon
                else:
                    preds = self.model(X_batch)
                    loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()

        updated_parameters = self.get_parameters()
        if self.dp_sigma is not None and self.dp_sigma > 0:
            updated_parameters = add_gaussian_noise(updated_parameters, self.dp_sigma)
        return updated_parameters, len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                if isinstance(outputs, tuple):  # autoencoder
                    outputs = outputs[0]
                y_true.append(y_batch.numpy())
                y_pred.append(outputs.cpu().numpy())
        y_true = np.concatenate(y_true).ravel()
        y_pred = np.concatenate(y_pred).ravel()
        acc = accuracy_score(y_true, y_pred > 0.5)
        auc = roc_auc_score(y_true, y_pred)
        return float(acc), len(self.val_loader.dataset), {"auc": float(auc)}


# ----------------------------------------------------------------------------------
# 5. Centralised XGBoost baseline (optional)
# ----------------------------------------------------------------------------------

def run_xgboost(X: np.ndarray, y: np.ndarray):
    import xgboost as xgb

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="auc",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred > 0.5)
    auc = roc_auc_score(y_test, y_pred)
    print(f"Centralised XGBoost  | Accuracy: {acc:.4f} | AUC: {auc:.4f}")


# ----------------------------------------------------------------------------------
# 6. Federated simulation entry point
# ----------------------------------------------------------------------------------

def run_federated(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    rounds: int,
    local_epochs: int,
    dp_sigma: float | None = None,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature scaling (global scaler for all clients)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Create non‑IID shards
    shards = partition_non_iid(X, y, n_clients)

    # Validation split inside each client (~20%)
    clients = []
    for Xi, yi in shards:
        Xi_train, Xi_val, yi_train, yi_val = train_test_split(
            Xi, yi, test_size=0.2, random_state=42, stratify=yi
        )
        train_loader = make_loader(Xi_train, yi_train, batch_size, train=True)
        val_loader = make_loader(Xi_val, yi_val, batch_size, train=False)

        if model_name == "mlp":
            model = MLP(in_features=Xi.shape[1])
        elif model_name == "autoencoder":
            model = AutoencoderClassifier(in_features=Xi.shape[1])
        elif model_name == "tabnet":
            assert (
                TabNetClassifier is not None
            ), "pytorch‑tabnet is not installed. Run pip install pytorch‑tabnet"
            model = TabNetClassifier(n_d=16, n_a=16, n_steps=3)  # wrapped below
        else:
            raise ValueError("Unknown model name")

        # For TabNet we wrap the fit/predict inside a small adapter
        if model_name == "tabnet":
            clients.append(
                TabNetClientAdapter(model, train_loader, val_loader, device, dp_sigma)
            )
        else:
            clients.append(
                CreditClient(model, train_loader, val_loader, device, dp_sigma)
            )

    # Flower simulation ----------------------------------------------------------------
    def client_fn(cid: str):
        return clients[int(cid)]

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # because dataset is small; every client participates
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_available_clients=n_clients,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs},
    )

    print(
        f"\nStarting simulation: {model_name.upper()}, {n_clients} clients, {rounds} rounds, DP sigma={dp_sigma}"
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": os.cpu_count(), "num_gpus": 0},
    )


# Adapter to make TabNet behave like a NumPyClient -----------------------------------
class TabNetClientAdapter(fl.client.NumPyClient):
    def __init__(
        self,
        tabnet: "TabNetClassifier",
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        dp_sigma: float | None = None,
    ):
        self.tabnet = tabnet
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dp_sigma = dp_sigma
        self.device = device
        self._init_params()

    def _init_params(self):
        # Extract numpy parameters (TabNet keeps them in .network.state_dict)
        self._state_keys = list(self.tabnet.network.state_dict().keys())

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.tabnet.network.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self._state_keys, parameters)
        }
        self.tabnet.network.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        X_train = torch.cat([batch[0] for batch in self.train_loader]).numpy()
        y_train = torch.cat([batch[1] for batch in self.train_loader]).numpy().ravel()
        self.tabnet.fit(X_train, y_train, max_epochs=config.get("local_epochs", 1), batch_size=1024, verbose=0)
        updated_parameters = self.get_parameters()
        if self.dp_sigma:
            updated_parameters = add_gaussian_noise(updated_parameters, self.dp_sigma)
        return updated_parameters, len(X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        X_val = torch.cat([batch[0] for batch in self.val_loader]).numpy()
        y_val = torch.cat([batch[1] for batch in self.val_loader]).numpy().ravel()
        preds = self.tabnet.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, preds > 0.5)
        auc = roc_auc_score(y_val, preds)
        return float(acc), len(X_val), {"auc": float(auc)}


# ----------------------------------------------------------------------------------
# 7. CLI entry
# ----------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Federated credit‑scoring experiments")
    p.add_argument("--model", choices=["mlp", "tabnet", "autoencoder", "xgb"], default="mlp")
    p.add_argument("--dataset", choices=["german", "gmsc"], default="german")
    p.add_argument("--gmsc_path", type=str, default="give_me_some_credit.csv")
    p.add_argument("--n_clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--dp_sigma", type=float, default=None, help="Gaussian noise stddev for DP")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--centralised", action="store_true", help="Run baseline XGBoost centrally")
    return p.parse_args()


def main():
    args = parse_args()

    # Load dataset ------------------------------------------------------------------
    if args.dataset == "german":
        X, y = load_german_credit()
    else:  # gmsc
        if not os.path.isfile(args.gmsc_path):
            raise FileNotFoundError(
                "Give‑Me‑Some‑Credit CSV not found. Download from Kaggle and pass --gmsc_path"
            )
        X, y = load_gmsc(args.gmsc_path)

    if args.centralised or args.model == "xgb":
        run_xgboost(X, y)
        return

    run_federated(
        model_name=args.model,
        X=X,
        y=y,
        n_clients=args.n_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        dp_sigma=args.dp_sigma,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

