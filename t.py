import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import flwr as fl
from opacus import PrivacyEngine
from collections import OrderedDict
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the German Credit Dataset
def load_german_credit_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = [
        "checking", "duration", "credit_history", "purpose", "credit_amount", "savings",
        "employment", "installment_rate", "personal_status", "other_debtors", "residence",
        "property", "age", "other_installment", "housing", "existing_credits", "job",
        "people_liable", "telephone", "foreign_worker", "credit_risk"
    ]
    data = pd.read_csv(url, sep=" ", header=None, names=columns)
    
    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=["object"]).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Features and labels
    X = data.drop("credit_risk", axis=1).values
    y = data["credit_risk"].values - 1  # Convert to 0 (good) and 1 (bad)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# Simulate non-i.i.d. data partitioning for 5 clients
def partition_data_non_iid(X, y, num_clients=5):
    # Sort data by a feature (e.g., credit_amount) to create non-i.i.d. partitions
    sorted_idx = np.argsort(X[:, 4])  # Assuming column 4 is credit_amount
    X_sorted = X[sorted_idx]
    y_sorted = y[sorted_idx]
    
    # Divide into num_clients partitions
    client_data = []
    samples_per_client = len(X) // num_clients
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X)
        client_X = X_sorted[start_idx:end_idx]
        client_y = y_sorted[start_idx:end_idx]
        client_data.append((client_X, client_y))
    return client_data

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Training function with differential privacy
def train(model, train_loader, optimizer, criterion, device, privacy_engine, epochs):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device).float()
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data).squeeze()
            loss = criterion(output, target)
            losses.append(loss.item())
            preds.extend(output.cpu().numpy())
            labels.extend(target.cpu().numpy())
    accuracy = accuracy_score(labels, np.round(preds))
    auc = roc_auc_score(labels, preds)
    return np.mean(losses), accuracy, auc

# Flower client
class CreditScoringClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, criterion, device, privacy_engine):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.privacy_engine = privacy_engine
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def get_parameters(self, config) -> List[np.ndarray]:
        return [param.cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        train(self.model, self.train_loader, self.optimizer, self.criterion, self.device, self.privacy_engine, epochs=1)
        return self.get_parameters(config), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss, accuracy, auc = evaluate(self.model, self.test_loader, self.criterion, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy), "auc": float(auc)}

# Main function to run federated learning
def main():
    # Load and partition data
    X, y = load_german_credit_data()
    client_data = partition_data_non_iid(X, y, num_clients=5)
    
    # Initialize global model
    input_dim = X.shape[1]
    global_model = MLP(input_dim).to(device)
    criterion = nn.BCELoss()
    
    # Create Flower clients
    clients = []
    for client_X, client_y in client_data:
        # Split client data into train/test
        X_train, X_test, y_train, y_test = train_test_split(client_X, client_y, test_size=0.2, random_state=42)
        
        # Create DataLoader
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Initialize model for client
        client_model = MLP(input_dim).to(device)
        
        # Set up differential privacy
        privacy_engine = PrivacyEngine()
        optimizer = optim.Adam(client_model.parameters(), lr=0.001)
        client_model, optimizer, train_loader = privacy_engine.make_private(
            module=client_model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.0,  # Controls privacy-accuracy trade-off
            max_grad_norm=1.0,
        )
        
        # Create Flower client
        client = CreditScoringClient(client_model, train_loader, test_loader, criterion, device, privacy_engine)
        clients.append(client)
    
    # Define Flower server strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )
    
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
        client_manager=fl.server.SimpleClientManager(),
        clients=clients,  # For simulation purposes
    )

if __name__ == "__main__":
    main()
