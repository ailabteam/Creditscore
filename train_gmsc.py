# train_gmsc.py
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------
# 1. Load dữ liệu (file đã upload cùng thư mục)
# ---------------------------------------------------------------------
train_df = pd.read_csv("cs-training.csv", index_col=0)
test_df  = pd.read_csv("cs-test.csv",  index_col=0)      # chưa dùng đến, để inference sau

FEATS = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]
LABEL = "SeriousDlqin2yrs"     # 1 = default

# ---------------------------------------------------------------------
# 2. Tiền xử lý: median-impute → standard-scale
# ---------------------------------------------------------------------
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
X_full = pipe.fit_transform(train_df[FEATS])
y_full = train_df[LABEL].astype(np.float32).values

# ---------------------------------------------------------------------
# 3. Chia train / validation (stratified 80 / 20)
# ---------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full,
    test_size=0.2, stratify=y_full, random_state=42
)

# ---------------------------------------------------------------------
# 4. DataLoader
# ---------------------------------------------------------------------
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)),
    batch_size=1024, shuffle=True, drop_last=False
)

# ---------------------------------------------------------------------
# 5. Mô hình MLP (logits output)
# ---------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)                 # logits, không sigmoid
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MLP(X_train.shape[1]).to(device)

# ---------------------------------------------------------------------
# 6. Hàm loss cân bằng class-imbalance (~1 : 14)
# ---------------------------------------------------------------------
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------------
# 7. Vòng huấn luyện
# ---------------------------------------------------------------------
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)

    epoch_loss /= len(train_loader.dataset)

    # Validation AUC mỗi epoch
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32).to(device))
        val_probs  = torch.sigmoid(val_logits).cpu().numpy().squeeze()
        val_auc    = roc_auc_score(y_val, val_probs)

    print(f"Epoch {epoch:2d}/{EPOCHS} | loss {epoch_loss:.4f} | val AUC {val_auc:.4f}")

# ---------------------------------------------------------------------
# 8. (Tuỳ chọn) Inference trên cs-test.csv
# ---------------------------------------------------------------------
X_test = pipe.transform(test_df[FEATS])
model.eval()
with torch.no_grad():
    test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
    test_probs  = torch.sigmoid(test_logits).cpu().numpy().squeeze()

# test_probs là xác suất vỡ nợ trong 2 năm tới cho từng dòng test
# Bạn có thể lưu ra CSV để nộp Kaggle:
# pd.DataFrame({"Id": test_df.index, "Probability": test_probs}).to_csv("submission.csv", index=False)

