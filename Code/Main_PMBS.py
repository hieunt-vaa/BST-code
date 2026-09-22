import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# =========================
# 0) Config
# =========================
CSV_PATH = "scaled_output_159bits_PMBS.csv"
SCALER_X_PATH = "scaler_x_input_min.save"           # scaler X (min-max?) để inverse_transform
SCALER_Y_PATH = "scaler_y_output_min.save"     # scaler Y (standard) để inverse_transform
SAVE_MODEL_PATH = "multitask_159bits_3real_PMBS.pth"

INPUT_COLS = ['P_MBS_1','P_MBS_2', 'P_MBS_3']
BIT_COLS = [f"bit_{i}" for i in range(1, 160)]
OUTPUT_COLS = ["z_1", "z_2", "z_3"]                 # 3 nghiệm thực (đã chuẩn hoá trong CSV)

BATCH_SIZE = 32
EPOCHS = 5000
LR = 1e-5
WEIGHT_DECAY = 1e-6
HIDDEN = 256
LAMBDA_BITS = 0.5
LAMBDA_REAL = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(10000)
np.random.seed(10000)

# =========================
# 1) Load data
# =========================
print(">>> Loading data...")
df = pd.read_csv(CSV_PATH)

# X đã scaled trong CSV
X_scaled = df[INPUT_COLS].values.astype(np.float32)

# Nhãn phân lớp 99 bit
Y_bits = df[BIT_COLS].values.astype(np.float32)

# Nhãn hồi quy 3 chiều (đã chuẩn hoá theo scaler_y)
Y_real_scaled = df[OUTPUT_COLS].values.astype(np.float32)

# Load scalers và lấy giá trị gốc để in minh hoạ / tính MSE gốc
scaler_x = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

X_original = scaler_x.inverse_transform(X_scaled)
# Với Y: inverse dùng khi đánh giá để có MSE ở không gian gốc
Y_real_original = scaler_y.inverse_transform(Y_real_scaled)

# =========================
# 2) Split train / val / test (70 / 15 / 15)
# =========================
X_train, X_temp, Yb_train, Yb_temp, Yr_train, Yr_temp, Xo_train, Xo_temp, Yro_train, Yro_temp = train_test_split(
    X_scaled, Y_bits, Y_real_scaled, X_original, Y_real_original,
    test_size=0.3, random_state=10000
)
X_val, X_test, Yb_val, Yb_test, Yr_val, Yr_test, Xo_val, Xo_test, Yro_val, Yro_test = train_test_split(
    X_temp, Yb_temp, Yr_temp, Xo_temp, Yro_temp,
    test_size=0.5, random_state=10000
)

# Tạo TensorDatasets
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Yb_train), torch.tensor(Yr_train))
val_dataset   = TensorDataset(torch.tensor(X_val),   torch.tensor(Yb_val),   torch.tensor(Yr_val))
test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(Yb_test),  torch.tensor(Yr_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# 3) Model: shared trunk + 2 heads (bits + real)
# =========================
class MultiTaskNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, bit_dim=159, real_dim=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Phân lớp 99 bit
        self.bit_head = nn.Sequential(
            nn.Linear(hidden_dim, bit_dim),
            nn.Sigmoid()
        )
        # Hồi quy 3 giá trị thực (không activation)
        # self.real_head = nn.Linear(hidden_dim, real_dim)
        self.real_head = nn.Linear(hidden_dim, real_dim)

    def forward(self, x):
        h = self.shared(x)
        out_bits = self.bit_head(h)
        out_real = self.real_head(h)
        return out_bits, out_real

model = MultiTaskNet(input_dim=3, hidden_dim=HIDDEN, bit_dim=159, real_dim=3).to(DEVICE)

# =========================
# 4) Loss & Optimizer
# =========================
criterion_bits = nn.BCELoss()   # vì đã dùng Sigmoid ở đầu ra nhị phân
criterion_real = nn.MSELoss()   # hồi quy
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# =========================
# 5) Training loop
#  - In: TrainAcc/Exact cho bits
#  - In: TrainMSE_scaled cho 3 nghiệm thực (không gian chuẩn hoá)
# =========================
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_acc, train_exact, train_mse = 0.0, 0.0, 0.0, 0.0

    for X_batch, Yb_batch, Yr_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        Yb_batch = Yb_batch.to(DEVICE)
        Yr_batch = Yr_batch.to(DEVICE)

        optimizer.zero_grad()
        out_bits, out_real = model(X_batch)

        loss_bits = criterion_bits(out_bits, Yb_batch)
        loss_real = criterion_real(out_real, Yr_batch)
        loss = LAMBDA_BITS * loss_bits + LAMBDA_REAL * loss_real
        loss.backward()
        optimizer.step()

        # ---- metrics bits ----
        preds_bits = (out_bits > 0.5).int()
        Yb_int = Yb_batch.int()
        acc = (preds_bits == Yb_int).float().mean().item()
        exact = preds_bits.eq(Yb_int).all(dim=1).float().mean().item()

        # ---- metrics real (MSE scaled) ----
        mse_scaled = loss_real.item()

        bs = X_batch.size(0)
        train_loss  += loss.item() * bs
        train_acc   += acc * bs
        train_exact += exact * bs
        train_mse   += mse_scaled * bs

    n_train = len(train_loader.dataset)
    train_loss  /= n_train
    train_acc   /= n_train
    train_exact /= n_train
    train_mse   /= n_train

    # ---- Validation ----
    model.eval()
    val_loss, val_acc, val_exact, val_mse = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for X_batch, Yb_batch, Yr_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            Yb_batch = Yb_batch.to(DEVICE)
            Yr_batch = Yr_batch.to(DEVICE)

            out_bits, out_real = model(X_batch)
            loss_bits = criterion_bits(out_bits, Yb_batch)
            loss_real = criterion_real(out_real, Yr_batch)
            loss = LAMBDA_BITS * loss_bits + LAMBDA_REAL * loss_real

            preds_bits = (out_bits > 0.5).int()
            Yb_int = Yb_batch.int()
            acc = (preds_bits == Yb_int).float().mean().item()
            exact = preds_bits.eq(Yb_int).all(dim=1).float().mean().item()
            mse_scaled = loss_real.item()

            bs = X_batch.size(0)
            val_loss  += loss.item() * bs
            val_acc   += acc * bs
            val_exact += exact * bs
            val_mse   += mse_scaled * bs

    n_val = len(val_loader.dataset)
    val_loss  /= n_val
    val_acc   /= n_val
    val_exact /= n_val
    val_mse   /= n_val

    print(
        f"Epoch {epoch:03d} | "
        f"TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.4f} | TrainMSE {train_mse:.6f} | "
        f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f} | ValMSE {val_mse:.6f}"
    )

# =========================
# 6) Test evaluation
#  - In: TestAcc/Exact cho bits
#  - In: TestMSE_scaled (chuẩn hoá)
#  - In: TestMSE_orig (đã inverse về không gian gốc)
# =========================
model.eval()
test_loss, test_acc, test_exact = 0.0, 0.0, 0.0
test_mse_scaled_sum = 0.0
test_mse_orig_sum = 0.0
num_samples = 0

with torch.no_grad():
    for X_batch, Yb_batch, Yr_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        Yb_batch = Yb_batch.to(DEVICE)
        Yr_batch = Yr_batch.to(DEVICE)

        out_bits, out_real = model(X_batch)

        loss_bits = criterion_bits(out_bits, Yb_batch)
        loss_real = criterion_real(out_real, Yr_batch)
        loss = LAMBDA_BITS * loss_bits + LAMBDA_REAL * loss_real

        # ---- bits metrics ----
        preds_bits = (out_bits > 0.5).int()
        Yb_int = Yb_batch.int()
        acc = (preds_bits == Yb_int).float().mean().item()
        exact = preds_bits.eq(Yb_int).all(dim=1).float().mean().item()

        # ---- real metrics (scaled) ----
        mse_scaled = loss_real.item()

        bs = X_batch.size(0)
        test_loss += loss.item() * bs
        test_acc  += acc * bs
        test_exact += exact * bs
        test_mse_scaled_sum += mse_scaled * bs
        num_samples += bs

        # ---- real metrics (original, inverse_transform) ----
        pred_real_scaled = out_real.detach().cpu().numpy()        # (bs, 3)
        true_real_scaled = Yr_batch.detach().cpu().numpy()
        pred_real_orig = scaler_y.inverse_transform(pred_real_scaled)
        true_real_orig = scaler_y.inverse_transform(true_real_scaled)

        # MSE theo không gian gốc: mean over all elements
        se = (pred_real_orig - true_real_orig) ** 2               # (bs, 3)
        test_mse_orig_sum += np.sum(se) / 3.0                     # chia trung bình theo 3 chiều cho từng mẫu, rồi cộng dồn

# Chuẩn hoá theo số mẫu
test_loss /= num_samples
test_acc  /= num_samples
test_exact /= num_samples
test_mse_scaled = test_mse_scaled_sum / num_samples
test_mse_orig   = test_mse_orig_sum / num_samples

print(
    f"TestLoss {test_loss:.4f} | TestAcc {test_acc:.4f} | "
    f"TestMSE {test_mse_scaled:.6f} "
)

# =========================
# 7) Lưu model & demo dự đoán vài mẫu
# =========================
torch.save(model.state_dict(), SAVE_MODEL_PATH)
print(f">>> Saved model to {SAVE_MODEL_PATH}")

# Demo: in thử 5 mẫu test (bits + real) với inverse về gốc
print("\n>>> Demo 5 samples from test set:")
with torch.no_grad():
    X_sample = torch.tensor(X_test[:15]).to(DEVICE)
    out_bits, out_real = model(X_sample)

    preds_bits = (out_bits > 0.5).int().cpu().numpy()
    true_bits = Yb_test[:15]

    pred_real_scaled = out_real.cpu().numpy()
    true_real_scaled = Yr_test[:15]

    pred_real_orig = scaler_y.inverse_transform(pred_real_scaled)
    true_real_orig = scaler_y.inverse_transform(true_real_scaled)

    for i in range(len(X_sample)):
        print(f"Sample {i}:")
        print("  X_original:", Xo_test[i])
        print("  Pred bits (first 10):", preds_bits[i][:10], "...")
        print("  True bits (first 10):", true_bits[i][:10], "...")
        print("  Pred real (scaled):", pred_real_scaled[i])
        print("  True real (scaled):", true_real_scaled[i])
        print("  Pred real (orig):  ", pred_real_orig[i])
        print("  True real (orig):  ", true_real_orig[i])
        print("-" * 60)
