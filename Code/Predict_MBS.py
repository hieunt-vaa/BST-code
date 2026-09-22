import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# =========================
# 0) Config
# =========================
CSV_PATH = "scaled_output_159bits_PMBS.csv"
MODEL_PATH = "multitask_159bits_3real_PMBS.pth"
SCALER_X_PATH = "scaler_x_input_min.save"
SCALER_Y_PATH = "scaler_y_output_min.save"
OUTPUT_CSV = "prediction_results_PMBS.csv"
INPUT_COLS = ["P_MBS_1", "P_MBS_2", "P_MBS_3"]
BIT_COLS = [f"bit_{i}" for i in range(1, 160)]
REAL_COLS = ["z_1", "z_2", "z_3"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 1) Model Definition (must match training architecture!)
# =========================
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, num_bits, num_real):
        super(MultiTaskModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.bit_head = nn.Sequential(
            nn.Linear(256, num_bits),
            nn.Sigmoid()
        )
        self.real_head = nn.Linear(256, num_real)

    def forward(self, x):
        feat = self.shared(x)
        bits = self.bit_head(feat)
        reals = self.real_head(feat)
        return bits, reals


# =========================
# 2) Load model
# =========================
model = MultiTaskModel(input_dim=len(INPUT_COLS), num_bits=159, num_real=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================
# 3) Load data & scalers
# =========================
df = pd.read_csv(CSV_PATH)

X_scaled = df[INPUT_COLS].values.astype(np.float32)
scaler_x = joblib.load(SCALER_X_PATH)
X_original = scaler_x.inverse_transform(X_scaled)

scaler_y = joblib.load(SCALER_Y_PATH)

# =========================
# 4) Predict
# =========================
with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    pred_bits, pred_reals = model(X_tensor)

    pred_bits = (pred_bits.cpu().numpy() > 0.5).astype(int)  # threshold 0.5
    pred_reals_scaled = pred_reals.cpu().numpy()
    pred_reals_origin = scaler_y.inverse_transform(pred_reals_scaled)

# =========================
# 5) Save results
# =========================
result_df = pd.DataFrame(X_original, columns=INPUT_COLS)
for i in range(159):
    result_df[f"bit_{i+1}"] = pred_bits[:, i]
for j in range(3):
    result_df[f"z_{j+1}"] = pred_reals_origin[:, j]

result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ Saved predictions to {OUTPUT_CSV}")
