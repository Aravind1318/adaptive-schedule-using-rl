# save this as your Streamlit app (e.g., adaptive_rl_app.py)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

st.set_page_config(page_title="🤖 AI-Driven Adaptive Scheduling", layout="wide")

# =========================
st.markdown("""
    <style>
    /* Main background with black-gold swirl theme */
    .stApp {
        background: linear-gradient(
            135deg,
            #000000 0%,
            #1a1a1a 20%,
            #4d3b1f 40%,
            #b8860b 60%,
            #ffd700 80%,
            #000000 100%
        );
        background-attachment: fixed;
        background-size: 300% 300%;
        animation: swirlGradient 25s ease infinite;
        font-family: 'Segoe UI', sans-serif;
        color: white;
    }

    @keyframes swirlGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    /* Model Accuracy styled same as prediction cards */
.model-accuracy-card {
    background: linear-gradient(145deg, #000000, #1a1a1a, #2c1a1a);
    border-radius: 12px;
    padding: 16px;
    margin: 10px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.6);
    font-size: 1.1rem;
    font-weight: 600;
    color: #FFD700;
    border: 1px solid #FFD700;
}


    /* Titles */
    h1, h2, h3, h4 {
        color: #FFD700; /* Gold */
        font-weight: 800;
        text-shadow: 2px 2px 6px black;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #FFD700, #4d3b1f, #000000) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6em 1.2em !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease-in-out !important;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.6);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ffb700, #b8860b, #2c1a1a) !important;
        transform: scale(1.05) !important;
    }
    .stButton > button:active {
        transform: scale(0.95) !important;
    }

    /* Number Input Fields */
    .stNumberInput > div > div > input {
        background-color: #1a1a1a !important; 
        color: #FFD700 !important;
        border-radius: 8px !important;
        border: 1px solid #FFD700 !important;
        padding: 6px 10px !important;
    }

    /* Selectbox */
    .stSelectbox > div > div > select {
        background-color: #2c1a1a !important;
        color: #FFD700 !important;
        border-radius: 8px !important;
        border: 1px solid #FFD700 !important;
        padding: 6px 10px !important;
    }

    /* MultiSelect */
    .stMultiSelect > div > div {
        background-color: #000000 !important;
        color: #FFD700 !important;
        border-radius: 8px !important;
        border: 1px solid #FFD700 !important;
        padding: 6px 10px !important;
    }

    /* DataFrame table */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 2px solid #FFD700 !important;
    }

    /* Success / Info boxes */
    .stSuccess {
        background-color: rgba(218,165,32,0.2) !important;
        border-left: 6px solid #FFD700 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        color: white !important;
    }
    .stInfo {
        background-color: rgba(255,215,0,0.15) !important;
        border-left: 6px solid #DAA520 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        color: white !important;
    }

    /* Custom Prediction Cards */
    .metric-card {
        background: linear-gradient(145deg, #000000, #1a1a1a, #2c1a1a);
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.6);
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFD700;
        border: 1px solid #FFD700;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Driven Adaptive Scheduling")

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Production_Load" in df and "Deadline_Hours" in df:
        df["urgency"] = df["Production_Load"] / (df["Deadline_Hours"] + 1e-3)
    if "Available_Operators" in df and "Available_Machines" in df:
        df["operator_machine_ratio"] = df["Available_Operators"] / (df["Available_Machines"] + 1)
    if "Expected_Runtime_Min" in df and "Machine_Efficiency" in df:
        df["adjusted_runtime"] = df["Expected_Runtime_Min"] / (df["Machine_Efficiency"] + 1e-3)
    if "Production_Load" in df and "Available_Operators" in df:
        df["load_per_operator"] = df["Production_Load"] / (df["Available_Operators"] + 1)
    if "Shift" in df:
        df["shift_binary"] = df["Shift"].apply(lambda x: 1 if str(x).lower() == "night" else 0)
    return df

# ------------------------
# Reinforcement Learning policy (continuous)
# ------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[128, 128]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        # output mean for each target dim
        self.mean_head = nn.Linear(prev, output_dim)
        # log std parameter (learnable, one per output)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std)
        return mean, std

def train_policy_reinforce(X_train_np, y_train_np, X_val_np=None, y_val_np=None,
                           epochs=500, lr=1e-3, batch_size=64, seed=42):
    torch.manual_seed(seed)
    X = torch.from_numpy(X_train_np).float()
    y = torch.from_numpy(y_train_np).float()
    n_samples, input_dim = X.shape
    output_dim = y.shape[1]

    policy = PolicyNetwork(input_dim, output_dim)
    opt = optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        # shuffle
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb = X[idx]
            yb = y[idx]

            mean, std = policy(xb)                       # mean: (B, out), std: (out,)
            dist = D.Normal(mean, std)                  # independent normals
            actions = dist.rsample()                    # (B, out)
            # reward: negative MSE per sample (higher is better)
            mse = torch.mean((actions - yb) ** 2, dim=1)  # (B,)
            rewards = -mse                                # (B,)

            # REINFORCE loss (we maximize reward => minimize -reward)
            log_prob = dist.log_prob(actions).sum(dim=1)  # sum across outputs => (B,)
            loss = - (log_prob * rewards).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * xb.size(0)

        # optional: early stopping or val logging could be added
        # (kept minimal to not change app logic)
    return policy

# ------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = add_engineered_features(df)

    engineered_features = ["urgency", "operator_machine_ratio", "adjusted_runtime",
                           "load_per_operator", "shift_binary"]

    st.write("✅ Dataset loaded successfully with engineered features!")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    st.subheader("⚙️ Select Features and Target Columns")

    input_cols = st.multiselect(
        "Select Input Columns (X)",
        [c for c in all_columns if c not in engineered_features],
        default=[c for c in all_columns if c not in engineered_features]
    )
    output_cols = st.multiselect(
        "Select Output Columns (y)",
        [c for c in all_columns if c not in engineered_features],
        default=[c for c in all_columns if c not in engineered_features and c not in input_cols]
    )

    if input_cols and output_cols and st.button("🚀 Train Model"):
        X = df[input_cols]
        y = df[output_cols]

        # keep the same preprocessing step style (one-hot)
        X_encoded = pd.get_dummies(X, drop_first=True)

        # convert to numpy arrays for RL training
        X_np = X_encoded.values.astype(np.float32)
        # ensure y is 2D (n_samples, n_targets)
        if isinstance(y, pd.Series):
            y_np = y.values.reshape(-1, 1).astype(np.float32)
        else:
            y_np = y.values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.2, random_state=42
        )

        # RL training hyperparams (exposed minimally)
        epochs = st.number_input("RL epochs", min_value=10, max_value=5000, value=500, step=10)
        lr = float(st.number_input("Learning rate", min_value=1e-5, max_value=1.0, value=1e-3, format="%.6f"))

        with st.spinner("Training RL policy (REINFORCE)... this may take time depending on epochs"):
            policy = train_policy_reinforce(X_train, y_train, epochs=int(epochs), lr=lr)

        # Make deterministic predictions: use policy mean
        policy.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).float()
            mean, _ = policy(X_test_t)
            y_pred = mean.numpy()

        # compute R^2 in the same way as original code
        r2 = r2_score(y_test, y_pred, multioutput="uniform_average")

        st.subheader("📊 Model Accuracy")
        st.markdown(
            f'<div class="model-accuracy-card">R² Score: {r2*100:.2f}%</div>',
            unsafe_allow_html=True
        )

        # save useful objects to session state for prediction UI
        st.session_state["policy"] = policy
        st.session_state["features"] = X_encoded.columns.tolist()
        st.session_state["output_cols"] = output_cols
        st.session_state["input_cols"] = input_cols
        st.session_state["df"] = df
        # Also save feature means for default inputs
        st.session_state["feature_means"] = dict(zip(X_encoded.columns, X_encoded.mean()))

if "policy" in st.session_state:
    st.subheader("🔧 Predict for New Input")

    df = st.session_state["df"]
    input_cols = st.session_state["input_cols"]
    output_cols = st.session_state["output_cols"]
    features = st.session_state["features"]

    input_data = {}
    for col in input_cols:
        if df[col].dtype in ["int64", "float64"]:
            default_val = float(df[col].mean())
            val = st.number_input(
                f"{col}",
                min_value=0.0,
                max_value=10000.0,
                value=default_val
            )
            input_data[col] = val
        else:
            options = df[col].unique().tolist()
            val = st.selectbox(f"{col}", options)
            input_data[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_df = add_engineered_features(input_df)
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Reindex to training features and fill missing with 0
        input_encoded = input_encoded.reindex(columns=features, fill_value=0)

        # convert to tensor and run through policy
        input_tensor = torch.from_numpy(input_encoded.values.astype(np.float32))
        policy = st.session_state["policy"]
        policy.eval()
        with torch.no_grad():
            mean, _ = policy(input_tensor)
            prediction = mean.numpy().flatten()

        st.success("🎯 Predictions:")
        for i, col in enumerate(output_cols):
            if col.lower() in ["machine", "manpower"]:
                val = int(round(prediction[i]))
            else:
                val = round(float(prediction[i]), 2)
            st.markdown(f'<div class="metric-card">{col}: {val}</div>', unsafe_allow_html=True)
else:
    st.info("📥 Please upload a CSV, select columns, and click 🚀 Train Model")
