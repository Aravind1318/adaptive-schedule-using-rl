import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="🤖 AI-Driven Adaptive Scheduling (RL)", layout="wide")
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
st.title("🤖 AI-Driven Adaptive Scheduling (Reinforcement Learning)")

uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

# =====================
# RL Policy Network
# =====================
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 🔒 constrain output between 0 and 1
        x = torch.sigmoid(self.fc_out(x))
        return x

# =====================
# Main Logic
# =====================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("📊 Dataset Preview:", df.head())

    # Assume last two columns are target (Machine, Manpower)
    X = df.iloc[:, :-2].values
    y = df.iloc[:, -2:].values

    # Scale features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # RL settings
    rl_epochs = st.sidebar.number_input("RL epochs", min_value=100, max_value=5000, value=500, step=100)
    learning_rate = st.sidebar.number_input("Learning rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")

    policy = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # =====================
    # Training Loop
    # =====================
    for epoch in range(rl_epochs):
        policy.train()
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        preds = policy(X_tensor)

        # Reward = -MSE
        loss = loss_fn(preds, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # =====================
    # Evaluation
    # =====================
    policy.eval()
    with torch.no_grad():
        y_pred_test = policy(torch.tensor(X_test, dtype=torch.float32)).numpy()

    # Inverse transform to original scale
    y_pred_rescaled = scaler_y.inverse_transform(y_pred_test)
    y_test_rescaled = scaler_y.inverse_transform(y_test)

    # Calculate R²
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # =====================
    # Streamlit Display
    # =====================
    st.subheader("📈 Model Accuracy")
    st.markdown(
        f"<div style='background-color:black; color:gold; padding:10px; border-radius:10px;'>"
        f"<b>R² Score:</b> {r2*100:.2f}%"
        f"</div>",
        unsafe_allow_html=True
    )

    st.subheader("🎯 Predictions")
    sample_input = X_test[0].reshape(1, -1)
    with torch.no_grad():
        pred_sample = policy(torch.tensor(sample_input, dtype=torch.float32)).numpy()
    pred_rescaled = scaler_y.inverse_transform(pred_sample)

    st.markdown(
        f"<div style='background-color:black; color:gold; padding:10px; border-radius:10px;'>"
        f"<b>Machine:</b> {pred_rescaled[0][0]:.2f} <br>"
        f"<b>Manpower:</b> {pred_rescaled[0][1]:.2f}"
        f"</div>",
        unsafe_allow_html=True
    )
