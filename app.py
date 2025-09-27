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
    h1, h2, h3, h4 {
        color: #FFD700;
        font-weight: 800;
        text-shadow: 2px 2px 6px black;
    }
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
        return torch.sigmoid(self.fc_out(x))  # outputs between 0 and 1

# =====================
# Main Logic
# =====================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Dataset Preview:", df.head())

    all_columns = df.columns.tolist()

    # User selects input & target columns
    input_cols = st.multiselect("Select input columns (features)", all_columns)
    target_cols = st.multiselect(
        "Select target columns (outputs)", 
        [c for c in all_columns if c not in input_cols]
    )

    if input_cols and target_cols:
        # Ensure numeric only & handle NaN
        X = df[input_cols].select_dtypes(include=[np.number]).fillna(0).values
        y = df[target_cols].select_dtypes(include=[np.number]).fillna(0).values

        # Scale features & targets
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)

        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

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

        y_pred_rescaled = scaler_y.inverse_transform(y_pred_test)
        y_test_rescaled = scaler_y.inverse_transform(y_test)

        r2 = r2_score(y_test_rescaled, y_pred_rescaled)

        st.subheader("📈 Model Accuracy")
        st.markdown(
            f"<div class='metric-card'><b>R² Score:</b> {r2:.4f}</div>",
            unsafe_allow_html=True
        )

        st.subheader("🎯 Predictions (Example Row)")
        sample_input = X_test[0].reshape(1, -1)
        with torch.no_grad():
            pred_sample = policy(torch.tensor(sample_input, dtype=torch.float32)).numpy()
        pred_rescaled = scaler_y.inverse_transform(pred_sample)

        result_str = "<br>".join(
            [f"<b>{col}:</b> {val:.2f}" for col, val in zip(target_cols, pred_rescaled[0])]
        )
        st.markdown(f"<div class='metric-card'>{result_str}</div>", unsafe_allow_html=True)
