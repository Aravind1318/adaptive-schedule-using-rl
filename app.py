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
