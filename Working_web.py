import os
import math as m
import statistics as s
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import io, sys

# Hide TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Try importing ML libraries
ML_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    ML_AVAILABLE = False
    print("⚠ TensorFlow / scikit-learn not installed. Neural network features disabled.")

# -------------------------------
# Helper function to build LSTM
# -------------------------------
def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units // 2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------------
# Main Analysis Function
# -------------------------------
def run_analysis(T, prd, p1, p2, seq_len, test_ratio, epochs, batch_size, predict_days):
    logs = io.StringIO()
    sys.stdout = logs
    try:
        date_end = dt.date.today() - relativedelta(days=1)
        date_str_end = date_end.strftime("%d-%m-%y")
        date1y = dt.date.today() - relativedelta(years=1) + relativedelta(days=1)
        print(date1y.strftime("%d-%m-%y"), "to", date_str_end)

        # Download stock data
        D = yf.download(T, period=prd, progress=False)
        if D is None or D.empty:
            raise ValueError("No data downloaded. Check ticker symbol and internet connection.")

        # Handle DataFrame vs Series
        if "Close" in D.columns:
            close_series = D["Close"]
        elif ("Close", T) in D.columns:
            close_series = D[("Close", T)]
        else:
            raise KeyError("Could not find 'Close' prices in downloaded data.")

        CL = close_series.squeeze().tolist()
        xCL = [i + 1 for i in range(len(CL))]

        # Moving averages
        x1, x2, y1, y2 = [], [], [], []

        # Short MA
        for i in range(p1, len(CL) + 1):
            y1.append(s.mean(CL[i - p1 : i]))
            x1.append(i)

        # Long MA
        for i in range(p2, len(CL) + 1):
            y2.append(s.mean(CL[i - p2 : i]))
            x2.append(i)

        # Crossovers
        g_cr, d_cr = [], []
        s_idx = max(p1, p2)
        y1c, y2c = np.array(y1[s_idx - p1 :]), np.array(y2[s_idx - p2 :])
        if len(y1c) == len(y2c) and len(y1c) > 1:
            cross_idx_local = (np.where(np.diff(np.sign(y1c - y2c)))[0] + s_idx).tolist()
            for i in cross_idx_local:
                v1 = y1[i - p1] if (i - p1) < len(y1) else None
                v2 = y2[i - p2] if (i - p2) < len(y2) else None
                if v1 is None or v2 is None:
                    continue
                if v1 > v2:
                    g_cr.append((i, v1))
                else:
                    d_cr.append((i, v1))

        gx, gy = (list(x) for x in zip(*g_cr)) if g_cr else ([], [])
        dx, dy = (list(x) for x in zip(*d_cr)) if d_cr else ([], [])

        # -----------------------------
        # Plot 1: Moving averages
        # -----------------------------
        plt.figure(figsize=(8, 4))
        plt.plot(xCL, CL, color="blue", label="Closing Price")
        if y1:
            plt.plot(x1, y1, color="r", label=f"MA - {p1} Days")
        if y2:
            plt.plot(x2, y2, color="g", label=f"MA - {p2} Days")
        if gx:
            plt.scatter(gx, gy, marker="^", facecolor="green", label="Golden Cross", s=40, zorder=10)
        if dx:
            plt.scatter(dx, dy, marker="v", facecolor="black", label="Death Cross", s=40, zorder=10)

        plt.legend()
        plt.xlabel(f"Day count from {date1y}")
        plt.ylabel("Price")
        plt.title(f"{T} | MA({p1},{p2}) | upto {date_str_end}")
        plt.grid(True)
        tab1.pyplot(plt)

        # Profit Calculation
        by, sl = [], []
        sby, ssl = 0, 0
        rg = min(len(g_cr), len(d_cr))
        for i in range(rg):
            by.append(g_cr[i][0])
            sl.append(d_cr[i][0])
        for j in by:
            sby += CL[j]
        for k in sl:
            ssl += CL[k]
        prf = ssl - sby

        # -----------------------------
        # Neural network part
        # -----------------------------
        if ML_AVAILABLE:
            prices = np.array(CL).reshape(-1, 1)
            scaler = MinMaxScaler((0, 1))
            prices_scaled = scaler.fit_transform(prices)

            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(seq_length, len(data)):
                    X.append(data[i - seq_length : i, 0])
                    y.append(data[i, 0])
                X, y = np.array(X), np.array(y)
                return X.reshape((X.shape[0], X.shape[1], 1)), y

            X, y = create_sequences(prices_scaled, seq_len)
            split = int(len(X) * (1 - test_ratio))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_lstm_model((seq_len, 1))
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

            pred_test = scaler.inverse_transform(model.predict(X_test))
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Plot 2: Test predictions
            plt.figure(figsize=(8, 4))
            plt.plot(range(len(CL)), CL, color="blue", label="Historical")
            start_idx = len(CL) - len(y_test_inv)
            plt.plot(range(start_idx, len(CL)), y_test_inv, color="orange", label="Actual present data")
            plt.plot(range(start_idx, len(CL)), pred_test, color="green", label="Predicted data")
            plt.legend()
            plt.title(f"{T} | LSTM Test Predictions")
            plt.xlabel(f"Day count from {date1y}")
            plt.ylabel("Price")
            plt.grid(True)
            tab2.pyplot(plt)

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test_inv, pred_test))
            r2 = r2_score(y_test_inv, pred_test)
            accuracy = max(0, r2) * 100

            tab4.write(f"💰 3 Year Trend Profit per share: {prf}")
            tab4.write(f"✅ Accuracy: {accuracy:.2f}%")
            tab4.write(f"📌 RMSE: {rmse:.2f}")
            tab4.write(f"📌 R² Score: {r2:.3f}")

            # Plot 3: Future predictions
            last_seq = prices_scaled[-seq_len:]
            future_preds = []
            current_seq = last_seq.reshape((1, seq_len, 1))

            for _ in range(predict_days):
                next_pred = model.predict(current_seq)[0, 0]
                future_preds.append(next_pred)
                current_seq = np.append(current_seq[:, 1:, :], [[[next_pred]]], axis=1)

            future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

            plt.figure(figsize=(8, 4))
            plt.plot(range(len(CL)), CL, color="blue", label="Historical")
            plt.plot(range(len(CL), len(CL) + predict_days), future_preds_inv, color="red", label="Future Prediction")
            plt.legend()
            plt.title(f"{T} | {predict_days}-Day Future Prediction")
            plt.xlabel(f"Day count from {date1y}")
            plt.ylabel("Price")
            plt.grid(True)
            tab3.pyplot(plt)

        else:
            print("⚠ Skipping neural network part (TensorFlow not installed).")

    except Exception as e:
        print("Error occurred:", e)

    sys.stdout = sys.__stdout__  # restore stdout
    return logs.getvalue()

# -------------------------------
# Streamlit Web App (improved UI)
# -------------------------------
st.set_page_config(page_title="📈 Stock Analyzer", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    h1, h2, h3 {color: #2c3e50;}
    .stButton>button {
        background-color: #e74c3c; 
        color: white; 
        font-weight: bold; 
        border-radius: 10px;
        padding: 0.5em 1em;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Stock Analyzer")
st.write("Analyze stock price trends, moving averages, and predict future values using LSTM.")

# Help/How-to Expander
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    - **Ticker**: Stock symbol (like `TSLA` for Tesla).
    - **Period**: Time period for data (like `3y` for 3 years)
                  (A period of 3 years is optimal)
                  (Higher period takes more data for higher accuracy but takes more time).
    - **MA-1 / MA-2**: Moving Average windows.
                      (An average moving over a period, removing the last day and adding the next day)
                      (Optimal short MA is usually 1/15 of the trading days)
                      (Optimal long MA is 4 times short MA)
                      (Any other data set could be used according to your preference)
    - **Golden Crossovers are buy signals in past data at a lower closing price
          A crossover happening between two days, for which the previous day had Short MA above Long MA
    - **Death Crossovers are sell/cautious signals in past data at a higher closing price
          A crossover happening between two days, for which the previous day had Short MA below Long MA
     
    - **MACHINE LEARNING AND PREDICTION**
                
    - **Seq Len**: Sequence length for LSTM training. 
                (No of days taken in each iteration) (optimally 30 days)
    - **Test Ratio**: Portion of data used for testing (optimally 12% to 20%).
    - **Epochs / Batch Size**: Training parameters for LSTM.
                (Higher no of Epochs correspond to more no of iterations run by ML)
                (Optimal Epochs is 1/15 to 1/10 of trading days)
                (Batch size should be kept between 15-30, higher batch size consumes more data for efficiency)
                (Running more Epochs would consume more time but provide higher accuracy)
    - **Predict Days**: Number of future days to forecast. (Predicting 5-15 days is optimal)
                (WARNING ! : Predicting any higher data set would exponentially decrease accuracy)

    ℹ️ **Our Model trains itself using MAC data, in multiple sets of iterations to predict upcoming trends**
    """)

# Sidebar inputs
st.sidebar.header("⚙️ Parameters")
labels = ["Ticker", "Period", "MA-1", "MA-2", "Seq Len", "Test Ratio", "Epochs", "Batch Size", "Predict Days"]
defaults = ["TSLA", "3y", "50", "200", "30", "0.2", "25", "16", "10"]
entries = {}
for lbl, dft in zip(labels, defaults):
    entries[lbl] = st.sidebar.text_input(lbl, dft)

# Tabs for output
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Moving Averages", "🤖 LSTM Predictions", "🔮 Future Forecast", "💰 Profit & Accuracy", "📜 Logs"]
)

if st.button("🚀 Run Analysis"):
    with st.spinner("⚙️ Running analysis... please wait"):
        logs = run_analysis(
            entries["Ticker"].upper().strip(),
            entries["Period"].strip(),
            int(entries["MA-1"]),
            int(entries["MA-2"]),
            int(entries["Seq Len"]),
            float(entries["Test Ratio"]),
            int(entries["Epochs"]),
            int(entries["Batch Size"]),
            int(entries["Predict Days"])
        )
    tab5.text(logs)
