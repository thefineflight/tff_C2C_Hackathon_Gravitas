import os
import datetime as dt
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

# === ANN imports (LM) ===
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf_levenberg_marquardt as lm

print("Compiling Tasks")

# Today label
today = dt.date.today()
date_str = today.strftime('%d-%m-%y')

# ---------------- USER INPUTS ----------------
T = input("Enter Ticker Symbol (eg. TSLA): ").strip().upper()
prd = input("Enter period (e.g., '3y' for 3 years, '1y', '6mo', 'ytd'): ").strip()
ma_type = input("Moving Average Type (SMA/EMA): ").strip().upper()
p1 = int(input("MA -1 (e.g., 20): "))
p2 = int(input("MA -2 (e.g., 50): "))

# Indicator settings
rsi_period = 14

# Directory for ANN training plots
ann_plot_dir = f"ANN_plots_{T}_{date_str}"
os.makedirs(ann_plot_dir, exist_ok=True)

# ---------------- PERIOD PARSING ----------------
def parse_period_to_start(period: str, ref_date: dt.date) -> dt.date:
    p = period.lower().strip()
    if p == 'ytd':
        return dt.date(ref_date.year, 1, 1)
    if p.endswith('y'):
        years = int(p[:-1])
        return ref_date - relativedelta(years==years)
    if p.endswith('mo'):
        months = int(p[:-2])
        return ref_date - relativedelta(months=months)
    if p.endswith('wk'):
        weeks = int(p[:-2])
        return ref_date - relativedelta(weeks=weeks)
    if p.endswith('d'):
        days = int(p[:-1])
        return ref_date - relativedelta(days=days)
    if p.isdigit():
        return ref_date - relativedelta(years=int(p))
    raise ValueError("Unsupported period format. Use '3y', '1y', '6mo', '4wk', '30d', or 'ytd'.")

# User-chosen visible window start
base_start = parse_period_to_start(prd, today)

# Buffer so first visible day has valid MA/RSI values
if prd.lower().endswith('y'):
    buffer_rel = relativedelta(years=1)  # extra history for yearly windows
else:
    buffer_days = max(p1, p2, rsi_period) + 5
    buffer_rel = relativedelta(days=buffer_days)

download_start = base_start - buffer_rel
download_end = today

# ---------------- DOWNLOAD CLOSE PRICES ----------------
def download_close(ticker: str, start_date: dt.date, end_date: dt.date) -> pd.Series:
    common_kwargs = dict(
        start=start_date.isoformat(),
        end=(end_date + relativedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        interval='1d'
    )
    try:
        df = yf.download(
            tickers=[ticker],
            multi_level_index=False,
            **common_kwargs
        )
    except TypeError:
        df = yf.download(
            tickers=ticker,
            **common_kwargs
        )

    if df is None or df.empty:
        raise ValueError("No data returned for the given ticker and dates.")

    # Handle single / multi-index columns safely
    if isinstance(df.columns, pd.MultiIndex):
        if ('Close', ticker) in df.columns:
            close = df[('Close', ticker)].rename('Close')
        elif 'Close' in df.columns.get_level_values(0):
            close = df['Close'].iloc[:, 0]
        else:
            df.columns = df.columns.get_level_values(0)
            close = df['Close']
    else:
        close = df['Close']

    close = close.dropna()
    return close

close = download_close(T, download_start, download_end)
if close.empty:
    raise ValueError("No close prices available after download.")

# ---------------- COMPUTE MAs ----------------
def compute_ma(series: pd.Series, period: int, mode: str) -> pd.Series:
    mode = mode.upper()
    if mode == 'SMA':
        return series.rolling(window=period, min_periods=period).mean()
    if mode == 'EMA':
        return series.ewm(span=period, adjust=False, min_periods=period).mean()
    raise ValueError("Invalid MA type. Use 'SMA' or 'EMA'.")

ma1_full = compute_ma(close, p1, ma_type)
ma2_full = compute_ma(close, p2, ma_type)

# ---------------- RSI ----------------
delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
rs = avg_gain / avg_loss
rsi_full = 100 - (100 / (1 + rs))
rsi_full = rsi_full.clip(lower=0, upper=100)

# ---------------- VISIBLE WINDOW ARRAYS ----------------
visible_close = close.loc[pd.Timestamp(base_start): pd.Timestamp(today)]
if visible_close.empty:
    raise ValueError("No visible data in the requested window; check the period or ticker.")

N = len(visible_close)
px = np.arange(1, N + 1)

ma1 = ma1_full.loc[visible_close.index].to_numpy()
ma2 = ma2_full.loc[visible_close.index].to_numpy()
py = visible_close.to_numpy()
rsi = rsi_full.loc[visible_close.index].to_numpy()

# ---------------- CROSSOVER DETECTION ----------------
mask_valid = np.isfinite(ma1) & np.isfinite(ma2)
if not mask_valid.all():
    first_valid = int(np.argmax(mask_valid))
    if not mask_valid[first_valid]:
        first_valid = N
else:
    first_valid = 0

ma1v = ma1[first_valid:]
ma2v = ma2[first_valid:]
xv = px[first_valid:]

gx, gy, dx, dy = [], [], [], []
if len(ma1v) > 1:
    sign = np.sign(ma1v - ma2v)
    cross_idx = np.where(np.diff(sign) != 0)[0] + 1
    for ci in cross_idx:
        xi = int(xv[ci])
        yi = float(ma1v[ci])
        if ma1v[ci] > ma2v[ci]:
            gx.append(xi)
            gy.append(yi)
        else:
            dx.append(xi)
            dy.append(yi)

# ---------------- PRICE + MA + RSI PLOT ----------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 8),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

ax1.plot(px, py, label='Closing Price')
ax1.plot(px, ma1, label=f'{ma_type} - {p1} Days')
ax1.plot(px, ma2, label=f'{ma_type} - {p2} Days')

if gx:
    ax1.scatter(gx, gy, marker='^', s=60, label='Golden Crossover')
if dx:
    ax1.scatter(dx, dy, marker='v', s=60, label='Death Crossover')

ax1.set_ylabel("Price / MAs")
ax1.legend(loc='best')

ax2.plot(px, rsi, label=f'RSI {rsi_period}')
ax2.axhline(70, linestyle='--', linewidth=1)
ax2.axhline(30, linestyle='--', linewidth=1)
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI")
ax2.legend(loc='upper left')

ax1.set_xlim(1, int(px.max()))
ax1.margins(x=0)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.set_xlabel(f"Day count from {visible_close.index[0].date().isoformat()}")

title_tkr = T.replace('.', '')
plt.suptitle(f"{prd} {ma_type} ({p1} | {p2}) | {T} | up to {date_str}")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{date_str} {title_tkr}.png", dpi=600)
plt.show()

print("Running ANN (LM, predicting next-day return)")

# ---------------- ANN DATA PREP (VISIBLE WINDOW ONLY) ----------------
close_np = visible_close.to_numpy().astype(np.float32)
ma1_np = ma1.astype(np.float32)
ma2_np = ma2.astype(np.float32)
rsi_np = rsi.astype(np.float32)

# Features: Close, MA1, MA2, RSI
features_all = np.stack([close_np, ma1_np, ma2_np, rsi_np], axis=1)  # (N, 4)

# Compute next-day returns: r_{t+1} = (Close_{t+1} - Close_t) / Close_t
returns_np = (close_np[1:] - close_np[:-1]) / close_np[:-1]  # length N-1

# ---------------- ANN CONFIG ----------------
WINDOW = 60        # lookback days for each input
PRED_DAYS = 5      # last 5 days to predict & match
CONTEXT_DAYS = 30  # last 30 days for plotting window

def build_dataset(feat_arr, close_arr, returns_arr, window, dates):
    """
    feat_arr: shape (N, num_features)
    close_arr: shape (N,)
    returns_arr: shape (N-1,), r_i = return from day i -> i+1
    dates: DatetimeIndex for these arrays
    Returns:
        X: (num_samples, window, num_features)
        y: (num_samples,) target returns
        y_dates: DatetimeIndex of target next-day close
        prev_close: (num_samples,) previous close used to reconstruct price
    """
    X, y, y_dates, prev_close = [], [], [], []
    N_local = len(close_arr)

    num_samples = N_local - window
    for i in range(num_samples):
        window_feats = feat_arr[i:i + window]  # (window, F)

        # Skip any window that contains NaN or inf in features
        if not np.isfinite(window_feats).all():
            continue

        ret_idx = i + window - 1               # index in returns_arr
        if ret_idx >= len(returns_arr):
            break

        target_return = returns_arr[ret_idx]   # r_{t+1}
        target_close_idx = i + window         # index of Close_{t+1}

        X.append(window_feats.astype(np.float32))
        y.append(float(target_return))
        y_dates.append(dates[target_close_idx])
        prev_close.append(float(close_arr[target_close_idx - 1]))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    y_dates = pd.to_datetime(y_dates)
    prev_close = np.array(prev_close, dtype=np.float32)
    return X, y, y_dates, prev_close

dates_visible = visible_close.index

X_all_raw, y_all, dates_all, prev_close_all = build_dataset(
    features_all, close_np, returns_np, WINDOW, dates_visible
)

num_samples = X_all_raw.shape[0]
if num_samples < (PRED_DAYS + 5):
    raise ValueError(
        f"Not enough data points for ANN training after removing NaN windows. "
        f"Increase period or reduce WINDOW. Samples: {num_samples}"
    )

# ---------------- TRAIN / TEST SPLIT ----------------
split_idx = num_samples - PRED_DAYS
X_train_raw, X_test_raw = X_all_raw[:split_idx], X_all_raw[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]
dates_train, dates_test = dates_all[:split_idx], dates_all[split_idx:]
prev_close_train, prev_close_test = (
    prev_close_all[:split_idx],
    prev_close_all[split_idx:],
)

print(f"Total samples: {num_samples}, train: {len(X_train_raw)}, test(pred): {len(X_test_raw)}")
print(f"Prediction dates (last {PRED_DAYS} closing days): {list(dates_test)}")

# ---------------- NORMALIZATION (TRAIN STATS ONLY) ----------------
# Features
train_feat_flat = X_train_raw.reshape(-1, X_train_raw.shape[-1])  # (?, num_features)
feat_mean = train_feat_flat.mean(axis=0, keepdims=True)
feat_std = train_feat_flat.std(axis=0, keepdims=True)
feat_std[feat_std == 0] = 1.0  # avoid division by zero

def normalize_windows(X_raw, mean, std):
    return (X_raw - mean) / std

X_train = normalize_windows(X_train_raw, feat_mean, feat_std)
X_test = normalize_windows(X_test_raw, feat_mean, feat_std)

# Flatten inputs for MLP
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))
input_dim = X_train_flat.shape[1]

# Target (returns) normalization
y_mean = y_train.mean()
y_std = y_train.std()
if y_std == 0:
    y_std = 1.0

y_train_n = (y_train - y_mean) / y_std
y_test_n = (y_test - y_mean) / y_std

# ---------------- MLP MODEL (SUITABLE FOR LM) ----------------
def build_mlp(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(1),  # predicted next-day return (normalized)
        ]
    )
    return model

model_lm = build_mlp(input_dim)

# LM wrapper
model_wrapper = lm.model.ModelWrapper(model_lm)
model_wrapper.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),
    loss=lm.loss.MeanSquaredError(),
    metrics=[],
)

# ---------------- TRAINING HYPERPARAMS ----------------
MAX_EPOCHS = 250      # upper limit on epochs
BATCH_SIZE = 32
PATIENCE = 25         # early stopping patience (epochs with no improvement)
MIN_DELTA = 1e-6      # minimal improvement in val_loss to count as "improved"
TARGET_VAL_MSE = 1e-4 # "good enough" validation MSE threshold

# ---------------- PLOTTING FUNCTION (LAST 30 DAYS VIEW) ----------------
def plot_predictions_epoch(
    model,
    X_test_flat,
    y_test,
    y_mean,
    y_std,
    prev_close_test,
    dates_test,
    visible_close,
    context_days,
    epoch_num,
):
    """
    For the given epoch (with current model weights), show:
    - Last `context_days` days of actual close.
    - On the last PRED_DAYS of those, overlay actual vs predicted closes.
    X-axis = 1..context_days (no date ticks).
    """
    # Predict normalized returns -> de-normalize
    y_pred_n = model.predict(X_test_flat, verbose=0).flatten()
    y_pred_returns = y_pred_n * y_std + y_mean  # predicted returns

    # Reconstruct predicted closes: Close_{t+1} = Close_t * (1 + r_{t+1})
    pred_closes = prev_close_test * (1.0 + y_pred_returns)

    all_dates = visible_close.index

    # Last CONTEXT_DAYS dates for cleaner view
    if len(all_dates) >= context_days:
        plot_dates = all_dates[-context_days:]
    else:
        plot_dates = all_dates

    plot_actual = visible_close.loc[plot_dates].to_numpy()
    x_full = np.arange(1, len(plot_dates) + 1)

    # Positions of the prediction dates among these last context_days
    pred_positions = []
    for d in dates_test:
        idx_arr = np.where(plot_dates == d)[0]
        if len(idx_arr) > 0:
            pred_positions.append(idx_arr[0])

    if len(pred_positions) == 0:
        return

    pred_positions = np.array(pred_positions, dtype=int)
    x_pred = x_full[pred_positions]

    # Actual closes on those prediction dates
    actual_pred_window = plot_actual[pred_positions]

    # Plot
    plt.figure(figsize=(12, 4))

    # Actual close for last context window
    plt.plot(x_full, plot_actual, label="Actual Close (last 30 days)", linestyle="-")

    # Actual & predicted only on last PRED_DAYS
    plt.plot(x_pred, actual_pred_window, label="Actual (last 5 days)", linestyle="-")
    plt.plot(x_pred, pred_closes, label="Predicted (last 5 days)", linestyle="--")

    # Vertical line: where last PRED_DAYS region starts
    first_pred_pos = pred_positions[0]
    plt.axvline(x=x_full[first_pred_pos], linestyle=":", linewidth=1)

    start_date_str = plot_dates[0].date().isoformat()
    end_date_str = plot_dates[-1].date().isoformat()

    plt.xlabel(
        f"Day index in last {len(plot_dates)} days "
        f"(1 = {start_date_str}, {len(plot_dates)} = {end_date_str})"
    )
    plt.ylabel("Close Price")
    plt.title(
        f"LM ANN (returns): Best-so-far model, epoch {epoch_num}"
    )
    plt.legend()
    fname = os.path.join(ann_plot_dir, f"lm_pred_epoch_{epoch_num:03d}.png")
    plt.tight_layout()
    plt.xticks(
        ticks=np.arange(1, len(plot_dates) + 1),
        labels=np.arange(1, len(plot_dates) + 1)
    )
    plt.savefig(fname, dpi=300)
    plt.close()

# ---------------- CUSTOM TRAINING LOOP WITH "ACCEPT ONLY BETTER" ----------------
history_lm = {"loss": [], "val_loss": []}
best_val = None
best_weights = model_lm.get_weights()
epochs_no_improve = 0
actual_epochs = 0

for epoch in range(1, MAX_EPOCHS + 1):
    # 1. Train one epoch (weights will move)
    model_wrapper.fit(
        X_train_flat,
        y_train_n,
        epochs=1,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_flat, y_test_n),
        verbose=0,
    )

    # 2. Evaluate with current weights
    train_loss_current = model_wrapper.evaluate(X_train_flat, y_train_n, verbose=0)
    val_loss_current = model_wrapper.evaluate(X_test_flat, y_test_n, verbose=0)

    # Guard: stop if NaNs ever appear
    if (not np.isfinite(train_loss_current)) or (not np.isfinite(val_loss_current)):
        print(
            f"Epoch {epoch}: train_loss={train_loss_current}, "
            f"val_loss={val_loss_current} -> NaN/inf detected, stopping."
        )
        break

    improved = False
    if (best_val is None) or ((best_val - val_loss_current) > MIN_DELTA):
        # Improvement -> accept new weights
        best_val = val_loss_current
        best_weights = model_lm.get_weights()
        epochs_no_improve = 0
        improved = True
    else:
        # No improvement -> revert to previous best weights
        model_lm.set_weights(best_weights)
        epochs_no_improve += 1
        # After reverting, recompute train loss (val loss is best_val)
        train_loss_current = model_wrapper.evaluate(X_train_flat, y_train_n, verbose=0)
        val_loss_current = best_val

    history_lm["loss"].append(train_loss_current)
    history_lm["val_loss"].append(val_loss_current)
    actual_epochs = epoch

    # 3. Status line
    print(
        f"Epoch {epoch}: train_loss={train_loss_current:.6f}, "
        f"val_loss={val_loss_current:.6f}, no_improve={epochs_no_improve}, "
        f"improved={improved}"
    )

    # 4. Save prediction plot ONLY when there is an improvement
    if improved:
        plot_predictions_epoch(
            model_lm,
            X_test_flat,
            y_test,
            y_mean,
            y_std,
            prev_close_test,
            dates_test,
            visible_close,
            CONTEXT_DAYS,
            epoch,
        )

    # 5. Stopping conditions

    # (a) reached target "good" MSE
    if (best_val is not None) and (best_val <= TARGET_VAL_MSE):
        print(
            f"Stopping at epoch {epoch}: "
            f"val_loss={best_val:.6f} <= TARGET_VAL_MSE={TARGET_VAL_MSE:.6f}"
        )
        break

    # (b) too many epochs with no improvement
    if epochs_no_improve >= PATIENCE:
        print(
            f"Early stopping at epoch {epoch} "
            f"(no val_loss improvement for {PATIENCE} epochs)."
        )
        break

# Restore best weights at the end
model_lm.set_weights(best_weights)

# ---------------- LOSS CURVE ----------------
epochs_range = range(1, actual_epochs + 1)
plt.figure()
plt.plot(epochs_range, history_lm["loss"], label="train_loss")
plt.plot(epochs_range, history_lm["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("MSE (on normalized returns)")
plt.title("LM ANN (returns): Loss vs Epoch (monotone best val_loss)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ann_plot_dir, "lm_loss_curve.png"), dpi=300)
plt.close()

print(f"ANN training complete after {actual_epochs} epochs. Plots saved in folder: {ann_plot_dir}")
