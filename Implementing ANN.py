import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from matplotlib.ticker import MaxNLocator

print("Compiling Tasks")

# Today label
today = dt.date.today()
date_str = today.strftime('%d-%m-%y')

# Inputs
T = input("Enter Ticker Symbol (eg. TSLA): ").strip().upper()
prd = input("Enter period (e.g., '1y' for 1 year): ").strip()
ma_type = input("Moving Average Type (SMA/EMA): ").strip().upper()
p1 = int(input("MA -1: "))
p2 = int(input("MA -2: "))

# Indicator settings
rsi_period = 14

# Parse period (supports: Nd, Nwk, Nmo, Ny, and ytd)
def parse_period_to_start(period: str, ref_date: dt.date) -> dt.date:
    p = period.lower().strip()
    if p == 'ytd':
        return dt.date(ref_date.year, 1, 1)
    if p.endswith('y'):
        years = int(p[:-1])
        return ref_date - relativedelta(years=years)
    if p.endswith('mo'):
        months = int(p[:-2])
        return ref_date - relativedelta(months=months)
    if p.endswith('wk'):
        weeks = int(p[:-2])
        return ref_date - relativedelta(weeks=weeks)
    if p.endswith('d'):
        days = int(p[:-1])
        return ref_date - relativedelta(days=days)
    # Fallback: assume years if only a number was passed
    if p.isdigit():
        return ref_date - relativedelta(years=int(p))
    raise ValueError("Unsupported period format. Use like '1y', '6mo', '4wk', '30d', or 'ytd'.")

base_start = parse_period_to_start(prd, today)  # visible window start
# Buffer so first visible day has valid MA/RSI values
if prd.lower().endswith('y'):
    buffer_rel = relativedelta(years=1)  # add 1 more year for yearly windows
else:
    buffer_days = max(p1, p2, rsi_period) + 5
    buffer_rel = relativedelta(days=buffer_days)

download_start = base_start - buffer_rel
download_end = today

# Download data robustly (handle multi-index vs single-index columns)
def download_close(ticker: str, start_date: dt.date, end_date: dt.date) -> pd.Series:
    try:
        df = yf.download(
            tickers=[ticker],
            start=start_date.isoformat(),
            end=(end_date + relativedelta(days=1)).isoformat(),  # inclusive end
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
            interval='1d'
        )
    except TypeError:
        df = yf.download(
            tickers=ticker,
            start=start_date.isoformat(),
            end=(end_date + relativedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
            interval='1d'
        )
    if df is None or df.empty:
        raise ValueError("No data returned for the given ticker and dates.")
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

# Compute MAs (SMA/EMA)
def compute_ma(series: pd.Series, period: int, mode: str) -> pd.Series:
    mode = mode.upper()
    if mode == 'SMA':
        return series.rolling(window=period, min_periods=period).mean()
    if mode == 'EMA':
        return series.ewm(span=period, adjust=False, min_periods=period).mean()
    raise ValueError("Invalid MA type. Use 'SMA' or 'EMA'.")

ma1_full = compute_ma(close, p1, ma_type)
ma2_full = compute_ma(close, p2, ma_type)

# RSI (Wilder smoothing via ewm alpha=1/period)
delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
rs = avg_gain / avg_loss
rsi_full = 100 - (100 / (1 + rs))
rsi_full = rsi_full.clip(lower=0, upper=100)

# MACD
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd_full = ema12 - ema26
signal_full = macd_full.ewm(span=9, adjust=False).mean()

#Bollinger Bands
bb_mid_full = close.rolling(20).mean()
bb_std_full = close.rolling(20).std()
bb_upper_full = bb_mid_full + 2 * bb_std_full
bb_lower_full = bb_mid_full - 2 * bb_std_full

# Slice visible window (start at base_start to today)
visible_close = close.loc[pd.Timestamp(base_start): pd.Timestamp(today)]
if visible_close.empty:
    raise ValueError("No visible data in the requested window; check the period or ticker.")
N = len(visible_close)
x_all = np.arange(1, N + 1)

# Align indicator series to visible window
ma1 = ma1_full.loc[visible_close.index].to_numpy()
ma2 = ma2_full.loc[visible_close.index].to_numpy()
px = x_all
py = visible_close.to_numpy()
rsi = rsi_full.loc[visible_close.index].to_numpy()
macd = macd_full.loc[visible_close.index].to_numpy()
signal = signal_full.loc[visible_close.index].to_numpy()
hist = macd - signal
bb_mid = bb_mid_full.loc[visible_close.index].to_numpy()
bb_upper = bb_upper_full.loc[visible_close.index].to_numpy()
bb_lower = bb_lower_full.loc[visible_close.index].to_numpy()

# Detect crossovers on visible window
# Detect MACD crossovers
macd_buy_x, macd_buy_y = [], []
macd_sell_x, macd_sell_y = [], []

sign = np.sign(macd - signal)
cross_idx = np.where(np.diff(sign) != 0)[0] + 1

for i in cross_idx:
    if macd[i] > signal[i]:
        macd_buy_x.append(px[i])
        macd_buy_y.append(macd[i])
    else:
        macd_sell_x.append(px[i])
        macd_sell_y.append(macd[i])
# Ensure numeric arrays without NaN before detecting sign changes
mask_valid = np.isfinite(ma1) & np.isfinite(ma2)
if not mask_valid.all():
    # If buffer was insufficient, trim to the first fully valid index
    first_valid = int(np.argmax(mask_valid))
    if not mask_valid[first_valid]:
        # No valid region
        first_valid = N
else:
    first_valid = 0

ma1v = ma1[first_valid:]
ma2v = ma2[first_valid:]
xv = px[first_valid:]

gx, gy, dx, dy = [], [], [], []
if len(ma1v) > 1:
    sign = np.sign(ma1v - ma2v)
    cross_idx = np.where(np.diff(sign) != 0)[0] + 1  # indices in ma1v/ma2v
    for ci in cross_idx:
        xi = int(xv[ci])
        yi = float(ma1v[ci])
        if ma1v[ci] > ma2v[ci]:
            gx.append(xi)
            gy.append(yi)
        else:
            dx.append(xi)
            dy.append(yi)

# Plot: price + MAs + filled crossovers on top; RSI below
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(12, 10), sharex=True,
    gridspec_kw={'height_ratios': [3, 1, 1]}
)

# Top: price and MAs
ax1.plot(px, py, color='blue', label='Closing Price')
ax1.plot(px, ma1, color='red', label=f'{ma_type} - {p1} Days')
ax1.plot(px, ma2, color='green', label=f'{ma_type} - {p2} Days')

# Filled crossover markers: green up for golden, red down for death
if gx:
    ax1.scatter(gx, gy, marker='^', s=60, color='green', linewidths=0, zorder=3, label='Golden Crossover')
if dx:
    ax1.scatter(dx, dy, marker='v', s=60, color='red', linewidths=0, zorder=3, label='Death Crossover')
# Bollinger Bands lines
ax1.plot(px, bb_upper, linestyle='--', linewidth=1, label='BB Upper')
ax1.plot(px, bb_mid, linestyle='--', linewidth=1, label='BB Mid')
ax1.plot(px, bb_lower, linestyle='--', linewidth=1, label='BB Lower')

# Shade between bands
ax1.fill_between(px, bb_lower, bb_upper, alpha=0.15)
ax1.set_ylabel("Price / MAs")
ax1.legend(loc='best')

# Bottom: RSI
ax2.plot(px, rsi, color='purple', label=f'RSI {rsi_period}')
ax2.axhline(70, color='black', linestyle='--', linewidth=1)
ax2.axhline(30, color='black', linestyle='--', linewidth=1)
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI")
ax2.legend(loc='upper left')

#MACD Plot
ax3.plot(px, macd, label='MACD')
ax3.plot(px, signal, label='Signal')
ax3.bar(px, hist, alpha=0.3, label='Histogram')
ax3.set_ylabel("MACD")
ax3.legend(loc='upper left')
# Plot crossover dots
ax3.scatter(macd_buy_x, macd_buy_y, color='green', s=10, zorder=3, label='MACD Buy')
ax3.scatter(macd_sell_x, macd_sell_y, color='red', s=10, zorder=3, label='MACD Sell')

# X axis starts at 1 and aligns flush left
ax1.set_xlim(1, int(px.max()))
ax1.margins(x=0)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.set_xlabel(f"Day count from {visible_close.index[0].date().isoformat()}")

# Vertical lines for Golden Cross
for x in gx:
    ax1.axvline(x=x, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax2.axvline(x=x, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax3.axvline(x=x, color='green', linestyle='--', linewidth=1, alpha=0.6)

# Vertical lines for Death Cross
for x in dx:
    ax1.axvline(x=x, color='red', linestyle='--', linewidth=1, alpha=0.6)
    ax2.axvline(x=x, color='red', linestyle='--', linewidth=1, alpha=0.6)
    ax3.axvline(x=x, color='red', linestyle='--', linewidth=1, alpha=0.6)

# Title and save
title_tkr = T.replace('.', '')
plt.suptitle(f"{prd} {ma_type} ({p1} | {p2}) | {T} | up to {date_str}")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{date_str} {title_tkr}.png", dpi=600)
plt.show()
