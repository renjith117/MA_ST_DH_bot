import os
import time
import logging
import schedule
import requests
import pytz
import pandas as pd

import pandas_ta as ta
from datetime import datetime, timedelta
import threading
from flask import Flask
from kiteconnect import KiteConnect

IST = pytz.timezone("Asia/Kolkata")

app = Flask(__name__)

@app.route("/")
def health():
    return "OK", 200
 
def run_health_server():
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
# ─────────────────────────────────────────────
#  CONFIGURATION  (edit before running)
# ─────────────────────────────────────────────
API_KEY      = os.environ["KITE_API_KEY"]
ACCESS_TOKEN = os.environ["KITE_ACCESS_TOKEN"]
TELEGRAM_TOKEN     = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID      = os.environ["TELEGRAM_CHAT_ID"]



#Watchlist: list of (exchange, tradingsymbol) tuples
#WATCHLIST = [
#    ("NSE", "RELIANCE"),
#    ("NSE", "INFY"),
#   ("NSE", "TCS"),
#   ("NSE", "ASKAUTOLTD"),
#   ("NSE", "NIFTY 50"),   # Index (use NFO options carefully)
#]

x = pd.read_csv('NIFTY_50_V1.csv')
WATCHLIST = list(zip(x["Exchange"], x["Stock"]))


# SuperTrend settings (Daily)
ST_PERIOD = 7
ST_MULTIPLIER = 3.0

# EMA settings (Hourly)
EMA_FAST = 5
EMA_SLOW = 10

# How many candles to fetch (keep enough for indicator warmup)
DAILY_CANDLES  = 100
HOURLY_CANDLES = 100

# Scan interval (minutes) – runs every N minutes during market hours
SCAN_INTERVAL_MINUTES = 5

# Send alert on SELL crossover too? Set False for buy-only alerts
ALERT_ON_SELL = True


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("indicator.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  KITE CONNECT SETUP
# ─────────────────────────────────────────────
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)


# ─────────────────────────────────────────────
#  TELEGRAM HELPER
# ─────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    """Send a message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        log.info("Telegram  message sent")
        return True
    except requests.RequestException as e:
        log.error(f"Telegram   {e}")
        return False


# ─────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────
def get_instrument_token(exchange: str, symbol: str) -> int | None:
    """Resolve symbol → instrument token (cached after first call)."""
    try:
        instruments = kite.instruments(exchange)
        for inst in instruments:
            if inst["tradingsymbol"] == symbol:
                return inst["instrument_token"]
    except Exception as e:
        log.error(f"Instrument lookup failed for {symbol}: {e}")
    return None


def fetch_ohlc(token: int, interval: str, days_back: int) -> pd.DataFrame:
    """
    Fetch historical OHLC data.
    interval: "day" | "60minute" | "15minute" etc.
    """
    now     = datetime.now(IST)
    from_dt = (now - timedelta(days=days_back)).replace(hour=9, minute=0, second=0, microsecond=0)
    to_dt   = now

    # strip timezone — Kite expects naive datetimes
    from_date = from_dt.replace(tzinfo=None) 
    to_date = to_dt.replace(tzinfo=None)

    try:
        records = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
        )
        df = pd.DataFrame(records)
        df.rename(columns={"date": "datetime"}, inplace=True)
        df.set_index("datetime", inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        return df
    except Exception as e:
        log.error(f"Data fetch error (token={token}, interval={interval}): {e}")
        return pd.DataFrame()


def calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculate SuperTrend using pandas_ta.
    Returns df with 'supertrend' (price line) and 'supertrend_direction' columns.
    direction: 1 = bullish (green), -1 = bearish (red)
    """
    st = ta.supertrend(
        df["high"], df["low"], df["close"],
        length=period, multiplier=multiplier
    )
    # pandas_ta names columns: SUPERT_P_M, SUPERTd_P_M, SUPERTl_P_M, SUPERTs_P_M
    dir_col = [c for c in st.columns if c.startswith("SUPERTd")][0]
    df = df.copy()
    df["st_direction"] = st[dir_col]   # 1 = bull, -1 = bear
    return df


def calc_ema_crossover(df: pd.DataFrame, fast: int = 5, slow: int = 10) -> pd.DataFrame:
    """
    Add fast/slow EMA and crossover signal.
    cross_signal:  1 = golden cross (fast crosses above slow)
                  -1 = death cross  (fast crosses below slow)
                   0 = no cross
    """
    df = df.copy()
    df[f"ema{fast}"]  = ta.ema(df["close"], length=fast)
    df[f"ema{slow}"]  = ta.ema(df["close"], length=slow)

    prev_fast  = df[f"ema{fast}"].shift(1)
    prev_slow  = df[f"ema{slow}"].shift(1)
    curr_fast  = df[f"ema{fast}"]
    curr_slow  = df[f"ema{slow}"]

    df["cross_signal"] = 0
    # Golden cross: fast was below slow, now above
    df.loc[(prev_fast < prev_slow) & (curr_fast >= curr_slow), "cross_signal"] =  1
    # Death cross: fast was above slow, now below
    df.loc[(prev_fast > prev_slow) & (curr_fast <= curr_slow), "cross_signal"] = -1
    return df

# ─────────────────────────────────────────────
#  ALERT MESSAGE BUILDER
# ─────────────────────────────────────────────
def build_alert(symbol: str, signal: int,
                close_h: float, ema5: float, ema10: float,
                st_dir: int) -> str:
    now   = datetime.now().strftime("%d-%b-%Y %H:%M")
    emoji = "🟢" if signal == 1 else "🔴"
    stype = "BUY 🚀" if signal == 1 else "SELL 📉"
    st_str = "🟢 Bullish" if st_dir == 1 else "🔴 Bearish"

    return (
        f"{emoji} <b>ALERT — {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 Signal      : <b>{stype}</b>\n"
        f"🕐 Time        : {now}\n"
        f"\n"
        f"<b>Daily  SuperTrend</b> : {st_str}\n"
        f"<b>Hourly EMA Cross</b>  : EMA{EMA_FAST} {'>' if signal==1 else '<'} EMA{EMA_SLOW}\n"
        f"\n"
        f"💰 Close (1H)  : ₹{close_h:.2f}\n"
        f"📊 EMA{EMA_FAST}         : ₹{ema5:.2f}\n"
        f"📊 EMA{EMA_SLOW}        : ₹{ema10:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Supertrend({ST_PERIOD},{ST_MULTIPLIER}) | "
        f"EMA({EMA_FAST},{EMA_SLOW})</i>"
    )



# ─────────────────────────────────────────────
#  CORE SCAN LOGIC
# ─────────────────────────────────────────────
# Track last cross direction to avoid duplicate alerts
_last_signal: dict[str, int] = {}


def scan_symbol(exchange: str, symbol: str):
    log.info(f"Scanning {exchange}:{symbol} …")

    token = get_instrument_token(exchange, symbol)
    if token is None:
        log.warning(f"  Token not found for {symbol}, skipping.")
        return

    # ── Daily data → SuperTrend ──────────────────
    daily_df = fetch_ohlc(token, "day", days_back=DAILY_CANDLES)
    if daily_df.empty or len(daily_df) < ST_PERIOD + 5:
        log.warning(f"  Insufficient daily data for {symbol}")
        return

    daily_df = calc_supertrend(daily_df, ST_PERIOD, ST_MULTIPLIER)
    latest_st_dir = int(daily_df["st_direction"].iloc[-1])

    if latest_st_dir != 1:
        log.info(f"  {symbol}: Daily SuperTrend is BEARISH — skipping cross check.")
        return

    log.info(f"  {symbol}: Daily SuperTrend is BULLISH ")

    # ── Hourly data → EMA crossover ─────────────
    # 100 hourly candles ≈ ~14 trading days
    hourly_df = fetch_ohlc(token, "60minute", days_back=20)
    if hourly_df.empty or len(hourly_df) < EMA_SLOW + 5:
        log.warning(f"  Insufficient hourly data for {symbol}")
        return

    hourly_df = calc_ema_crossover(hourly_df, EMA_FAST, EMA_SLOW)
    last_row   = hourly_df.iloc[-1]
    cross      = int(last_row["cross_signal"])

    if cross == 0:
        log.info(f"  {symbol}: No EMA crossover on last hourly candle.")
        return

    if cross == -1 and not ALERT_ON_SELL:
        log.info(f"  {symbol}: SELL cross detected but ALERT_ON_SELL is False.")
        return

    # Avoid re-sending the same signal
    prev = _last_signal.get(symbol, 0)
    if cross == prev:
        log.info(f"  {symbol}: Signal ({cross}) already sent, skipping duplicate.")
        return

    _last_signal[symbol] = cross

    msg = build_alert(
        symbol      = symbol,
        signal      = cross,
        close_h     = float(last_row["close"]),
        ema5        = float(last_row[f"ema{EMA_FAST}"]),
        ema10       = float(last_row[f"ema{EMA_SLOW}"]),
        st_dir      = latest_st_dir,
    )

    log.info(f"  {symbol}: {'BUY' if cross==1 else 'SELL'} alert triggered!")
    send_telegram(msg)

def scan_all():
    log.info("=" * 50)
    log.info(f"Running scan at {datetime.now().strftime('%H:%M:%S')}")
    for exchange, symbol in WATCHLIST:
        try:
            scan_symbol(exchange, symbol)
        except Exception as e:
            log.error(f"Unexpected error scanning {symbol}: {e}")
        time.sleep(0.5)   # gentle rate-limit between symbols
    log.info("Scan complete.")
# ─────────────────────────────────────────────
#  SCHEDULER
# ─────────────────────────────────────────────
def is_market_hours() -> bool:
    """Return True if current IST time is within NSE market hours."""
    now = datetime.now()
    # NSE: Mon–Fri, 09:15–15:30 IST
    if now.weekday() >= 5:       # Saturday / Sunday
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def scheduled_scan():
    scan_all()

if __name__ == "__main__":
    t = threading.Thread(target=run_health_server, daemon=True)
    t.start()
    log.info("SuperTrend + EMA Crossover Alert Bot started")
    send_telegram(
        "🤖 <b>Alert Bot Started</b>\n"
        f"Watching {len(WATCHLIST)} symbols\n" 
        f"Strategy: Daily ST({ST_PERIOD},{ST_MULTIPLIER}) + "
        f"Hourly EMA({EMA_FAST},{EMA_SLOW})\n"
        f"Scan every {SCAN_INTERVAL_MINUTES} min"
    )

    # Run once immediately, then on schedule
    scheduled_scan()
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(scheduled_scan)

    while True:
        schedule.run_pending()
        time.sleep(30)
   