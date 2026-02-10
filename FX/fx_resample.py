import pandas as pd
from typing import Dict, Iterable

# ------------------------------------------------------------
# TIMEFRAME MAP: human-readable → pandas resample rule
# ------------------------------------------------------------
TIMEFRAME_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
}


# ------------------------------------------------------------
# 1) LOAD RAW 10-SECOND FX DATA
# ------------------------------------------------------------
def load_raw_ohlc(csv_path: str) -> pd.DataFrame:
    """
    Load raw FX OHLC data from CSV with columns:
    timestamp, symbol, open, high, low, close
    
    Returns a DataFrame indexed by timestamp (UTC), sorted by time.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column.")

    # Parse timestamp as timezone-aware and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # Ensure OHLC columns exist and are numeric
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"CSV must contain an '{col}' column.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where any OHLC is missing
    df = df.dropna(subset=["open", "high", "low", "close"])

    return df


# ------------------------------------------------------------
# 2) RESAMPLE TO A SINGLE TIMEFRAME
# ------------------------------------------------------------
def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    df_resampled = df.resample(rule).agg(agg_dict)
    df_resampled = df_resampled.dropna()

    return df_resampled


# ------------------------------------------------------------
# 3) BUILD MULTIPLE TIMEFRAMES AT ONCE
# ------------------------------------------------------------
def build_timeframes(
    csv_path: str,
    timeframes: Iterable[str] = ("1m", "5m", "15m", "30m"),
) -> Dict[str, pd.DataFrame]:
    df_raw = load_raw_ohlc(csv_path)
    out = {}

    for tf in timeframes:
        if tf not in TIMEFRAME_MAP:
            raise ValueError(
                f"Unknown timeframe '{tf}'. "
                f"Allowed: {list(TIMEFRAME_MAP.keys())}"
            )

        rule = TIMEFRAME_MAP[tf]
        df_tf = resample_ohlc(df_raw, rule)
        out[tf] = df_tf

    return out


# ------------------------------------------------------------
# 4) OPTIONAL: QUICK TEST / DEMO
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example usage: change this path to your AUD/USD CSV
    csv_path = r"C:\Data\Job\UK\GoodMarkets\questdb-audusd.csv"

    frames = build_timeframes(csv_path, timeframes=["1m", "5m", "15m", "30m"])

    for tf, df_tf in frames.items():
        print(f"\n=== {tf} ===")
        print(df_tf.head())
        print("Rows:", len(df_tf))
