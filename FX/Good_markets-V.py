import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fx_resample import build_timeframes

from analysis_plots import (
    plot_price_with_ema,
    plot_forward_return_comparison,
    plot_trend_comparison,
    plot_direction_accuracy_series,
    plot_accuracy_vs_abs_pred,
    summarize_confidence_filter,
    plot_price_prediction_nextclose
)

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"C:\Data\Job\UK\GoodMarkets"
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMEFRAME = "5m"       # "5m" or "10m" recommended for trend
FUTURE_STEPS = 100       # multi-step horizon in bars (e.g., 50x10m = 500min)

EVAL_H = 24  # try 6, 12, 24, 50
CONF_PERCENTILE = 75

# ----- SAFE PANDAS FREQ (FIX) -----
# Use this for pd.date_range (do NOT use freq=TIMEFRAME directly)
PANDAS_FREQ_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
}
if TIMEFRAME not in PANDAS_FREQ_MAP:
    raise ValueError(f"Unsupported TIMEFRAME='{TIMEFRAME}'. Allowed={list(PANDAS_FREQ_MAP.keys())}")
PANDAS_FREQ = PANDAS_FREQ_MAP[TIMEFRAME]

# ----- DATE FILTERING -----
USE_DATE_FILTER = True
START_DATE = "2024-01-01T00:00:00Z"
END_DATE   = "2025-11-14T21:59:50Z"

# ----- TRAIN / TEST SPLIT USING DATES -----
TRAIN_END = "2025-06-01T00:00:00Z"
TEST_END  = "2025-11-14T21:59:50Z"

# ----- WALK-FORWARD CONFIG -----
# IMPORTANT: these are sequence-window sizes (not raw bars).
WINDOW_SIZE = 20000
STEP_SIZE   = 5000
WFV_TEST_SLICE = 10         # how many sequences to test per window in WFV

# ----- OPTUNA -----
N_TRIALS = 3                 # increase later
SKIP_TUNING_IF_BEST_EXISTS = True  # <--- you requested this behavior

# ----- STRICT WFV TRAINING -----
WFV_VAL_SPLIT = 0.2
WFV_MIN_SAMPLES = 300
WFV_MAX_EPOCHS = 30
WFV_PATIENCE = 3

# ----- MODEL TRAINING (FINAL) -----
FINAL_EPOCHS = 20

# ----- FEATURE SET (minimal but trend-aligned) -----
FEATURE_COLS = ["ret1", "ema20", "ema50", "ema20_slope", "atr14", "close_minus_ema20"]
GLOBAL_SEED = 42


# Global containers
TRIAL_HISTORY = []      # per-pair trial history
SUMMARY = []            # overall metrics summary across pairs


# ============================================================
# DATA LOADING (UTC SAFE) + DATE FILTER
# ============================================================
def load_data(filename):
    path = os.path.join(BASE_DIR, filename)
    frames = build_timeframes(path, [TIMEFRAME])
    df = frames[TIMEFRAME].dropna()

    df.index = pd.to_datetime(df.index, utc=True)

    if USE_DATE_FILTER:
        start_ts = pd.to_datetime(START_DATE, utc=True)
        end_ts   = pd.to_datetime(END_DATE, utc=True)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    return df


# ============================================================
# TREND FEATURES
# ============================================================
def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_close"] = np.log(df["close"])
    df["ret1"] = df["log_close"].diff()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema20_slope"] = df["ema20"].diff()

    df["hl_range"] = df["high"] - df["low"]
    df["atr14"] = df["hl_range"].rolling(14).mean()

    df["close_minus_ema20"] = df["close"] - df["ema20"]

    #df = df.dropna()
    return df


# ============================================================
# BUILD MULTI-STEP TREND TARGET PATH (forward log returns)
#   For each t: y_path[j] = log(C_{t+j} / C_t), j=1..FUTURE_STEPS
# ============================================================
def create_supervised_sequences(df: pd.DataFrame, feature_cols, time_step: int, future_steps: int):
    X, Y, idx = [], [], []

    feat_vals = df[feature_cols].values
    close_vals = df["close"].values
    ts_index = df.index

    # Need enough room for future_steps
    for i in range(time_step, len(df) - future_steps):
        X.append(feat_vals[i - time_step:i])
        c0 = close_vals[i]
        future = close_vals[i + 1:i + 1 + future_steps]
        y_path = np.log(future / c0)
        Y.append(y_path)
        idx.append(ts_index[i])

    return np.asarray(X), np.asarray(Y), pd.Index(idx)

# ============================================================
# MODEL BUILDER (multi-feature input)
# ============================================================
def build_model(time_step, n_features, filters, kernel, dense_units, dropout, lr, future_steps):
    inputs = Input(shape=(time_step, n_features))

    x = Conv1D(filters, kernel, padding="causal", activation="relu")(inputs)
    x = Conv1D(filters, kernel, padding="causal", activation="relu")(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(filters * 2, kernel, padding="causal", activation="relu")(x)
    x = MaxPooling1D(2)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)

    outputs = Dense(future_steps)(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


# ============================================================
# PARAM CACHE HELPERS (SAVE/LOAD BEST PARAMS)
# ============================================================
def best_params_path(pair_name: str) -> str:
    # include timeframe and future steps to avoid accidental reuse across different settings
    fname = f"{pair_name}_best_params_{TIMEFRAME}_fs{FUTURE_STEPS}.json"
    return os.path.join(OUTPUT_DIR, fname)

def load_best_params(pair_name: str):
    path = best_params_path(pair_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_best_params(pair_name: str, params: dict):
    path = best_params_path(pair_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return path


# ============================================================
# OPTUNA OBJECTIVE FACTORY (WFV ON TRAIN ONLY)
# ============================================================
def make_objective(train_df_scaled: pd.DataFrame, feature_cols, future_steps: int):

    train_len = len(train_df_scaled)

    def objective(trial):
        # Reproducibility
        tf.keras.utils.set_random_seed(GLOBAL_SEED + int(trial.number))
        np.random.seed(GLOBAL_SEED + int(trial.number))

        time_step  = trial.suggest_int("time_step", 40, 80)
        filters    = trial.suggest_categorical("filters", [32, 64])
        kernel     = trial.suggest_categorical("kernel", [3, 5])
        dense_units= trial.suggest_categorical("dense_units", [32, 64])
        dropout    = trial.suggest_float("dropout", 0.05, 0.25)
        lr         = trial.suggest_float("lr", 5e-4, 2e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256])

        print(
            f"\n[Trial {trial.number}] ts={time_step}, filters={filters}, kernel={kernel}, "
            f"dense={dense_units}, dropout={dropout:.3f}, lr={lr:.6f}, batch={batch_size}",
            flush=True
        )

        # Build sequences for the whole train-only series for this time_step
        X_all, Y_all, _ = create_supervised_sequences(
            train_df_scaled,
            feature_cols=feature_cols,
            time_step=time_step,
            future_steps=future_steps
        )

        if len(X_all) < max(500, WINDOW_SIZE + 50):
            # Not enough sequences to do meaningful WFV
            mean_rmse = 1e9
            TRIAL_HISTORY.append({
                "trial": trial.number,
                "time_step": time_step,
                "filters": filters,
                "kernel": kernel,
                "dense_units": dense_units,
                "dropout": dropout,
                "learning_rate": lr,
                "batch_size": batch_size,
                "mean_rmse": mean_rmse,
            })
            return mean_rmse

        rmses = []

        max_idx = len(X_all) - WINDOW_SIZE
        if max_idx <= 0:
            raise ValueError("WINDOW_SIZE is too large compared to available sequences in training.")

        for start_idx in range(0, max_idx, STEP_SIZE):
            end_idx = start_idx + WINDOW_SIZE

            X_train = X_all[start_idx:end_idx]
            y_train = Y_all[start_idx:end_idx]

            X_test = X_all[end_idx:end_idx + WFV_TEST_SLICE]
            y_test = Y_all[end_idx:end_idx + WFV_TEST_SLICE]
            if len(X_test) == 0:
                break

            tf.keras.backend.clear_session()
            model = build_model(
                time_step=time_step,
                n_features=len(feature_cols),
                filters=filters,
                kernel=kernel,
                dense_units=dense_units,
                dropout=dropout,
                lr=lr,
                future_steps=future_steps
            )

            # Time-based validation split inside the window
            n = len(X_train)
            if n >= WFV_MIN_SAMPLES:
                split = int(n * (1.0 - WFV_VAL_SPLIT))
                X_tr, y_tr = X_train[:split], y_train[:split]
                X_val, y_val = X_train[split:], y_train[split:]

                es = EarlyStopping(monitor="val_loss", patience=WFV_PATIENCE, restore_best_weights=True)
                model.fit(
                    X_tr, y_tr,
                    validation_data=(X_val, y_val),
                    epochs=WFV_MAX_EPOCHS,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[es],
                )
            else:
                es = EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=8, batch_size=batch_size, verbose=0, callbacks=[es])

            pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test.flatten(), pred.flatten())))
            rmses.append(rmse)

            if len(rmses) % 5 == 0:
                print(f"  [Trial {trial.number}] window {start_idx}/{max_idx}, last RMSE={rmse:.6f}", flush=True)

        mean_rmse = float(np.mean(rmses)) if rmses else 1e9
        print(f"[Trial {trial.number}] mean WFV RMSE = {mean_rmse:.6f}", flush=True)

        TRIAL_HISTORY.append({
            "trial": trial.number,
            "time_step": time_step,
            "filters": filters,
            "kernel": kernel,
            "dense_units": dense_units,
            "dropout": dropout,
            "learning_rate": lr,
            "batch_size": batch_size,
            "mean_rmse": mean_rmse,
        })

        return mean_rmse

    return objective

# ============================================================
# MAIN: RUN ONE PAIR
# ============================================================
def run_optuna_forecast(filename, pair_name):
    global TRIAL_HISTORY, SUMMARY
    TRIAL_HISTORY = []

    # --------------------------------------------------------
    # 1) Load + features
    # --------------------------------------------------------
    df = load_data(filename)
    df_feat = add_trend_features(df)

    feat_csv_path = os.path.join(OUTPUT_DIR, f"{pair_name}_{TIMEFRAME}_features.csv")
    df_feat.to_csv(feat_csv_path)
    print(f"[{pair_name}] Saved feature bars CSV:", feat_csv_path)

    # --------------------------------------------------------
    # 2) Date split
    # --------------------------------------------------------
    train_ts = pd.to_datetime(TRAIN_END, utc=True)
    test_ts  = pd.to_datetime(TEST_END,  utc=True)

    train_df = df_feat.loc[:train_ts].copy()
    test_df  = df_feat.loc[train_ts:test_ts].copy()

    print(f"\n{pair_name}: Train bars = {len(train_df)}, Test bars = {len(test_df)}")
    if len(train_df) < 1000:
        raise ValueError(
            f"{pair_name}: Not enough training bars after filters/features. "
            f"Increase date range or use smaller timeframe."
        )
    if len(test_df) < 200:
        print(f"[{pair_name}] Warning: small test set. Consider extending TEST_END/date range.")

    # --------------------------------------------------------
    # 3) Scale features TRAIN-ONLY
    # --------------------------------------------------------
    feat_scaler = MinMaxScaler()
    train_feat_scaled = feat_scaler.fit_transform(train_df[FEATURE_COLS].values)
    test_feat_scaled  = feat_scaler.transform(test_df[FEATURE_COLS].values)

    train_df_scaled = pd.DataFrame(train_feat_scaled, index=train_df.index, columns=FEATURE_COLS)
    test_df_scaled  = pd.DataFrame(test_feat_scaled,  index=test_df.index,  columns=FEATURE_COLS)

    # Keep close (unscaled) for target path creation
    train_df_scaled["close"] = train_df["close"].values
    test_df_scaled["close"]  = test_df["close"].values

    # --------------------------------------------------------
    # 4) Hyperparameter selection (cached or Optuna)
    # --------------------------------------------------------
    cached = load_best_params(pair_name) if SKIP_TUNING_IF_BEST_EXISTS else None

    if cached is not None:
        best_params = cached["best_params"]
        best_rmse = float(cached.get("best_mean_wfv_rmse", np.nan))
        print(f"\n[{pair_name}] Loaded cached best params:", best_params)
        print(f"[{pair_name}] Cached best mean WFV RMSE:", best_rmse)
    else:
        objective_fn = make_objective(train_df_scaled, FEATURE_COLS, FUTURE_STEPS)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_fn, n_trials=N_TRIALS)

        best_params = study.best_params
        best_rmse   = float(study.best_value)

        print(f"\n[{pair_name}] BEST PARAMS:", best_params)
        print(f"[{pair_name}] BEST mean WFV RMSE: {best_rmse:.6f}")

        trial_df = pd.DataFrame(TRIAL_HISTORY)
        trial_hist_path = os.path.join(OUTPUT_DIR, f"{pair_name}_trial_history.csv")
        trial_df.to_csv(trial_hist_path, index=False)
        print(f"[{pair_name}] Saved trial history:", trial_hist_path)

        cache_payload = {
            "pair": pair_name,
            "timeframe": TIMEFRAME,
            "future_steps": FUTURE_STEPS,
            "train_end": TRAIN_END,
            "test_end": TEST_END,
            "best_mean_wfv_rmse": best_rmse,
            "best_params": best_params,
        }
        saved_path = save_best_params(pair_name, cache_payload)
        print(f"[{pair_name}] Saved best params cache:", saved_path)

    ts = int(best_params["time_step"])
    f  = int(best_params["filters"])
    k  = int(best_params["kernel"])
    d  = int(best_params["dense_units"])
    dr = float(best_params["dropout"])
    lr = float(best_params["lr"])
    bs = int(best_params["batch_size"])

    # --------------------------------------------------------
    # 5) Train final model
    # --------------------------------------------------------
    tf.keras.utils.set_random_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    tf.keras.backend.clear_session()

    X_train_all, Y_train_all, _ = create_supervised_sequences(
        train_df_scaled, FEATURE_COLS, time_step=ts, future_steps=FUTURE_STEPS
    )
    if len(X_train_all) == 0:
        raise ValueError(
            f"{pair_name}: Not enough training rows to build sequences for "
            f"time_step={ts}, FUTURE_STEPS={FUTURE_STEPS}"
        )

    model = build_model(ts, len(FEATURE_COLS), f, k, d, dr, lr, FUTURE_STEPS)

    n = len(X_train_all)
    split = int(n * 0.8)
    X_tr, y_tr = X_train_all[:split], Y_train_all[:split]
    X_val, y_val = X_train_all[split:], Y_train_all[split:]

    es = EarlyStopping(monitor="val_loss", patience=WFV_PATIENCE, restore_best_weights=True)
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=FINAL_EPOCHS,
        batch_size=bs,
        verbose=1,
        callbacks=[es],
    )

    # --------------------------------------------------------
    # 6) Test evaluation + ALL analysis plots (NO REPEATS)
    # --------------------------------------------------------
    X_test_all, Y_test_all, idx_test = create_supervised_sequences(
        test_df_scaled, FEATURE_COLS, time_step=ts, future_steps=FUTURE_STEPS
    )

    test_metrics = {}

    if len(X_test_all) > 0:
        Y_test_pred = model.predict(X_test_all, verbose=0)

        # Global error metrics (all horizons)
        test_rmse = float(np.sqrt(mean_squared_error(Y_test_all.flatten(), Y_test_pred.flatten())))
        test_mae  = float(mean_absolute_error(Y_test_all.flatten(), Y_test_pred.flatten()))

        # Trend comparison (full horizon)
        plot_trend_comparison(
            idx=idx_test,
            y_true_path=Y_test_all,
            y_pred_path=Y_test_pred,
            pair_name=pair_name,
            timeframe=TIMEFRAME,
            output_dir=OUTPUT_DIR,
            horizon_label=f"{FUTURE_STEPS}bars",
        )

        H = int(EVAL_H)
        if H < 1 or H > FUTURE_STEPS:
            raise ValueError(f"EVAL_H must be in [1, {FUTURE_STEPS}], got {H}")

        # Use cumulative forward return at horizon H (already cumulative by construction)
        y_true_H = Y_test_all[:, H - 1]
        y_pred_H = Y_test_pred[:, H - 1]

        dir_acc = float(np.mean(np.sign(y_true_H) == np.sign(y_pred_H)))

        threshold = float(np.percentile(np.abs(y_pred_H), CONF_PERCENTILE))
        conf_mask = np.abs(y_pred_H) >= threshold

        y_true_f = y_true_H[conf_mask]
        y_pred_f = y_pred_H[conf_mask]
        idx_test_f = idx_test[conf_mask]

        dir_acc_f = float(np.mean(np.sign(y_true_f) == np.sign(y_pred_f))) if len(y_true_f) > 0 else np.nan
        trade_coverage = float(np.mean(conf_mask))


        plot_forward_return_comparison(idx_test, y_true_H, y_pred_H, pair_name, TIMEFRAME, OUTPUT_DIR, skip=50)
        plot_direction_accuracy_series(idx_test, y_true_H, y_pred_H, pair_name, TIMEFRAME, OUTPUT_DIR, roll_window=100)
        plot_accuracy_vs_abs_pred(y_true_H, y_pred_H, pair_name, TIMEFRAME, OUTPUT_DIR, n_bins=12)
        summarize_confidence_filter(y_true_H, y_pred_H, pair_name, TIMEFRAME, OUTPUT_DIR, percentiles=[50,60,70,80,90])


        # Optional: filtered rolling accuracy (only if enough kept trades)
        if len(y_true_f) >= 200:
            plot_direction_accuracy_series(
                idx_test_f, y_true_f, y_pred_f,
                pair_name + "_FILTERED", TIMEFRAME, OUTPUT_DIR, roll_window=100
            )

        # Logging
        print(f"\n[{pair_name}] TEST RMSE (all steps) = {test_rmse:.6f}")
        print(f"[{pair_name}] TEST MAE  (all steps) = {test_mae:.6f}")
        print(f"[{pair_name}] Direction Accuracy (1-step, all) = {dir_acc:.4f}")
        print(
            f"[{pair_name}] Direction Accuracy (|pred|>p{CONF_PERCENTILE}) = "
            f"{dir_acc_f:.4f}  (coverage={trade_coverage:.2%})"
        )

        # Save predictions CSV (kept + adds abs_pred + trade mask)
        pred_df = pd.DataFrame(index=idx_test)
        pred_df[f"y_true_H{H}"] = y_true_H
        pred_df[f"y_pred_H{H}"] = y_pred_H
        pred_df[f"abs_pred_H{H}"] = np.abs(y_pred_H)
        pred_df[f"take_trade_H{H}"] = conf_mask.astype(int)
        pred_df["take_trade"] = conf_mask.astype(int)

        for h in [5, 10, 20, min(49, FUTURE_STEPS - 1)]:
            if h < FUTURE_STEPS:
                pred_df[f"y_true_h{h+1}"] = Y_test_all[:, h]
                pred_df[f"y_pred_h{h+1}"] = Y_test_pred[:, h]

        pred_csv_path = os.path.join(OUTPUT_DIR, f"{pair_name}_test_predictions_returns.csv")
        pred_df.to_csv(pred_csv_path)
        print(f"[{pair_name}] Saved test predictions CSV:", pred_csv_path)

        # Filtered trades CSV (new)
        trades_df = pred_df.loc[idx_test_f].copy()
        trades_csv_path = os.path.join(OUTPUT_DIR, f"{pair_name}_test_trades_filtered.csv")
        trades_df.to_csv(trades_csv_path)
        print(f"[{pair_name}] Saved filtered trades CSV:", trades_csv_path)

        # Metrics saved to summary CSV
        test_metrics.update({
            "test_rmse_all_steps": test_rmse,
            "test_mae_all_steps": test_mae,
            "test_dir_acc_1step": dir_acc,
            "test_dir_acc_1step_filtered": dir_acc_f,
            "conf_percentile": CONF_PERCENTILE,
            "conf_threshold": threshold,
            "conf_trade_coverage": trade_coverage,
            "filtered_trades": int(conf_mask.sum()),
            "test_sequences": int(len(X_test_all)),
        })
    else:
        print(f"[{pair_name}] No test sequences produced (test too small for time_step+future_steps).")

    # --------------------------------------------------------
    # 7) Price plot (next-close reconstruction) -> plot file
    # --------------------------------------------------------
    df_all_scaled = pd.concat([train_df_scaled, test_df_scaled], axis=0)
    X_all, _, idx_all = create_supervised_sequences(
        df_all_scaled, FEATURE_COLS, time_step=ts, future_steps=FUTURE_STEPS
    )
    Y_all_pred = model.predict(X_all, verbose=0)
    y1_pred = Y_all_pred[:, 0]

    close_series = df_feat["close"]
    close_t = close_series.reindex(idx_all).values
    close_pred_next = close_t * np.exp(y1_pred)

    # ============================================================
    # 7.6) SAVE ACTUAL vs PREDICTED FEATURES (side-by-side)
    #   Predicted columns are based on predicted NEXT close at t+1.
    # ============================================================

    # 1) Start from the raw bars (NOT df_feat, because df_feat was dropna’d)
    df_base = df.copy()
    df_base.index = pd.to_datetime(df_base.index, utc=True)

    # 2) Build a predicted-close series on the same index as df_base
    close_pred_series = df_base["close"].copy()

    # Map anchor timestamp t -> next timestamp t+1, then assign close_pred at t+1
    for t, p in zip(idx_all, close_pred_next):
        loc = df_base.index.get_indexer([t])[0]
        if loc != -1 and (loc + 1) < len(df_base.index):
            t1 = df_base.index[loc + 1]
            close_pred_series.loc[t1] = p

    # 3) Compute features for ACTUAL and PREDICTED (same formulas)
    df_act = add_trend_features(df_base)
    df_pred_base = df_base.copy()
    df_pred_base["close"] = close_pred_series
    df_pred = add_trend_features(df_pred_base)

    # 4) Build comparison dataframe
    cols = [
        "close",
        "log_close","ret1","ema20","ema50","ema20_slope",
        "hl_range","atr14","close_minus_ema20"
    ]

    cmp = pd.DataFrame(index=df_base.index)
    cmp.index.name = "timestamp"

    for c in cols:
        # Actual & predicted
        cmp[f"{c}_actual"] = df_act[c]
        cmp[f"{c}_pred"]   = df_pred[c]

        # Absolute error
        cmp[f"{c}_abserr"] = abs(cmp[f"{c}_pred"] - cmp[f"{c}_actual"])

        # Relative error (%) = 100 * (pred - actual) / |actual|
        denom = np.abs(cmp[f"{c}_actual"]).replace(0.0, np.nan)
        cmp[f"{c}_relerr_pct"] = abs(100.0 * cmp[f"{c}_abserr"] / denom)

    # 5) Drop rows where key fields are not available (burn-in from EMA/ATR, plus last bar)
    # You can tighten/loosen this rule. This keeps the table “complete”.
    need = ["close_actual","close_pred","ema20_actual","ema20_pred","atr14_actual","atr14_pred"]
    cmp = cmp.dropna(subset=need)

    # 6) Save
    cmp_path = os.path.join(OUTPUT_DIR, f"{pair_name}_{TIMEFRAME}_actual_pred_features.csv")
    cmp.to_csv(cmp_path)
    print(f"[{pair_name}] Saved actual/pred features CSV:", cmp_path)

    N = len(df_feat)
    timestamps = df_feat.index
    actual_prices = df_feat["close"].values

    train_plot = np.full(N, np.nan)
    test_plot  = np.full(N, np.nan)

    train_end_ts = pd.to_datetime(TRAIN_END, utc=True)
    loc_map = {t: i for i, t in enumerate(timestamps)}

    for t, p in zip(idx_all, close_pred_next):
        j = loc_map.get(t, None)
        if j is None:
            continue
        if t <= train_end_ts:
            train_plot[j] = p
        else:
            test_plot[j] = p

    plot_price_prediction_nextclose(
        timestamps=timestamps,
        actual_prices=actual_prices,
        train_plot=train_plot,
        test_plot=test_plot,
        pair_name=pair_name,
        timeframe=TIMEFRAME,
        output_dir=OUTPUT_DIR,
        marker_skip=100
    )

    plot_price_with_ema(
        df_actual=df_act,
        df_pred=df_pred,
        pair_name=pair_name,
        timeframe=TIMEFRAME,
        output_dir=OUTPUT_DIR,
        suffix="_actual_vs_pred",
        last_days=5,
    )

    # --------------------------------------------------------
    # 8) Forecast CSV (uses PANDAS_FREQ)
    # --------------------------------------------------------
    train_cut = df_feat.loc[:train_end_ts]
    if len(train_cut) >= (ts + 1):
        anchor_ts = train_cut.index[-1]

        df_anchor = df_all_scaled.loc[:anchor_ts].copy()
        X_anchor, _, _ = create_supervised_sequences(
            df_anchor, FEATURE_COLS, time_step=ts, future_steps=FUTURE_STEPS
        )

        if len(X_anchor) > 0:
            last_input = X_anchor[-1].reshape(1, ts, len(FEATURE_COLS))
            pred_future_ret = model.predict(last_input, verbose=0)[0]

            anchor_price = float(df_feat.loc[anchor_ts, "close"])
            forecast_prices = anchor_price * np.exp(pred_future_ret)

            future_ts = pd.date_range(anchor_ts, periods=FUTURE_STEPS + 1, freq=PANDAS_FREQ)[1:]

            forecast_df = pd.DataFrame({
                "timestamp": future_ts,
                "forecast_forward_logret": pred_future_ret,
                "forecast_price": forecast_prices
            })
            forecast_path = os.path.join(OUTPUT_DIR, f"{pair_name}_forecast.csv")
            forecast_df.to_csv(forecast_path, index=False)
            print(f"[{pair_name}] Saved {FUTURE_STEPS}-step forecast CSV:", forecast_path)
        else:
            print(f"[{pair_name}] Could not form anchor sequence for forecast at TRAIN_END.")
    else:
        print(f"[{pair_name}] Not enough data before TRAIN_END to form forecast anchor.")

    # --------------------------------------------------------
    # 9) Summary row
    # --------------------------------------------------------
    summary_row = {
        "pair": pair_name,
        "timeframe": TIMEFRAME,
        "future_steps": FUTURE_STEPS,
        "train_bars": int(len(train_df)),
        "test_bars": int(len(test_df)),
        "train_start": train_df.index[0] if len(train_df) else None,
        "train_end":   train_df.index[-1] if len(train_df) else None,
        "test_start":  test_df.index[0] if len(test_df) else None,
        "test_end":    test_df.index[-1] if len(test_df) else None,
        "best_mean_wfv_rmse": float(best_rmse) if not (best_rmse is None or (isinstance(best_rmse, float) and math.isnan(best_rmse))) else None,
        "best_time_step": ts,
        "best_filters": f,
        "best_kernel": k,
        "best_dense_units": d,
        "best_dropout": dr,
        "best_learning_rate": lr,
        "best_batch_size": bs,
    }
    summary_row.update(test_metrics)
    SUMMARY.append(summary_row)

# ============================================================
# RUN FOR ALL PAIRS
# ============================================================
if __name__ == "__main__":

    pairs = [
        ("questdb-eurusd.csv", "EURUSD"),
        #("questdb-eurgbp.csv", "EURGBP"),
        #("questdb-audusd.csv", "AUDUSD"),
    ]

    for fname, pname in pairs:
        run_optuna_forecast(fname, pname)

    if SUMMARY:
        summary_df = pd.DataFrame(SUMMARY)
        summary_path = os.path.join(OUTPUT_DIR, "conv1d_wfv_trend_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print("\nSaved overall summary metrics:", summary_path)

    print("\nALL DONE.")
