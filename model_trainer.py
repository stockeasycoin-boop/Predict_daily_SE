"""
model_trainer.py — Ensemble model with Optuna tuning.

Architecture:
  OPEN prediction  (gap direction)  : XGBoost + LightGBM ensemble + regression
  CLOSE prediction (intraday dir)   : XGBoost + LightGBM ensemble + regression
  Signal only fires when BOTH models agree AND confidence ≥ threshold.

Optuna finds optimal hyperparameters for each model separately.
"""

import pandas as pd
import numpy as np
import joblib, json, warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False
    print("LightGBM not installed. Run: pip install lightgbm")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False
    print("Optuna not installed. Run: pip install optuna")

from feature_engineering import FEATURE_COLS, FEATURE_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def _optuna_xgb(X_tr, y_tr, X_val, y_val, n_trials: int = 60) -> dict:
    """Find best XGBoost hyperparameters using Optuna."""
    if not OPTUNA_OK:
        return {}

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 0.95),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 0.95),
            "min_child_weight":  trial.suggest_int("min_child_weight", 2, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 3.0),
            "eval_metric": "logloss",
            "random_state": 42, "n_jobs": -1,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return accuracy_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _optuna_lgb(X_tr, y_tr, X_val, y_val, n_trials: int = 60) -> dict:
    """Find best LightGBM hyperparameters."""
    if not OPTUNA_OK or not LGB_OK:
        return {}

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 80),
            "subsample":         trial.suggest_float("subsample", 0.5, 0.95),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 0.95),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 3.0),
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        return accuracy_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _default_xgb(**kw):
    p = dict(n_estimators=400, max_depth=4, learning_rate=0.04,
             subsample=0.75, colsample_bytree=0.75, min_child_weight=4,
             gamma=0.15, reg_alpha=0.1, reg_lambda=1.5,
             eval_metric="logloss",
             random_state=42, n_jobs=-1)
    p.update(kw)
    return xgb.XGBClassifier(**p)


def _default_lgb(**kw):
    p = dict(n_estimators=400, max_depth=5, learning_rate=0.04,
             num_leaves=40, subsample=0.75, colsample_bytree=0.75,
             min_child_samples=10, reg_alpha=0.1, reg_lambda=1.5,
             random_state=42, n_jobs=-1, verbose=-1)
    p.update(kw)
    return lgb.LGBMClassifier(**p)


def _default_xgb_reg(**kw):
    p = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
             subsample=0.75, colsample_bytree=0.75,
             reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=-1)
    p.update(kw)
    return xgb.XGBRegressor(**p)


def _cv_score(model_fn, X, y, n_splits=5, test_size=50):
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    scores = []
    for tr, te in tscv.split(X):
        m = model_fn()
        m.fit(X[tr], y[tr])
        scores.append(accuracy_score(y[te], m.predict(X[te])))
    return np.mean(scores), np.std(scores)


def train_model(feature_df: pd.DataFrame,
                model_dir: str = "models",
                verbose: bool = True,
                use_optuna: bool = True) -> tuple:
    """
    Train ensemble models for Nifty 50 open and close prediction.

    Saves per model:
      xgb_open.pkl, lgb_open.pkl        — open gap direction classifiers
      xgb_close.pkl, lgb_close.pkl      — close direction classifiers
      xgb_open_reg.pkl                  — open gap % regressor
      xgb_close_reg.pkl                 — close return % regressor
      scaler.pkl, feature_list.pkl
      metadata.json, feature_importance.csv
    """
    from settings import OPTUNA_TRIALS
    Path(model_dir).mkdir(exist_ok=True)

    df    = feature_df.copy()
    avail = [f for f in FEATURE_COLS if f in df.columns]
    n_trials = OPTUNA_TRIALS if use_optuna and OPTUNA_OK else 0

    scaler = StandardScaler()

    # ── Validation split (last 20% held out for Optuna) ───────────────────
    split = int(len(df) * 0.80)
    df_tr, df_val = df.iloc[:split], df.iloc[split:]

    def prep(data, target_col, reg=False):
        sub  = data.dropna(subset=avail + [target_col])
        X    = sub[avail].values.astype(np.float32)
        y    = sub[target_col].values
        if reg: y = y.astype(np.float32)
        else:   y = y.astype(int)
        return X, y

    def prep_chained(data, target_col, extra_cols):
        """
        Build a CHAINED regression matrix: scaled base features + raw upstream
        target columns (teacher forcing). At inference these upstream columns are
        filled with the model's own predictions, so each OHLC level influences
        the next (open → close → high → low).
        """
        need = avail + extra_cols + [target_col]
        sub  = data.dropna(subset=need)
        X_base = scaler.transform(sub[avail].values.astype(np.float32))
        if extra_cols:
            X_extra = sub[extra_cols].values.astype(np.float32)
            X = np.hstack([X_base, X_extra])
        else:
            X = X_base
        y = sub[target_col].values.astype(np.float32)
        return X, y

    # ═══════════════════════════════════════════════════════════════════════
    # 1. OPEN GAP MODEL
    # ═══════════════════════════════════════════════════════════════════════
    if verbose: print("\n── Training OPEN GAP models ──")

    X_tr_o, y_tr_o = prep(df_tr,  "open_target")
    X_val_o,y_val_o= prep(df_val, "open_target")
    X_all_o,y_all_o= prep(df,     "open_target")

    X_tr_o_sc  = scaler.fit_transform(X_tr_o)
    X_val_o_sc = scaler.transform(X_val_o)
    X_all_o_sc = scaler.transform(X_all_o)

    # Optuna search on val split
    xgb_o_params, lgb_o_params = {}, {}
    if n_trials > 0:
        if verbose: print(f"  Optuna XGB ({n_trials} trials)…")
        xgb_o_params = _optuna_xgb(X_tr_o_sc, y_tr_o, X_val_o_sc, y_val_o, n_trials)
        if LGB_OK:
            if verbose: print(f"  Optuna LGB ({n_trials} trials)…")
            lgb_o_params = _optuna_lgb(X_tr_o_sc, y_tr_o, X_val_o_sc, y_val_o, n_trials)

    xgb_open = _default_xgb(**{k: v for k, v in xgb_o_params.items()
                                if k not in ["eval_metric","use_label_encoder","random_state","n_jobs"]})
    xgb_open.fit(X_all_o_sc, y_all_o)
    joblib.dump(xgb_open, f"{model_dir}/xgb_open.pkl")

    lgb_open = None
    if LGB_OK:
        lgb_open = _default_lgb(**{k: v for k, v in lgb_o_params.items()
                                   if k not in ["random_state","n_jobs","verbose"]})
        lgb_open.fit(X_all_o_sc, y_all_o)
        joblib.dump(lgb_open, f"{model_dir}/lgb_open.pkl")

    # CV score
    cv_open_mean, cv_open_std = _cv_score(
        lambda: _default_xgb(**{k: v for k, v in xgb_o_params.items()
                                 if k not in ["eval_metric","use_label_encoder","random_state","n_jobs"]}),
        X_all_o_sc, y_all_o
    )
    if verbose:
        _open_base = max(y_all_o.mean(), 1 - y_all_o.mean())
        print(f"  XGB Open CV: {cv_open_mean:.3f} ± {cv_open_std:.3f}  "
              f"(gap-up rate: {y_all_o.mean():.1%}, baseline: {_open_base:.1%}, "
              f"skill: {cv_open_mean - _open_base:+.1%})")

    # Open regression (gap % magnitude)
    X_reg_o, y_reg_o = prep(df, "open_ret_pct", reg=True)
    X_reg_o_sc       = scaler.transform(X_reg_o)
    xgb_open_reg     = _default_xgb_reg()
    xgb_open_reg.fit(X_reg_o_sc, y_reg_o)
    joblib.dump(xgb_open_reg, f"{model_dir}/xgb_open_reg.pkl")
    mae_open = mean_absolute_error(y_reg_o, xgb_open_reg.predict(X_reg_o_sc))
    if verbose: print(f"  Open regression MAE: {mae_open:.3f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. CLOSE (INTRADAY) MODEL
    # ═══════════════════════════════════════════════════════════════════════
    if verbose: print("\n── Training CLOSE (intraday) models ──")

    X_tr_c, y_tr_c  = prep(df_tr,  "close_target")
    X_val_c,y_val_c = prep(df_val, "close_target")
    X_all_c,y_all_c = prep(df,     "close_target")

    X_tr_c_sc  = scaler.transform(X_tr_c)
    X_val_c_sc = scaler.transform(X_val_c)
    X_all_c_sc = scaler.transform(X_all_c)

    xgb_c_params, lgb_c_params = {}, {}
    if n_trials > 0:
        if verbose: print(f"  Optuna XGB ({n_trials} trials)…")
        xgb_c_params = _optuna_xgb(X_tr_c_sc, y_tr_c, X_val_c_sc, y_val_c, n_trials)
        if LGB_OK:
            if verbose: print(f"  Optuna LGB ({n_trials} trials)…")
            lgb_c_params = _optuna_lgb(X_tr_c_sc, y_tr_c, X_val_c_sc, y_val_c, n_trials)

    xgb_close = _default_xgb(**{k: v for k, v in xgb_c_params.items()
                                  if k not in ["eval_metric","use_label_encoder","random_state","n_jobs"]})
    xgb_close.fit(X_all_c_sc, y_all_c)
    joblib.dump(xgb_close, f"{model_dir}/xgb_close.pkl")

    lgb_close = None
    if LGB_OK:
        lgb_close = _default_lgb(**{k: v for k, v in lgb_c_params.items()
                                    if k not in ["random_state","n_jobs","verbose"]})
        lgb_close.fit(X_all_c_sc, y_all_c)
        joblib.dump(lgb_close, f"{model_dir}/lgb_close.pkl")

    cv_close_mean, cv_close_std = _cv_score(
        lambda: _default_xgb(**{k: v for k, v in xgb_c_params.items()
                                  if k not in ["eval_metric","use_label_encoder","random_state","n_jobs"]}),
        X_all_c_sc, y_all_c
    )
    if verbose:
        _close_base = max(y_all_c.mean(), 1 - y_all_c.mean())
        print(f"  XGB Close CV: {cv_close_mean:.3f} ± {cv_close_std:.3f}  "
              f"(bull rate: {y_all_c.mean():.1%}, baseline: {_close_base:.1%}, "
              f"skill: {cv_close_mean - _close_base:+.1%})")

    # Close regression (intraday return %) — CHAINED on predicted open gap
    X_reg_c, y_reg_c = prep_chained(df, "close_ret_pct", ["open_ret_pct"])
    xgb_close_reg    = _default_xgb_reg()
    xgb_close_reg.fit(X_reg_c, y_reg_c)
    joblib.dump(xgb_close_reg, f"{model_dir}/xgb_close_reg.pkl")
    mae_close = mean_absolute_error(y_reg_c, xgb_close_reg.predict(X_reg_c))
    if verbose: print(f"  Close regression MAE: {mae_close:.3f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # 2b. HIGH / LOW MODELS (chained — influenced by open & close)
    # ═══════════════════════════════════════════════════════════════════════
    if verbose: print("\n── Training HIGH / LOW models ──")

    # High regression — chained on predicted open + close
    X_reg_h, y_reg_h = prep_chained(df, "high_ret_pct", ["open_ret_pct", "close_ret_pct"])
    xgb_high_reg     = _default_xgb_reg()
    xgb_high_reg.fit(X_reg_h, y_reg_h)
    joblib.dump(xgb_high_reg, f"{model_dir}/xgb_high_reg.pkl")
    mae_high = mean_absolute_error(y_reg_h, xgb_high_reg.predict(X_reg_h))
    if verbose: print(f"  High regression MAE: {mae_high:.3f}%")

    # Low regression — chained on predicted open + close + high
    X_reg_l, y_reg_l = prep_chained(df, "low_ret_pct", ["open_ret_pct", "close_ret_pct", "high_ret_pct"])
    xgb_low_reg      = _default_xgb_reg()
    xgb_low_reg.fit(X_reg_l, y_reg_l)
    joblib.dump(xgb_low_reg, f"{model_dir}/xgb_low_reg.pkl")
    mae_low = mean_absolute_error(y_reg_l, xgb_low_reg.predict(X_reg_l))
    if verbose: print(f"  Low regression MAE: {mae_low:.3f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. SHARED ARTEFACTS
    # ═══════════════════════════════════════════════════════════════════════
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    joblib.dump(avail,  f"{model_dir}/feature_list.pkl")
    # Backward-compat aliases
    joblib.dump(xgb_open,  f"{model_dir}/xgb_model.pkl")
    joblib.dump(xgb_close, f"{model_dir}/xgb_direction.pkl")

    imp = pd.DataFrame({"feature": avail,
                         "importance": xgb_close.feature_importances_}
                       ).sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(f"{model_dir}/feature_importance.csv", index=False)

    meta = {
        "trained_at":      datetime.now().isoformat(),
        "n_samples":       len(X_all_c),
        "n_features":      len(avail),
        "cv_open":         round(cv_open_mean,  4),
        "cv_open_std":     round(cv_open_std,   4),
        "cv_close":        round(cv_close_mean, 4),
        "cv_close_std":    round(cv_close_std,  4),
        # Majority-class baseline = accuracy of always predicting the more common
        # class. Skill = how much the model beats that naive baseline.
        "open_base_rate":     round(float(y_all_o.mean()), 4),
        "close_base_rate":    round(float(y_all_c.mean()), 4),
        "open_baseline_acc":  round(float(max(y_all_o.mean(), 1 - y_all_o.mean())), 4),
        "close_baseline_acc": round(float(max(y_all_c.mean(), 1 - y_all_c.mean())), 4),
        "open_skill":         round(float(cv_open_mean  - max(y_all_o.mean(), 1 - y_all_o.mean())), 4),
        "close_skill":        round(float(cv_close_mean - max(y_all_c.mean(), 1 - y_all_c.mean())), 4),
        "mae_open_pct":    round(mae_open,  4),
        "mae_close_pct":   round(mae_close, 4),
        "mae_high_pct":    round(mae_high,  4),
        "mae_low_pct":     round(mae_low,   4),
        "lgb_available":   LGB_OK,
        "optuna_trials":   n_trials,
        "optuna_used":     OPTUNA_OK and n_trials > 0,
        "train_start":     str(df["date"].iloc[0])[:10],
        "train_end":       str(df["date"].iloc[-1])[:10],
    }
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Open  model CV accuracy:  {cv_open_mean:.1%}")
        print(f"  Close model CV accuracy:  {cv_close_mean:.1%}")
        print(f"  Open  regression MAE:     {mae_open:.3f}%")
        print(f"  Close regression MAE:     {mae_close:.3f}%")
        print(f"  High  regression MAE:     {mae_high:.3f}%")
        print(f"  Low   regression MAE:     {mae_low:.3f}%")
        print(f"  Ensemble active:          {'Yes (XGB+LGB)' if LGB_OK else 'No (XGB only)'}")
        print(f"  Optuna tuning:            {'Yes (' + str(n_trials) + ' trials)' if n_trials > 0 else 'No'}")
        print(f"{'='*50}\n")

    return xgb_open, xgb_close, scaler, avail, meta


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_today(feature_df: pd.DataFrame, model_dir: str = "models") -> dict:
    """
    Run all models on the latest row and return complete prediction dict.

    Returns
    -------
    open_direction    : 1 = gap-up, 0 = gap-down/flat
    open_confidence   : ensemble confidence for open direction
    open_pred_pct     : predicted gap % (regression)
    open_range        : (low_est, high_est) predicted open price range
    close_direction   : 1 = close > open (bullish), 0 = bearish
    close_confidence  : ensemble confidence for close direction
    close_pred_pct    : predicted intraday return % from open
    close_range       : (low_est, high_est) predicted close price range
    ensemble_agree    : True if XGB and LGB agree on both open AND close
    trade_signal      : 'BUY_CE' / 'BUY_PE' / 'NO_TRADE'
    signal_reason     : plain text reason for NO_TRADE if applicable
    atr_pct, india_vix: for options sizing
    """
    from settings import MIN_CONFIDENCE, ENSEMBLE_AGREE_ONLY, SKIP_EXPIRY_DAY, REGIME_EMA

    scaler    = joblib.load(f"{model_dir}/scaler.pkl")
    feat_list = joblib.load(f"{model_dir}/feature_list.pkl")

    avail   = [f for f in feat_list if f in feature_df.columns]
    missing = [f for f in feat_list if f not in feature_df.columns]
    if missing:
        sample = ", ".join(missing[:6]) + ("…" if len(missing) > 6 else "")
        raise RuntimeError(
            f"Model expects {len(feat_list)} features but only {len(avail)} are in "
            f"the current data — {len(missing)} are missing (e.g. {sample}). "
            "This usually means the feature pipeline changed since training. "
            "Go to Model Health → Train model now to retrain with current data."
        )
    row = feature_df[avail].tail(1).copy()
    row = row.fillna(feature_df[avail].mean())
    X   = scaler.transform(row.values.astype(np.float32))

    def _load(fname):
        p = Path(f"{model_dir}/{fname}")
        return joblib.load(p) if p.exists() else None

    xgb_open_m  = _load("xgb_open.pkl")
    lgb_open_m  = _load("lgb_open.pkl")
    xgb_close_m = _load("xgb_close.pkl")
    lgb_close_m = _load("lgb_close.pkl")
    xgb_open_r  = _load("xgb_open_reg.pkl")
    xgb_close_r = _load("xgb_close_reg.pkl")
    xgb_high_r  = _load("xgb_high_reg.pkl")
    xgb_low_r   = _load("xgb_low_reg.pkl")

    # ── Open prediction ───────────────────────────────────────────────────
    open_xgb_dir  = int(xgb_open_m.predict(X)[0])  if xgb_open_m  else 0
    open_xgb_prob = float(xgb_open_m.predict_proba(X)[0][open_xgb_dir]) if xgb_open_m else 0.5

    open_lgb_dir  = int(lgb_open_m.predict(X)[0])  if lgb_open_m  else open_xgb_dir
    open_lgb_prob = float(lgb_open_m.predict_proba(X)[0][open_lgb_dir]) if lgb_open_m else open_xgb_prob

    open_agree    = (open_xgb_dir == open_lgb_dir)
    open_conf     = float(np.mean([open_xgb_prob, open_lgb_prob])) if open_agree else 0.5
    open_dir      = open_xgb_dir if open_agree else open_xgb_dir   # XGB wins on disagree

    # ── Chained regression: open → close → high → low ─────────────────────
    open_pred_pct = float(xgb_open_r.predict(X)[0]) if xgb_open_r else 0.0

    # ── Close prediction ──────────────────────────────────────────────────
    close_xgb_dir  = int(xgb_close_m.predict(X)[0])  if xgb_close_m else 0
    close_xgb_prob = float(xgb_close_m.predict_proba(X)[0][close_xgb_dir]) if xgb_close_m else 0.5

    close_lgb_dir  = int(lgb_close_m.predict(X)[0])  if lgb_close_m else close_xgb_dir
    close_lgb_prob = float(lgb_close_m.predict_proba(X)[0][close_lgb_dir]) if lgb_close_m else close_xgb_prob

    close_agree   = (close_xgb_dir == close_lgb_dir)
    close_conf    = float(np.mean([close_xgb_prob, close_lgb_prob])) if close_agree else 0.5
    close_dir     = close_xgb_dir

    # Close regressor is chained on predicted open gap
    if xgb_close_r:
        X_close = np.hstack([X, np.array([[open_pred_pct]], dtype=np.float32)])
        try:
            close_pred_pct = float(xgb_close_r.predict(X_close)[0])
        except Exception:
            # Backward-compat: old base-only close regressor
            close_pred_pct = float(xgb_close_r.predict(X)[0])
    else:
        close_pred_pct = 0.0

    # High regressor chained on predicted open + close
    if xgb_high_r:
        X_high = np.hstack([X, np.array([[open_pred_pct, close_pred_pct]], dtype=np.float32)])
        high_pred_pct = float(xgb_high_r.predict(X_high)[0])
    else:
        high_pred_pct = max(close_pred_pct, 0.0) + 0.3  # fallback estimate

    # Low regressor chained on predicted open + close + high
    if xgb_low_r:
        X_low = np.hstack([X, np.array([[open_pred_pct, close_pred_pct, high_pred_pct]], dtype=np.float32)])
        low_pred_pct = float(xgb_low_r.predict(X_low)[0])
    else:
        low_pred_pct = min(close_pred_pct, 0.0) - 0.3  # fallback estimate

    ensemble_agree = open_agree and close_agree

    # ── Price ranges ──────────────────────────────────────────────────────
    last_close  = float(feature_df["close"].iloc[-1]) if "close" in feature_df.columns else 23000.0
    atr_pct     = float(feature_df["atr_pct"].iloc[-1]) if "atr_pct" in feature_df.columns else 0.8
    india_vix   = float(feature_df["india_vix"].iloc[-1]) if "india_vix" in feature_df.columns else 16.0

    # Open range estimate
    open_mid    = last_close * (1 + open_pred_pct / 100)
    open_range  = (round(open_mid * (1 - atr_pct * 0.25 / 100)),
                   round(open_mid * (1 + atr_pct * 0.25 / 100)))

    # Close range estimate (from predicted open)
    close_mid   = open_mid * (1 + close_pred_pct / 100)
    close_range = (round(close_mid * (1 - atr_pct * 0.35 / 100)),
                   round(close_mid * (1 + atr_pct * 0.35 / 100)))

    # Predicted daily HIGH / LOW (chained, relative to predicted open)
    high_mid = open_mid * (1 + high_pred_pct / 100)
    low_mid  = open_mid * (1 + low_pred_pct  / 100)
    # Enforce OHLC consistency: high ≥ max(open,close), low ≤ min(open,close)
    predicted_high = round(max(high_mid, open_mid, close_mid))
    predicted_low  = round(min(low_mid,  open_mid, close_mid))
    daily_range    = (predicted_low, predicted_high)

    # ── Trade signal logic ────────────────────────────────────────────────
    is_tuesday  = feature_df["date"].iloc[-1].weekday() == 1 if "date" in feature_df.columns else False
    above_ema21 = float(feature_df.get("above_ema21", pd.Series([1])).iloc[-1]) > 0.5

    trade_signal  = "NO_TRADE"
    signal_reason = ""

    if SKIP_EXPIRY_DAY and is_tuesday:
        signal_reason = "Skipping Tuesday (expiry day) — max-pain pinning makes direction unpredictable."
    elif ENSEMBLE_AGREE_ONLY and not ensemble_agree:
        signal_reason = f"XGBoost and LightGBM disagree — no trade. XGB: {'↑' if close_xgb_dir else '↓'}, LGB: {'↑' if close_lgb_dir else '↓'}."
    elif close_conf < MIN_CONFIDENCE:
        signal_reason = f"Close confidence {close_conf:.0%} below threshold {MIN_CONFIDENCE:.0%}."
    elif india_vix > 25:
        signal_reason = f"India VIX = {india_vix:.1f} — options premiums too expensive."
    else:
        # Regime filter: only take trades in trend direction
        if close_dir == 1 and not above_ema21:
            signal_reason = "Bullish signal but price below 21-EMA (downtrend) — skipping."
        elif close_dir == 0 and above_ema21:
            signal_reason = "Bearish signal but price above 21-EMA (uptrend) — skipping."
        else:
            trade_signal = "BUY_CE" if close_dir == 1 else "BUY_PE"

    return {
        "open_direction":   open_dir,
        "open_confidence":  round(open_conf,   4),
        "open_pred_pct":    round(open_pred_pct, 3),
        "open_range":       open_range,
        "open_agree":       open_agree,

        "close_direction":  close_dir,
        "close_confidence": round(close_conf,  4),
        "close_pred_pct":   round(close_pred_pct, 3),
        "close_range":      close_range,
        "close_agree":      close_agree,

        "ensemble_agree":   ensemble_agree,
        "trade_signal":     trade_signal,
        "signal_reason":    signal_reason,

        "last_close":       round(last_close, 2),
        "predicted_open":   round(open_mid,   2),
        "predicted_close":  round(close_mid,  2),

        "high_pred_pct":    round(high_pred_pct, 3),
        "low_pred_pct":     round(low_pred_pct,  3),
        "predicted_high":   predicted_high,
        "predicted_low":    predicted_low,
        "daily_range":      daily_range,

        "atr_pct":          round(atr_pct,   3),
        "india_vix":        round(india_vix, 2),

        # Legacy keys for options_engine compatibility
        "direction":        close_dir,
        "confidence":       round(close_conf, 4),
        "predicted_move_pct": round(abs(close_pred_pct), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REASONING
# ─────────────────────────────────────────────────────────────────────────────

def reasoning_for_prediction(feature_df: pd.DataFrame,
                               model_dir: str = "models",
                               top_n: int = 6) -> dict:
    """
    Explain the close-direction prediction using signed feature contributions.
    contribution = feature_importance × (value − mean) / std
    """
    try:
        scaler    = joblib.load(f"{model_dir}/scaler.pkl")
        feat_list = joblib.load(f"{model_dir}/feature_list.pkl")
        model     = joblib.load(f"{model_dir}/xgb_close.pkl")
    except Exception:
        return {"bullish_factors":[], "bearish_factors":[], "summary_text":"Model not trained."}

    avail = [f for f in feat_list if f in feature_df.columns]
    row   = feature_df[avail].tail(1).copy().fillna(feature_df[avail].mean())
    vals  = row.values.astype(np.float32)[0]

    pop_mean = feature_df[avail].mean().values.astype(np.float32)
    pop_std  = feature_df[avail].std().values.astype(np.float32)  + 1e-9
    norm_dev = (vals - pop_mean) / pop_std
    imp      = model.feature_importances_
    contrib  = norm_dev * imp

    rows = []
    for i, feat in enumerate(avail):
        label   = FEATURE_LABELS.get(feat, feat.replace("_"," ").title())
        score   = float(contrib[i])
        val     = float(vals[i])
        val_str = _fmt(feat, val)
        rows.append((label, score, val_str))

    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    bullish = [(l,s,v) for l,s,v in rows if s >  0.004][:top_n]
    bearish = [(l,s,v) for l,s,v in rows if s < -0.004][:top_n]

    X  = scaler.transform(row.values.astype(np.float32))
    d  = int(model.predict(X)[0])
    dw = "bullish" if d == 1 else "bearish"
    top = (bullish[0][0].lower() if bullish else (bearish[0][0].lower() if bearish else "mixed signals"))
    summary = f"Signal is {dw} — primarily driven by {top}."

    return {"bullish_factors": bullish, "bearish_factors": bearish, "summary_text": summary}


def _fmt(feat: str, val: float) -> str:
    PCT = {"gap_pct","ret_1d","ret_2d","ret_3d","ret_5d","ret_10d","ret_20d",
           "c_vs_ema9","c_vs_ema21","c_vs_ema50","atr_pct","vol_5d",
           "bb_width","vix_pct_chg","gift_vs_prev","prev_oc_ret","prev_range_pct",
           "macd_hist","prev_intra_morning_ret","prev_intra_afternoon_ret",
           "prev_intra_intraday_range","prev_intra_day_oc_ret"}
    BOOL = {"gap_up","gap_down","above_ema21","above_ema50","above_ema200",
            "rsi_ob","rsi_os","macd_bull","macd_xbull","macd_xbear",
            "bb_squeeze","bb_upper_touch","bb_lower_touch",
            "vix_spike","vix_calm","vix_high","vix_extreme",
            "fii_bull","fii_bear","fii_trend","gift_bull","gift_bear",
            "pcr_high","pcr_low","vol_surge","is_monday","is_tuesday",
            "is_friday","expiry_week","prev_oc_bull","prev_intra_reversal",
            "prev_intra_breakout","near_52w_high","near_52w_low"}
    if feat in BOOL: return "Yes" if val > 0.5 else "No"
    if feat in PCT:  return f"{val:+.2f}%"
    if feat == "rsi_14": return f"{val:.1f}"
    if feat == "india_vix": return f"{val:.1f}"
    if feat == "pcr":    return f"{val:.2f}"
    if feat in {"fii_net","dii_net","fii_5d_ma","fii_3d_sum"}:
        return f"₹{val:+,.0f}cr"
    if feat == "days_to_expiry": return f"{int(val)}d"
    if feat == "day_of_week":
        return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][int(val)] if 0 <= int(val) <= 6 else "?"
    return f"{val:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def model_exists(model_dir: str = "models") -> bool:
    return all(Path(f"{model_dir}/{f}").exists()
               for f in ["xgb_open.pkl","xgb_close.pkl","scaler.pkl","feature_list.pkl"])


def load_metadata(model_dir: str = "models") -> dict:
    try:
        with open(f"{model_dir}/metadata.json") as f: return json.load(f)
    except FileNotFoundError: return {}


def load_importance(model_dir: str = "models") -> pd.DataFrame:
    try: return pd.read_csv(f"{model_dir}/feature_importance.csv")
    except FileNotFoundError: return pd.DataFrame(columns=["feature","importance"])
