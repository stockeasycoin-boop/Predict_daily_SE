"""
meta_labeling.py — "when to trust the direction model" layer (López de Prado).

The per-horizon XGB+LGB models predict DIRECTION. This adds a second model per
horizon that predicts P(the direction model is correct) from the same features
plus the primary's own probabilities. Gating on this meta-probability beats
gating on raw confidence — measured lift +1.5–2.2 pts on 30/60-min horizons in
leakage-free walk-forward backtest (see scratchpad/meta_label_backtest.py).

Artifacts written to models/:
  intraday_meta_<hz>.pkl   — meta classifier per horizon
  meta_config.json         — per-horizon {meta_gate, exp_acc, coverage}

Meta feature vector at inference = [scaled base features..., conf, xgb_up, lgb_up, agree]
built identically in training and serving so there is no train/serve skew in the
base features. (Primary probs are uncalibrated in OOF vs isotonic-calibrated at
serve time — a monotonic remap the tree meta-model largely absorbs; the base
features dominate the signal.)
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import xgboost as xgb
from intraday_predictor import (
    build_intraday_features, build_horizon_targets,
    get_feature_cols_intraday, HORIZONS, LGB_OK,
)
if LGB_OK:
    import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# Primary configs — MUST match train_intraday_models() so OOF reflects production.
_XGB_PRIMARY = dict(n_estimators=300, max_depth=4, learning_rate=0.04,
                    subsample=0.75, colsample_bytree=0.75, eval_metric="logloss",
                    use_label_encoder=False, random_state=42, n_jobs=-1)
_LGB_PRIMARY = dict(n_estimators=300, max_depth=5, learning_rate=0.04,
                    num_leaves=40, subsample=0.75, colsample_bytree=0.75,
                    random_state=42, n_jobs=-1, verbose=-1)
_META_MODEL = dict(n_estimators=300, max_depth=3, learning_rate=0.02,
                   subsample=0.80, colsample_bytree=0.70, min_child_weight=10,
                   reg_alpha=1.0, reg_lambda=4.0, eval_metric="logloss",
                   use_label_encoder=False, random_state=42, n_jobs=-1)

_MIN_TRAIN = 20000
_STEP = 3000
_TARGET_ACC = 0.70        # aim for a >=70% held-out accuracy floor per bucket
_MIN_GATED = 100          # need at least this many held-out gated signals to trust it
_MAX_Q = 0.98             # never gate tighter than the top 2% most-confident


def build_meta_features(X_scaled: np.ndarray, xgb_up: np.ndarray,
                        lgb_up: np.ndarray) -> np.ndarray:
    """Assemble the meta feature matrix. xgb_up/lgb_up = P(up) from each primary.

    Extra columns (order fixed — inference must match): conf, xgb_up, lgb_up, agree.
    conf = mean prob on the predicted side when the two primaries agree, else 0.5.
    """
    xdir = (xgb_up >= 0.5).astype(int)
    ldir = (lgb_up >= 0.5).astype(int)
    agree = (xdir == ldir).astype(float)
    xconf = np.where(xdir == 1, xgb_up, 1 - xgb_up)
    lconf = np.where(ldir == 1, lgb_up, 1 - lgb_up)
    conf = np.where(agree == 1, (xconf + lconf) / 2.0, 0.5)
    extra = np.column_stack([conf, xgb_up, lgb_up, agree]).astype(np.float32)
    return np.column_stack([X_scaled, extra]).astype(np.float32)


def _oof_primary(X_sc: np.ndarray, y: np.ndarray):
    """Leakage-free expanding walk-forward -> OOF P(up) for xgb and lgb."""
    n = len(X_sc)
    xgb_up = np.full(n, np.nan, dtype=np.float32)
    lgb_up = np.full(n, np.nan, dtype=np.float32)
    s = _MIN_TRAIN
    while s < n:
        e = min(s + _STEP, n)
        xm = xgb.XGBClassifier(**_XGB_PRIMARY)
        xm.fit(X_sc[:s], y[:s], verbose=False)
        xgb_up[s:e] = xm.predict_proba(X_sc[s:e])[:, 1]
        if LGB_OK:
            lm = lgb.LGBMClassifier(**_LGB_PRIMARY)
            lm.fit(X_sc[:s], y[:s])
            lgb_up[s:e] = lm.predict_proba(X_sc[s:e])[:, 1]
        else:
            lgb_up[s:e] = xgb_up[s:e]
        s = e
    return xgb_up, lgb_up


def train_meta_models(df_5min: pd.DataFrame, model_dir: str = "models",
                      verbose: bool = True) -> dict:
    """Train + save one meta model per horizon and derive gate thresholds."""
    model_dir = str(model_dir)
    Path(model_dir).mkdir(exist_ok=True)

    feat_df = build_horizon_targets(build_intraday_features(df_5min))
    avail = [f for f in get_feature_cols_intraday() if f in feat_df.columns]

    # Reuse the SAME scaler the production models use, if present.
    scaler_path = Path(model_dir) / "intraday_scaler.pkl"
    Xraw = feat_df[avail].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
    Xraw = Xraw.values.astype(np.float32)
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_sc_all = scaler.transform(Xraw)
    else:
        X_sc_all = StandardScaler().fit_transform(Xraw)

    cfg = {}
    for hz in HORIZONS:
        col = f"target_{hz}"
        if col not in feat_df.columns:
            continue
        valid = feat_df[col].notna().values
        X_sc = X_sc_all[valid]
        y = feat_df.loc[valid, col].values.astype(int)
        if len(X_sc) <= _MIN_TRAIN + _STEP:
            if verbose:
                print(f"  {hz}: not enough data for meta ({len(X_sc)}), skipping")
            continue

        xgb_up, lgb_up = _oof_primary(X_sc, y)
        oof = ~np.isnan(xgb_up)
        Xm = build_meta_features(X_sc[oof], xgb_up[oof], lgb_up[oof])
        xdir = (xgb_up[oof] >= 0.5).astype(int)
        meta_y = (xdir == y[oof]).astype(int)

        # Honest OOS estimate: fit meta on the first 70% of the OOF region and
        # tune the gate on the held-out last 30% (recent regime). Evaluating on the
        # training rows would inflate exp_acc.
        split = int(len(Xm) * 0.70)
        eval_meta = xgb.XGBClassifier(**_META_MODEL)
        eval_meta.fit(Xm[:split], meta_y[:split], verbose=False)
        hp = eval_meta.predict_proba(Xm[split:])[:, 1]
        hy = meta_y[split:]

        # Target a >=70% accuracy FLOOR: raise the threshold (lower coverage) until
        # held-out accuracy clears 70%, keeping the MOST coverage that still does.
        # If no threshold reaches 70% (e.g. 5-min's ~65% ceiling), fall back to the
        # most-selective point and flag meets_70=False — never fake the number.
        best = None
        for q in np.arange(0.50, _MAX_Q + 1e-9, 0.01):
            thr_q = float(np.quantile(hp, q))
            g = hp >= thr_q
            if g.sum() < _MIN_GATED:
                continue
            acc_q = float(hy[g].mean())
            if acc_q >= _TARGET_ACC:
                best = (thr_q, acc_q, float(g.mean()), True)
                break
        if best is None:
            thr_q = float(np.quantile(hp, _MAX_Q))     # tightest allowed
            g = hp >= thr_q
            acc_q = float(hy[g].mean()) if g.sum() else 0.0
            best = (thr_q, acc_q, float(g.mean()), False)
        thr, exp_acc, cov, meets_70 = best[0], best[1] * 100, best[2], best[3]

        # Deploy a meta model refit on ALL OOF rows (more data = better serving).
        meta = xgb.XGBClassifier(**_META_MODEL)
        meta.fit(Xm, meta_y, verbose=False)
        joblib.dump(meta, f"{model_dir}/intraday_meta_{hz}.pkl")

        cfg[hz] = {"meta_gate": round(thr, 4),
                   "exp_acc": round(exp_acc, 1),
                   "coverage": round(cov, 3),
                   "meets_70": meets_70}
        if verbose:
            flag = "OK>=70" if meets_70 else "BELOW70(ceiling)"
            print(f"  {hz:8s}: meta gate p>={thr:.3f} -> {exp_acc:.1f}% acc @ "
                  f"{cov:.0%} coverage  [{flag}]  (n_oof={oof.sum()})")

    from datetime import datetime
    out = {"_comment": "Meta-labeling gates: P(primary direction correct) >= meta_gate "
                       "marks a high-conviction signal. exp_acc/coverage measured on "
                       "leakage-free walk-forward OOF.",
           "trained_at": datetime.now().isoformat(),
           "n_meta_features": len(avail) + 4,
           "horizons": cfg}
    with open(f"{model_dir}/meta_config.json", "w") as f:
        json.dump(out, f, indent=2)
    if verbose:
        print(f"\nMeta models + meta_config.json saved to {model_dir}/")
    return cfg


# ── Inference helpers ─────────────────────────────────────────────────────────

_META_CACHE = {}


def load_meta_config(model_dir: str = "models") -> dict:
    try:
        with open(Path(model_dir) / "meta_config.json") as f:
            return json.load(f).get("horizons", {})
    except Exception:
        return {}


def meta_predict(horizon: str, X_scaled_row: np.ndarray,
                 xgb_up: float, lgb_up: float, model_dir: str = "models"):
    """Return P(primary correct) for one live row, or None if no meta model."""
    key = (model_dir, horizon)
    if key not in _META_CACHE:
        p = Path(model_dir) / f"intraday_meta_{horizon}.pkl"
        _META_CACHE[key] = joblib.load(p) if p.exists() else None
    model = _META_CACHE[key]
    if model is None:
        return None
    Xm = build_meta_features(np.asarray(X_scaled_row, dtype=np.float32).reshape(1, -1),
                             np.array([xgb_up], dtype=np.float32),
                             np.array([lgb_up], dtype=np.float32))
    try:
        return float(model.predict_proba(Xm)[0, 1])
    except Exception:
        return None


def meta_models_exist(model_dir: str = "models") -> bool:
    return (Path(model_dir) / "intraday_meta_close.pkl").exists()
