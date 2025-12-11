# train_risk_model.py
"""
Train a memory-safe, accurate binary classifier for Delay>15.
Run: python train_risk_model.py
Outputs:
- risk_model.pkl
- risk_feature_map.json
- risk_scores.csv (probability per candidate)
"""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import joblib, json
from pathlib import Path

CLEAN = "clean_flights_for_bi.csv"
CAND = "risk_input_candidates.csv"
OUT_MODEL = "risk_model.pkl"
OUT_SCORES = "risk_scores.csv"
OUT_MAP = "risk_feature_map.json"
SAMPLE_FRAC = 1.0   # keep 1.0 for full, reduce to 0.2 if needed

def load():
    df = pd.read_csv(CLEAN, low_memory=False)
    return df

def freq_encode_train(df, cols):
    enc = {}
    for c in cols:
        vc = df[c].fillna("NA").astype(str).value_counts()
        enc_map = vc.to_dict()
        enc[c] = enc_map
        df[c + "_FREQ"] = df[c].fillna("NA").astype(str).map(enc_map).fillna(0)
    return df, enc

def prepare(df):
    # labels
    df = df.dropna(subset=["IS_DELAY"])
    df["IS_DELAY"] = df["IS_DELAY"].astype(int)
    # feature list: numeric + freq-encoded categorical
    cat_cols = [c for c in ["OP_CARRIER","ORIGIN","DEST","ROUTE"] if c in df.columns]
    num_cols = [c for c in ["DISTANCE","ROUTE_MEAN_ARR_DELAY","CARRIER_MEAN_ARR_DELAY","CRS_DEP_HOUR"] if c in df.columns]
    df, enc = freq_encode_train(df, cat_cols)
    feat_cols = num_cols + [c + "_FREQ" for c in cat_cols]
    df[feat_cols] = df[feat_cols].fillna(0)
    return df, feat_cols, enc

def train(df, features):
    if SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=42)
    X = df[features].values
    y = df["IS_DELAY"].values
    # holdout split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    # xgboost with early stopping via cv (use DMatrix)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "seed": 42,
        "tree_method": "hist"
    }
    evals = [(dtrain, "train"), (dtest, "eval")]
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=30, verbose_eval=10)
    # eval metrics
    preds = model.predict(dtest)
    auc = roc_auc_score(y_test, preds)
    print("Holdout AUC:", auc)
    print(classification_report(y_test, (preds>0.5).astype(int), digits=4))
    return model

def score_candidates(model, enc, features):
    cand = pd.read_csv(CAND, low_memory=False)
    # apply same freq encoding
    for k,v in enc.items():
        cand[k + "_FREQ"] = cand[k].fillna("NA").astype(str).map(v).fillna(0)
    # ensure numeric features exist
    for n in ["DISTANCE","ROUTE_MEAN_ARR_DELAY","CARRIER_MEAN_ARR_DELAY","CRS_DEP_HOUR"]:
        if n not in cand.columns:
            cand[n] = 0
    feat_cols = features
    Xc = cand[feat_cols].fillna(0).values
    dmat = xgb.DMatrix(Xc)
    probs = model.predict(dmat)
    cand_out = cand.copy()
    cand_out["prob_delay"] = probs
    cand_out = cand_out.sort_values("prob_delay", ascending=False)
    cand_out.to_csv(OUT_SCORES, index=False)
    print("Wrote", OUT_SCORES)

def main():
    df = load()
    df, features, enc = prepare(df)
    model = train(df, features)
    # save model and encoder
    joblib.dump(model, OUT_MODEL)
    with open(OUT_MAP,"w") as f:
        json.dump({"features": features, "freq_map_keys": {k: list(v.keys())[:20] for k,v in enc.items()}}, f, indent=2)
    print("Saved model & map")
    score_candidates(model, enc, features)

if __name__ == "__main__":
    main()
