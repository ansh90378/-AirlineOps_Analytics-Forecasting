# model_predict_full.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
from datetime import timedelta

IN_FILE = "processed_delays.csv"
MODEL_FILE = "unified_delay_model_full.pkl"
OUT_PRED = "predictions_nextday.csv"
SAMPLE_FRAC = 1.0   # set to <1.0 to speed up training on a sample (e.g., 0.3)

def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    return df

def prepare_features(df):
    # Ensure columns exist
    num_feats = [c for c in ["DISTANCE","ROUTE_MEAN_ARR_DELAY","CARRIER_MEAN_ARR_DELAY","CRS_DEP_HOUR"] if c in df.columns]
    cat_feats = [c for c in ["OP_CARRIER","ORIGIN","DEST","ROUTE"] if c in df.columns]
    df = df.dropna(subset=["TARGET_CLASS"])
    # Fill missing
    df[num_feats] = df[num_feats].fillna(df[num_feats].median())
    df[cat_feats] = df[cat_feats].fillna("NA")
    X = df[num_feats + cat_feats]
    y = df["TARGET_CLASS"].astype(int)
    return X, y, num_feats, cat_feats

def build_pipeline(num_features, cat_features):
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features)
    ], remainder="drop")
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def train_model(X, y, pipe):
    if SAMPLE_FRAC < 1.0:
        X, _, y, _ = train_test_split(X, y, train_size=SAMPLE_FRAC, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training on", len(X_train), "rows")
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    return pipe

def build_nextday(df, pipe, num_feats, cat_feats):
    # pick top combos to forecast
    combos = df.groupby(["OP_CARRIER","ROUTE"]).size().reset_index(name="n").sort_values("n", ascending=False).head(200)
    rows = []
    for _, r in combos.iterrows():
        sample = {}
        sample["OP_CARRIER"] = r["OP_CARRIER"]
        sample["ROUTE"] = r["ROUTE"]
        if "-" in r["ROUTE"]:
            o,d = r["ROUTE"].split("-",1)
            sample["ORIGIN"], sample["DEST"] = o,d
        else:
            sample["ORIGIN"], sample["DEST"] = "NA","NA"
        sample["ROUTE_MEAN_ARR_DELAY"] = df.loc[df["ROUTE"]==r["ROUTE"], "ROUTE_MEAN_ARR_DELAY"].median() if "ROUTE_MEAN_ARR_DELAY" in df.columns else 0
        sample["CARRIER_MEAN_ARR_DELAY"] = df.loc[df["OP_CARRIER"]==r["OP_CARRIER"], "CARRIER_MEAN_ARR_DELAY"].median() if "CARRIER_MEAN_ARR_DELAY" in df.columns else 0
        sample["DISTANCE"] = df.loc[df["ROUTE"]==r["ROUTE"], "DISTANCE"].median() if "DISTANCE" in df.columns else 0
        sample["CRS_DEP_HOUR"] = df.loc[df["ROUTE"]==r["ROUTE"], "CRS_DEP_HOUR"].median() if "CRS_DEP_HOUR" in df.columns else 0
        rows.append(sample)
    grid = pd.DataFrame(rows)
    feature_cols = num_feats + cat_feats
    grid = grid[feature_cols].fillna(0)
    proba = pipe.predict_proba(grid)
    preds = pipe.predict(grid)
    out = grid.copy()
    out["pred_class"] = preds
    out["prob_on_time"] = proba[:,0]
    out["prob_delayed"] = proba[:,1]
    out["prob_cancelled"] = proba[:,2]
    # attach identification
    out["OP_CARRIER"] = [r["OP_CARRIER"] for r in rows]
    out["ROUTE"] = [r["ROUTE"] for r in rows]
    out.to_csv(OUT_PRED, index=False)
    print("Wrote", OUT_PRED)

def main():
    print("Loading processed data...")
    df = load_data(IN_FILE)
    X, y, num_feats, cat_feats = prepare_features(df)
    pipe = build_pipeline(num_feats, cat_feats)
    pipe = train_model(X, y, pipe)
    joblib.dump(pipe, MODEL_FILE)
    print("Saved model:", MODEL_FILE)
    build_nextday(df, pipe, num_feats, cat_feats)

if __name__ == "__main__":
    main()
