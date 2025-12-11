# analytics_export.py
"""
Create BI-ready files:
- clean_flights_for_bi.csv
- aggregated_metrics.csv
- risk_input_candidates.csv
Run: python analytics_export.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

IN = "processed_delays.csv"   # produced earlier
OUT_CLEAN = "clean_flights_for_bi.csv"
OUT_AGG = "aggregated_metrics.csv"
OUT_CAND = "risk_input_candidates.csv"

def load():
    df = pd.read_csv(IN, low_memory=False)
    return df

def feature_engineer(df):
    # Normalize names for BI
    df = df.rename(columns=lambda c: c.strip())
    # Create ROUTE if missing
    if "ROUTE" not in df.columns and "ORIGIN" in df.columns and "DEST" in df.columns:
        df["ROUTE"] = df["ORIGIN"].astype(str) + "-" + df["DEST"].astype(str)
    # Daypart
    if "CRS_DEP_HOUR" in df.columns:
        df["DAYPART"] = pd.cut(df["CRS_DEP_HOUR"].fillna(0),
                               bins=[-1,5,11,16,21,24],
                               labels=["Night","Morning","Afternoon","Evening","LateEvening"])
    else:
        df["DAYPART"] = "Unknown"
    # Convert target flags if absent
    if "TARGET_CLASS" in df.columns:
        df["IS_DELAY"] = (df["TARGET_CLASS"]==1).astype(int)
        df["IS_CANCELLED"] = (df["TARGET_CLASS"]==2).astype(int)
    else:
        df["IS_DELAY"] = (df.get("ARR_DELAY",0) > 15).astype(int)
        df["IS_CANCELLED"] = (df.get("CANCELLED",0) == 1).astype(int)
    # Useful business columns
    df["ARR_DELAY_MIN"] = df.get("ARR_DELAY", np.nan)
    df["DEP_DELAY_MIN"] = df.get("DEP_DELAY", np.nan)
    return df

def create_aggregates(df):
    # KPIs by carrier
    by_carrier = df.groupby(["OP_CARRIER"]).agg(
        flights=("ROUTE","count"),
        avg_arr_delay=("ARR_DELAY_MIN","mean"),
        pct_delayed=("IS_DELAY","mean"),
        pct_cancelled=("IS_CANCELLED","mean")
    ).reset_index()
    by_carrier["level"] = "carrier"
    # By route
    by_route = df.groupby(["ROUTE"]).agg(
        flights=("ROUTE","count"),
        avg_arr_delay=("ARR_DELAY_MIN","mean"),
        pct_delayed=("IS_DELAY","mean"),
        pct_cancelled=("IS_CANCELLED","mean")
    ).reset_index()
    by_route["level"] = "route"
    # By airport
    by_origin = df.groupby(["ORIGIN"]).agg(
        flights=("ROUTE","count"),
        avg_arr_delay=("ARR_DELAY_MIN","mean"),
        pct_delayed=("IS_DELAY","mean"),
        pct_cancelled=("IS_CANCELLED","mean")
    ).reset_index().assign(level="origin")
    by_dest = df.groupby(["DEST"]).agg(
        flights=("ROUTE","count"),
        avg_arr_delay=("ARR_DELAY_MIN","mean"),
        pct_delayed=("IS_DELAY","mean"),
        pct_cancelled=("IS_CANCELLED","mean")
    ).reset_index().assign(level="dest")
    agg = pd.concat([by_carrier, by_route, by_origin, by_dest], axis=0, ignore_index=True, sort=False)
    # Round for BI
    for c in ["avg_arr_delay","pct_delayed","pct_cancelled"]:
        if c in agg.columns:
            agg[c] = agg[c].round(4)
    return agg

def pick_risk_candidates(df, top_n_routes=300):
    # Build candidate list for next-day scoring: top frequent routes + carriers
    combos = df.groupby(["OP_CARRIER","ROUTE"]).size().reset_index(name="n").sort_values("n",ascending=False).head(top_n_routes)
    rows = []
    for _,r in combos.iterrows():
        row = {"OP_CARRIER": r["OP_CARRIER"], "ROUTE": r["ROUTE"]}
        if "-" in r["ROUTE"]:
            o,d = r["ROUTE"].split("-",1)
            row["ORIGIN"]=o; row["DEST"]=d
        else:
            row["ORIGIN"]="NA"; row["DEST"]="NA"
        # historical stats
        sub = df[df["ROUTE"]==r["ROUTE"]]
        row["ROUTE_MEAN_ARR_DELAY"] = sub["ARR_DELAY_MIN"].median()
        row["CARRIER_MEAN_ARR_DELAY"] = df[df["OP_CARRIER"]==r["OP_CARRIER"]]["ARR_DELAY_MIN"].median()
        row["DISTANCE"] = sub["DISTANCE"].median() if "DISTANCE" in df.columns else np.nan
        row["CRS_DEP_HOUR"] = sub["CRS_DEP_HOUR"].median() if "CRS_DEP_HOUR" in sub.columns else np.nan
        rows.append(row)
    cand = pd.DataFrame(rows)
    return cand

def main():
    df = load()
    df = feature_engineer(df)
    df.to_csv(OUT_CLEAN, index=False)
    print("Wrote", OUT_CLEAN)
    agg = create_aggregates(df)
    agg.to_csv(OUT_AGG, index=False)
    print("Wrote", OUT_AGG)
    cand = pick_risk_candidates(df, top_n_routes=300)
    cand.to_csv(OUT_CAND, index=False)
    print("Wrote", OUT_CAND)

if __name__ == "__main__":
    main()
