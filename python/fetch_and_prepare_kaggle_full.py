# fetch_and_prepare_kaggle_full.py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob

# CONFIG
FLIGHTS_GLOB = "flights*.*"   # will match 'flights' file (csv)
AIRLINES_FILE = "airlines"
AIRPORTS_FILE = "airports"
OUT_FILE = "processed_delays.csv"
CHUNKSIZE = 200_000          # tune for your RAM
DATE_COLS = []               # dataset uses YEAR/MONTH/DAY (no combined date) but we will keep them

# Helpers
def find_file_like(prefix):
    files = glob.glob(prefix + "*")
    if not files:
        return None
    # choose CSV/xlsx/etc - prefer .csv
    for ext in [".csv", ".txt", ".gzip", ".gz", ".zip", ".parquet", ""]:
        for f in files:
            if f.lower().endswith(ext) or ext == "":
                return f
    return files[0]

def detect_flights_file():
    path = find_file_like("flights")
    if path:
        return Path(path)
    # fallback: first large file in folder named flights*
    f = find_file_like(FLIGHTS_GLOB)
    return Path(f) if f else None

def read_lookup(fname):
    p = find_file_like(fname)
    if not p:
        print(f"Lookup file {fname} not found")
        return None
    # try CSV then Excel
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        try:
            return pd.read_excel(p)
        except Exception as e:
            print("Cannot read lookup:", p, e)
            return None

def map_columns_header(df):
    rename_map = {}

    col_map_dict = {
        # carrier
        "AIRLINE": "OP_CARRIER",
        "CARRIER": "OP_CARRIER",
        "UNIQUE_CARRIER": "OP_CARRIER",

        # airports
        "ORIGIN": "ORIGIN",
        "ORIGIN_AIRPORT": "ORIGIN",
        "ORIGIN_CITY": "ORIGIN",

        "DEST": "DEST",
        "DESTINATION_AIRPORT": "DEST",
        "DEST_CITY": "DEST",

        # flight id + aircraft
        "FLIGHT_NUMBER": "FLIGHT_NUM",
        "TAIL_NUMBER": "TAIL_NUM",
        "TAILNUM": "TAIL_NUM",

        # delays
        "DEPARTURE_DELAY": "DEP_DELAY",
        "DEP_DELAY": "DEP_DELAY",
        "ARRIVAL_DELAY": "ARR_DELAY",
        "ARR_DELAY": "ARR_DELAY",

        # schedule times
        "SCHEDULED_DEPARTURE": "CRS_DEP_TIME",
        "CRS_DEP_TIME": "CRS_DEP_TIME",
        "SCHEDULED_ARRIVAL": "CRS_ARR_TIME",
        "CRS_ARR_TIME": "CRS_ARR_TIME",

        # cancellation
        "CANCELLED": "CANCELLED",
        "CANCELLATION_REASON": "CANCELLATION_CODE",
        "CANCELLATION_CODE": "CANCELLATION_CODE",

        # misc
        "DISTANCE": "DISTANCE",
        "DAY_OF_WEEK": "DAY_OF_WEEK",
        "DAY": "DAY",
        "MONTH": "MONTH",
        "YEAR": "YEAR"
    }

    for c in df.columns:
        key = c.strip().upper()
        if key in col_map_dict:
            rename_map[c] = col_map_dict[key]

    df = df.rename(columns=rename_map)
    return df


def compute_aggregates(fpath):
    # compute per-route and per-carrier mean ARR_DELAY (two-pass to save mem)
    route_sum = {}
    carrier_sum = {}
    route_count = {}
    carrier_count = {}

    print("Computing aggregates (route/carrier mean ARR_DELAY)...")
    for chunk in pd.read_csv(fpath, chunksize=CHUNKSIZE, low_memory=False):
        chunk = map_columns_header(chunk)
        # ensure ARR_DELAY numeric
        if "ARR_DELAY" in chunk.columns:
            arr = pd.to_numeric(chunk["ARR_DELAY"], errors="coerce")
            chunk["ARR_DELAY_NUM"] = arr
        else:
            chunk["ARR_DELAY_NUM"] = np.nan
        # route
        if "ORIGIN" in chunk.columns and "DEST" in chunk.columns:
            chunk["ROUTE"] = chunk["ORIGIN"].astype(str)+"-"+chunk["DEST"].astype(str)
        else:
            chunk["ROUTE"] = "NA-NA"
        # carrier
        if "OP_CARRIER" not in chunk.columns:
            chunk["OP_CARRIER"] = "UNK"
        # accumulate
        grp_r = chunk.groupby("ROUTE")["ARR_DELAY_NUM"].agg(["sum","count"])
        for route, row in grp_r.iterrows():
            route_sum[route] = route_sum.get(route, 0.0) + (0 if np.isnan(row["sum"]) else row["sum"])
            route_count[route] = route_count.get(route, 0) + int(row["count"])
        grp_c = chunk.groupby("OP_CARRIER")["ARR_DELAY_NUM"].agg(["sum","count"])
        for carr, row in grp_c.iterrows():
            carrier_sum[carr] = carrier_sum.get(carr, 0.0) + (0 if np.isnan(row["sum"]) else row["sum"])
            carrier_count[carr] = carrier_count.get(carr, 0) + int(row["count"])
    # compute means
    route_mean = {r: (route_sum[r] / route_count[r]) if route_count[r]>0 else 0.0 for r in route_sum}
    carrier_mean = {c: (carrier_sum[c] / carrier_count[c]) if carrier_count[c]>0 else 0.0 for c in carrier_sum}
    return route_mean, carrier_mean

def process_and_write(fpath, route_mean, carrier_mean, airlines_df, airports_df):
    print("Second pass: processing chunks and writing", OUT_FILE)
    first = True
    cols_to_keep = [
        "YEAR","MONTH","DAY","DAY_OF_WEEK","CRS_DEP_TIME","CRS_ARR_TIME","CRS_DEP_HOUR",
        "OP_CARRIER","FLIGHT_NUM","TAIL_NUM","ORIGIN","DEST","ROUTE",
        "DISTANCE","ARR_DELAY","DEP_DELAY","CANCELLED","CANCELLATION_CODE",
        "ROUTE_MEAN_ARR_DELAY","CARRIER_MEAN_ARR_DELAY","TARGET_CLASS",
        "OP_CARRIER_NAME","ORIGIN_NAME","DEST_NAME"
    ]
    for chunk in tqdm(pd.read_csv(fpath, chunksize=CHUNKSIZE, low_memory=False)):
        chunk = map_columns_header(chunk)
        # Create ROUTE, numeric conversions
        if "ORIGIN" in chunk.columns and "DEST" in chunk.columns:
            chunk["ROUTE"] = chunk["ORIGIN"].astype(str) + "-" + chunk["DEST"].astype(str)
        else:
            chunk["ROUTE"] = "NA-NA"
        # numeric casts
        for col in ["ARR_DELAY","DEP_DELAY","DISTANCE","MONTH","DAY","DAY_OF_WEEK","CRS_DEP_TIME","CRS_ARR_TIME"]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        # scheduled hour
        if "CRS_DEP_TIME" in chunk.columns:
            chunk["CRS_DEP_HOUR"] = ((chunk["CRS_DEP_TIME"].fillna(0).astype(int) // 100) % 24).astype(float)
        else:
            chunk["CRS_DEP_HOUR"] = np.nan
        # aggregates
        chunk["ROUTE_MEAN_ARR_DELAY"] = chunk["ROUTE"].map(route_mean).fillna(0.0)
        chunk["CARRIER_MEAN_ARR_DELAY"] = chunk["OP_CARRIER"].map(carrier_mean).fillna(0.0)
        # target classes
        chunk["is_cancelled"] = (chunk.get("CANCELLED", 0) == 1).astype(int)
        chunk["dep_delay_flag"] = (chunk.get("DEP_DELAY", np.nan) > 15).astype(int)
        chunk["arr_delay_flag"] = (chunk.get("ARR_DELAY", np.nan) > 15).astype(int)
        chunk["is_delayed"] = ((chunk["is_cancelled"]==0) & ((chunk["dep_delay_flag"]==1) | (chunk["arr_delay_flag"]==1))).astype(int)
        chunk["TARGET_CLASS"] = np.where(chunk["is_cancelled"]==1, 2, np.where(chunk["is_delayed"]==1, 1, 0))
        # join airlines (carrier name) and airports (names)
        if airlines_df is not None and "OP_CARRIER" in chunk.columns:
            # airlines df commonly has columns: Code, Description or carrier, name
            # try common col names
            possible_code = [c for c in airlines_df.columns if c.lower() in ("code","carrier","iata","icao","carrier_code")]
            possible_name = [c for c in airlines_df.columns if c.lower() in ("name","description","airline","carrier_name")]
            if possible_code and possible_name:
                chunk = chunk.merge(airlines_df[[possible_code[0], possible_name[0]]].rename(columns={possible_code[0]:"OP_CARRIER","Description":possible_name[0]}), on="OP_CARRIER", how="left")
                chunk = chunk.rename(columns={possible_name[0]:"OP_CARRIER_NAME"})
            else:
                # fallback: try "carrier" column
                if "name" in airlines_df.columns:
                    chunk = chunk.rename(columns={"OP_CARRIER":"OP_CARRIER_CODE"})
        if airports_df is not None:
            # airports table often has 'iata' or 'ident' and 'name'
            # try to map ORIGIN and DEST to airport names
            id_col = None
            name_col = None
            for c in airports_df.columns:
                if c.lower() in ("iata","iata_code","ident","iata_code"):
                    id_col = c
                if c.lower() in ("name","airport_name"):
                    name_col = c
            if id_col and name_col:
                airports_small = airports_df[[id_col,name_col]].rename(columns={id_col:"AP_CODE", name_col:"AP_NAME"})
                # origin
                chunk = chunk.merge(airports_small.rename(columns={"AP_CODE":"ORIGIN","AP_NAME":"ORIGIN_NAME"}), on="ORIGIN", how="left")
                chunk = chunk.merge(airports_small.rename(columns={"AP_CODE":"DEST","AP_NAME":"DEST_NAME"}), on="DEST", how="left")
        # select keep columns that exist
        keep = [c for c in cols_to_keep if c in chunk.columns]
        out = chunk[keep].copy()
        # append to CSV
        if first:
            out.to_csv(OUT_FILE, index=False, mode="w")
            first = False
        else:
            out.to_csv(OUT_FILE, index=False, header=False, mode="a")
    print("Finished writing", OUT_FILE)

def main():
    flights_file = detect_flights_file()
    if flights_file is None:
        print("Cannot find flights file. Place flights CSV in folder as 'flights' prefix.")
        return
    print("Flights file found:", flights_file)
    # read lookups
    airlines_df = read_lookup(AIRLINES_FILE)
    airports_df = read_lookup(AIRPORTS_FILE)
    # Compute route & carrier aggregates
    route_mean, carrier_mean = compute_aggregates(flights_file)
    # Second pass to produce processed_delays.csv
    process_and_write(flights_file, route_mean, carrier_mean, airlines_df, airports_df)

if __name__ == "__main__":
    main()
