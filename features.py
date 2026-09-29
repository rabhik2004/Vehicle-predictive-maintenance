"""Preprocessing + feature engineering (rolling statistics per vehicle)."""
import numpy as np
import pandas as pd
from generate_data import SENSORS

WIN = 10
DYNAMIC = [s for s in SENSORS if s != "operating_hours"]


def add_features(df):
    df = df.sort_values(["vehicle_id", "t"]).copy()
    g = df.groupby("vehicle_id")
    for s in DYNAMIC:
        df[f"{s}_mean"] = g[s].transform(lambda x: x.rolling(WIN, min_periods=1).mean())
        df[f"{s}_std"] = g[s].transform(lambda x: x.rolling(WIN, min_periods=2).std()).fillna(0)
        df[f"{s}_slope"] = g[s].transform(lambda x: x.diff(WIN).fillna(0) / WIN)
    return df


def feature_columns(df):
    drop = {"vehicle_id", "t", "label", "rul"}
    return [c for c in df.columns if c not in drop]


def base_sensor(feature_name):
    for s in sorted(SENSORS, key=len, reverse=True):
        if feature_name.startswith(s):
            return s
    return feature_name
