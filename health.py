"""Failure probability -> Vehicle Health Score -> Maintenance alert + XAI reasons."""
import numpy as np
import pandas as pd
import shap
from features import base_sensor

# healthy operating ranges (from the training-data healthy baseline)
PHRASES = {
    "vibration": ("High vibration", 1),
    "engine_temp": ("Rising engine temperature", 1),
    "coolant_temp": ("Rising coolant temperature", 1),
    "oil_pressure": ("Abnormal oil pressure", -1),
    "battery_voltage": ("Low battery voltage", -1),
    "fuel_consumption": ("Elevated fuel consumption", 1),
    "rpm": ("Abnormal RPM", 1),
    "vehicle_speed": ("Abnormal speed pattern", 1),
    "operating_hours": ("High operating hours", 1),
}
W = {"vibration": .25, "engine_temp": .2, "oil_pressure": .2, "coolant_temp": .15,
     "battery_voltage": .1, "fuel_consumption": .1}


def sensor_deviation(row, base):
    """0 (at healthy mean) .. 1 (>=3 sigma in the bad direction)."""
    tot = 0
    for s, w in W.items():
        z = (row[s] - base[s][0]) / base[s][1]
        z *= PHRASES[s][1]
        tot += w * float(np.clip((z - 1) / 3, 0, 1))
    return tot


def status(p):
    return "HEALTHY" if p < 0.35 else "WARNING" if p < 0.75 else "CRITICAL"


class HealthMonitor:
    def __init__(self, model, feats, healthy_df):
        self.model, self.feats = model, feats
        self.base = {s: (healthy_df[s].mean(), healthy_df[s].std()) for s in W}
        self.explainer = shap.TreeExplainer(model)

    def assess(self, X_row_df, raw_row, top_k=3):
        p = float(self.model.predict_proba(X_row_df[self.feats])[0, 1])
        dev = sensor_deviation(raw_row, self.base)
        health = 100 * (0.5 * (1 - p) + 0.5 * (1 - dev))
        sv = self.explainer.shap_values(X_row_df[self.feats])
        sv = sv[1] if isinstance(sv, list) else sv
        sv = np.asarray(sv).reshape(-1)
        agg = {}
        for f, v in zip(self.feats, sv):
            b = base_sensor(f)
            agg[b] = agg.get(b, 0) + v
        top = [PHRASES[b][0] for b, v in sorted(agg.items(), key=lambda kv: -kv[1])[:top_k] if v > 0] if status(p) != "HEALTHY" else []
        return dict(health=round(health), failure_prob=round(100 * p),
                    status=status(p), factors=top)

    @staticmethod
    def report(r):
        lines = [f"Vehicle Health: {r['health']}%",
                 f"Failure Probability: {r['failure_prob']}%",
                 f"Status: {r['status']}"]
        if r["factors"]:
            lines.append("Major contributing factors:")
            lines += [f"{i}. {f}" for i, f in enumerate(r["factors"], 1)]
        return "\n".join(lines)
