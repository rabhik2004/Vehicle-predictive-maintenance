"""Synthetic vehicle sensor simulator.

Each vehicle produces T readings. Healthy vehicles stay near baseline; vehicles
that fail degrade progressively before the failure step. Label = 1 if a failure
occurs within the next HORIZON readings.
"""
import numpy as np
import pandas as pd

SENSORS = ["engine_temp", "rpm", "oil_pressure", "vibration", "battery_voltage",
           "coolant_temp", "fuel_consumption", "vehicle_speed", "operating_hours"]
HORIZON = 20


def simulate_vehicle(vid, T, rng):
    fails = rng.random() < 0.55
    fail_t = int(rng.integers(int(T * 0.5), T)) if fails else None
    wear = rng.uniform(0.9, 1.1)                      # vehicle-specific baseline
    t = np.arange(T)
    load = np.clip(rng.normal(0.5, 0.15, T) + 0.1 * np.sin(t / 15), 0.1, 1)  # driving load

    # degradation ramp 0->1 over the 60 steps before failure (0 for healthy)
    deg = np.zeros(T)
    if fails:
        lead = int(rng.integers(30, 80))
        deg = np.clip((t - (fail_t - lead)) / lead, 0, 1) ** 1.5
        deg[fail_t:] = 1
        deg *= rng.uniform(0.5, 1.0)          # severity varies per vehicle
    sens = rng.uniform(0.3, 1.0, 6)            # each sensor reacts differently

    n = lambda s: rng.normal(0, s, T)
    df = pd.DataFrame({
        "vehicle_id": vid, "t": t,
        "engine_temp": 88 + 12 * load * wear + 18 * deg * sens[0] + n(3.0),
        "rpm": 1500 + 2500 * load + 150 * deg + n(200),
        "oil_pressure": 45 + 15 * load - 16 * deg * sens[1] + n(3.5),
        "vibration": 1.2 + 0.8 * load + 3.2 * deg * sens[2] + np.abs(n(0.6)),
        "battery_voltage": 13.9 - 0.6 * load - 1.4 * deg * sens[3] + n(0.35),
        "coolant_temp": 85 + 8 * load * wear + 14 * deg * sens[4] + n(2.8),
        "fuel_consumption": 6 + 5 * load + 2.5 * deg * sens[5] + n(0.9),
        "vehicle_speed": 20 + 90 * load + n(6),
        "operating_hours": 1000 * wear + t * 1.5 + rng.uniform(0, 2000),
    })
    if fails:
        df = df[df.t <= fail_t].copy()
        df["label"] = (fail_t - df.t <= HORIZON).astype(int)
        df["rul"] = fail_t - df.t
    else:
        df["label"] = 0
        df["rul"] = 999
    return df


def generate(n_vehicles=400, T=250, seed=42):
    rng = np.random.default_rng(seed)
    return pd.concat([simulate_vehicle(i, T, rng) for i in range(n_vehicles)],
                     ignore_index=True)


if __name__ == "__main__":
    d = generate()
    d.to_csv("vehicle_sensor_data.csv", index=False)
    print(d.shape, "positive rate: %.3f" % d.label.mean())
