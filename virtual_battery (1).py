"""
virtual_battery.py
==================
Complete EV Virtual Battery Simulator using Synthetic CAN Data

Components included:
  1.  OCV-SOC lookup table         (NMC chemistry)
  2.  2RC Thevenin cell model       (accurate transient voltage)
  3.  Arrhenius aging model         (calendar + cycle fade, knee-point)
  4.  Battery pack with cell imbalance (96S, per-cell SOH drift)
  5.  Realistic CAN sensor noise    (drift + quantization + offset)
  6.  Drive cycle profiles          (urban, highway, aggressive, idle)
  7.  Synthetic CAN frame generator (BMS-style output)
  8.  Feature extractor             (CAN window → ML features)
  9.  SOH predictor                 (GradientBoosting, trained on synthetic data)
  10. VirtualBattery twin class     (ticks in real-time, logs history, fires alerts)
  11. Dataset generator             (bulk multi-age / multi-cycle Parquet export)
  12. NASA dataset loader           (optional validation against B0005-B0008)
  13. Demo runner                   (CLI demo with printed output)

Requirements:
  pip install numpy pandas scikit-learn scipy

Optional (for Parquet export):
  pip install pyarrow

Usage:
  python virtual_battery.py              # runs interactive CLI demo
  python virtual_battery.py --generate   # generates full synthetic dataset
  python virtual_battery.py --train      # trains and evaluates SOH model
  python virtual_battery.py --all        # generate + train + demo
"""

import argparse
import math
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. OCV-SOC LOOKUP TABLE  (NMC 811 chemistry, per cell)
# ─────────────────────────────────────────────────────────────────────────────

_OCV_SOC_POINTS = np.array([
    0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00
])

_OCV_VOLTAGE_POINTS = np.array([
    3.00, 3.45, 3.53, 3.60, 3.65, 3.68, 3.71, 3.73,
    3.75, 3.77, 3.79, 3.82, 3.84, 3.87, 3.90, 3.93,
    3.97, 4.00, 4.05, 4.08, 4.13, 4.17, 4.20
])


def get_cell_ocv(soc: float) -> float:
    """Interpolate open-circuit voltage for a single NMC cell at given SOC."""
    soc = float(np.clip(soc, 0.0, 1.0))
    return float(np.interp(soc, _OCV_SOC_POINTS, _OCV_VOLTAGE_POINTS))


def get_pack_ocv(soc: float, n_series: int = 96) -> float:
    """Pack OCV = cell OCV × number of series cells."""
    return get_cell_ocv(soc) * n_series


# ─────────────────────────────────────────────────────────────────────────────
# 2. TWO-RC THEVENIN CELL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TwoRCThevenin:
    """
    Equivalent circuit:
        V_terminal = OCV(SOC) - I*R0 - V1 - V2

    R0          : ohmic (instantaneous) resistance
    R1, C1      : fast RC pair  (τ1 = R1*C1 ≈ 30–60 s)
    R2, C2      : slow RC pair  (τ2 = R2*C2 ≈ 500–2000 s, diffusion)

    All parameters degrade with age and temperature.
    """

    def __init__(self, soh: float = 1.0, age_years: float = 0.0,
                 temp_c: float = 25.0, n_series: int = 96):
        self.soh = soh
        self.age = age_years
        self.T   = temp_c
        self.n   = n_series

        # Temperature correction (cold increases resistance)
        t_factor = 1.0 + max(0.0, (25.0 - temp_c) * 0.015)

        # Per-cell parameters (scale with age and temperature)
        self.R0 = (0.003 + 0.001 * age_years) * t_factor          # Ω per cell
        self.R1 = (0.001 + 0.0005 * age_years) * t_factor
        self.C1 = max(500.0, 3000.0 - 80.0 * age_years)
        self.R2 = (0.0008 + 0.0003 * age_years) * t_factor
        self.C2 = max(5000.0, 18000.0 - 400.0 * age_years)

        self.V1 = 0.0   # state variable: voltage across RC1
        self.V2 = 0.0   # state variable: voltage across RC2

    def step(self, soc: float, I_pack: float, dt: float = 0.1) -> float:
        """
        Advance model by dt seconds with pack current I_pack (A).
        Returns pack terminal voltage (V).
        """
        # Convert pack current to cell current (series string → same current)
        I_cell = I_pack  # same current flows through each series cell

        # Euler integration of RC dynamics
        self.V1 += dt * (I_cell / self.C1 - self.V1 / (self.R1 * self.C1))
        self.V2 += dt * (I_cell / self.C2 - self.V2 / (self.R2 * self.C2))

        cell_ocv     = get_cell_ocv(soc)
        V_cell       = cell_ocv - I_cell * self.R0 - self.V1 - self.V2
        V_pack       = V_cell * self.n
        return float(V_pack)

    def pack_resistance(self) -> float:
        """Effective DC pack resistance in Ohms."""
        return (self.R0 + self.R1 + self.R2) * self.n


# ─────────────────────────────────────────────────────────────────────────────
# 3. ARRHENIUS AGING MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ArrheniusAging:
    """
    Models capacity fade and resistance growth using:

    Calendar aging : SEI layer growth — Q_cal  = k_cal  × √(days)
    Cycle aging    : Ah-throughput law — Q_cyc  = k_cyc  × N_cycles × DoD^1.5
    Temperature    : Arrhenius acceleration factor
    Knee-point     : Exponential acceleration after ~80% SOH
    """

    Ea    = 31_500   # activation energy (J/mol) — NMC typical
    R_gas = 8.314    # universal gas constant
    T_ref = 298.15   # 25 °C reference temperature (K)

    def _arrhenius(self, temp_c: float) -> float:
        T_k = temp_c + 273.15
        return math.exp(-self.Ea / self.R_gas * (1.0 / T_k - 1.0 / self.T_ref))

    def capacity_fade(self, days: float, n_cycles: int,
                      avg_temp_c: float = 25.0, avg_dod: float = 0.8) -> float:
        """
        Returns SOH in [0.6, 1.0].
        """
        af = self._arrhenius(avg_temp_c)
        k_cal  = 1.5e-4 * af
        k_cyc  = 4.0e-5 * af * (avg_dod ** 1.5)

        q_cal  = k_cal * math.sqrt(max(days, 0.0))
        q_cyc  = k_cyc * n_cycles

        linear_fade = q_cal + q_cyc

        # Knee-point: once linear fade > 15%, accelerate
        if linear_fade > 0.15:
            knee_extra = 0.8 * (linear_fade - 0.15) ** 1.6
        else:
            knee_extra = 0.0

        soh = max(0.60, 1.0 - linear_fade - knee_extra)
        return round(soh, 5)

    def resistance_growth(self, soh: float) -> float:
        """
        Internal resistance grows as capacity fades (per cell, Ohms).
        """
        fade = 1.0 - soh
        return 0.003 + 0.015 * fade + 0.04 * (fade ** 2)

    def cycles_from_age(self, age_years: float,
                        cycle: str = "urban") -> int:
        """Estimate cycle count from age + drive cycle intensity."""
        cycles_per_year = {
            "urban": 300, "highway": 250, "aggressive": 400, "idle": 50
        }
        return int(age_years * cycles_per_year.get(cycle, 300))


# ─────────────────────────────────────────────────────────────────────────────
# 4. BATTERY PACK WITH CELL IMBALANCE
# ─────────────────────────────────────────────────────────────────────────────

class BatteryPack:
    """
    96 cells in series (typical 400 V EV pack).
    Each cell has slight manufacturing variance → natural imbalance that
    grows with age.
    """

    def __init__(self, n_series: int = 96, age_years: float = 0.0,
                 avg_temp_c: float = 25.0):
        self.n       = n_series
        self.age     = age_years
        self.aging   = ArrheniusAging()

        n_cycles = self.aging.cycles_from_age(age_years)
        base_soh = self.aging.capacity_fade(
            days=age_years * 365, n_cycles=n_cycles, avg_temp_c=avg_temp_c
        )

        # Per-cell variance: ±0.3% SOH, ±0.1% SOC, ±5% IR
        rng = np.random.default_rng(seed=42)
        soh_var  = rng.normal(0, 0.003, n_series)
        soc_var  = rng.normal(0, 0.005, n_series)
        ir_scale = rng.normal(1.0, 0.05, n_series)

        base_ir = self.aging.resistance_growth(base_soh)

        self.cells = [
            {
                "soh": float(np.clip(base_soh + soh_var[i], 0.60, 1.0)),
                "soc": float(np.clip(0.80 + soc_var[i], 0.05, 1.0)),
                "ir":  float(np.clip(base_ir * ir_scale[i], 0.001, 0.1)),
            }
            for i in range(n_series)
        ]

    # ── public helpers ──────────────────────────────────────────────────────

    def weakest_soh(self) -> float:
        return min(c["soh"] for c in self.cells)

    def mean_soh(self) -> float:
        return float(np.mean([c["soh"] for c in self.cells]))

    def soc_spread(self) -> float:
        socs = [c["soc"] for c in self.cells]
        return float(max(socs) - min(socs))

    def update_soc(self, I: float, dt: float, capacity_ah: float):
        """Apply current to every cell's SOC (Coulomb counting)."""
        d_soc = (I * dt) / (capacity_ah * 3600.0)
        for c in self.cells:
            c["soc"] = float(np.clip(c["soc"] - d_soc, 0.02, 1.0))

    def pack_voltage(self, I: float) -> tuple:
        """
        Returns (pack_voltage, min_cell_v, max_cell_v).
        Uses simple OCV - IR model per cell.
        """
        cell_vs = []
        for c in self.cells:
            ocv = get_cell_ocv(c["soc"])
            v   = ocv - I * c["ir"]
            cell_vs.append(v)
        total = sum(cell_vs)
        return total, min(cell_vs), max(cell_vs)

    def mean_soc(self) -> float:
        return float(np.mean([c["soc"] for c in self.cells]))


# ─────────────────────────────────────────────────────────────────────────────
# 5. CAN SENSOR NOISE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CANSensorModel:
    """
    Adds realistic sensor imperfections on top of true electrical values:
      - Voltage sensor : slow drift + Gaussian noise + 0.1 V quantization
      - Current sensor : fixed offset + Gaussian noise + 0.5 A quantization
      - Temperature    : slow drift + Gaussian noise + 0.5 °C quantization
      - SOC (BMS est.) : Coulomb-counting error accumulation
    """

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._v_drift   = 0.0
        self._t_drift   = 0.0
        self._i_offset  = float(rng.normal(0, 0.3))   # fixed hall-sensor offset
        self._soc_err   = 0.0
        self._rng       = rng

    def voltage(self, true_v: float, dt: float = 0.1) -> float:
        self._v_drift += float(self._rng.normal(0, 0.002)) * dt
        self._v_drift  = np.clip(self._v_drift, -3.0, 3.0)
        noisy = true_v + self._v_drift + float(self._rng.normal(0, 0.15))
        return round(noisy * 10) / 10          # 0.1 V resolution

    def current(self, true_i: float) -> float:
        noisy = true_i + self._i_offset + float(self._rng.normal(0, 0.4))
        return round(noisy * 2) / 2            # 0.5 A resolution

    def temperature(self, true_t: float, dt: float = 0.1) -> float:
        self._t_drift += float(self._rng.normal(0, 0.001)) * dt
        self._t_drift  = np.clip(self._t_drift, -1.5, 1.5)
        noisy = true_t + self._t_drift + float(self._rng.normal(0, 0.2))
        return round(noisy * 2) / 2            # 0.5 °C resolution

    def soc_estimate(self, true_soc: float, I: float, dt: float) -> float:
        """Simulate BMS Coulomb-counting error (integrates small current errors)."""
        self._soc_err += float(self._rng.normal(0, 5e-6)) * abs(I) * dt
        self._soc_err  = np.clip(self._soc_err, -0.05, 0.05)
        return float(np.clip(true_soc + self._soc_err, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 6. DRIVE CYCLE PROFILES
# ─────────────────────────────────────────────────────────────────────────────

DRIVE_CYCLES = {
    "urban": {
        "I_rms":   80.0,    # A RMS pack current during discharge
        "I_peak":  180.0,   # A peak (acceleration)
        "freq_hz": 0.05,    # oscillation frequency of load pattern
        "regen":   0.25,    # fraction of time regenerative braking
        "I_regen": -30.0,   # A during regen (negative = charging)
        "temp_rise": 8.0,   # °C above ambient
        "stress":  1.0,     # aging stress multiplier
    },
    "highway": {
        "I_rms":   120.0,
        "I_peak":  200.0,
        "freq_hz": 0.02,
        "regen":   0.10,
        "I_regen": -20.0,
        "temp_rise": 12.0,
        "stress":  0.75,
    },
    "aggressive": {
        "I_rms":   200.0,
        "I_peak":  380.0,
        "freq_hz": 0.08,
        "regen":   0.20,
        "I_regen": -60.0,
        "temp_rise": 22.0,
        "stress":  1.6,
    },
    "idle": {
        "I_rms":   5.0,
        "I_peak":  10.0,
        "freq_hz": 0.001,
        "regen":   0.0,
        "I_regen": 0.0,
        "temp_rise": 1.0,
        "stress":  0.05,
    },
}


def sample_current(cycle: str, t: float, rng: np.random.Generator) -> float:
    """
    Generate instantaneous pack current (A) for a given drive cycle and time.
    Positive = discharge, negative = regen / charging.
    """
    p   = DRIVE_CYCLES[cycle]
    sin = math.sin(2 * math.pi * p["freq_hz"] * t)
    I   = p["I_rms"] + p["I_peak"] * 0.35 * sin
    I  += float(rng.normal(0, p["I_rms"] * 0.06))

    # Occasional regenerative braking
    if rng.random() < p["regen"] * 0.02:
        I = p["I_regen"] + float(rng.normal(0, 3.0))

    return float(I)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SYNTHETIC CAN FRAME GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticCANGenerator:
    """
    Produces BMS-style CAN frames at a configurable sample rate.
    Internally uses the 2RC Thevenin model + Arrhenius aging + pack imbalance.
    """

    def __init__(self, age_years: float = 0.0, ambient_temp: float = 25.0,
                 cycle: str = "urban", n_series: int = 96,
                 nominal_cap_ah: float = 200.0, seed: int = 42):
        self.age    = age_years
        self.T_amb  = ambient_temp
        self.cycle  = cycle
        self.n      = n_series
        self.cap_ah = nominal_cap_ah

        self._aging  = ArrheniusAging()
        n_cycles     = self._aging.cycles_from_age(age_years, cycle)
        self.soh     = self._aging.capacity_fade(
            days=age_years * 365, n_cycles=n_cycles, avg_temp_c=ambient_temp
        )
        self.cap_now = nominal_cap_ah * self.soh

        self._pack   = BatteryPack(n_series, age_years, ambient_temp)
        self._ecm    = TwoRCThevenin(self.soh, age_years, ambient_temp, n_series)
        self._sensor = CANSensorModel(seed=seed)
        self._rng    = np.random.default_rng(seed)

        self._t      = 0.0
        self._temp   = ambient_temp + DRIVE_CYCLES[cycle]["temp_rise"] * 0.1

    def next_frame(self, dt: float = 0.1) -> dict:
        """Advance simulation by dt seconds and return one CAN-style record."""
        I_true = sample_current(self.cycle, self._t, self._rng)

        # Thermal model: cell temp converges toward steady-state
        T_ss = self.T_amb + DRIVE_CYCLES[self.cycle]["temp_rise"]
        self._temp += (T_ss - self._temp) * 0.001 * dt

        soc_true = self._pack.mean_soc()

        # 2RC voltage
        V_true = self._ecm.step(soc_true, I_true, dt)

        # Pack imbalance
        _, V_min, V_max = self._pack.pack_voltage(I_true)
        V_spread = V_max - V_min

        # Update SOC of each cell
        self._pack.update_soc(I_true, dt, self.cap_now)

        # Apply sensor noise
        V_meas   = self._sensor.voltage(V_true, dt)
        I_meas   = self._sensor.current(I_true)
        T_meas   = self._sensor.temperature(self._temp, dt)
        SOC_meas = self._sensor.soc_estimate(soc_true, I_true, dt)

        frame = {
            "timestamp":    round(self._t, 2),
            "voltage":      V_meas,
            "current":      I_meas,
            "soc":          round(SOC_meas, 4),
            "soh":          round(self.soh, 4),
            "temp_cell":    T_meas,
            "ir_pack":      round(self._ecm.pack_resistance(), 4),
            "v_cell_min":   round(V_min, 3),
            "v_cell_max":   round(V_max, 3),
            "v_cell_spread":round(V_spread, 3),
            "soc_spread":   round(self._pack.soc_spread(), 4),
            "weakest_soh":  round(self._pack.weakest_soh(), 4),
            "age_years":    self.age,
            "cycle":        self.cycle,
        }

        self._t += dt
        return frame

    def generate_session(self, duration_s: float = 1800.0,
                         dt: float = 0.5) -> pd.DataFrame:
        """Generate a full driving session as a DataFrame."""
        frames = [self.next_frame(dt)
                  for _ in range(int(duration_s / dt))]
        return pd.DataFrame(frames)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE EXTRACTOR  (CAN window → ML features)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> dict:
    """
    Derive health-relevant features from a window of CAN frames.
    These mimic what a real BMS or edge compute unit would calculate.
    """
    v = df["voltage"]
    i = df["current"]
    t = df["temp_cell"]

    # dV/dQ — indicator of capacity fade (peak shift)
    q = i.cumsum() * 0.5 / 3600          # rough Ah integral
    dv = v.diff()
    dq = q.diff().replace(0, np.nan)
    dvdq_mean = (dv / dq).abs().dropna().mean()

    # IR estimate from voltage sag during high-current pulses
    high_i_mask = i.abs() > i.abs().quantile(0.75)
    if high_i_mask.sum() > 5:
        ir_est = (v[~high_i_mask].mean() - v[high_i_mask].mean()) / \
                 (i[high_i_mask].mean() - i[~high_i_mask].mean() + 1e-9)
    else:
        ir_est = df["ir_pack"].mean() if "ir_pack" in df.columns else 0.0

    features = {
        "mean_voltage":       v.mean(),
        "std_voltage":        v.std(),
        "min_voltage":        v.min(),
        "voltage_range":      v.max() - v.min(),
        "mean_current":       i.mean(),
        "peak_current":       i.abs().max(),
        "rms_current":        math.sqrt((i ** 2).mean()),
        "mean_temp":          t.mean(),
        "max_temp":           t.max(),
        "temp_rise":          t.max() - t.min(),
        "ir_estimate":        abs(ir_est),
        "soc_swing":          df["soc"].max() - df["soc"].min(),
        "dvdq_mean":          dvdq_mean if not math.isnan(dvdq_mean) else 0.0,
        "v_cell_spread_mean": df["v_cell_spread"].mean() if "v_cell_spread" in df.columns else 0.0,
        "soc_spread_mean":    df["soc_spread"].mean() if "soc_spread" in df.columns else 0.0,
    }
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 9. SOH PREDICTOR  (trained on synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

class SOHPredictor:
    """
    GradientBoosting regressor trained on windows of synthetic CAN data.
    Call .train(dataset_df) once, then .predict(window_df) any time.
    """

    def __init__(self):
        self.model   = GradientBoostingRegressor(
            n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            random_state=42
        )
        self.scaler  = StandardScaler()
        self.trained = False
        self._feature_names = None

    def _build_training_set(self, dataset: pd.DataFrame,
                            window_s: int = 300, dt: float = 0.5):
        """
        Slide a window over the dataset to extract (features, soh) pairs.
        """
        window_rows = int(window_s / dt)
        X_rows, y_rows = [], []

        for (age, cycle), grp in dataset.groupby(["age_years", "cycle"]):
            grp = grp.reset_index(drop=True)
            for start in range(0, len(grp) - window_rows, window_rows):
                window = grp.iloc[start: start + window_rows]
                feats  = extract_features(window)
                soh    = window["soh"].mean()
                X_rows.append(feats)
                y_rows.append(soh)

        X = pd.DataFrame(X_rows)
        y = np.array(y_rows)
        return X, y

    def train(self, dataset: pd.DataFrame, test_size: float = 0.2):
        print("[SOHPredictor] Building training set …")
        X, y = self._build_training_set(dataset)
        self._feature_names = X.columns.tolist()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        print(f"[SOHPredictor] Training on {len(X_tr)} samples …")
        self.model.fit(X_tr_s, y_tr)
        self.trained = True

        y_pred = self.model.predict(X_te_s)
        mae    = mean_absolute_error(y_te, y_pred)
        r2     = r2_score(y_te, y_pred)
        print(f"[SOHPredictor] Test MAE: {mae*100:.3f}%  |  R²: {r2:.4f}")
        return {"mae": mae, "r2": r2}

    def predict(self, window_df: pd.DataFrame) -> float:
        """Predict SOH from a DataFrame window of CAN frames."""
        if not self.trained:
            raise RuntimeError("Call .train() before .predict()")
        feats = extract_features(window_df)
        X     = pd.DataFrame([feats])[self._feature_names]
        X_s   = self.scaler.transform(X)
        return float(np.clip(self.model.predict(X_s)[0], 0.60, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 10. VIRTUAL BATTERY  (the digital twin class)
# ─────────────────────────────────────────────────────────────────────────────

class VirtualBattery:
    """
    The digital twin. Wraps the generator, ticks forward in time,
    accumulates history, predicts SOH from recent data, and fires alerts.

    Usage:
        twin = VirtualBattery(age_years=3, cycle="urban", ambient_temp=28)
        for _ in range(1000):
            state = twin.tick()
            print(state)
    """

    ALERT_SOH_WARN    = 0.80
    ALERT_SOH_CRIT    = 0.70
    ALERT_TEMP_WARN   = 45.0
    ALERT_TEMP_CRIT   = 55.0
    ALERT_VMIN_WARN   = 3.10   # per cell
    ALERT_SOC_LOW     = 0.15

    def __init__(self, age_years: float = 0.0, cycle: str = "urban",
                 ambient_temp: float = 25.0, dt: float = 0.5,
                 predictor: "SOHPredictor | None" = None,
                 predict_window_s: int = 300):
        self._gen      = SyntheticCANGenerator(
            age_years=age_years, ambient_temp=ambient_temp,
            cycle=cycle, seed=int(age_years * 100)
        )
        self.dt              = dt
        self._predictor      = predictor
        self._predict_window = int(predict_window_s / dt)
        self.history: list   = []
        self.alerts:  list   = []

    # ── main tick ──────────────────────────────────────────────────────────

    def tick(self) -> dict:
        """Advance simulation by one dt step. Returns current state dict."""
        frame = self._gen.next_frame(self.dt)
        self.history.append(frame)

        # ML SOH prediction from recent window
        frame["predicted_soh"] = None
        if self._predictor and len(self.history) >= self._predict_window:
            window = pd.DataFrame(self.history[-self._predict_window:])
            try:
                frame["predicted_soh"] = self._predictor.predict(window)
            except Exception:
                pass

        self._check_alerts(frame)
        return frame

    def run(self, n_steps: int) -> pd.DataFrame:
        """Run n_steps ticks and return full history as DataFrame."""
        for _ in range(n_steps):
            self.tick()
        return pd.DataFrame(self.history)

    # ── alerts ─────────────────────────────────────────────────────────────

    def _check_alerts(self, frame: dict):
        t  = frame["timestamp"]
        s  = frame["soh"]
        tc = frame["temp_cell"]
        sc = frame["soc"]
        vc = frame.get("v_cell_min", 999)

        if s < self.ALERT_SOH_CRIT:
            self._alert(t, "CRITICAL", f"SOH critically low: {s*100:.1f}%")
        elif s < self.ALERT_SOH_WARN:
            self._alert(t, "WARNING",  f"SOH below 80%: {s*100:.1f}%")

        if tc > self.ALERT_TEMP_CRIT:
            self._alert(t, "CRITICAL", f"Cell overtemperature: {tc:.1f}°C")
        elif tc > self.ALERT_TEMP_WARN:
            self._alert(t, "WARNING",  f"Cell temp elevated: {tc:.1f}°C")

        if sc < self.ALERT_SOC_LOW:
            self._alert(t, "WARNING",  f"Low SOC: {sc*100:.1f}%")

        if vc < self.ALERT_VMIN_WARN:
            self._alert(t, "WARNING",  f"Cell undervoltage: {vc:.3f} V")

    def _alert(self, t: float, level: str, msg: str):
        entry = {"t": t, "level": level, "msg": msg}
        # Deduplicate: only log if last alert was different
        if not self.alerts or self.alerts[-1]["msg"] != msg:
            self.alerts.append(entry)
            print(f"  [ALERT @ {t:.0f}s] {level}: {msg}")

    # ── helpers ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.history:
            return {}
        df  = pd.DataFrame(self.history)
        return {
            "duration_s":     df["timestamp"].max(),
            "mean_soh":       df["soh"].mean(),
            "final_soc":      df["soc"].iloc[-1],
            "max_temp":       df["temp_cell"].max(),
            "mean_ir_pack":   df["ir_pack"].mean(),
            "total_alerts":   len(self.alerts),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 11. DATASET GENERATOR  (bulk synthetic data for model training)
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    ages       = np.arange(0, 10.5, 0.5),
    cycles     = ("urban", "highway", "aggressive"),
    temps      = (15.0, 25.0, 35.0),
    duration_s = 1800.0,
    dt         = 0.5,
    output_path: str = "synthetic_can_dataset.parquet"
) -> pd.DataFrame:
    """
    Generate a comprehensive multi-condition synthetic dataset.

    Covers:
      - age 0 → 10 years (0.5 yr steps)
      - 3 drive cycles
      - 3 ambient temperatures
    Total: 21 × 3 × 3 = 189 sessions × 3600 frames each ≈ 680K rows
    """
    records = []
    total   = len(ages) * len(cycles) * len(temps)
    done    = 0

    print(f"[DataGen] Generating {total} sessions …")
    for age in ages:
        for cycle in cycles:
            for temp in temps:
                gen = SyntheticCANGenerator(
                    age_years=age, ambient_temp=temp,
                    cycle=cycle, seed=int(age * 1000 + ord(cycle[0]) + int(temp))
                )
                df = gen.generate_session(duration_s=duration_s, dt=dt)
                df["ambient_temp"] = temp
                records.append(df)
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  {done}/{total} sessions done …")

    dataset = pd.concat(records, ignore_index=True)

    try:
        dataset.to_parquet(output_path, index=False)
        print(f"[DataGen] Saved {len(dataset):,} rows → {output_path}")
    except ImportError:
        csv_path = output_path.replace(".parquet", ".csv")
        dataset.to_csv(csv_path, index=False)
        print(f"[DataGen] pyarrow not found — saved as CSV → {csv_path}")

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 12. NASA BATTERY DATASET LOADER  (optional validation)
# ─────────────────────────────────────────────────────────────────────────────

def load_nasa_battery(mat_file_path: str) -> pd.DataFrame:
    """
    Load NASA PCOE battery dataset (B0005-B0008 .mat files).

    Download from:
    https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

    Returns DataFrame with columns: cycle, capacity, soh (normalised to first cycle)
    """
    try:
        import scipy.io
    except ImportError:
        raise ImportError("Install scipy:  pip install scipy")

    mat    = scipy.io.loadmat(mat_file_path, simplify_cells=True)
    key    = [k for k in mat if not k.startswith("_")][0]
    cycles_raw = mat[key]["cycle"]

    discharge_caps = []
    for cyc in cycles_raw:
        try:
            if cyc["type"] == "discharge":
                cap = float(cyc["data"]["Capacity"])
                discharge_caps.append(cap)
        except (KeyError, TypeError):
            continue

    if not discharge_caps:
        raise ValueError(f"No discharge cycles found in {mat_file_path}")

    cap_series = pd.Series(discharge_caps)
    df = pd.DataFrame({
        "cycle":    range(len(cap_series)),
        "capacity": cap_series.values,
        "soh":      cap_series.values / cap_series.iloc[0],
    })
    print(f"[NASA] Loaded {len(df)} discharge cycles from {mat_file_path}")
    return df


def compare_with_nasa(nasa_df: pd.DataFrame, age_years_max: float = 8.0):
    """
    Print MAE between synthetic SOH curve and NASA SOH curve.
    Maps NASA cycle numbers → years (assuming ~1 cycle/day average).
    """
    aging    = ArrheniusAging()
    n_cycles = int(len(nasa_df))

    synth_soh = []
    nasa_soh  = []

    for i, row in nasa_df.iterrows():
        frac     = i / n_cycles
        days     = frac * age_years_max * 365
        n_cyc    = int(frac * n_cycles)
        s_soh    = aging.capacity_fade(days=days, n_cycles=n_cyc)
        synth_soh.append(s_soh)
        nasa_soh.append(row["soh"])

    mae = mean_absolute_error(nasa_soh, synth_soh)
    print(f"[Validation] SOH MAE vs NASA dataset: {mae*100:.2f}%")
    return mae


# ─────────────────────────────────────────────────────────────────────────────
# 13. DEMO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _print_frame(frame: dict, idx: int):
    pred = frame.get("predicted_soh")
    pred_str = f"{pred*100:.1f}%" if pred else "  n/a  "
    print(
        f"  t={frame['timestamp']:6.1f}s | "
        f"V={frame['voltage']:6.1f}V | "
        f"I={frame['current']:6.1f}A | "
        f"SOC={frame['soc']*100:5.1f}% | "
        f"SOH={frame['soh']*100:5.1f}% | "
        f"T={frame['temp_cell']:5.1f}°C | "
        f"IR={frame['ir_pack']*1000:5.1f}mΩ | "
        f"pred_SOH={pred_str}"
    )


def run_demo(predictor: "SOHPredictor | None" = None):
    scenarios = [
        {"age_years": 0.0,  "cycle": "urban",      "ambient_temp": 25.0},
        {"age_years": 3.0,  "cycle": "highway",     "ambient_temp": 30.0},
        {"age_years": 7.0,  "cycle": "aggressive",  "ambient_temp": 38.0},
        {"age_years": 10.0, "cycle": "urban",       "ambient_temp": 20.0},
    ]

    print("\n" + "═"*90)
    print("  VIRTUAL EV BATTERY DEMO  —  Synthetic CAN Data Simulation")
    print("═"*90)

    for sc in scenarios:
        age   = sc["age_years"]
        cycle = sc["cycle"]
        temp  = sc["ambient_temp"]
        print(f"\n▶ Scenario: age={age}yr | cycle={cycle} | ambient={temp}°C")
        print("─"*90)

        twin = VirtualBattery(
            age_years=age, cycle=cycle, ambient_temp=temp,
            predictor=predictor
        )

        # Print every 60 s for 600 s
        for step in range(1200):
            frame = twin.tick()
            if step % 120 == 0:
                _print_frame(frame, step)

        sm = twin.summary()
        print(f"\n  Summary → mean SOH: {sm['mean_soh']*100:.1f}%  |  "
              f"max temp: {sm['max_temp']:.1f}°C  |  "
              f"alerts fired: {sm['total_alerts']}")

    print("\n" + "═"*90)
    print("  Demo complete.")
    print("═"*90 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Virtual EV Battery Simulator")
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic CAN dataset")
    parser.add_argument("--train",    action="store_true",
                        help="Train SOH predictor on synthetic data")
    parser.add_argument("--all",      action="store_true",
                        help="Run generate + train + demo")
    args = parser.parse_args()

    predictor = None
    dataset   = None

    if args.generate or args.all:
        dataset = generate_dataset()

    if args.train or args.all:
        if dataset is None:
            print("[Train] Loading dataset …")
            try:
                dataset = pd.read_parquet("synthetic_can_dataset.parquet")
            except Exception:
                try:
                    dataset = pd.read_csv("synthetic_can_dataset.csv")
                except Exception:
                    print("[Train] Dataset not found — generating now …")
                    dataset = generate_dataset()

        predictor = SOHPredictor()
        predictor.train(dataset)

    # Always run the demo
    run_demo(predictor=predictor)


if __name__ == "__main__":
    main()
