"""Loading the ECSIM PEM fuel-cell dataset (polarisation + impedance).

Data: Hamidi, Haghighi, Askari (2020), doi:10.26434/chemrxiv.11902023,
CC BY 4.0, https://github.com/ECSIM/pem-dataset1.  Cathode and anode are
Pt/C (20 wt%) on carbon paper with a Nafion membrane, so *catalyst
chemistry is not varied* in this data; what varies is ionomer content,
membrane compression, pressure, humidity and activation history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_DIR

POL_REQUIRED = {"current_density", "cell_voltage", "power_density"}
EIS_REQUIRED = {"z_real", "z_img", "applied_voltage"}
CURVE_KEYS = ["source_file", "pressure", "relative_humidity",
              "membrane_compression", "nafion_percent", "operating_set"]


def _read(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_polarization() -> pd.DataFrame:
    """All polarisation tables, one row per measured point, plus a `curve_id`.

    A *curve* is one voltage sweep at fixed (file, pressure, RH, compression,
    Nafion %, activation set).  Curve level is the right unit for physics
    descriptors and for group-aware validation.
    """
    frames = []
    for path in sorted(RAW_DIR.rglob("*.csv")):
        df = _read(path)
        if df is None or not POL_REQUIRED.issubset(df.columns):
            continue
        out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce")
                            for c in ["current_density", "cell_voltage", "power_density"]})
        for c in ["pressure", "relative_humidity", "membrane_compression", "nafion_percent"]:
            out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
        out["operating_set"] = pd.to_numeric(df["set"], errors="coerce") if "set" in df else np.nan
        out["test_family"] = path.parent.name
        out["source_file"] = str(path.relative_to(RAW_DIR))
        frames.append(out)
    if not frames:
        raise FileNotFoundError(f"No polarisation CSVs under {RAW_DIR}. Run: python download_data.py")
    data = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["current_density", "cell_voltage"])
    data = data[(data.current_density > 0)].reset_index(drop=True)
    # NaN keys (e.g. "pressure not recorded") must still form a valid group.
    data["curve_id"] = data.groupby(CURVE_KEYS, dropna=False, sort=False).ngroup()
    return data


def load_eis() -> pd.DataFrame:
    """All impedance (Nyquist) tables.  No frequency column is provided."""
    frames = []
    for path in sorted(RAW_DIR.rglob("*.csv")):
        df = _read(path)
        if df is None or not EIS_REQUIRED.issubset(df.columns):
            continue
        out = df[list(EIS_REQUIRED)].apply(pd.to_numeric, errors="coerce")
        for c in ["pressure", "relative_humidity"]:
            out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
        out["operating_set"] = pd.to_numeric(df["set"], errors="coerce") if "set" in df else np.nan
        out["test_family"] = path.parent.name
        out["source_file"] = str(path.relative_to(RAW_DIR))
        frames.append(out)
    if not frames:
        raise FileNotFoundError("No impedance CSVs found.")
    return pd.concat(frames, ignore_index=True).dropna(subset=["z_real", "z_img"])


def eis_summary() -> pd.DataFrame:
    """Per (file, applied voltage, set/pressure/RH) Nyquist proxies.

    hfr_proxy = smallest z_real (high-frequency intercept ~ ohmic resistance)
    arc_width = max z_real - min z_real (sampled arc ~ charge transfer +
                mass-transport resistance)
    These are *proxies* because the frequency axis is not in the files.
    """
    e = load_eis()
    for c in ["pressure", "relative_humidity", "operating_set"]:
        e[c] = e[c].fillna(-1)          # groupby-safe placeholder for "not recorded"
    keys = ["source_file", "test_family", "applied_voltage", "pressure",
            "relative_humidity", "operating_set"]
    s = (e.groupby(keys)
          .agg(hfr_proxy=("z_real", "min"), z_real_max=("z_real", "max"),
               n_points=("z_real", "size"))
          .reset_index())
    s["arc_width"] = s["z_real_max"] - s["hfr_proxy"]
    for c in ["pressure", "relative_humidity", "operating_set"]:
        s[c] = s[c].replace(-1, np.nan)
    return s.drop(columns="z_real_max")
