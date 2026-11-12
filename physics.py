"""Polarisation-curve physics: descriptors, loss breakdown and failure screening.

Everything here is *semi-empirical*: it reads a polarisation curve the way an
electrochemist does (kinetic region -> ohmic/linear region -> mass-transport
knee) and turns it into a small set of numbers.  No black box is involved, so
each number can be defended and later compared with what an ML model says.

Units: current density in mA cm-2, voltage in V, power density in mW cm-2,
resistance in ohm cm2 (slope of V vs j, multiplied by 1000).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

E_REV_DEFAULT = 1.18   # V, thermodynamic cell voltage near 75 C (liquid water product)

# How each loss zone maps to a cell component.  Wording is deliberately hedged:
# a polarisation curve alone cannot prove a root cause.
ZONE_INFO = {
    "kinetic": {
        "label": "Activation / kinetic zone (low current)",
        "component": "cathode catalyst layer (ORR activity) or gas crossover",
        "kb_query": "low open circuit voltage kinetic loss cathode catalyst ORR activity crossover",
        "parameters": ["catalyst loading", "Pt utilisation / ECSA", "ionomer content", "humidity", "pressure"],
    },
    "ohmic": {
        "label": "Ohmic / linear zone (mid current)",
        "component": "membrane and ionomer hydration, contact resistance",
        "kb_query": "high ohmic resistance membrane dehydration contact resistance HFR humidity",
        "parameters": ["relative humidity", "membrane compression", "ionomer content", "temperature"],
    },
    "mass_transport": {
        "label": "Mass-transport zone (high current)",
        "component": "cathode GDL / catalyst layer (flooding or O2 transport)",
        "kb_query": "mass transport loss flooding oxygen transport high current density cathode",
        "parameters": ["pressure", "relative humidity", "ionomer film / pore structure", "gas flow rate"],
    },
}


@dataclass
class CurveDescriptors:
    n_points: int
    j_min: float
    j_max: float
    v_at_jmin: float
    intercept_v: float          # a in V = a - b*j (linear-region fit)
    asr_ohm_cm2: float          # b*1000, apparent area-specific resistance
    fit_r2: float
    j_knee: float               # first j where V falls >= knee_drop below the linear fit
    mt_loss_at_jmax: float      # V below the linear extrapolation at j_max
    p_max: float                # mW cm-2
    j_at_pmax: float
    j_at_0p6: float             # interpolated, NaN if 0.6 V is not bracketed
    tafel_mv_dec: float         # NaN when fewer than 3 low-current points

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _sorted_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    d = df.sort_values("current_density")
    return d["current_density"].to_numpy(float), d["cell_voltage"].to_numpy(float)


def curve_descriptors(df: pd.DataFrame, knee_drop: float = 0.03) -> CurveDescriptors | None:
    """Descriptors of one curve (needs >= 6 points, otherwise returns None)."""
    j, v = _sorted_curve(df)
    if len(j) < 6:
        return None
    # 1) linear region: 0.45-0.80 V is where kinetics have saturated and
    #    mass transport has not yet bent the curve.  Fall back to the middle
    #    half of the points when that window is too thin.
    mask = (v >= 0.45) & (v <= 0.80)
    if mask.sum() < 4:
        lo, hi = len(j) // 4, len(j) - len(j) // 4
        mask = np.zeros(len(j), bool)
        mask[lo:hi] = True
    slope, intercept = np.polyfit(j[mask], v[mask], 1)
    pred = slope * j[mask] + intercept
    ss_res, ss_tot = np.sum((v[mask] - pred) ** 2), np.sum((v[mask] - v[mask].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    v_lin = slope * j + intercept
    # 2) mass-transport knee: beyond the fit window, first point clearly below the line
    beyond = j > j[mask].max()
    deficit = v_lin - v
    knee_idx = np.where(beyond & (deficit >= knee_drop))[0]
    j_knee = float(j[knee_idx[0]]) if len(knee_idx) else np.nan
    mt_loss = float(max(0.0, deficit[-1])) if beyond.any() else 0.0
    # 3) power
    p = j * v
    ip = int(np.argmax(p))
    # 4) j at 0.6 V by interpolation (v is monotonic decreasing with j in practice)
    order = np.argsort(v)
    j06 = float(np.interp(0.6, v[order], j[order])) if v.min() <= 0.6 <= v.max() else np.nan
    # 5) Tafel-like slope on the low-current points that are still kinetic
    tmask = (v >= 0.70) & (j > 0)
    if tmask.sum() >= 3 and np.ptp(np.log10(j[tmask])) > 0.1:
        tafel = -np.polyfit(np.log10(j[tmask]), v[tmask], 1)[0] * 1000
    else:
        tafel = np.nan
    return CurveDescriptors(
        n_points=len(j), j_min=float(j.min()), j_max=float(j.max()), v_at_jmin=float(v[0]),
        intercept_v=float(intercept), asr_ohm_cm2=float(-slope * 1000), fit_r2=float(r2),
        j_knee=j_knee, mt_loss_at_jmax=mt_loss, p_max=float(p[ip]), j_at_pmax=float(j[ip]),
        j_at_0p6=j06, tafel_mv_dec=float(tafel),
    )


def loss_breakdown(desc: CurveDescriptors, j_ref: float, e_rev: float = E_REV_DEFAULT) -> dict:
    """Split the voltage deficit at `j_ref` into three additive parts (V).

    kinetic        = E_rev - a                 (intercept deficit: activation + crossover)
    ohmic          = b * j_ref                 (linear slope: ohmic + charge-transfer)
    mass_transport = V_linear(j_ref) - V(j_ref) is not available without the raw
                     curve, so it is estimated from the knee deficit scaled to j_ref.
    The three parts add up to E_rev - V(j_ref) by construction in `breakdown_from_curve`.
    """
    ohmic = desc.asr_ohm_cm2 * j_ref / 1000.0
    kinetic = e_rev - desc.intercept_v
    return {"kinetic": kinetic, "ohmic": ohmic, "mass_transport": desc.mt_loss_at_jmax}


def breakdown_from_curve(df: pd.DataFrame, j_ref: float | None = None,
                         e_rev: float = E_REV_DEFAULT) -> dict:
    """Exact additive breakdown at `j_ref` (defaults to the curve's j at max power)."""
    desc = curve_descriptors(df)
    if desc is None:
        return {}
    j, v = _sorted_curve(df)
    j_ref = float(j_ref if j_ref is not None else desc.j_at_pmax)
    v_ref = float(np.interp(j_ref, j, v))
    v_lin = desc.intercept_v - desc.asr_ohm_cm2 * j_ref / 1000.0
    parts = {"kinetic": e_rev - desc.intercept_v,
             "ohmic": desc.asr_ohm_cm2 * j_ref / 1000.0,
             "mass_transport": v_lin - v_ref}
    parts["total_deficit"] = e_rev - v_ref
    parts["j_ref"] = j_ref
    parts["v_ref"] = v_ref
    return parts


# ----------------------------------------------------------------------------
# Corpus level: descriptor table, peer comparison, failure screening
# ----------------------------------------------------------------------------
META_COLS = ["curve_id", "test_family", "source_file", "pressure", "relative_humidity",
             "membrane_compression", "nafion_percent", "operating_set"]


def build_curve_table(pol: pd.DataFrame) -> pd.DataFrame:
    """One row per curve: metadata + descriptors."""
    rows = []
    for cid, g in pol.groupby("curve_id"):
        d = curve_descriptors(g)
        if d is None:
            continue
        meta = g.iloc[0][META_COLS].to_dict()
        rows.append({**meta, **d.as_dict()})
    return pd.DataFrame(rows)


def _peer_key(t: pd.DataFrame) -> pd.Series:
    """Peers share operating conditions; if those are not recorded, the test family."""
    cond = t["pressure"].astype(str) + "|" + t["relative_humidity"].astype(str)
    return np.where(t["pressure"].isna(), "fam:" + t["test_family"], cond)


# direction = +1 means "higher is worse", -1 means "lower is worse"
SCREEN_RULES = {
    "kinetic": ("intercept_v", -1),
    "ohmic": ("asr_ohm_cm2", +1),
    "mass_transport": ("mt_loss_at_jmax", +1),
    "power": ("p_max", -1),
}


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad < 1e-9:
        mad = x.std(ddof=0) if x.std(ddof=0) > 1e-9 else 1.0
    return (x - med) / mad


def screen_curves(table: pd.DataFrame, z_flag: float = 1.5) -> pd.DataFrame:
    """Add peer-relative 'badness' z-scores and failure flags.

    badness_z > 0 always means *worse than peers* (same pressure/RH).  A flag
    is raised when badness_z >= `z_flag` (about the worst 7% if normal).  The
    thresholds are screening heuristics, not validated failure criteria.
    """
    t = table.copy()
    t["peer_group"] = _peer_key(t)
    for zone, (col, direction) in SCREEN_RULES.items():
        z = t.groupby("peer_group")[col].transform(robust_z)
        # peer groups of < 3 curves give meaningless z: neutralise them
        n = t.groupby("peer_group")[col].transform("size")
        t[f"bad_{zone}"] = np.where(n >= 3, direction * z, 0.0)
        t[f"flag_{zone}"] = t[f"bad_{zone}"] >= z_flag
    # Curves whose linear window is not linear (or has a non-physical slope) do not
    # follow the 3-zone model: say so instead of forcing a zone on them.
    t["fit_ok"] = (t["fit_r2"] >= 0.95) & (t["asr_ohm_cm2"] > 0)
    bad = t[[f"bad_{z}" for z in ZONE_INFO]].to_numpy()
    zone = np.where(bad.max(axis=1) >= z_flag,
                    np.array(list(ZONE_INFO))[bad.argmax(axis=1)], "none")
    t["dominant_zone"] = np.where(t["fit_ok"], zone, "unreliable_fit")
    return t


def diagnose_row(row: pd.Series) -> dict:
    """Turn one screened curve row into a structured diagnosis."""
    flags = [z for z in ZONE_INFO if bool(row.get(f"flag_{z}", False))]
    zone = row.get("dominant_zone", "none")
    info = ZONE_INFO.get(zone)
    if zone == "unreliable_fit":
        info = {"label": "Curve shape not captured by the 3-zone model (inspect manually)",
                "component": "undetermined", "parameters": [], "kb_query": "reading a polarisation curve zones"}
    severity = float(max(row.get(f"bad_{z}", 0.0) for z in ZONE_INFO))
    return {
        "curve_id": int(row["curve_id"]),
        "dominant_zone": zone,
        "flags": flags,
        "severity_z": round(severity, 2),
        "zone_label": info["label"] if info else "No zone stands out against peers",
        "likely_component": info["component"] if info else "-",
        "parameters_to_check": info["parameters"] if info else [],
        "kb_query": info["kb_query"] if info else "",
        "descriptors": {k: (None if pd.isna(row[k]) else round(float(row[k]), 4))
                        for k in ["intercept_v", "asr_ohm_cm2", "j_knee", "mt_loss_at_jmax",
                                  "p_max", "j_at_pmax", "j_at_0p6", "tafel_mv_dec"]},
        "conditions": {k: (None if pd.isna(row[k]) else row[k])
                       for k in ["test_family", "pressure", "relative_humidity",
                                 "membrane_compression", "nafion_percent", "operating_set"]},
    }


# ----------------------------------------------------------------------------
# Synthetic fault injection (how we test the screening without fault labels)
# ----------------------------------------------------------------------------
def inject_fault(df: pd.DataFrame, kind: str, severity: float) -> pd.DataFrame:
    """Return a copy of a healthy curve with a known, physically shaped fault.

    kinetic        : voltage shifted down by `severity` V at all currents
    ohmic          : extra series resistance of `severity` ohm cm2 (V drops with j)
    mass_transport : exponential knee that starts at ~60% of j_max and reaches
                     `severity` V at j_max
    """
    out = df.copy()
    j = out["current_density"].to_numpy(float)
    if kind == "kinetic":
        dv = np.full_like(j, severity)
    elif kind == "ohmic":
        dv = severity * j / 1000.0
    elif kind == "mass_transport":
        x = np.clip((j / j.max() - 0.6) / 0.4, 0, None)
        dv = severity * (np.exp(3 * x) - 1) / (np.exp(3) - 1)
    else:
        raise ValueError(kind)
    out["cell_voltage"] = (out["cell_voltage"] - dv).clip(lower=0.02)
    out["power_density"] = out["current_density"] * out["cell_voltage"]
    return out
