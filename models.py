"""Machine-learning layer (the escalation of Project 2).

Three jobs, each with its own honest evaluation:

1. ``compare_models``      voltage regression on all data, grouped CV by source file
                           (Project 2 logic, now with a neural network and timings).
2. ``oof_curve_residuals`` out-of-fold residuals per curve: a *second opinion* on
                           which curves under-perform what their conditions predict.
3. ``DesignSurrogate``     a small surrogate trained on the Nafion standard test, the only
                           file in which ionomer content and compression vary, used for
                           what-if screening *inside* the measured design space.
"""
from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SEED

NUMERIC = ["current_density", "pressure", "relative_humidity", "membrane_compression",
           "nafion_percent", "operating_set"]
CATEGORICAL = ["test_family"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "cell_voltage"
GROUP = "source_file"       # whole files are held out; power_density is never a feature (leakage)


def make_preprocessor(scale: bool) -> ColumnTransformer:
    num = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num.append(("scale", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(num), NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])


def model_zoo(seed: int = SEED) -> dict[str, Pipeline]:
    """Same preprocessing for every model so the comparison isolates the learner."""
    return {
        "Mean baseline": Pipeline([("pre", make_preprocessor(False)), ("m", DummyRegressor())]),
        "Ridge": Pipeline([("pre", make_preprocessor(True)), ("m", Ridge(alpha=1.0))]),
        "Random Forest": Pipeline([("pre", make_preprocessor(False)),
                                   ("m", RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=seed))]),
        "Extra Trees": Pipeline([("pre", make_preprocessor(False)),
                                 ("m", ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=seed))]),
        "Gradient Boosting": Pipeline([("pre", make_preprocessor(False)),
                                       ("m", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=seed))]),
        "Neural net (MLP 64-64)": Pipeline([("pre", make_preprocessor(True)),
                                            ("m", MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=600,
                                                               random_state=seed))]),
    }


def holdout_split(data: pd.DataFrame, test_size: float = 0.22, seed: int = SEED):
    """Hold out whole source files (as in Project 2) so test curves are never seen in training."""
    tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(data, groups=data[GROUP]))
    return data.iloc[tr].reset_index(drop=True), data.iloc[te].reset_index(drop=True)


def _metrics(y, p) -> dict:
    return {"MAE_V": mean_absolute_error(y, p), "RMSE_V": mean_squared_error(y, p) ** 0.5, "R2": r2_score(y, p)}


def compare_models(train: pd.DataFrame, n_splits: int = 5, models: dict | None = None) -> pd.DataFrame:
    """Grouped CV on the training files only; the test files stay untouched."""
    models = models or model_zoo()
    # Some folds have no recorded value for a column (e.g. Nafion %); the imputer then drops it
    # and warns.  That is expected here, so the warning is silenced.
    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
    X, y, g = train[FEATURES], train[TARGET], train[GROUP]
    folds = list(GroupKFold(n_splits).split(X, y, g))
    rows = []
    for name, pipe in models.items():
        scores, t_fit = [], 0.0
        for tr, va in folds:
            from sklearn.base import clone
            m = clone(pipe)
            t0 = time.perf_counter()
            m.fit(X.iloc[tr], y.iloc[tr])
            t_fit += time.perf_counter() - t0
            scores.append(_metrics(y.iloc[va], m.predict(X.iloc[va])))
        s = pd.DataFrame(scores)
        rows.append({"model": name, **{f"{c}_mean": s[c].mean() for c in s}, **{f"{c}_std": s[c].std() for c in s},
                     "fit_seconds_total": t_fit})
    return pd.DataFrame(rows).sort_values("MAE_V_mean").reset_index(drop=True)


def model_size_kb(pipe) -> float:
    import io
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    return buf.tell() / 1024


def final_test(pipe, train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit on all training files and score ONCE on the held-out files."""
    from sklearn.base import clone
    m = clone(pipe).fit(train[FEATURES], train[TARGET])
    pred = m.predict(test[FEATURES])
    out = test.assign(predicted_voltage=pred, residual=test[TARGET] - pred)
    return _metrics(test[TARGET], pred), out


# ----------------------------------------------------------------------------
# Second opinion on failures: out-of-fold residuals per curve
# ----------------------------------------------------------------------------
def oof_curve_residuals(data: pd.DataFrame, pipe: Pipeline | None = None, n_splits: int = 5) -> pd.DataFrame:
    """Mean (measured - predicted) voltage per curve, predicted by models that never saw that file.

    Strongly negative = the curve is worse than the conditions alone would predict.
    """
    from sklearn.base import clone
    pipe = pipe or model_zoo()["Gradient Boosting"]
    X, y, g = data[FEATURES], data[TARGET], data[GROUP]
    oof = np.zeros(len(data))
    for tr, va in GroupKFold(n_splits).split(X, y, g):
        oof[va] = clone(pipe).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
    r = data.assign(oof_pred=oof, resid=y - oof)
    c = r.groupby("curve_id").agg(mean_resid=("resid", "mean"), rmse=("resid", lambda s: float(np.sqrt((s ** 2).mean()))),
                                  n=("resid", "size")).reset_index()
    med = c["mean_resid"].median()
    mad = (c["mean_resid"] - med).abs().median() * 1.4826 or 1e-6
    c["resid_z"] = (c["mean_resid"] - med) / mad          # < -2 means clearly under-performing
    return c


# ----------------------------------------------------------------------------
# What-if surrogate inside the measured design space
# ----------------------------------------------------------------------------
DESIGN_VARS = ["pressure", "relative_humidity", "membrane_compression", "nafion_percent"]


class DesignSurrogate:
    """Gradient-boosting surrogate V(j; pressure, RH, compression, Nafion %).

    It is trained on the ECSIM Nafion standard test file with all four design variables
    recorded (one MEA design, 40+ conditions).  It can interpolate between the measured
    conditions; it refuses to extrapolate beyond the measured ranges.
    """

    def __init__(self, seed: int = SEED):
        self.model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed)
        self.features = ["current_density"] + DESIGN_VARS
        self.ranges: dict[str, tuple[float, float]] = {}
        self.levels: dict[str, list[float]] = {}
        self.cv_mae_: float | None = None

    @staticmethod
    def training_rows(pol: pd.DataFrame) -> pd.DataFrame:
        d = pol.dropna(subset=DESIGN_VARS + ["current_density", "cell_voltage"])
        return d.reset_index(drop=True)

    def fit(self, pol: pd.DataFrame) -> "DesignSurrogate":
        d = self.training_rows(pol)
        self.ranges = {c: (float(d[c].min()), float(d[c].max())) for c in ["current_density"] + DESIGN_VARS}
        self.levels = {c: sorted(float(x) for x in d[c].unique()) for c in DESIGN_VARS}
        # leave-one-curve-out style check: each curve is a unique condition combination
        from sklearn.base import clone
        oof = np.zeros(len(d))
        for tr, va in GroupKFold(6).split(d, groups=d["curve_id"]):
            oof[va] = clone(self.model).fit(d.iloc[tr][self.features], d.iloc[tr][TARGET]).predict(d.iloc[va][self.features])
        self.cv_mae_ = float(mean_absolute_error(d[TARGET], oof))
        self.model.fit(d[self.features], d[TARGET])
        return self

    def check(self, **cond) -> list[str]:
        """Warnings for conditions outside the measured design space."""
        warn = []
        for k, v in cond.items():
            lo, hi = self.ranges[k]
            if not lo <= v <= hi:
                warn.append(f"{k}={v} is outside the measured range [{lo:g}, {hi:g}]: no extrapolation allowed")
        return warn

    def predict_curve(self, pressure, relative_humidity, membrane_compression, nafion_percent,
                      j_grid: np.ndarray | None = None) -> pd.DataFrame:
        warn = self.check(pressure=pressure, relative_humidity=relative_humidity,
                          membrane_compression=membrane_compression, nafion_percent=nafion_percent)
        if warn:
            raise ValueError("; ".join(warn))
        lo, hi = self.ranges["current_density"]
        j = np.linspace(lo, hi, 60) if j_grid is None else np.asarray(j_grid, float)
        X = pd.DataFrame({"current_density": j, "pressure": pressure, "relative_humidity": relative_humidity,
                          "membrane_compression": membrane_compression, "nafion_percent": nafion_percent})
        v = self.model.predict(X[self.features])
        return pd.DataFrame({"current_density": j, "cell_voltage": v, "power_density": j * v})

    def screen_grid(self) -> pd.DataFrame:
        """Predicted peak power and current at 0.6 V for every combination of measured levels."""
        rows = []
        for p in self.levels["pressure"]:
            for rh in self.levels["relative_humidity"]:
                for mc in self.levels["membrane_compression"]:
                    for n in self.levels["nafion_percent"]:
                        c = self.predict_curve(p, rh, mc, n)
                        order = np.argsort(c["cell_voltage"].to_numpy())
                        j06 = float(np.interp(0.6, c["cell_voltage"].to_numpy()[order], c["current_density"].to_numpy()[order]))
                        rows.append({"pressure": p, "relative_humidity": rh, "membrane_compression": mc,
                                     "nafion_percent": n, "pred_p_max": float(c["power_density"].max()),
                                     "pred_j_at_0p6": j06})
        return pd.DataFrame(rows).sort_values("pred_p_max", ascending=False).reset_index(drop=True)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "DesignSurrogate":
        return joblib.load(path)
