"""ML for DFT-guided screening: predict the ORR limiting potential of candidates *without* running DFT on them.

Setting
-------
Training rows are candidates that already have adsorption energies (from your DFT / a database; the demo
table is synthetic).  Features are **cheap**: composition descriptors, facet, skin flag.  The model predicts
U_L (and via the volcano from a predicted dG_OH), with an uncertainty, so a scientist can decide **which
candidates deserve an expensive DFT calculation next**.

Evaluation is **leave-one-chemical-system-out** (hold out every Pt-Co row, then every Pt-Ni row, ...):
the realistic question is "how well do we do for an alloying element we have never computed?".
Metrics: MAE of U_L (V), Spearman rank correlation, top-10% recall, and for uncertainty-aware models the
coverage of the nominal 90% interval.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .dft import TRUSTED, build_feature_table, chemical_system, volcano_curve

Z90 = 1.645


def usable_features(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].nunique() > 1]


def make_model(name: str):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED)
    if name == "Random Forest":
        return RandomForestRegressor(300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    if name == "Gaussian Process":
        k = ConstantKernel(1.0) * RBF(length_scale=3.0) + WhiteKernel(0.01)
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1,
                                                                        random_state=SEED))
    raise ValueError(name)


MODELS = ["Ridge", "Gradient Boosting", "Random Forest", "Gaussian Process"]


def predict_with_std(name: str, model, X: pd.DataFrame):
    """Mean and standard deviation (None for models without an uncertainty)."""
    if name == "Gaussian Process":
        mu, sd = model.predict(X.values, return_std=True) if not hasattr(model, "steps") else \
            model[-1].predict(model[:-1].transform(X.values), return_std=True)
        return mu, sd
    if name == "Random Forest":
        per_tree = np.stack([t.predict(X.values) for t in model.estimators_])
        return per_tree.mean(0), per_tree.std(0)
    return model.predict(X.values), None


def rank_metrics(y_true, y_pred, frac: float = 0.1) -> dict:
    """Spearman rho and the share of the true top `frac` of candidates found in the predicted top `frac`."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else np.nan
    k = max(1, int(round(frac * len(y_true))))
    top_true, top_pred = set(np.argsort(-y_true)[:k]), set(np.argsort(-y_pred)[:k])
    return {"spearman": float(rho), "top10pct_recall": len(top_true & top_pred) / k}


# --------------------------------------------------------------------------- baselines (no ML)
def baseline_scores(df: pd.DataFrame, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rankings from one simple quantity.  Higher score = predicted better catalyst."""
    out = {"strain heuristic (compressive = better)": -X["lattice_mismatch_vs_Pt"].values,
           "electronegativity vs Pt (closer = better)": -np.abs(X["chi_mean"].values - 2.28)}
    if "d_band_center" in df and df["d_band_center"].notna().all():
        out["d-band centre (lower = better)"] = -df["d_band_center"].values
    if "dG_OH" in df:
        out["ORACLE: volcano from the DFT dG_OH (needs DFT)"] = volcano_curve(df["dG_OH"].values)
    return out


def evaluate_baselines(df: pd.DataFrame) -> pd.DataFrame:
    X = build_feature_table(df)
    rows = [{"method": n, "needs_DFT_for_candidate": n.startswith("ORACLE") or n.startswith("d-band"),
             **rank_metrics(df["U_L"].values, s)} for n, s in baseline_scores(df, X).items()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- leave-one-system-out CV
def loso_evaluate(df: pd.DataFrame, models=None, route: str = "U_L direct") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary per model, out-of-fold predictions).

    route 'U_L direct'      : regress U_L
    route 'dG_OH -> volcano': regress dG_OH, then U_L = volcano(dG_OH)   (physics-structured target)
    """
    models = models or MODELS
    X = build_feature_table(df)[usable_features(build_feature_table(df))]
    groups = df["chemical_system"] if "chemical_system" in df else df["composition"].map(chemical_system)
    target = df["U_L"].values if route == "U_L direct" else df["dG_OH"].values
    oof = {m: np.full(len(df), np.nan) for m in models}
    sd = {m: np.full(len(df), np.nan) for m in models}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in LeaveOneGroupOut().split(X, target, groups):
            for m in models:
                mod = clone(make_model(m)).fit(X.iloc[tr].values, target[tr])
                mu, s = predict_with_std(m, mod, X.iloc[te])
                oof[m][te] = mu
                if s is not None:
                    sd[m][te] = s
    rows = []
    for m in models:
        if route == "U_L direct":
            u_hat, u_sd = oof[m], sd[m]
        else:
            u_hat = volcano_curve(oof[m])
            u_sd = sd[m]                       # sd of dG_OH (the volcano slope is +-1, so the scale carries over)
        row = {"model": m, "route": route, "MAE_U_L_V": float(np.abs(u_hat - df["U_L"].values).mean()),
               **rank_metrics(df["U_L"].values, u_hat)}
        if not np.isnan(u_sd).all():
            inside = np.abs(u_hat - df["U_L"].values) <= Z90 * u_sd
            row["coverage_90pct"] = float(inside.mean())
            row["mean_sd_V"] = float(np.nanmean(u_sd))
        rows.append(row)
    oof_df = df[["system_id", "composition", "facet", "U_L"]].copy()
    for m in models:
        oof_df[f"pred_{m}"] = oof[m] if route == "U_L direct" else volcano_curve(oof[m])
        oof_df[f"sd_{m}"] = sd[m]
    return pd.DataFrame(rows).sort_values("MAE_U_L_V").reset_index(drop=True), oof_df


# --------------------------------------------------------------------------- screening new candidates
class Screener:
    """Fit on a DFT table, then score candidates that have no DFT value yet."""

    def __init__(self, df: pd.DataFrame, model_name: str = "Gaussian Process"):
        self.model_name = model_name
        self.train_df = df.reset_index(drop=True)
        Xall = build_feature_table(self.train_df)
        self.cols = usable_features(Xall)
        self.X = Xall[self.cols]
        self.scaler = StandardScaler().fit(self.X.values)
        Z = self.scaler.transform(self.X.values)
        d = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1)) + np.eye(len(Z)) * 1e9
        self.nn_scale = float(np.median(d.min(1)))            # typical nearest-neighbour distance inside the training set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = make_model(model_name).fit(self.X.values, self.train_df["U_L"].values)
        self.trusted_fraction = float(self.train_df["provenance"].isin(TRUSTED).mean())
        self.recommendation_allowed = self.trusted_fraction >= 0.5
        self.systems = set(self.train_df["chemical_system"]) if "chemical_system" in self.train_df else set()

    def score(self, candidates: pd.DataFrame, kappa: float = 1.0) -> pd.DataFrame:
        """candidates need `composition` and `facet` (and optional `structure`). No DFT values required."""
        Xc = build_feature_table(candidates)
        Xc = Xc.reindex(columns=self.cols, fill_value=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sd = predict_with_std(self.model_name, self.model, Xc)
        if sd is None:
            sd = np.full(len(mu), np.nan)
        Zc = self.scaler.transform(Xc.values)
        Z = self.scaler.transform(self.X.values)
        dist = np.sqrt(((Zc[:, None] - Z[None]) ** 2).sum(-1)).min(1) / self.nn_scale
        out = candidates.reset_index(drop=True).copy()
        out["pred_U_L"], out["pred_sd"], out["ood_ratio"] = mu, sd, dist
        out["system"] = out["composition"].map(chemical_system)
        out["new_system"] = ~out["system"].isin(self.systems)
        out["outside_training_distribution"] = (dist > 2.0) | out["new_system"]
        out["acquisition_UCB"] = mu + kappa * np.nan_to_num(sd)
        out["rank"] = out["pred_U_L"].rank(ascending=False, method="first").astype(int)
        out["confidence"] = [confidence_label(s, o) for s, o in zip(out["pred_sd"], out["outside_training_distribution"])]
        out["recommendation_allowed"] = self.trusted_fraction >= 0.5
        out["recommendation"] = [recommend(r) if self.recommendation_allowed else
                                 "NO RECOMMENDATION: the training table is not DFT/MLIP/literature data (synthetic or toy rows)."
                                 for _, r in out.iterrows()]
        out["provenance"] = "ml_predicted"                    # a prediction, never a calculated value
        return out.sort_values("rank").reset_index(drop=True)


def confidence_label(sd: float, ood: bool) -> str:
    if ood:
        return "Low (outside the training distribution)"
    if np.isnan(sd):
        return "Unknown (model has no uncertainty)"
    return "High" if sd < 0.06 else "Moderate" if sd < 0.12 else "Low"


def recommend(row) -> str:
    if row["outside_training_distribution"]:
        return "Run DFT adsorption calculations (OH, O, OOH) before trusting this ranking; the model has not seen this chemistry."
    if row["rank"] <= 3:
        return "Promising: confirm with DFT adsorption energies, check stability (dissolution, segregation), then consider synthesis."
    return "Low priority for DFT unless uncertainty is high; do not synthesise on this prediction alone."


def suggest_next_dft(scored: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """The k candidates with the highest upper confidence bound (high predicted U_L OR high uncertainty)."""
    return scored.sort_values("acquisition_UCB", ascending=False).head(k)[
        ["composition", "facet", "pred_U_L", "pred_sd", "acquisition_UCB", "confidence"]]
