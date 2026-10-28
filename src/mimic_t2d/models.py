"""Pre-specified model families used in identical held-out folds."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import ADAPTIVE_FEATURES, HISTORY_FEATURES, SNAPSHOT_FEATURES


def model_specifications() -> dict[str, tuple[list[str], object | None]]:
    ridge = lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    hgb = lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0, random_state=20261026))])
    return {
        "persistence": (["current_glucose"], None),
        "snapshot_ridge": (SNAPSHOT_FEATURES, ridge()),
        "history_ridge": (HISTORY_FEATURES, ridge()),
        "adaptive_ridge": (ADAPTIVE_FEATURES, ridge()),
        "adaptive_hgb": (ADAPTIVE_FEATURES, hgb()),
    }
