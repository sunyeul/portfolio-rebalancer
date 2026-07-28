"""Chronological, research-only forecasts over the Qlib validation snapshot."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from research.qlib_validation.artifacts import (
    canonical_bytes,
    staged_run,
    write_inputs,
    write_json,
)
from research.qlib_validation.contracts import Candle, SourceSnapshot
from research.qlib_validation.environment import environment_info
from research.qlib_validation.source import load_snapshot

ROOT = Path(__file__).resolve().parents[2]
FEATURE_NAMES = (
    "return_5",
    "return_20",
    "return_60",
    "volatility_20",
    "drawdown_60",
)
DEFAULT_HORIZON_SESSIONS = 20
DEFAULT_TEST_SESSIONS = 120
DEFAULT_EMBARGO_SESSIONS = 20
DEFAULT_MINIMUM_TRAIN_DATES = 100
DEFAULT_INNER_VALIDATION_SESSIONS = 60
LGB_NUM_BOOST_ROUND_CAP = 120
LGB_EARLY_STOPPING_ROUNDS = 20
LGB_RANDOM_SEED = 1729
LGB_NUM_THREADS = 1


class ForecastError(RuntimeError):
    """Raised when a point-in-time forecast cannot be validated safely."""


def _features(candles: tuple[Candle, ...], index: int) -> dict[str, float]:
    closes = np.array([item.close_price for item in candles], dtype=float)
    recent_daily_returns = (
        closes[index - 19 : index + 1] / closes[index - 20 : index] - 1.0
    )
    return {
        "return_5": float(closes[index] / closes[index - 5] - 1.0),
        "return_20": float(closes[index] / closes[index - 20] - 1.0),
        "return_60": float(closes[index] / closes[index - 60] - 1.0),
        "volatility_20": float(np.std(recent_daily_returns, ddof=0)),
        "drawdown_60": float(
            closes[index] / np.max(closes[index - 59 : index + 1]) - 1.0
        ),
    }


def _feature_frames(
    snapshot: SourceSnapshot, *, horizon_sessions: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for spec in snapshot.policy_specs:
        candles = tuple(
            sorted(snapshot.candles_for(spec.key), key=lambda item: item.available_at)
        )
        if len(candles) <= 60 + horizon_sessions:
            raise ForecastError(f"forecast history is too short for series: {spec.key}")
        for index in range(60, len(candles)):
            candle = candles[index]
            row = {
                "key": spec.key,
                "instrument": spec.key.replace("/", "_"),
                "decision_session": candle.session_date.isoformat(),
                "decision_timestamp": candle.available_at.isoformat(),
                **_features(candles, index),
            }
            if index + horizon_sessions < len(candles):
                future = candles[index + horizon_sessions]
                labeled.append(
                    {
                        **row,
                        "actual_return": float(
                            future.close_price / candle.close_price - 1.0
                        ),
                        "target_session": future.session_date.isoformat(),
                    }
                )
            if index == len(candles) - 1:
                current.append(row)
    if not labeled or not current:
        raise ForecastError("forecast data contains no labeled or current rows")
    return pd.DataFrame(labeled), pd.DataFrame(current)


def _verify_qlib_static_loader(frame: pd.DataFrame) -> dict[str, bool]:
    """Run the prediction feature panel through Qlib's concrete static loader."""
    from qlib.data.dataset.loader import StaticDataLoader

    expected = (
        frame.assign(datetime=pd.to_datetime(frame["decision_session"]))
        .set_index(["datetime", "instrument"])[list(FEATURE_NAMES)]
        .sort_index()
    )
    loaded = StaticDataLoader(expected).load().sort_index()
    assert_frame_equal(loaded, expected, check_dtype=False, check_names=True)
    return {"matched": True}


class _FrameDataset:
    """Minimal tabular dataset adapter for Qlib's native tabular models."""

    def __init__(
        self,
        *,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        test_features: pd.DataFrame,
        valid_features: pd.DataFrame | None = None,
        valid_labels: pd.Series | None = None,
    ) -> None:
        self.segments = {"train": (), "test": ()}
        self._features = {"train": train_features, "test": test_features}
        self._labels = {"train": train_labels}
        if valid_features is not None and valid_labels is not None:
            self.segments["valid"] = ()
            self._features["valid"] = valid_features
            self._labels["valid"] = valid_labels

    def prepare(self, segment: str, *, col_set, data_key: str) -> pd.DataFrame:
        del data_key
        features = self._features[segment]
        if col_set == "feature":
            return features
        if col_set == ["feature", "label"] and segment in self._labels:
            return pd.concat(
                {
                    "feature": features,
                    "label": self._labels[segment].to_frame("LABEL0"),
                },
                axis=1,
            )
        raise ForecastError(f"unsupported Qlib dataset request: {segment}/{col_set}")


def _standardize(
    train: pd.DataFrame, *others: pd.DataFrame
) -> tuple[pd.DataFrame, ...]:
    means = train.mean()
    scales = train.std(ddof=0).replace(0.0, 1.0)
    return tuple((frame - means) / scales for frame in (train, *others))


@contextmanager
def _qlib_tracking_run() -> Iterator[Path]:
    """Give Qlib's model workflow an isolated, disposable experiment store."""
    import qlib
    from mlflow.tracking import MlflowClient
    from qlib.workflow import R
    from qlib.workflow.recorder import MLflowRecorder

    with TemporaryDirectory(prefix="qlib-lightgbm-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        provider = temporary_root / "provider"
        artifacts = temporary_root / "artifacts"
        provider.mkdir()
        artifacts.mkdir()
        tracking_uri = f"sqlite:///{temporary_root / 'tracking.db'}"
        qlib.init(
            provider_uri=provider,
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": tracking_uri,
                    "default_exp_name": "research-forecast",
                },
            },
        )
        MlflowClient(tracking_uri=tracking_uri).create_experiment(
            "research-forecast",
            artifact_location=artifacts.as_uri(),
        )
        original_code_logger = MLflowRecorder._log_uncommitted_code
        MLflowRecorder._log_uncommitted_code = lambda _: None
        try:
            with R.start(experiment_name="research-forecast", recorder_name="lightgbm"):
                yield temporary_root
        finally:
            MLflowRecorder._log_uncommitted_code = original_code_logger


def _qlib_native_lgb_predict(
    *,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    test_features: pd.DataFrame,
    valid_features: pd.DataFrame | None = None,
    valid_labels: pd.Series | None = None,
    num_boost_round: int = LGB_NUM_BOOST_ROUND_CAP,
) -> tuple[pd.Series, int]:
    """Fit and predict through Qlib's native ``LGBModel`` implementation."""
    from qlib.contrib.model.gbdt import LGBModel

    dataset = _FrameDataset(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        valid_features=valid_features,
        valid_labels=valid_labels,
    )
    model = LGBModel(
        num_boost_round=num_boost_round,
        early_stopping_rounds=LGB_EARLY_STOPPING_ROUNDS,
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=20,
        lambda_l2=1.0,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        seed=LGB_RANDOM_SEED,
        feature_fraction_seed=LGB_RANDOM_SEED,
        bagging_seed=LGB_RANDOM_SEED,
        data_random_seed=LGB_RANDOM_SEED,
        num_threads=LGB_NUM_THREADS,
    )
    model.fit(dataset, verbose_eval=0)
    selected_rounds = (
        int(model.model.best_iteration)
        if model.model is not None and model.model.best_iteration > 0
        else num_boost_round
    )
    return model.predict(dataset, segment="test"), selected_rounds


def _rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(
        sqrt(float(np.mean(np.square(actual.to_numpy() - predicted.to_numpy()))))
    )


def _safe_correlation(
    left: pd.Series, right: pd.Series, *, rank: bool = False
) -> float | None:
    if rank:
        left, right = left.rank(), right.rank()
    value = left.corr(right)
    return float(value) if value is not None and np.isfinite(value) else None


def _mean_cross_sectional_rank_ic(frame: pd.DataFrame) -> float | None:
    values = [
        _safe_correlation(group["predicted_return"], group["actual_return"], rank=True)
        for _, group in frame.groupby("decision_session")
        if len(group) >= 3
    ]
    valid = [item for item in values if item is not None]
    return float(np.mean(valid)) if valid else None


def _split(
    labeled: pd.DataFrame,
    *,
    test_sessions: int,
    embargo_sessions: int,
    minimum_train_dates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = sorted(labeled["decision_session"].unique())
    needed = minimum_train_dates + embargo_sessions + test_sessions
    if len(sessions) < needed:
        raise ForecastError(
            "insufficient chronological history for train, embargo, and holdout"
        )
    train_sessions = set(sessions[: -(embargo_sessions + test_sessions)])
    test_sessions_set = set(sessions[-test_sessions:])
    train = labeled[labeled["decision_session"].isin(train_sessions)].copy()
    holdout = labeled[labeled["decision_session"].isin(test_sessions_set)].copy()
    if not holdout.empty:
        holdout_start = pd.to_datetime(holdout["decision_session"]).min()
        train = train[pd.to_datetime(train["target_session"]) < holdout_start].copy()
    if train.empty or holdout.empty:
        raise ForecastError("chronological train or holdout frame is empty")
    if train["decision_session"].nunique() < minimum_train_dates:
        raise ForecastError("insufficient chronological history after label embargo")
    return train, holdout


def _split_inner_validation(
    train: pd.DataFrame,
    *,
    embargo_sessions: int,
    validation_sessions: int,
    minimum_train_dates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = sorted(train["decision_session"].unique())
    needed = minimum_train_dates + embargo_sessions + validation_sessions
    if len(sessions) < needed:
        raise ForecastError(
            "insufficient chronological history for LightGBM inner validation"
        )
    train_sessions = set(sessions[: -(embargo_sessions + validation_sessions)])
    validation_sessions_set = set(sessions[-validation_sessions:])
    inner_train = train[train["decision_session"].isin(train_sessions)].copy()
    validation = train[train["decision_session"].isin(validation_sessions_set)].copy()
    if not validation.empty:
        validation_start = pd.to_datetime(validation["decision_session"]).min()
        inner_train = inner_train[
            pd.to_datetime(inner_train["target_session"]) < validation_start
        ].copy()
    if inner_train.empty or validation.empty:
        raise ForecastError("LightGBM inner train or validation frame is empty")
    if inner_train["decision_session"].nunique() < minimum_train_dates:
        raise ForecastError(
            "insufficient chronological history after LightGBM label embargo"
        )
    return inner_train, validation


def _json_rows(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            column: (
                float(row[column])
                if isinstance(row[column], (float, np.floating))
                else row[column]
            )
            for column in columns
        }
        for _, row in frame.sort_values(["decision_session", "key"]).iterrows()
    ]


def forecast_snapshot(
    snapshot: SourceSnapshot,
    *,
    horizon_sessions: int = DEFAULT_HORIZON_SESSIONS,
    test_sessions: int = DEFAULT_TEST_SESSIONS,
    embargo_sessions: int = DEFAULT_EMBARGO_SESSIONS,
    minimum_train_dates: int = DEFAULT_MINIMUM_TRAIN_DATES,
) -> dict[str, Any]:
    """Train once on past labels, score a later holdout, then forecast the latest rows."""
    if horizon_sessions < 1 or test_sessions < 1 or embargo_sessions < horizon_sessions:
        raise ForecastError(
            "invalid forecast horizon, holdout, or embargo configuration"
        )
    labeled, current = _feature_frames(snapshot, horizon_sessions=horizon_sessions)
    qlib_static_loader = _verify_qlib_static_loader(labeled)
    train, holdout = _split(
        labeled,
        test_sessions=test_sessions,
        embargo_sessions=embargo_sessions,
        minimum_train_dates=minimum_train_dates,
    )
    inner_train, validation = _split_inner_validation(
        train,
        embargo_sessions=embargo_sessions,
        validation_sessions=DEFAULT_INNER_VALIDATION_SESSIONS,
        minimum_train_dates=minimum_train_dates,
    )
    inner_train_features, validation_features = _standardize(
        inner_train[list(FEATURE_NAMES)],
        validation[list(FEATURE_NAMES)],
    )
    train_features, holdout_features = _standardize(
        train[list(FEATURE_NAMES)],
        holdout[list(FEATURE_NAMES)],
    )
    all_features, current_features = _standardize(
        labeled[list(FEATURE_NAMES)], current[list(FEATURE_NAMES)]
    )
    with _qlib_tracking_run():
        _, selected_rounds = _qlib_native_lgb_predict(
            train_features=inner_train_features,
            train_labels=inner_train["actual_return"],
            valid_features=validation_features,
            valid_labels=validation["actual_return"],
            test_features=validation_features,
        )
        holdout_predictions, _ = _qlib_native_lgb_predict(
            train_features=train_features,
            train_labels=train["actual_return"],
            test_features=holdout_features,
            num_boost_round=selected_rounds,
        )
        current_predictions, _ = _qlib_native_lgb_predict(
            train_features=all_features,
            train_labels=labeled["actual_return"],
            test_features=current_features,
            num_boost_round=selected_rounds,
        )
    holdout["predicted_return"] = holdout_predictions.to_numpy()
    baseline_return = float(train["actual_return"].mean())
    baseline = pd.Series(baseline_return, index=holdout.index)
    actual = holdout["actual_return"]
    model_rmse = _rmse(actual, holdout["predicted_return"])
    baseline_rmse = _rmse(actual, baseline)
    metrics = {
        "model_mae": float(np.mean(np.abs(actual - holdout["predicted_return"]))),
        "model_rmse": model_rmse,
        "baseline_rmse": baseline_rmse,
        "model_beats_mean_baseline": model_rmse < baseline_rmse,
        "pearson_ic": _safe_correlation(holdout["predicted_return"], actual),
        "rank_ic": _safe_correlation(holdout["predicted_return"], actual, rank=True),
        "mean_cross_sectional_rank_ic": _mean_cross_sectional_rank_ic(holdout),
    }
    interpretation = (
        "holdout_model_rmse_is_lower_than_mean_baseline"
        if metrics["model_beats_mean_baseline"]
        else "holdout_model_rmse_is_not_lower_than_mean_baseline"
    )
    current["predicted_return"] = current_predictions.to_numpy()
    current_forecasts = _json_rows(
        current,
        (
            "key",
            "decision_session",
            "decision_timestamp",
            "predicted_return",
        ),
    )
    for item in current_forecasts:
        item["horizon_sessions"] = horizon_sessions
    return {
        "forecast_protocol": {
            "model": {
                "family": "lightgbm_regression",
                "backend": "qlib_native",
                "qlib_component": "qlib.contrib.model.gbdt.LGBModel",
                "model_selection": "chronological_inner_validation",
                "num_boost_round_cap": LGB_NUM_BOOST_ROUND_CAP,
                "early_stopping_rounds": LGB_EARLY_STOPPING_ROUNDS,
                "random_seed": LGB_RANDOM_SEED,
                "num_threads": LGB_NUM_THREADS,
                "experiment_tracking": "isolated_temporary_sqlite",
                "holdout_refit": "outer_train_at_selected_rounds",
            },
            "horizon_sessions": horizon_sessions,
            "target": "native_close_return_after_horizon_sessions",
            "currency_conversion": "not_applied",
            "feature_names": list(FEATURE_NAMES),
            "test_sessions": test_sessions,
            "embargo_sessions": embargo_sessions,
            "minimum_train_dates": minimum_train_dates,
            "overlapping_horizon_labels": True,
        },
        "qlib_static_loader": qlib_static_loader,
        "holdout": {
            "train_first_session": str(train["decision_session"].min()),
            "train_last_session": str(train["decision_session"].max()),
            "test_first_session": str(holdout["decision_session"].min()),
            "test_last_session": str(holdout["decision_session"].max()),
            "embargo_sessions": embargo_sessions,
            "row_count": len(holdout),
            "metrics": metrics,
            "interpretation": interpretation,
            "model_selection": {
                "train_first_session": str(inner_train["decision_session"].min()),
                "train_last_session": str(inner_train["decision_session"].max()),
                "validation_first_session": str(validation["decision_session"].min()),
                "validation_last_session": str(validation["decision_session"].max()),
                "embargo_sessions": embargo_sessions,
                "validation_sessions": DEFAULT_INNER_VALIDATION_SESSIONS,
                "selected_boosting_rounds": selected_rounds,
                "holdout_refit": "outer_train_at_selected_rounds",
            },
        },
        "holdout_predictions": _json_rows(
            holdout,
            (
                "key",
                "decision_session",
                "decision_timestamp",
                "target_session",
                "actual_return",
                "predicted_return",
            ),
        ),
        "current_forecasts": current_forecasts,
    }


def run_forecast(
    *, database: Path, as_of: datetime, output: Path, universe: str = "current-holdings"
) -> dict[str, Any]:
    """Publish one immutable, research-only actual forecast run."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    snapshot = load_snapshot(database, as_of=as_of, universe=universe)
    result = forecast_snapshot(snapshot)
    input_fingerprint = sha256(
        canonical_bytes(
            {
                "as_of": as_of.isoformat(),
                "policy_hash": snapshot.policy_record["policy_hash"],
                "research_universe": snapshot.research_universe,
                "forecast_protocol": result["forecast_protocol"],
                "candles": [item.record() for item in snapshot.candles],
            }
        )
    ).hexdigest()
    run_id = f"forecast-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{input_fingerprint[:12]}"
    with staged_run(output / run_id) as run_dir:
        input_manifest = write_inputs(snapshot, run_dir, repository_root=ROOT)
        summary = {
            "run_id": run_id,
            "as_of": as_of.isoformat(),
            "policy_hash": snapshot.policy_record["policy_hash"],
            "research_universe": snapshot.research_universe,
            "prediction_horizon_sessions": result["forecast_protocol"][
                "horizon_sessions"
            ],
            "model": result["forecast_protocol"]["model"],
            "qlib_static_loader": result["qlib_static_loader"],
            "holdout": result["holdout"],
            "current_forecast_count": len(result["current_forecasts"]),
            "source_reproducible": not input_manifest["relevant_source_dirty"],
            "research_only": True,
        }
        manifest = {
            "run_id": run_id,
            "as_of": as_of.isoformat(),
            "policy_hash": snapshot.policy_record["policy_hash"],
            "research_universe": snapshot.research_universe,
            "forecast_protocol": result["forecast_protocol"],
            "environment": environment_info(),
            "source_manifest": input_manifest["source_manifest"],
            "input_manifest": input_manifest["input_manifest"],
        }
        write_json(run_dir / "forecast-protocol.json", result["forecast_protocol"])
        write_json(run_dir / "holdout-predictions.json", result["holdout_predictions"])
        write_json(run_dir / "current-forecasts.json", result["current_forecasts"])
        write_json(run_dir / "manifest.json", manifest)
        write_json(run_dir / "summary.json", summary)
    return summary
