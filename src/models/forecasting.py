"""Forecasting service for Twinergy 2.0.

Wraps the trained LightGBM model artifact (twinergy_lgbm.pkl), preserving the exact
22-feature input contract and providing predictions in both log and physical (kWh) units.

NOTE: This service implements the existing 1-step-ahead nowcast interface using the
pre-engineered lag/rolling feature contract. True recursive multi-step forecasting
is deliberately deferred to subsequent steps.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.data.loader import DataLoader, get_default_project_root
from src.data.schema import LIGHTGBM_FEATURES

# Optional type hinting for lightgbm
try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def resolve_lgbm_model_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve LightGBM model file path with precedence."""
    if custom_path is not None:
        p = Path(custom_path)
        if not p.is_absolute():
            p = (get_default_project_root() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Configured LightGBM model path does not exist: {p}")
        return p

    env_path = os.getenv("TWINERGY_LGBM_PATH")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = (get_default_project_root() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"TWINERGY_LGBM_PATH does not exist: {p}")
        return p

    root = get_default_project_root()
    candidates = [
        root / "twinergy_lgbm.pkl",
        root / "models" / "twinergy_lgbm.pkl",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"No LightGBM model artifact found under {root}. "
        "Please provide a path or set TWINERGY_LGBM_PATH."
    )


class EnergyForecaster:
    """Reusable service for LightGBM energy forecasting."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize and load the trained LightGBM booster.

        Args:
            model_path: Optional path to twinergy_lgbm.pkl.
        """
        self.model_path = resolve_lgbm_model_path(model_path)
        self.model = self._load_model(self.model_path)
        self.feature_names: List[str] = list(LIGHTGBM_FEATURES)
        self._verify_feature_alignment()

    def _load_model(self, path: Path) -> Any:
        """Load pickled LightGBM booster."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    def _verify_feature_alignment(self) -> None:
        """Verify model feature names align with the canonical 22-feature schema."""
        model_features = getattr(self.model, "feature_name", None)
        if callable(model_features):
            names = model_features()
            if names != self.feature_names:
                raise ValueError(
                    f"Model feature mismatch!\nExpected: {self.feature_names}\nFound: {names}"
                )

    def prepare_features(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Validate and align feature matrix to the exact 22-feature contract.

        Args:
            X: Input DataFrame or array containing the 22 features.

        Returns:
            pd.DataFrame with exactly LIGHTGBM_FEATURES in proper order.
        """
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Input data is missing required LightGBM features: {missing}")
            # Ensure correct column ordering
            df = X[self.feature_names].copy()
            # Preserve baseline behavior: fill missing square_feet if any
            if df["square_feet"].isnull().any():
                median_sqft = df["square_feet"].median()
                df["square_feet"] = df["square_feet"].fillna(median_sqft if not np.isnan(median_sqft) else 10000.0)
            return df
        elif isinstance(X, np.ndarray):
            if X.ndim != 2 or X.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Expected 2D array with {len(self.feature_names)} features, got shape {X.shape}"
                )
            return pd.DataFrame(X, columns=self.feature_names)
        else:
            raise TypeError(f"Unsupported feature type: {type(X)}. Expected pd.DataFrame or np.ndarray.")

    def predict_log(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict energy consumption on log scale (meter_reading_log).

        Args:
            X: Feature matrix containing the 22 LightGBM features.

        Returns:
            1D array of predictions on ln(1 + kWh) scale.
        """
        features_df = self.prepare_features(X)
        preds = self.model.predict(features_df)
        return np.asarray(preds, dtype=np.float64)

    def predict_kwh(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict energy consumption in physical units (kWh).

        Inverts the log1p transform via expm1(preds_log), clamping negative values to 0.

        Args:
            X: Feature matrix containing the 22 LightGBM features.

        Returns:
            1D array of predicted kWh values.
        """
        preds_log = self.predict_log(X)
        preds_kwh = np.expm1(preds_log)
        return np.clip(preds_kwh, 0.0, None)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_kwh: bool = True,
    ) -> np.ndarray:
        """Predict energy consumption.

        Args:
            X: Feature matrix containing the 22 LightGBM features.
            return_kwh: If True, returns physical kWh values. If False, returns log1p values.

        Returns:
            1D array of predictions.
        """
        return self.predict_kwh(X) if return_kwh else self.predict_log(X)

    def forecast_24h(
        self,
        building_id: int,
        meter: int = 0,
        cutoff_time: Optional[Union[str, pd.Timestamp]] = None,
        history_df: Optional[pd.DataFrame] = None,
        future_weather: Optional[Union[pd.DataFrame, pd.Series, Sequence[float]]] = None,
        loader: Optional[DataLoader] = None,
    ) -> pd.DataFrame:
        """Generate a true recursive 24-hour multi-step energy forecast.

        Recursively forecasts t+1 through t+24:
        1. Predicts t+1 on log scale (meter_reading_log) using historical lags and rolling stats.
        2. Feeds the t+1 prediction back into the autoregressive buffer for t+2.
        3. Dynamically updates all 7 lag and rolling metrics:
           - lag_1hr, lag_24hr, lag_168hr
           - rolling_mean_6hr, rolling_mean_24hr, rolling_std_24hr, rolling_mean_168hr
           using only information available at each step, strictly preventing target leakage.
        4. Reconstructs calendar features (hour, dayofweek, month, dayofyear, is_weekend,
           is_business_hours, hour_sin, hour_cos, month_sin, month_cos) from future timestamps.
        5. Computes interaction features (temp_x_hour, temp_x_weekend, occupancy_proxy).
        6. Converts predicted log1p values to physical consumption (kWh) via expm1(pred_log).

        Args:
            building_id: ASHRAE building identifier.
            meter: Meter type identifier (default 0 for Electricity).
            cutoff_time: Historical cutoff timestamp (inclusive). The 24-hour forecast
                         begins strictly after cutoff_time (cutoff_time + 1 hour).
                         If None and history_df is None, defaults to 24 hours prior to
                         the latest available timestamp in the dataset for this building/meter.
            history_df: Optional historical DataFrame containing at least 168 consecutive
                        hourly records up to cutoff_time. Must contain 'timestamp' and
                        either 'meter_reading_log' or 'meter_reading'.
            future_weather: Optional sequence or DataFrame containing at least 24 hourly
                            'air_temperature' values for the forecast horizon.
                            If None, retrieves observed weather from the dataset.
            loader: Optional DataLoader instance for querying history and weather.

        Returns:
            pd.DataFrame with exactly 24 rows containing:
                - timestamp: Forecast timestamp (consecutive hourly, starting after cutoff)
                - step: Horizon step index (1 to 24)
                - predicted_kwh: Forecasted consumption in physical units (kWh)
                - predicted_log: Forecasted consumption on log1p scale
                - building_id: Target building ID
                - meter: Target meter ID
                - weather_source: 'dataset_observed' or 'provided'
                - All 22 canonical LightGBM features computed at each step

        Raises:
            ValueError: If historical records are fewer than 168 hours, discontinuous,
                        or if future weather is unavailable in the dataset without an explicit input.
        """
        if loader is None:
            loader = DataLoader()

        # 1. Resolve cutoff timestamp
        if history_df is not None:
            if "timestamp" not in history_df.columns:
                raise ValueError("history_df must contain a 'timestamp' column.")
            cutoff = pd.to_datetime(history_df["timestamp"].max())
            if cutoff_time is not None:
                cutoff = pd.to_datetime(cutoff_time)
        elif cutoff_time is not None:
            cutoff = pd.to_datetime(cutoff_time)
        else:
            max_ts = loader.get_latest_timestamp(building_id=building_id, meter=meter)
            cutoff = max_ts - pd.Timedelta(hours=24)

        # 2. Retrieve and validate historical window (at least 168 hours)
        if history_df is None:
            hist_start = cutoff - pd.Timedelta(hours=167)
            cols_needed = ["timestamp", "meter_reading_log", "square_feet", "air_temperature"]
            hist_slice = loader.load(
                building_id=building_id,
                meter=meter,
                start_time=hist_start,
                end_time=cutoff,
                columns=cols_needed,
            )
        else:
            hist_slice = history_df[pd.to_datetime(history_df["timestamp"]) <= cutoff].copy()
            if "meter_reading_log" not in hist_slice.columns:
                if "meter_reading" in hist_slice.columns:
                    hist_slice["meter_reading_log"] = np.log1p(hist_slice["meter_reading"])
                else:
                    raise ValueError(
                        "history_df must contain either 'meter_reading_log' or 'meter_reading'."
                    )

        if len(hist_slice) < 168:
            raise ValueError(
                f"Insufficient historical information for building {building_id}, meter {meter} up to {cutoff}. "
                f"Required at least 168 consecutive hourly records to construct lag_168hr and rolling_mean_168hr, "
                f"but found only {len(hist_slice)} rows."
            )

        # Validate strictly consecutive hourly history
        hist_168 = hist_slice.iloc[-168:].copy()
        expected_range = pd.date_range(end=cutoff, periods=168, freq="h")
        actual_timestamps = pd.to_datetime(hist_168["timestamp"]).values

        if not (actual_timestamps == expected_range.values).all():
            raise ValueError(
                f"Discontinuous historical timestamps detected for building {building_id}, meter {meter} up to {cutoff}. "
                f"Expected 168 strictly consecutive hourly observations from {expected_range[0]} to {expected_range[-1]}."
            )

        hist_logs = hist_168["meter_reading_log"].values.astype(np.float64)
        if np.isnan(hist_logs).any():
            raise ValueError(
                f"Historical meter_reading_log contains NaN values up to cutoff {cutoff}."
            )

        # 3. Retrieve building square_feet
        if "square_feet" in hist_168.columns and not hist_168["square_feet"].isnull().all():
            sqft = float(hist_168["square_feet"].dropna().iloc[-1])
        else:
            b_meta = loader.get_building_metadata(building_id)
            sqft = float(b_meta["square_feet"])

        # 4. Resolve future forecast timestamps (consecutive hourly strictly after cutoff)
        forecast_timestamps = pd.date_range(
            start=cutoff + pd.Timedelta(hours=1),
            periods=24,
            freq="h",
        )

        # 5. Resolve future weather baseline
        if future_weather is not None:
            if isinstance(future_weather, pd.DataFrame):
                if "air_temperature" not in future_weather.columns:
                    raise ValueError("future_weather DataFrame must contain 'air_temperature' column.")
                weather_vals = future_weather["air_temperature"].values[:24].astype(np.float64)
            elif isinstance(future_weather, pd.Series):
                weather_vals = future_weather.values[:24].astype(np.float64)
            elif isinstance(future_weather, (list, tuple, np.ndarray)):
                weather_vals = np.asarray(future_weather, dtype=np.float64)[:24]
            else:
                raise TypeError(f"Unsupported future_weather type: {type(future_weather)}")

            if len(weather_vals) < 24:
                raise ValueError(
                    f"future_weather must provide at least 24 hourly observations, found {len(weather_vals)}."
                )
            if np.isnan(weather_vals).any():
                raise ValueError("future_weather contains NaN values.")
            weather_source = "provided"
        else:
            t_start = forecast_timestamps[0]
            t_end = forecast_timestamps[-1]
            weather_df = loader.load(
                building_id=building_id,
                meter=meter,
                start_time=t_start,
                end_time=t_end,
                columns=["timestamp", "air_temperature"],
            )
            if len(weather_df) < 24:
                raise ValueError(
                    f"Dataset does not contain complete weather observations for future horizon "
                    f"[{t_start} to {t_end}]. Found {len(weather_df)} records. In accordance with Step 3A "
                    f"requirements, no weather forecasting model is invented. Please provide 'future_weather' "
                    f"or choose a cutoff_time where future weather exists in the dataset."
                )
            if weather_df["air_temperature"].isnull().any():
                raise ValueError(
                    f"Dataset contains null air_temperature values in future horizon [{t_start} to {t_end}]."
                )
            weather_vals = weather_df["air_temperature"].values.astype(np.float64)
            weather_source = "dataset_observed"

        # 6. Execute recursive 24-step forecasting
        log_buffer: List[float] = [float(v) for v in hist_logs]
        forecast_rows: List[Dict[str, Any]] = []

        for h in range(1, 25):
            t_h = forecast_timestamps[h - 1]
            hour = int(t_h.hour)
            dayofweek = int(t_h.dayofweek)
            month = int(t_h.month)
            dayofyear = int(t_h.dayofyear)
            is_weekend = int(dayofweek >= 5)
            is_business_hours = int(is_weekend == 0 and 8 <= hour <= 18)
            hour_sin = float(np.sin(2.0 * np.pi * hour / 24.0))
            hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))
            month_sin = float(np.sin(2.0 * np.pi * month / 12.0))
            month_cos = float(np.cos(2.0 * np.pi * month / 12.0))

            # Autoregressive lags from active rolling buffer
            lag_1hr = float(log_buffer[-1])
            lag_24hr = float(log_buffer[-24])
            lag_168hr = float(log_buffer[-168])

            # Rolling statistics strictly from buffer (prior hours, excluding t_h)
            rolling_mean_6hr = float(np.mean(log_buffer[-6:]))
            rolling_mean_24hr = float(np.mean(log_buffer[-24:]))
            rolling_std_24hr = float(np.std(log_buffer[-24:], ddof=1))
            rolling_mean_168hr = float(np.mean(log_buffer[-168:]))

            air_temp = float(weather_vals[h - 1])
            temp_x_hour = float(air_temp * hour)
            temp_x_weekend = float(air_temp * is_weekend)
            occupancy_proxy = float(is_business_hours * rolling_mean_6hr)

            feature_dict = {
                "hour": hour,
                "dayofweek": dayofweek,
                "month": month,
                "dayofyear": dayofyear,
                "is_weekend": is_weekend,
                "is_business_hours": is_business_hours,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "lag_1hr": lag_1hr,
                "lag_24hr": lag_24hr,
                "lag_168hr": lag_168hr,
                "rolling_mean_6hr": rolling_mean_6hr,
                "rolling_mean_24hr": rolling_mean_24hr,
                "rolling_std_24hr": rolling_std_24hr,
                "rolling_mean_168hr": rolling_mean_168hr,
                "temp_x_hour": temp_x_hour,
                "temp_x_weekend": temp_x_weekend,
                "occupancy_proxy": occupancy_proxy,
                "air_temperature": air_temp,
                "square_feet": sqft,
            }

            # Prepare aligned feature frame and predict single step
            step_features = pd.DataFrame([feature_dict])[self.feature_names]
            pred_log = float(self.predict_log(step_features)[0])
            pred_kwh = float(np.clip(np.expm1(pred_log), 0.0, None))

            # Recursively update buffer with predicted log value
            log_buffer.append(pred_log)

            forecast_rows.append({
                "timestamp": t_h,
                "step": h,
                "predicted_kwh": pred_kwh,
                "predicted_log": pred_log,
                "building_id": building_id,
                "meter": meter,
                "weather_source": weather_source,
                **feature_dict,
            })

        return pd.DataFrame(forecast_rows)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Extract global tree-split feature importance from the LightGBM booster.

        WARNING: This is tree split gain/split frequency, NOT SHAP values.
        For local additive SHAP attribution, use ModelExplainer in explainability.py.

        Args:
            importance_type: 'gain' (total information gain) or 'split' (split count).

        Returns:
            DataFrame sorted descending by importance.
        """
        raw_imp = self.model.feature_importance(importance_type=importance_type)
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": raw_imp,
            "importance_type": importance_type,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp

    def get_model_metadata(self) -> Dict[str, Any]:
        """Return operational metadata about the serialized model."""
        num_trees = getattr(self.model, "num_trees", None)
        return {
            "model_path": str(self.model_path),
            "model_type": type(self.model).__name__,
            "num_trees": num_trees() if callable(num_trees) else None,
            "num_features": len(self.feature_names),
            "features": list(self.feature_names),
            "target_scale": "log1p(meter_reading)",
        }


def load_forecaster(model_path: Optional[Union[str, Path]] = None) -> EnergyForecaster:
    """Convenience factory function to instantiate EnergyForecaster."""
    return EnergyForecaster(model_path=model_path)
