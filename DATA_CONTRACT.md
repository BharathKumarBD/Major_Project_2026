# Twinergy 2.0 — Data Contract & Foundation Specification

## 1. Overview & Dataset Identity

- **Dataset Name:** `twinergy_clean.csv` / `twinergy_clean.parquet`
- **Canonical File Paths:**
  - Raw cleaned CSV: `twinergy_clean.csv` (Size: 6,344,297,066 bytes / ~5.91 GB)
  - Columnar Parquet: `twinergy_clean.parquet` (Size: ~534 MB, ZSTD compressed)
- **Source:** ASHRAE Great Energy Predictor III (GEPIII) Kaggle Competition dataset, merged with building metadata and site weather observations.
- **Total Records:** `18,266,719` rows
- **Total Columns:** `37` columns
- **Temporal Range:** `2016-01-02 00:00:00` to `2016-12-31 23:00:00` (Hourly resolution, full calendar year 2016 minus the first day dropped during lag creation)
- **Unique Buildings:** `1,449`
- **Unique Sites:** `16` (`site_id` 0 through 15)
- **Unique Meter Types:** `4` (`0`: Electricity, `1`: Chilled Water, `2`: Steam, `3`: Hot Water)
- **Duplicate Rows:** `0` duplicate readings (`building_id`, `meter`, `timestamp`, `meter_reading`)
- **Duplicate Timestamps:** `0` duplicate timestamps within any `(building_id, meter)` time series

---

## 2. Schema Specification & Column Dictionary

| Column Name | Physical Type | Logical Type | Null Count | Null % | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `building_id` | `INT64` | Identifier | 0 | 0.00% | Unique building identifier (`0` to `1448`) |
| `meter` | `INT64` | Category | 0 | 0.00% | Meter type code (`0`=Electricity, `1`=Chilled Water, `2`=Steam, `3`=Hot Water) |
| `timestamp` | `TIMESTAMP` | Temporal | 0 | 0.00% | Hourly observation timestamp (UTC) |
| `meter_reading` | `FLOAT64` | Measurement | 0 | 0.00% | Raw energy meter reading (native unit, strictly > 0) |
| `site_id` | `INT64` | Identifier | 0 | 0.00% | Geographic site location index (`0` to `15`) |
| `primary_use` | `VARCHAR` | Category | 0 | 0.00% | ASHRAE primary facility classification (e.g., Education, Office) |
| `square_feet` | `INT64` | Dimension | 0 | 0.00% | Gross floor area in square feet |
| `year_built` | `FLOAT64` | Metadata | 10,992,109 | 60.18% | Year of building construction (sparse in source metadata) |
| `floor_count` | `VARCHAR` | Metadata | 15,034,385 | 82.30% | Number of building floors (sparse in source metadata) |
| `air_temperature` | `FLOAT64` | Weather | 0 | 0.00% | Ambient outdoor dry-bulb temperature (°C) |
| `cloud_coverage` | `FLOAT64` | Weather | 7,928,466 | 43.40% | Cloud coverage in oktas (`0` to `9`) |
| `dew_temperature` | `FLOAT64` | Weather | 0 | 0.00% | Dew point temperature (°C) |
| `precip_depth_1_hr` | `FLOAT64` | Weather | 3,499,655 | 19.16% | Precipitation depth in 1 hour (mm) |
| `sea_level_pressure` | `FLOAT64` | Weather | 771,617 | 4.22% | Barometric sea-level pressure (hPa/millibar) |
| `wind_direction` | `FLOAT64` | Weather | 1,321,079 | 7.23% | Compass wind direction (degrees, `0` to `360`) |
| `wind_speed` | `FLOAT64` | Weather | 0 | 0.00% | Wind speed (m/s) |
| `meter_reading_log` | `FLOAT64` | Target | 0 | 0.00% | Target variable: $\ln(1 + \text{meter\_reading})$ |
| `hour` | `INT64` | Time Feature | 0 | 0.00% | Hour of day (`0` to `23`) |
| `dayofweek` | `INT64` | Time Feature | 0 | 0.00% | Day of week (`0` = Monday, `6` = Sunday) |
| `month` | `INT64` | Time Feature | 0 | 0.00% | Month of year (`1` to `12`) |
| `dayofyear` | `INT64` | Time Feature | 0 | 0.00% | Day of year (`1` to `366`) |
| `is_weekend` | `INT64` | Time Feature | 0 | 0.00% | Binary indicator: `1` if Saturday or Sunday, else `0` |
| `is_business_hours` | `INT64` | Time Feature | 0 | 0.00% | Binary indicator: `1` if 08:00–18:00 on non-weekend, else `0` |
| `hour_sin` | `FLOAT64` | Cyclical | 0 | 0.00% | Sine cyclic encoding: $\sin(2\pi \cdot \text{hour} / 24)$ |
| `hour_cos` | `FLOAT64` | Cyclical | 0 | 0.00% | Cosine cyclic encoding: $\cos(2\pi \cdot \text{hour} / 24)$ |
| `month_sin` | `FLOAT64` | Cyclical | 0 | 0.00% | Sine cyclic encoding: $\sin(2\pi \cdot \text{month} / 12)$ |
| `month_cos` | `FLOAT64` | Cyclical | 0 | 0.00% | Cosine cyclic encoding: $\cos(2\pi \cdot \text{month} / 12)$ |
| `lag_1hr` | `FLOAT64` | Autoregressive | 0 | 0.00% | Previous hour $\text{meter\_reading\_log}$ ($t-1$) |
| `lag_24hr` | `FLOAT64` | Autoregressive | 0 | 0.00% | Same hour previous day $\text{meter\_reading\_log}$ ($t-24$) |
| `lag_168hr` | `FLOAT64` | Autoregressive | 0 | 0.00% | Same hour previous week ($t-168$), imputed with `rolling_mean_24hr` for week 1 |
| `rolling_mean_6hr` | `FLOAT64` | Rolling Stat | 0 | 0.00% | Rolling mean of $\text{meter\_reading\_log}$ over prior 6 hours ($t-6$ to $t-1$) |
| `rolling_mean_24hr` | `FLOAT64` | Rolling Stat | 0 | 0.00% | Rolling mean of $\text{meter\_reading\_log}$ over prior 24 hours ($t-24$ to $t-1$) |
| `rolling_std_24hr` | `FLOAT64` | Rolling Stat | 0 | 0.00% | Rolling standard deviation of $\text{meter\_reading\_log}$ over prior 24 hours |
| `rolling_mean_168hr` | `FLOAT64` | Rolling Stat | 0 | 0.00% | Rolling mean of $\text{meter\_reading\_log}$ over prior 168 hours ($t-168$ to $t-1$) |
| `temp_x_hour` | `FLOAT64` | Interaction | 0 | 0.00% | Interaction: $\text{air\_temperature} \times \text{hour}$ |
| `temp_x_weekend` | `FLOAT64` | Interaction | 0 | 0.00% | Interaction: $\text{air\_temperature} \times \text{is\_weekend}$ |
| `occupancy_proxy` | `FLOAT64` | Heuristic Proxy | 0 | 0.00% | Heuristic proxy: $\text{is\_business\_hours} \times \text{rolling\_mean\_6hr}$ |

---

## 3. Machine Learning Model Contracts

### 3.1 LightGBM Regressor (`twinergy_lgbm.pkl`)
- **Objective:** Regression (`root_mean_squared_error`)
- **Target:** `meter_reading_log` ($\ln(1 + \text{meter\_reading})$)
- **Input Features (22 Ordered Features):**
  1. `hour`
  2. `dayofweek`
  3. `month`
  4. `dayofyear`
  5. `is_weekend`
  6. `is_business_hours`
  7. `hour_sin`
  8. `hour_cos`
  9. `month_sin`
  10. `month_cos`
  11. `lag_1hr`
  12. `lag_24hr`
  13. `lag_168hr`
  14. `rolling_mean_6hr`
  15. `rolling_mean_24hr`
  16. `rolling_std_24hr`
  17. `rolling_mean_168hr`
  18. `temp_x_hour`
  19. `temp_x_weekend`
  20. `occupancy_proxy`
  21. `air_temperature`
  22. `square_feet`
- **Output Inversion:** Predictions are inverted using $\text{actual} = \exp(\hat{y}) - 1$.

### 3.2 Isolation Forest Anomaly Detector (`twinergy_isoforest.pkl`)
- **Objective:** Unsupervised out-of-distribution anomaly detection
- **Target:** None (unsupervised score: `-1` = anomaly, `1` = nominal)
- **Input Features (8 Ordered Features):**
  1. `meter_reading_log`
  2. `rolling_mean_6hr`
  3. `rolling_std_24hr`
  4. `lag_1hr`
  5. `lag_24hr`
  6. `hour`
  7. `is_weekend`
  8. `air_temperature`

---

## 4. Preprocessing & Feature Engineering Semantics

1. **Target Log Transformation:**
   - Generated strictly as `np.log1p(df['meter_reading'])`.
   - Invertible via `np.expm1(...)`.
2. **Missing Weather Imputation:**
   - Four core weather features (`air_temperature`, `dew_temperature`, `wind_speed`, `sea_level_pressure`) were grouped by `site_id` and filled via forward-fill followed by backward-fill (`transform(lambda x: x.ffill().bfill())`).
   - `cloud_coverage`, `precip_depth_1_hr`, and `wind_direction` were intentionally left unfilled as they are not consumed by the trained models.
3. **Outlier & Zero Filtering:**
   - Non-positive readings (`meter_reading <= 0`) were dropped. In the ASHRAE GEPIII dataset, Site 0 meters (Building IDs 0 to 104) had an unmetered sensor dropout logging constant `0.0` from January 1 to May 20, 2016 before meter activation on May 21, 2016. Dropping non-positive readings prevents corrupted lag features and false zero-load baselines.
   - Extreme physical outliers above the 99.9th percentile (`quantile(0.999)`) computed per meter type were removed.
4. **Lag Feature Construction:**
   - Grouped by `['building_id', 'meter']` and shifted chronologically:
     - `lag_1hr = group.shift(1)`
     - `lag_24hr = group.shift(24)`
     - `lag_168hr = group.shift(168)`
   - To preserve all 12 calendar months without dropping the initial 168 rows per series, initial NaN values in `lag_168hr` were imputed using `rolling_mean_24hr`.
5. **Rolling Features:**
   - Computed on `meter_reading_log` grouped by `['building_id', 'meter']`.
   - Crucially, each rolling window applied a 1-step lag before rolling:
     `x.shift(1).rolling(window, min_periods=1).mean()`
   - Thus, rolling windows cover history up to $t-1$ and do not include timestamp $t$.

---

## 5. Critical Domain Definitions & Proxy Disclosures

### 5.1 Occupancy Proxy (`occupancy_proxy`)
> [!IMPORTANT]
> **EXPLICIT PROXY DISCLOSURE:**
> The feature `occupancy_proxy` **DOES NOT** represent ground-truth room occupancy, physical headcounts, turnstile taps, badge swipes, or sensor-measured human occupancy.
>
> It is strictly an engineered mathematical heuristic:
> $$\text{occupancy\_proxy} = \text{is\_business\_hours} \times \text{rolling\_mean\_6hr}$$
>
> During non-business hours or weekends, `occupancy_proxy` is identically `0.0`. During weekday business hours (08:00–18:00), it scales proportionally with the prior 6-hour mean energy consumption.
> Any UI presentation or reporting must present this variable strictly as an estimated proxy, never as actual measured occupancy.

---

## 6. Leakage Audit & Time-Series Evaluation Concerns

1. **Intra-timestamp Target Leakage (Verified Negative):**
   - In `twinergy_clean.csv`, all rolling features (`rolling_mean_6hr`, `rolling_mean_24hr`, etc.) were verified against ground-truth shifted windows: difference $| \text{CSV} - \text{shifted}(1) | = 0.00\text{e}+00$. The current target value $y_t$ is **NOT** included in the rolling features of that row.
   - The unshifted rolling baseline identified in the previous repository audit (`demo_building['baseline'] = demo_building['meter_reading'].rolling(24*7).mean()`) was confined to an exploratory visualization cell (`Cell 25` in `notebooks/phase1.ipynb`) and was **never incorporated into the clean dataset or model training features**.
2. **Autoregressive Multi-step Horizon Consideration:**
   - Because `lag_1hr`, `lag_24hr`, and `rolling_mean_6hr` exist across all test timestamps in `twinergy_clean.csv`, current predictions in `app.py` represent **1-step-ahead nowcasting** (predicting $t$ given actual ground-truth observations up to $t-1$).
   - For recursive multi-step forecasting into an unobserved future (e.g. 7-day or 30-day ahead projections), true future lags must be iteratively rolled from prior model predictions rather than looked up from ground truth.
3. **Weather Backward-fill Imputation:**
   - The use of `bfill()` for early weather records introduces a mild theoretical lookahead across initial observations for sites with missing initial weather readings. However, all weather features represent exogenous observations.

---

## 7. Storage & Query Foundation

- **Modernized Parquet Storage:** `twinergy_clean.parquet` provides an 11x reduction in file size (534 MB vs. 5.91 GB) and sub-second point queries without in-memory buffering.
- **DuckDB Analytical Pushdown:** Querying by `building_id`, `meter`, and temporal window bypasses RAM exhaustion by executing pushdown scans directly on the Parquet representation.
