# Water Usage Data Cleaning & Clustering with CAMEO Compression

This project builds a pipeline that cleans raw water-meter time series data, compresses each meter's daily usage pattern using a custom **CAMEO** (Critical-point Autocorrelation-preserving Method for Efficient cOmpression) algorithm, and clusters meters into usage-behavior groups with DTW-based K-Means.

It's designed to run as a Google Colab notebook (`watermeterdataschema_CAMEO.ipynb`) against a Parquet dataset stored in Google Drive.

## What it does

1. **Load data** — Reads a Parquet file of meter readings (`meter_uuid`, `timestamp_UTC`, `water_usage_m3`) from Google Drive.
2. **Clean data** — Replaces negative, null, and NaN water-usage values with `0` and reports how many rows were affected.
3. **Split into monthly files** — Scans the dataset for all distinct `year_month` values and writes a cleaned Parquet file per month to `cleaned_monthly_files_with_compression/`.
4. **Compress with CAMEO** — For a chosen month, aggregates readings to daily sums per meter, pivots to a wide (meter × day) matrix, and compresses each meter's daily series down to a fixed number of points (`TARGET_LENGTH`, default 10) while preserving the most temporally significant points.
5. **Cluster** — Runs `TimeSeriesKMeans` (DTW metric) on the compressed series to group meters into `OPTIMAL_K` (default 4) usage-pattern clusters, saves cluster assignments to CSV, and plots the results.

## How CAMEO compression works

For each meter's daily usage series:

1. **Autocorrelation (ACF) analysis** identifies lags with significant temporal dependency (`ACF_THRESHOLD`, default 0.3).
2. **Critical points** are selected: the series' start/end, points at significant ACF lag intervals, and local extrema (peaks/troughs) ranked by deviation from the mean.
3. The critical points are **linearly interpolated/resampled** to a fixed length (`TARGET_LENGTH`), producing a compact representation that preserves the shape and periodic structure of the original series far better than naive downsampling.

This lets clustering run efficiently on short, comparable-length vectors while retaining the meaningful structure (spikes, troughs, weekly/periodic patterns) of the original daily readings.

## Requirements

- Google Colab (uses `google.colab.drive` to mount Drive) — or a local environment with the mount step adapted to read from disk
- Python 3.9+
- `polars`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tslearn`
- `statsmodels`
- `scipy`

Install locally with:

```bash
pip install polars numpy matplotlib scikit-learn tslearn statsmodels scipy
```

## Configuration

Key parameters are set near the top of the notebook:

| Parameter | Default | Description |
|---|---|---|
| `TARGET_LENGTH` | 10 | Number of points each daily series is compressed to (8–15 recommended) |
| `ACF_THRESHOLD` | 0.3 | Minimum absolute ACF value to treat a lag as significant |
| `OPTIMAL_K` | 4 | Number of clusters for K-Means |

## Outputs

All outputs are written to `cleaned_monthly_files_with_compression/`:

- `cleaned_water_usage_<YYYY-MM>.parquet` — cleaned data, one file per month
- `compression_sample_<YYYY-MM>.png` — plot comparing one meter's original vs. CAMEO-compressed series
- `clusters_<YYYY-MM>.png` — plot of each cluster's member series and centroid
- `meter_clusters_<YYYY-MM>_compressed.csv` — final `meter_uuid` → `cluster` assignments


## Notes / possible extensions

- The notebook imports `train_test_split`, `KNeighborsClassifier`, and `accuracy_score` from scikit-learn but doesn't yet use them these look intended for a future step (e.g., training a classifier on cluster labels or validating compression quality against a held-out set).
- `metermetadata` is currently loaded from the same file as `df`; if separate meter metadata exists, point it at the correct file.
- To run outside Colab, replace the `drive.mount(...)` call and `file_path` with a local or cloud-storage path.