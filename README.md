# Context-Aware-Temprature-Analysis
A framework based on geospatial foundation model for contextual urban thermal anomaly assessment using satellite imagery and AlphaEarth embeddings.
Setup

Python 3.10+ recommended. Create a Python virtual environment and install the dependencies:
```bash

pip install -r requirements.txt
```
Enable Google Earth Engine for your Google Cloud project and authenticate once:
```bash
earthengine authenticate
```
Run all four experiments:
```bash
python run.py --project YOUR_EARTH_ENGINE_PROJECT_ID --output-dir outputs --skip-maps
```
To additionally download true satellite rasters and create full satellite-backed maps:
```bash
python run.py --project YOUR_EARTH_ENGINE_PROJECT_ID --output-dir outputs
```

File structure
```text
Context-aware/
├── run.py                         # CLI orchestrator
├── requirements.txt
├── stages/
│   ├── 01_setup.py                # EE auth, settings, four Dubai study windows
│   ├── 02_earth_observation.py    # Landsat/Sentinel/Dynamic World/AlphaEarth preprocessing
│   ├── 03_sample_features.py     # Spatial sample locations, embeddings and features
│   ├── 04_monthly_lst.py         # Monthly 2024–2025 Landsat LST sampling
│   ├── 05_data_quality_plots.py  # Observed-temperature plots
│   ├── 06_e1_lst_prediction.py   # E1: RF prediction and spatial/year validation
│   ├── 07_e2_context_reference.py# E2: GeoFM contexts, reference temperatures and anomalies
│   ├── 08_e3_geoheat.py          # E3: GeoHeat scoring, ablations and weight sensitivity
│   ├── 09_e4_temporal_stability.py# E4: matched 2024–2025 stability

├── scripts/
│   ├── check_summary_results.py
│   └── plot_reference_mae.py
└
```
Research data and assumptions:-
Four sampled Dubai neighborhoods: Downtown, Al Quoz, Dubai Marina and Mirdif.
Eight summer months across 2024 and 2025 (June–September each year).
Monthly clear-sky Landsat 8/9 Collection 2 L2ST LST, sampled at 100 m; no claim of 10 m thermal ground truth.
2024 Sentinel-2 features, Dynamic World probabilities and AlphaEarth annual 64D embeddings.
E1 RF baseline, AlphaEarth-only and fused features; site holdout 2024 and same-site 2025 transfer.
E2 Dynamic World broad strata, up to 3 clusters/stratum on 2024 embeddings; cross-site median reference with context/landcover/city fallback (min 15 reference records).
E3 GeoHeat weights: absolute LST 0.4, anomaly 0.2, persistence 0.2, projected GHSL residential population 0.2.
E4 rank correlation and top-decile agreement across 2024 and 2025.

Attribution and data availability
AlphaEarth Foundations Satellite Embedding is a Google/Google DeepMind-produced product. Satellite observations are retrieved using Google Earth Engine and retain their respective data provider licenses and attribution requirements. 

