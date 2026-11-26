"""Initialize Earth Engine, output paths, sampling parameters, and study windows.

Automatically refactored from the original HeatContext experiment notebook.
Execution requires run.py, which supplies a shared experiment namespace.
"""

from pathlib import Path
from datetime import date
import json, io, os, time, zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ee
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

# Avoid interactive windows when invoked as a command-line pipeline.
import matplotlib
matplotlib.use('Agg')

# Set your real Google Cloud *project ID* here, or use Colab Secrets -> EE_PROJECT.
# If you assigned EE_PROJECT in a previous cell, this cell will preserve it.
EE_PROJECT = os.environ.get('EE_PROJECT') or globals().get('EE_PROJECT', '')
if not EE_PROJECT or EE_PROJECT == 'YOUR_GCP_PROJECT_ID':
    try:
        from google.colab import userdata
        EE_PROJECT = userdata.get('EE_PROJECT') or ''
    except Exception:
        EE_PROJECT = ''
if not EE_PROJECT:
    raise ValueError('Set EE_PROJECT in the environment or pass --project to run.py')
if not EE_PROJECT or EE_PROJECT == 'YOUR_GCP_PROJECT_ID':
    raise ValueError('A real Google Cloud Project ID is required (not the project display name).')
try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)
print('Connected Earth Engine project:', EE_PROJECT)

SEED = 42
np.random.seed(SEED)
OUT = Path(os.environ.get('HEATCONTEXT_OUTPUT_DIR', './outputs')).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
REFRESH_DATA = os.environ.get('HEATCONTEXT_REFRESH_DATA', '0') == '1'
POINTS_PER_SITE = 300       # 300*4 = 1,200 candidate 100m observations per month
SCALE_M = 100
MONTHS = [6, 7, 8, 9]       # June–September
YEARS = [2024, 2025]        # 2025 = temporally held out
TRAIN_YEAR = 2024
TEST_YEAR = 2025
N_CONTEXTS_PER_STRATUM = 3
MIN_CONTEXT_REF = 15
MIN_VALID_MONTHS = 2
PROJ = ee.Projection('EPSG:32640').atScale(SCALE_M)

# Four *illustrative* sites, not the full geographic area of Dubai.
# (lon, lat), ~4 km wide rectangles; adjust before publication if required.
SITES = {
    'Downtown': (55.2810, 25.1950),
    'Al_Quoz': (55.2360, 25.1440),
    'Dubai_Marina': (55.1430, 25.0840),
    'Mirdif': (55.4170, 25.2200),
}
HALF_WIDTH_DEG = 0.018
HALF_HEIGHT_DEG = 0.017
REGIONS = {
    name: ee.Geometry.Rectangle([lon-HALF_WIDTH_DEG, lat-HALF_HEIGHT_DEG,
                                 lon+HALF_WIDTH_DEG, lat+HALF_HEIGHT_DEG],
                                geodesic=False)
    for name, (lon,lat) in SITES.items()
}
ROI = ee.FeatureCollection([
    ee.Feature(g, {'site': name}) for name,g in REGIONS.items()
]).geometry()
print('Earth Engine initialized. Output:', OUT)
print('Sites:', list(SITES), '| Years:', YEARS, '| Months:', MONTHS)
