"""Generate sampling locations and extract satellite/static predictors.

Execution requires run.py, which supplies a shared experiment namespace.
"""

# Generate reproducible 100m-grid point locations within each site.
points = ee.FeatureCollection([])
for idx,(site,region) in enumerate(REGIONS.items()):
    sample = (ee.Image.constant(1).rename('seed')
              .sample(region=region, projection=PROJ, numPixels=POINTS_PER_SITE,
                      seed=SEED+idx, geometries=True, tileScale=4))
    def tag_point(feat, site_name=site):
        xy = feat.geometry().coordinates()
        return (ee.Feature(feat.geometry())
                .set('site',site_name)
                .set('pixel_id',ee.String(site_name).cat('_').cat(ee.String(feat.get('system:index'))))
                .set('lon',xy.get(0)).set('lat',xy.get(1)))
    points = points.merge(sample.map(tag_point))
print('Sample point candidates:',points.size().getInfo())

s2 = S2COL.median()
ls = LS2024.median()
dw = DWC.select(DW_RAW).mean().rename(DW_NAMES)
aef = AEF.mosaic().select(EMBED_BANDS)
# Projected 2025 GHSL cell count; allow valid zero-population locations.
pop = ee.Image('JRC/GHSL/P2023A/GHS_POP/2025').select('population_count').rename('pop_2025_proj').unmask(0)
static_image = ee.Image.cat([
    s2.select(S2_NAMES+['s2_ndvi','s2_ndbi','s2_mndwi']),
    ls.select(LS_NAMES+['ls_ndvi','ls_ndbi','ls_mndwi']),
    dw, aef, pop
])
STATIC_BANDS = (S2_NAMES+['s2_ndvi','s2_ndbi','s2_mndwi']+
                LS_NAMES+['ls_ndvi','ls_ndbi','ls_mndwi']+DW_NAMES+
                EMBED_BANDS+['pop_2025_proj'])
STATIC_PROPS = ['pixel_id','site','lon','lat']


def download_ee_csv(fc, path, selectors):
    """Download actual GEE table, with retry; no fake/synthetic fallback."""
    path = Path(path)
    if path.exists() and not REFRESH_DATA:
        print('Cached:',path.name)
        return pd.read_csv(path)
    last_error = None
    for attempt in range(3):
        try:
            url = fc.getDownloadURL(filetype='csv',selectors=selectors,filename=path.stem)
            response = requests.get(url,timeout=180)
            response.raise_for_status()
            data = pd.read_csv(io.BytesIO(response.content))
            if data.empty:
                raise RuntimeError('Earth Engine returned an empty CSV')
            path.parent.mkdir(parents=True,exist_ok=True)
            data.to_csv(path,index=False)
            print(f'Downloaded {path.name}: {len(data):,} rows')
            return data
        except Exception as e:
            last_error=e
            print(f'Attempt {attempt+1}/3 failed: {str(e)[:250]}')
            time.sleep(3*(attempt+1))
    raise RuntimeError(f'Download of real Earth Engine data failed for {path.name}: {last_error}')

static_fc = static_image.sampleRegions(
    collection=points,properties=STATIC_PROPS,
    projection=PROJ,tileScale=8,geometries=False
)
static_df = download_ee_csv(static_fc,OUT/'static_features_2024.csv', STATIC_PROPS+STATIC_BANDS)
static_df['pixel_id'] = static_df['pixel_id'].astype(str)
static_df = static_df.dropna(subset=EMBED_BANDS+S2_NAMES+['ls_ndvi','dw_built']).drop_duplicates('pixel_id')
print('Static observations by site:')
print(static_df.groupby('site').size().to_string())
if static_df.site.nunique()<3 or len(static_df)<150:
    raise RuntimeError('Too few valid static points: inspect masked areas or enlarge the sampled windows.')
