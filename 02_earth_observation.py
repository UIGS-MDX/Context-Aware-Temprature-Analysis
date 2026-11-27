"""Define Earth Engine sources and cloud-filtered satellite preprocessors.

Execution requires run.py, which supplies a shared experiment namespace.
"""

EMBED_BANDS = [f'A{i:02d}' for i in range(64)]
S2_RAW = ['B2','B3','B4','B8','B11','B12']
S2_NAMES = ['s2_blue','s2_green','s2_red','s2_nir','s2_swir1','s2_swir2']
L8_RAW = ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']
LS_NAMES = ['ls_blue','ls_green','ls_red','ls_nir','ls_swir1','ls_swir2']
DW_RAW = ['built','bare','trees','grass','shrub_and_scrub','water']
DW_NAMES = ['dw_built','dw_bare','dw_trees','dw_grass','dw_shrub','dw_water']


def prepare_landsat(img):
    qa = img.select('QA_PIXEL')
    clear = qa.bitwiseAnd(63).eq(0)   # bits 0–5: fill/dilated/cirrus/cloud/shadow/snow
    clear = clear.And(img.select('QA_RADSAT').eq(0))
    optical = img.select(L8_RAW, LS_NAMES).multiply(0.0000275).add(-0.2)
    ndvi = optical.normalizedDifference(['ls_nir','ls_red']).rename('ls_ndvi')
    ndbi = optical.normalizedDifference(['ls_swir1','ls_nir']).rename('ls_ndbi')
    mndwi = optical.normalizedDifference(['ls_green','ls_swir1']).rename('ls_mndwi')
    lst = img.select('ST_B10').multiply(0.00341802).add(149).subtract(273.15).rename('lst_c')
    valid_lst = lst.gt(10).And(lst.lt(80))
    return (ee.Image.cat([optical, ndvi, ndbi, mndwi, lst])
            .updateMask(clear).updateMask(valid_lst)
            .copyProperties(img, ['system:time_start','LANDSAT_PRODUCT_ID']))


def prepare_sentinel(img):
    scl = img.select('SCL')
    # Keep vegetation, non-vegetation, dark, and water SCL classes.
    good = (scl.neq(0).And(scl.neq(1)).And(scl.neq(3))
            .And(scl.neq(7)).And(scl.neq(8)).And(scl.neq(9))
            .And(scl.neq(10)).And(scl.neq(11)))
    sr = img.select(S2_RAW,S2_NAMES).multiply(0.0001).updateMask(good)
    ndvi = sr.normalizedDifference(['s2_nir','s2_red']).rename('s2_ndvi')
    ndbi = sr.normalizedDifference(['s2_swir1','s2_nir']).rename('s2_ndbi')
    mndwi = sr.normalizedDifference(['s2_green','s2_swir1']).rename('s2_mndwi')
    return ee.Image.cat([sr,ndvi,ndbi,mndwi]).copyProperties(img,['system:time_start'])


def landsat_collection(start,end):
    def one(asset):
        return (ee.ImageCollection(asset)
                .filterBounds(ROI)
                .filterDate(start,end)
                .filter(ee.Filter.eq('PROCESSING_LEVEL','L2SP'))
                .filter(ee.Filter.lt('CLOUD_COVER',90)))
    return one('LANDSAT/LC08/C02/T1_L2').merge(one('LANDSAT/LC09/C02/T1_L2')).map(prepare_landsat)

summer_start, summer_end = '2024-05-01','2024-10-01'
S2COL = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
         .filterBounds(ROI).filterDate(summer_start,summer_end)
         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',80))
         .map(prepare_sentinel))
DWC = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
       .filterBounds(ROI).filterDate(summer_start,summer_end))
AEF = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
       .filterDate('2024-01-01','2025-01-01')
       .filterBounds(ROI))
LS2024 = landsat_collection(summer_start, summer_end)

counts = {
    'landsat_2024_May_Sep': LS2024.size().getInfo(),
    'sentinel2_2024_May_Sep': S2COL.size().getInfo(),
    'dynamic_world_2024_May_Sep': DWC.size().getInfo(),
    'alphaearth_2024_tiles': AEF.size().getInfo(),
}
print('Available source images:',counts)
if any(v==0 for v in counts.values()):
    raise RuntimeError('At least one essential Earth Engine collection is empty. Review date, ROI, or dataset access.')
for yr in YEARS:
    for mo in MONTHS:
        begin, end = f'{yr}-{mo:02d}-01', (date(yr+1,1,1).isoformat() if mo==12 else f'{yr}-{mo+1:02d}-01')
        count = landsat_collection(begin,end).size().getInfo()
        print(f'{yr}-{mo:02d}: {count} Landsat 8/9 L2SP scenes before per-pixel QA masking')
        if count==0:
            raise RuntimeError(f'No Landsat scenes for {yr}-{mo:02d}; change MONTHS if needed.')
