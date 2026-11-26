"""Extract monthly clear-sky Landsat 2024–2025 surface temperatures.

Automatically refactored from the original HeatContext experiment notebook.
Execution requires run.py, which supplies a shared experiment namespace.
"""

THERMAL_PROPS = ['pixel_id','site','lon','lat','lst_c','valid_scenes']
monthly_frames=[]
for yr in YEARS:
    for mo in MONTHS:
        start=f'{yr}-{mo:02d}-01'
        end=f'{yr}-{mo+1:02d}-01' if mo<12 else f'{yr+1}-01-01'
        col=landsat_collection(start,end)
        monthly_img=ee.Image.cat([
            col.select('lst_c').median().rename('lst_c'),
            col.select('lst_c').count().rename('valid_scenes')
        ])
        observed=monthly_img.sampleRegions(
            collection=points,properties=STATIC_PROPS,
            projection=PROJ,tileScale=8,geometries=False
        )
        obs_df=download_ee_csv(observed,OUT/f'landsat_LST_{yr}_{mo:02d}.csv',THERMAL_PROPS)
        obs_df['year'], obs_df['month'] = yr,mo
        monthly_frames.append(obs_df)

thermal_df=pd.concat(monthly_frames,ignore_index=True)
thermal_df['pixel_id']=thermal_df['pixel_id'].astype(str)
for c in ['lst_c','valid_scenes']:
    thermal_df[c]=pd.to_numeric(thermal_df[c],errors='coerce')
thermal_df=thermal_df[(thermal_df.valid_scenes>=1)&thermal_df.lst_c.between(10,80)].copy()
thermal_df=thermal_df.drop_duplicates(['pixel_id','year','month'])
# Prevent coordinate fields duplicating/overwriting the static sample coordinates.
df = thermal_df.drop(columns=['site','lon','lat']).merge(
    static_df,on='pixel_id',how='inner',validate='many_to_one'
)
assert df.year.nunique()==2 and df.site.nunique()>=3
print('Observed monthly rows:',len(df),'| unique 100m sample locations:',df.pixel_id.nunique())
print(df.groupby(['year','month','site']).agg(n=('lst_c','size'),
          med_C=('lst_c','median'),mean_scenes=('valid_scenes','mean')).round(2).to_string())
print('LST percentiles:',df.lst_c.quantile([0,.01,.5,.99,1]).round(2).to_dict())
df.to_csv(OUT/'joined_real_satellite_samples.csv',index=False)
