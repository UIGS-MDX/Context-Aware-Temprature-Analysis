"""E4: cross-year rank correlations and hotspot retention.

Automatically refactored from the original HeatContext experiment notebook.
Execution requires run.py, which supplies a shared experiment namespace.
"""

# Per-pixel, per-year persistence for each reference baseline.
method_year=(anomalies.groupby(['pixel_id','year'],as_index=False)
             .agg(city_score=('city_score','median'),
                  lc_score=('lc_score','median'),ctx_score=('ctx_score','median'),
                  city_persist=('city_hot','mean'),lc_persist=('lc_hot','mean'),
                  ctx_persist=('ctx_hot','mean'),n_months=('month','nunique'),
                  site=('site','first'),lon=('lon','first'),lat=('lat','first')))
method_year=method_year[method_year.n_months>=MIN_VALID_MONTHS]


def temporal_comparison(frame,value_col):
    wide=(frame.pivot_table(index='pixel_id',columns='year',values=value_col,aggfunc='first')
          .dropna(subset=[2024,2025]))
    n=len(wide)
    if n<30: raise RuntimeError(f'Insufficient repeated points for {value_col}: {n}')
    k=max(1,int(np.ceil(0.1*n)))
    early=set(wide.nlargest(k,2024).index)
    later=set(wide.nlargest(k,2025).index)
    return dict(metric=value_col,N_paired=n,
                Spearman_2024_2025=float(spearmanr(wide[2024],wide[2025]).statistic),
                Top10_Jaccard=jaccard(early,later),
                Top10_retention=len(early&later)/len(early))

stability=[]
for v in ['city_score','lc_score','ctx_score','city_persist','lc_persist','ctx_persist']:
    stability.append(temporal_comparison(method_year,v))
stability.append(temporal_comparison(idx,'GeoHeat'))
stability=pd.DataFrame(stability)
stability.to_csv(OUT/'E4_temporal_stability.csv',index=False)
print(stability.round(3).to_string(index=False))

# Paired contexts, one-to-one location linking, allows inspection of stable locations.
paired=(method_year.pivot_table(index=['pixel_id','site','lon','lat'],columns='year',
          values=['ctx_score','ctx_persist'],aggfunc='first'))
paired.columns=[f'{a}_{b}' for a,b in paired.columns]
paired=paired.dropna().reset_index()
paired.to_csv(OUT/'E4_paired_observations.csv',index=False)
plt.figure(figsize=(6,5))
plt.scatter(paired.ctx_score_2024,paired.ctx_score_2025,s=10,alpha=0.45)
plt.xlabel('2024 median contextual anomaly score')
plt.ylabel('2025 median contextual anomaly score')
plt.title('E4: context-anomaly temporal stability at paired locations')
plt.tight_layout();plt.savefig(OUT/'fig08_E4_context_stability.png',dpi=240,bbox_inches='tight');plt.show()
plt.figure(figsize=(8,4))
plt.bar(stability['metric'], stability['Top10_retention'])
plt.ylim(0,1);plt.xticks(rotation=45,ha='right')
plt.ylabel('Fraction of 2024 top-decile points still top-decile in 2025')
plt.title('E4: observed temporal hotspot retention')
plt.tight_layout();plt.savefig(OUT/'fig09_E4_temporal_hotspot_retention.png',dpi=240,bbox_inches='tight');plt.show()
