"""E2: context clustering, leave-site-out references, and anomalies.
Execution requires run.py, which supplies a shared experiment namespace.
"""

# Assign broad land-cover strata from average 2024 Dynamic World probabilities.
point_info=static_df.copy()
point_info['dw_green']=point_info[['dw_trees','dw_grass','dw_shrub']].sum(axis=1)
point_info['stratum']=np.select(
    [point_info.dw_water>=0.4,
     point_info.dw_built>=0.35,
     point_info.dw_green>=0.4],
    ['water','built','green'],default='bare_or_mixed'
)
# Explicitly exclude likely water from an urban-surface study.
point_info=point_info[point_info.stratum!='water'].copy()
point_info['context']=''
for stratum,group in point_info.groupby('stratum'):
    k=max(1,min(N_CONTEXTS_PER_STRATUM,len(group)//60))
    if k==1: labels=np.zeros(len(group),dtype=int)
    else:
        model=KMeans(n_clusters=k,n_init=10,random_state=SEED)
        labels=model.fit_predict(group[EMBED_BANDS].to_numpy())
    point_info.loc[group.index,'context']=[f'{stratum}_c{v}' for v in labels]

labels=point_info[['pixel_id','stratum','context']]
dat=df.merge(labels,on='pixel_id',how='inner',validate='many_to_one')
print('Context sizes, by unique 100m locations:')
print(labels.groupby(['stratum','context']).size().to_string())
labels.to_csv(OUT/'E2_2024_environmental_contexts.csv',index=False)


def quartile_stats(series):
    a=np.asarray(series.dropna(),float)
    if len(a)==0: return None
    q25,med,q75=np.percentile(a,[25,50,75])
    return {'n':len(a),'median':float(med),'iqr':float(max(q75-q25,1.0))}


def spatial_excluded_references(data,min_n=MIN_CONTEXT_REF):
    """All baselines exclude target study-area locations from contemporaneous references."""
    out=[]
    for (yr,month,site),target in data.groupby(['year','month','site'],sort=True):
        refs=data[(data.year==yr)&(data.month==month)&(data.site!=site)]
        if refs.empty: continue
        city=quartile_stats(refs.lst_c)
        lc_stats={k:quartile_stats(g.lst_c) for k,g in refs.groupby('stratum')}
        context_stats={k:quartile_stats(g.lst_c) for k,g in refs.groupby('context')}
        piece=target.copy()
        for method in ['city','lc','ctx']:
            medians=[];iqrs=[];fallbacks=[];ref_counts=[]
            for _,r in piece.iterrows():
                if method=='city':
                    stats,origin=city,'city'
                elif method=='lc':
                    possible=lc_stats.get(r.stratum)
                    stats,origin=(possible,'lc') if possible is not None and possible['n']>=min_n else (city,'city_fallback')
                else:
                    possible=context_stats.get(r.context)
                    if possible is not None and possible['n']>=min_n:
                        stats,origin=possible,'ctx'
                    else:
                        lc=lc_stats.get(r.stratum)
                        stats,origin=(lc,'lc_fallback') if lc is not None and lc['n']>=min_n else (city,'city_fallback')
                medians.append(stats['median']);iqrs.append(stats['iqr'])
                fallbacks.append(origin);ref_counts.append(stats['n'])
            piece[f'{method}_ref_c']=medians
            piece[f'{method}_anom_c']=piece.lst_c-np.asarray(medians)
            piece[f'{method}_score']=(piece.lst_c-np.asarray(medians))/np.asarray(iqrs)
            piece[f'{method}_ref_n']=ref_counts
            piece[f'{method}_source']=fallbacks
        out.append(piece)
    if not out: raise RuntimeError('No valid cross-site references. Add more study sites.')
    return pd.concat(out,ignore_index=True)

anomalies=spatial_excluded_references(dat)
print('Reference provenance:',anomalies.ctx_source.value_counts(normalize=True).round(3).to_dict())
if (anomalies.ctx_source=='ctx').mean()<0.5:
    print('WARNING: frequent context-reference fallbacks. Reduce k / enlarge areas / sample more points.')
anomalies.to_csv(OUT/'E2_contextual_anomalies_all_months.csv',index=False)

# Per-month top-decile hotspots calculated *after* reference matching.
for method in ['city','lc','ctx']:
    anomalies[f'{method}_hot']=(
        anomalies.groupby(['year','month'])[f'{method}_score']
        .transform(lambda s:s.rank(method='average',pct=True)>=0.90)
    )

def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a|b else np.nan

monthly_overlap=[]
for (yr,mo),g in anomalies.groupby(['year','month']):
    city=set(g.loc[g.city_hot,'pixel_id'])
    lc=set(g.loc[g.lc_hot,'pixel_id'])
    ctx=set(g.loc[g.ctx_hot,'pixel_id'])
    monthly_overlap.append({'year':yr,'month':mo,'Jaccard_ctx_vs_city':jaccard(ctx,city),
                            'Jaccard_ctx_vs_landcover':jaccard(ctx,lc),
                            'N':len(g)})
overlap=pd.DataFrame(monthly_overlap)
overlap.to_csv(OUT/'E2_monthly_hotspot_overlap.csv',index=False)
print(overlap.round(3).to_string(index=False))

# Actual site-level spatial results: mean anomaly from multiple observations.
map24=(anomalies[anomalies.year==2024].groupby('pixel_id',as_index=False)
       .agg(context=('context','first'), stratum=('stratum','first'),
            lon=('lon','first'),lat=('lat','first'),
            ctx_anom_c=('ctx_anom_c','median'), ctx_score=('ctx_score','median'),
            n_months=('month','nunique')))
plt.figure(figsize=(9,6))
sc=plt.scatter(map24.lon,map24.lat,c=map24.ctx_anom_c,cmap='coolwarm',s=16,
               vmin=-8,vmax=8,edgecolors='none')
plt.colorbar(sc,label='Observed minus matched-context LST (°C)')
plt.xlabel('Longitude');plt.ylabel('Latitude');plt.title('E2: cross-site contextual thermal anomalies, summer 2024')
plt.tight_layout();plt.savefig(OUT/'fig04_E2_contextual_anomaly_map.png',dpi=240,bbox_inches='tight');plt.show()

plt.figure(figsize=(9,6))
codes,uniques=pd.factorize(map24.context)
sc=plt.scatter(map24.lon,map24.lat,c=codes,cmap='tab20',s=16,edgecolors='none')
plt.xlabel('Longitude');plt.ylabel('Latitude');plt.title('E2: AlphaEarth-derived environmental context groups')
plt.tight_layout();plt.savefig(OUT/'fig05_E2_environmental_context_map.png',dpi=240,bbox_inches='tight');plt.show()
print('Context mapping:',{int(i):label for i,label in enumerate(uniques)})
