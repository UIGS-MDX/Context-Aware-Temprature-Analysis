"""E3: GeoHeat index, component ablations and weight sensitivity.

Automatically refactored from the original HeatContext experiment notebook.
Execution requires run.py, which supplies a shared experiment namespace.
"""

anomalies['H_month']=(anomalies.groupby(['year','month']).lst_c
                      .transform(lambda s:s.rank(pct=True)))
positive_rank=(anomalies.groupby(['year','month']).ctx_score
               .transform(lambda s:s.clip(lower=0).rank(pct=True)))
anomalies['A_month']=np.where(anomalies.ctx_score>0,positive_rank,0.0)
# context hotspot flags were computed in E2 using held-out-site references.
# Static projected-population percentile: rank is fixed for temporal comparison.
popinfo=(anomalies[['pixel_id','pop_2025_proj']]
         .drop_duplicates('pixel_id').copy())
popinfo['E']=np.log1p(popinfo.pop_2025_proj.clip(lower=0)).rank(pct=True)
idx=(anomalies.groupby(['pixel_id','year'],as_index=False)
     .agg(H=('H_month','mean'),A=('A_month','mean'),
          P=('ctx_hot','mean'),n_months=('month','nunique'),
          site=('site','first'),lon=('lon','first'),lat=('lat','first')))
idx=idx.merge(popinfo[['pixel_id','E']],on='pixel_id',validate='many_to_one')
idx=idx[idx.n_months>=MIN_VALID_MONTHS].copy()
BASE_WEIGHTS={'H':0.4,'A':0.2,'P':0.2,'E':0.2}

def weighted_index(data,weights):
    total=sum(weights.values())
    return sum(data[k]*v for k,v in weights.items())/total

idx['GeoHeat']=weighted_index(idx,BASE_WEIGHTS)
for exclude in BASE_WEIGHTS:
    remaining={k:v for k,v in BASE_WEIGHTS.items() if k!=exclude}
    idx[f'without_{exclude}']=weighted_index(idx,remaining)
idx.to_csv(OUT/'E3_heat_priority_scores_and_ablations.csv',index=False)
print(idx.groupby('year')[['H','A','P','E','GeoHeat']].describe().round(3).to_string())

# Rank consistency and top-decile overlap in 2024 when removing each component.
index24=idx[idx.year==2024].copy()
top_base=set(index24.nlargest(max(1,int(np.ceil(.1*len(index24)))),'GeoHeat').pixel_id)
ab_rows=[]
for omit in BASE_WEIGHTS:
    col=f'without_{omit}'
    top_other=set(index24.nlargest(max(1,int(np.ceil(.1*len(index24)))),col).pixel_id)
    ab_rows.append({'ablation':f'Remove {omit}',
                    'Spearman_rank_vs_full':spearmanr(index24.GeoHeat,index24[col]).statistic,
                    'Top10_Jaccard_vs_full':jaccard(top_base,top_other)})
ablations=pd.DataFrame(ab_rows)
ablations.to_csv(OUT/'E3_component_ablation_metrics.csv',index=False)
print(ablations.round(3).to_string(index=False))

# Weight-sensitivity: 150 sampled weight combinations centered on 0.4/0.2/0.2/0.2.
rng=np.random.default_rng(SEED)
weight_draws=rng.dirichlet([12,6,6,6],size=150)
sens=[]
for w in weight_draws:
    weight_map=dict(zip(['H','A','P','E'],w))
    scores=weighted_index(index24,weight_map)
    top=set(index24.loc[scores.nlargest(len(top_base)).index,'pixel_id'])
    sens.append({'top10_jaccard':jaccard(top_base,top),
                 **weight_map})
sensitivity=pd.DataFrame(sens)
sensitivity.to_csv(OUT/'E3_weight_sensitivity.csv',index=False)
print('Top-decile Jaccard across alternative weights (5%, median, 95%):',
      sensitivity.top10_jaccard.quantile([.05,.5,.95]).round(3).to_dict())

plt.figure(figsize=(8,4))
plt.bar(ablations['ablation'], ablations['Top10_Jaccard_vs_full'])
plt.ylim(0,1.0);plt.ylabel('Top-decile Jaccard vs full index')
plt.title('E3: index-component ablation')
plt.tight_layout();plt.savefig(OUT/'fig06_E3_index_ablation.png',dpi=240,bbox_inches='tight');plt.show()
plt.figure(figsize=(9,6))
sc=plt.scatter(index24.lon,index24.lat,c=index24.GeoHeat,s=18,cmap='magma',vmin=0,vmax=1)
plt.colorbar(sc,label='Experimental surface-heat-priority score (0–1)')
plt.xlabel('Longitude');plt.ylabel('Latitude');plt.title('E3: experimental summer-2024 heat priority score')
plt.tight_layout();plt.savefig(OUT/'fig07_E3_heat_priority_map.png',dpi=240,bbox_inches='tight');plt.show()
