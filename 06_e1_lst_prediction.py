"""E1: conventional vs AlphaEarth vs fused Random Forest predictors.

Automatically refactored from the original HeatContext experiment notebook.
Execution requires run.py, which supplies a shared experiment namespace.
"""

CONVENTIONAL = (S2_NAMES+['s2_ndvi','s2_ndbi','s2_mndwi']+
                LS_NAMES+['ls_ndvi','ls_ndbi','ls_mndwi']+DW_NAMES)
FEATURE_SETS = {
    'Conventional': CONVENTIONAL+['month'],
    'AlphaEarth': EMBED_BANDS+['month'],
    'Fusion': CONVENTIONAL+EMBED_BANDS+['month'],
}

def fit_rf(train,test,features):
    # Static predictors are real; rows with missing predictors were excluded in step 2.
    model=RandomForestRegressor(n_estimators=120,min_samples_leaf=3,
                                max_features=0.7,n_jobs=-1,random_state=SEED)
    model.fit(train[features],train['lst_c'])
    pred=model.predict(test[features])
    return model,pred

def regression_metrics(y,pred):
    return {
        'MAE_C':float(mean_absolute_error(y,pred)),
        'RMSE_C':float(np.sqrt(mean_squared_error(y,pred))),
        'R2':float(r2_score(y,pred)),
        'N':int(len(y))
    }

train24=df[df.year==TRAIN_YEAR].copy()
test25=df[df.year==TEST_YEAR].copy()
results=[]
for heldout in sorted(train24.site.unique()):
    tr=train24[train24.site!=heldout]
    te=train24[train24.site==heldout]
    if len(te)<15: continue
    for setting,features in FEATURE_SETS.items():
        _,pred=fit_rf(tr,te,features)
        results.append(dict(split='Spatial_LOSO_2024',fold=heldout,features=setting,
                            **regression_metrics(te.lst_c,pred)))
    print('Finished spatial holdout:',heldout)

# Held-out-year 2025: fit only on 2024 targets.
temporal_predictions=test25[['pixel_id','site','year','month','lst_c']].copy()
trained_models={}
for setting,features in FEATURE_SETS.items():
    model,pred=fit_rf(train24,test25,features)
    trained_models[setting]=model
    temporal_predictions[f'pred_{setting}']=pred
    results.append(dict(split='Temporal_2025',fold='all_sites',features=setting,
                        **regression_metrics(test25.lst_c,pred)))

metrics=pd.DataFrame(results)
metrics.to_csv(OUT/'E1_model_metrics.csv',index=False)
temporal_predictions.to_csv(OUT/'E1_temporal_predictions_2025.csv',index=False)
print(metrics.sort_values(['split','fold','RMSE_C']).round(3).to_string(index=False))

summary=(metrics.groupby(['split','features'],as_index=False)
         .agg(RMSE_C=('RMSE_C','mean'),MAE_C=('MAE_C','mean'),R2=('R2','mean')))
summary.to_csv(OUT/'E1_comparison_summary.csv',index=False)
plt.figure(figsize=(8,4))
pivot=summary.pivot(index='split',columns='features',values='RMSE_C')
pivot.plot(kind='bar',ax=plt.gca())
plt.title('E1: prediction RMSE (°C); lower is better')
plt.xticks(rotation=10)
plt.tight_layout();plt.savefig(OUT/'fig03_E1_feature_comparison.png',dpi=240,bbox_inches='tight');plt.show()
