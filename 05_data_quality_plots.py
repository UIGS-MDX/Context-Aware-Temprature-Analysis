"""Generate preliminary real-data LST plots.

Execution requires run.py, which supplies a shared experiment namespace.
"""

plt.figure(figsize=(9,4))
labels=[]; values=[]
for (month,year), g in df.groupby(['month','year'],sort=True):
    labels.append(f'{month:02d}/{year}'); values.append(g['lst_c'].to_numpy())
plt.boxplot(values, tick_labels=labels)
plt.xticks(rotation=45)
plt.title('Observed clear-sky monthly median Landsat LST: Dubai study sites')
plt.ylabel('Land surface temperature (°C)')
plt.tight_layout()
plt.savefig(OUT/'fig01_landsat_monthly_boxplot.png',dpi=220,bbox_inches='tight')
plt.show()

base24=(df[df.year==2024].groupby('pixel_id',as_index=False)
        .agg(lst_c=('lst_c','median'),lon=('lon','first'),lat=('lat','first'),site=('site','first')))
plt.figure(figsize=(9,6))
sc=plt.scatter(base24.lon,base24.lat,c=base24.lst_c,s=15,cmap='inferno',edgecolor='none')
plt.colorbar(sc,label='Summer 2024 median LST (°C)')
for site,(lon,lat) in SITES.items(): plt.text(lon,lat,site.replace('_',' '),fontsize=9)
plt.xlabel('Longitude');plt.ylabel('Latitude');plt.title('Actual sampled Landsat temperatures (four Dubai study windows)')
plt.tight_layout()
plt.savefig(OUT/'fig02_LST_2024_sites.png',dpi=240,bbox_inches='tight')
plt.show()
