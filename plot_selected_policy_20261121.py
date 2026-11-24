"""Plot frozen selected policies using executed Re-Fed reuse; no cost estimates."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'supplementary_data/selected_policy_20261121_v5'
OUT.mkdir(parents=True, exist_ok=True)

def read(rel):
    with (ROOT / rel).open(newline='') as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

rows = []
sources = ['supplementary_data/feddia_method_suite_20261114_v1/main_per_seed.csv',
           'supplementary_data/recent_external_fcl_20261114_v1/main_per_seed.csv',
           'supplementary_data/evidence_bridge_20261121_v1/external_formal_per_seed.csv']
for r in read(sources[0]):
    if r['dataset'] == 'CIC_IDS' and r['role'] == 'budget_final' and r['arm'] in ('M0','M4','M5'):
        rows.append(dict(arm=r['arm'], seed=int(r['seed']), schedule=r['schedule'], F1_percent=float(r['final_pooled_f1_percent']), duplex_bytes=int(r['duplex_bytes'])))
for r in read(sources[1]):
    if r['dataset'] == 'CIC_IDS' and r['role'] == 'budget_final' and r['arm'] == 'FedCLASS':
        rows.append(dict(arm=r['arm'], seed=int(r['seed']), schedule=r['schedule'], F1_percent=float(r['final_pooled_percent']), duplex_bytes=int(r['duplex_bytes'])))
for r in read(sources[2]):
    if r['variant'] == 'optimized':
        assert r['schedule'] == '10/6/6'
        rows.append(dict(arm='ReFed', seed=int(r['seed']), schedule=r['schedule'], F1_percent=float(r['final_pooled_macroF1_percent']), duplex_bytes=int(r['duplex_bytes'])))
assert len(rows) == 15
write('per_seed.csv', rows)
colors = ['#0072B2','#E69F00','#CC79A7','#008080','#882255']
markers = ['o','D','P','*','d']
labels = ['FedDIA (Ours)','Plain (M4)','Adapted DER++ (M5)','FedCLASS (adapted)','Re-Fed (executed reuse)']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(7.25,3.85))
fig.subplots_adjust(left=.105,right=.975,top=.90,bottom=.26)
summary=[]; handles=[]
cap = 37208796
for arm, color, marker, label in zip(['M0','M4','M5','FedCLASS','ReFed'],colors,markers,labels):
    group=[r for r in rows if r['arm']==arm]
    assert sorted(r['seed'] for r in group)==[42,43,123]
    assert all(r['duplex_bytes']<=cap for r in group)
    x=np.array([r['duplex_bytes']/2**20 for r in group]); y=np.array([r['F1_percent'] for r in group])
    assert np.all((y >= 97) & (y <= 100))
    ax.scatter(x,y,color=color,s=17,alpha=.4,zorder=2)
    ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),fmt=marker,color=color,markersize=9 if arm=='FedCLASS' else 6,capsize=3,elinewidth=1,zorder=3)
    handles.append(Line2D([],[],color=color,marker=marker,linestyle='None',label=label))
    summary.append(dict(arm=arm,n=3,mean_F1_percent=y.mean(),sample_SD_F1_percent=y.std(ddof=1),mean_duplex_MiB=x.mean(),sample_SD_duplex_MiB=x.std(ddof=1),schedule=group[0]['schedule']))
ax.axvline(cap/2**20,color='#555555',ls='--',lw=1)
ax.text(cap/2**20-.06,97.12,'Cap: 35.485 MiB',rotation=90,ha='right',va='bottom',fontsize=8,color='#555555')
ax.set(xlim=(31,36),ylim=(97,100),xlabel='Actual duplex model-state communication (MiB)',ylabel='Terminal pooled macro-F1 (%)',title='CIC-IDS2017: validation-selected policies')
ax.set_xticks(np.arange(31,37)); ax.set_yticks(np.arange(97,100.01,.5))
ax.grid(alpha=.18,lw=.5)
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False,fontsize=7.5,columnspacing=1.5)
for ext in ['pdf','svg','png']:
    fig.savefig(OUT / ('fig_selected_policy_common_cap.'+ext),dpi=600)
plt.close(fig)
write('summary.csv',summary)
caption='Current CIC-IDS2017 selected-policy comparison under the 35.485 MiB cap. Large points and bars show mean terminal pooled macro-F1 and sample SD across three seeds; faint points show individual seeds at their actual coordinates. Re-Fed uses executed reuse and the newly selected 10/6/6 schedule. M0, M4, M5 and FedCLASS retain their recorded selections. The horizontal axis reports actual mean duplex model-state bytes; only the selected high-F1 subset is shown. AF-FCL-small is reported separately because it falls outside this high-F1 view. Communication counts serialized model-state payloads, not network wire traffic.\n'
(OUT/'CAPTION.txt').write_text(caption)
(OUT/'VERIFICATION.json').write_text(json.dumps({'status':'PASS','n_methods':5,'n_seeds_each':3,'SD':'sample ddof=1','cost':'executed serialized payload bytes','theoretical_reuse_points':False,'Table_9_binding':'not verified against full manuscript','sources':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}},indent=2))
print(OUT)
print(json.dumps(summary,indent=2))
