"""Source-day-separated binary *statistical-only* CIC-IDS flow-record version.

Rows retain release CSV lineage and original subtype. No PCAP matching, byte
sequence or fictitious session identity is asserted. Selection precedes models.
"""
import argparse,hashlib,json,os,pickle
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Label'}
SPLITS={'Tuesday':'train','Wednesday':'train','Friday':'val','Thursday':'test'}
DATES={'Tuesday':4,'Wednesday':5,'Thursday':6,'Friday':7}
SEED=20261110

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def bucket(day,label):
 if label=='BENIGN':return 'normal'
 if day=='Tuesday':return {'FTP-Patator':'ftp','SSH-Patator':'ssh'}.get(label)
 return 'attack'

def build(raw,out):
 if (out/'audit_report.json').exists():return
 out.mkdir(parents=True,exist_ok=True)
 csvroot=raw/'GeneratedLabelledFlows/TrafficLabelling'
 pools={};counts=Counter();reasons=Counter();files=[];schema=None;source_support=Counter()
 for day in SPLITS:
  paths=sorted(csvroot.glob(day+'*.csv'))
  assert paths,day
  for path in paths:
   digest=sha(path);offset=0;valids=0
   for frame in pd.read_csv(path,chunksize=100000,encoding='latin1',low_memory=False,skip_blank_lines=False):
    frame.columns=[x.strip() for x in frame.columns]
    features=[c for c in frame.columns if c not in EXCLUDED]
    if schema is None:schema=features
    assert features==schema,'Feature order changed'
    rowids=offset+np.arange(len(frame))+1;offset+=len(frame)
    label=frame['Label'].astype('string').str.strip()
    for k,n in label.dropna().value_counts().items():counts[(day,str(k))]+=int(n)
    numeric=frame[features].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
    finite=np.isfinite(numeric).all(axis=1)&np.isfinite(numeric.astype(np.float32)).all(axis=1)
    metadata=frame[['Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp']].notna().all(axis=1).to_numpy()
    time=pd.to_datetime(frame['Timestamp'],format='%d/%m/%Y %H:%M',errors='coerce')
    alt=pd.to_datetime(frame['Timestamp'],format='%d/%m/%Y %H:%M:%S',errors='coerce');time=time.fillna(alt)
    date_ok=((time.dt.year==2017)&(time.dt.month==7)&(time.dt.day==DATES[day])).to_numpy()
    labels=label.fillna('').to_numpy(dtype=str)
    missing=labels=='';finite_ok=finite&~missing;meta_ok=finite_ok&metadata;accepted=meta_ok&date_ok
    reasons[(day,'missing_label')]+=int(missing.sum());reasons[(day,'nonfinite_or_nonnumeric_features')]+=int((~missing&~finite).sum())
    reasons[(day,'missing_flow_metadata')]+=int((finite_ok&~metadata).sum());reasons[(day,'timestamp_not_expected_capture_date')]+=int((meta_ok&~date_ok).sum())
    valids+=int(accepted.sum())
    for l in np.unique(labels[accepted]):source_support[(day,l)]+=int((accepted&(labels==l)).sum())
    for b in ('normal','ftp','ssh','attack'):
     select=accepted&np.asarray([bucket(day,l)==b for l in labels])
     indices=np.flatnonzero(select)
     if not len(indices):continue
     # Fixed uniform priority from original row identity; labels do not rank rows.
     identities=pd.Series([digest+':'+str(rowids[i]) for i in indices])
     priority=pd.util.hash_pandas_object(identities,index=False).to_numpy(dtype=np.uint64)
     keep=np.argsort(priority,kind='stable')[:10000];indices=indices[keep]
     part=frame.iloc[indices].copy();part['_row']=rowids[indices];part['_priority']=priority[keep]
     part['_path']=str(path.resolve());part['_sha']=digest;part['_day']=day;part['_bucket']=b
     key=(day,b)
     if key in pools:part=pd.concat([pools[key],part],ignore_index=True)
     pools[key]=part.nsmallest(10000,'_priority').copy()
   files.append({'path':str(path.resolve()),'sha256':digest,'parsed_data_rows':offset,'numeric_metadata_date_valid':valids,'source_group':'capture_day:'+day,'split':SPLITS[day]})
   print(day,path.name,offset,valids,flush=True)
 candidates=pd.concat(list(pools.values()),ignore_index=True)
 X=candidates[schema].apply(pd.to_numeric,errors='raise').to_numpy(dtype='<f4')
 hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in X]
 candidates['_feature_sha']=hashes
 # Quarantine all selected-pool members of any vector repeated across capture days.
 # Not a claim of duplicate physical packets: this is an extra model-input check.
 spans=candidates.groupby('_feature_sha')['_day'].nunique();cross=set(spans[spans>1].index)
 cross_count=int(candidates['_feature_sha'].isin(cross).sum())
 candidates=candidates[~candidates['_feature_sha'].isin(cross)].copy()
 # Same-day identical vectors with conflicting binary labels are also quarantined.
 candidates['_binary']=(candidates['Label'].str.strip()!='BENIGN').astype(int)
 conflict=candidates.groupby(['_day','_feature_sha'])['_binary'].nunique();bad=set(conflict[conflict>1].index)
 mask=np.asarray([(d,h) in bad for d,h in zip(candidates['_day'],candidates['_feature_sha'])])
 conflict_count=int(mask.sum());candidates=candidates[~mask]
 candidates=candidates.sort_values('_priority').drop_duplicates(['_day','_feature_sha'],keep='first')
 def take(day,b,n):
  return candidates[(candidates['_day']==day)&(candidates['_bucket']==b)].nsmallest(n,'_priority').copy()
 # Normal Tuesday samples are disjoint across its two training tasks.
 tn=take('Tuesday','normal',2400);ftp=take('Tuesday','ftp',1200);ssh=take('Tuesday','ssh',1200)
 k=min(1200,len(tn)//2,len(ftp),len(ssh));assert k>=300,('Tuesday support',k)
 wn=take('Wednesday','normal',1200);wa=take('Wednesday','attack',1200);w=min(len(wn),len(wa));assert w>=300
 pieces=[]
 def add(part,split,task):
  z=part.copy();z['_split']=split;z['_task']=task;pieces.append(z)
 add(tn.iloc[:k],'train',0);add(ftp.iloc[:k],'train',0);add(tn.iloc[k:2*k],'train',1);add(ssh.iloc[:k],'train',1)
 add(wn.iloc[:w],'train',2);add(wa.iloc[:w],'train',2)
 for day,split in [('Friday','val'),('Thursday','test')]:
  normal=take(day,'normal',1500);attack=take(day,'attack',1500);n=min(len(normal),len(attack));assert n>=500,(day,n)
  add(normal.iloc[:n],split,-1);add(attack.iloc[:n],split,-1)
 selected=pd.concat(pieces,ignore_index=True)
 ids=[hashlib.sha256((s+':'+str(int(r))).encode()).hexdigest() for s,r in zip(selected['_sha'],selected['_row'])]
 selected['_sample_id']=ids;assert len(set(ids))==len(ids)
 stat=selected[schema].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float32)
 y=(selected['Label'].str.strip()!='BENIGN').to_numpy(dtype=np.int64)
 # Stable simulated ownership; source identity is never a model feature.
 client=np.full(len(selected),-1,dtype=int)
 for task in (0,1,2):
  for label in (0,1):
   idx=np.flatnonzero((selected['_split']=='train').to_numpy()&(selected['_task']==task).to_numpy()&(y==label))
   client[idx]=np.arange(len(idx))%3
 for split in ('train','val','test'):
  mask=(selected['_split']==split).to_numpy()
  np.savez_compressed(out/f'{split}.npz',stat_raw=stat[mask],y=y[mask],sample_id=np.asarray(ids)[mask],task=selected['_task'].to_numpy()[mask],client=client[mask])
 metadata=[]
 for i,(_,r) in enumerate(selected.iterrows()):
  metadata.append({'sample_id':ids[i],'csv_path':r['_path'],'csv_sha256':r['_sha'],'csv_data_row_1based':int(r['_row']),
   'flow_record_id':str(r['Flow ID']),'source_ip':str(r['Source IP']),'source_port':int(r['Source Port']),'destination_ip':str(r['Destination IP']),
   'destination_port':int(r['Destination Port']),'protocol':int(r['Protocol']),'timestamp_raw':str(r['Timestamp']),'timestamp_precision':'CSV local minute/second; AM/PM ambiguity not resolved; calendar day verified',
   'flow_duration_us':float(r['Flow Duration']),'original_label':str(r['Label']).strip(),'binary_label':int(y[i]),'source_group_id':'capture_day:'+r['_day'],
   'split':r['_split'],'task':int(r['_task']),'client':int(client[i]),'model_input_sha256':r['_feature_sha'],'label_source':'original local release-format labelled flow CSV row; no packet-level label revalidation'})
 provenance=pd.DataFrame(metadata);provenance.to_parquet(out/'provenance.parquet',index=False)
 support=provenance.groupby(['binary_label','original_label','source_group_id','split']).size().reset_index(name='samples');support.to_csv(out/'class_source_split_support.csv',index=False)
 provenance.groupby(['binary_label','source_group_id','split']).size().reset_index(name='samples').to_csv(out/'binary_source_split_support.csv',index=False)
 pd.DataFrame([{'day':d,'split':SPLITS[d],'original_label':l,'rows':n} for (d,l),n in counts.items()]).to_csv(out/'raw_label_source_support.csv',index=False)
 pd.DataFrame([{'day':d,'split':SPLITS[d],'original_label':l,'valid_rows':n} for (d,l),n in source_support.items()]).to_csv(out/'valid_label_source_support.csv',index=False)
 train0=(selected['_split']=='train').to_numpy()&(selected['_task']==0).to_numpy()
 signed=np.sign(stat.astype(np.float64))*np.log1p(np.abs(stat.astype(np.float64)))
 mu=signed[train0].mean(0);scale=signed[train0].std(0);scale[scale==0]=1
 np.savez(out/'preprocessing.npz',mean=mu,scale=scale)
 for split in ('train','val','test'):
  assert np.isfinite(((signed[(selected['_split']==split).to_numpy()]-mu)/scale).astype(np.float32)).all()
 no_sources=provenance.groupby('source_group_id')['split'].nunique().max()==1
 no_vectors=provenance.groupby('model_input_sha256')['split'].nunique().max()==1
 assert no_sources and no_vectors
 # Independently reread selected rows by original CSV identity, not cached arrays.
 checks=[]
 for path,group in provenance.groupby('csv_path'):
  probe=group.sample(n=min(3,len(group)),random_state=SEED)
  wanted={int(r.csv_data_row_1based):r for r in probe.itertuples()};offset=0
  for frame in pd.read_csv(path,chunksize=100000,encoding='latin1',low_memory=False,skip_blank_lines=False):
   frame.columns=[x.strip() for x in frame.columns]
   for row,record in wanted.items():
    if offset<row<=offset+len(frame):
     r=frame.iloc[row-offset-1];values=pd.to_numeric(r[schema],errors='raise').to_numpy(dtype='<f4')
     ok=hashlib.sha256(values.tobytes()).hexdigest()==record.model_input_sha256 and str(r['Label']).strip()==record.original_label and str(r['Flow ID'])==record.flow_record_id
     checks.append({'sample_id':record.sample_id,'row_recomputed':bool(ok)})
   offset+=len(frame)
  assert sha(path)==group.iloc[0]['csv_sha256']
 assert all(x['row_recomputed'] for x in checks)
 save(out/'files.json',files);save(out/'feature_schema.json',{'version':'cicids_release_stat77_signedlog_train_task1_v1','fields':schema,'dim':len(schema),'exclude_from_model':sorted(EXCLUDED),'duplicate_measurement':'Fwd Header Length.1 retained as a distinct release field; no data-driven feature selection','sequence_used':False,'transform':'sign(x)*log1p(abs(x)); standardization fitted only Task1 training; no clipping or refit'})
 save(out/'audit_report.json',{'status':'PASS_FLOW_RECORD_CAPTURE_DAY_BINARY_STAT_ONLY','task':'binary domain-incremental detection, not multiclass class-incremental','source_unit':'local labelled-flow capture day','split_days':SPLITS,'samples':len(selected),'split_samples':provenance.groupby('split').size().to_dict(),'source_groups_disjoint':bool(no_sources),'selected_model_vectors_cross_split_disjoint':bool(no_vectors),'candidate_cross_day_vector_members_quarantined':cross_count,'candidate_binary_conflict_members_quarantined':conflict_count,'exclusions':[{'day':d,'reason':r,'rows':n} for (d,r),n in reasons.items()],
  'independent_csv_recomputations':checks,'test_source':'Thursday flow-record capture day held out from this protocol training/selection; corpus previously inspected, not a new collection','limits':['raw local CSV label truth inherited, not independently packet-validated','timestamp AM/PM unresolved; no claim of precise UTC/session reconstruction','no sequence and no CSV-PCAP pairing claim','single held-out acquisition day, same testbed; not new organization generalization','not the old two-train/two-fresh-val/one-fresh-test dual-modal confirmation protocol'],
  'subtype_source_coverage':'Attack subclasses often one day; only binary label has train/val/test source support; unseen subtype transfer interpreted explicitly'})
 save(out/'artifact_manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in {'artifact_manifest.json','EXPERIMENT_PLAN.md'}})
 print('PASS',len(selected),schema.__len__(),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,default=ROOT.parents[2]/'data/CIC_IDS');p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.raw_root.resolve(),a.output.resolve())
