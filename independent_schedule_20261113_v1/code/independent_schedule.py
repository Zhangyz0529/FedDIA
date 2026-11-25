"""Independent Rbot schedule comparison: frozen data, six runs, delayed testing.

Three class-stratified chronological portions of one training capture define
repeated-class tasks. This is not class-incremental learning or three sites.
All experiment artifacts are confined to --output; historical results are read
only. Test inference is a separate action after all six validation runs finish.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS=(42,43,123)
SCHEDULES={'front':[9,5,5],'late':[5,7,7]}
FEATURES=['Dur','TotPkts','TotBytes','SrcBytes']
LABELS={'BENIGN':0,'Rbot':1}
CAPTURES={'train':44,'val':45,'test':51}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def read(path):return json.loads(Path(path).read_text())


def write(path,rows):
 if not rows:return
 with Path(path).open('w',newline='') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys)
  w.writeheader();w.writerows(rows)


def seed_all(seed):
 import torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def make_model():
 from fcl_models_v4 import V4DualBackbone
 return V4DualBackbone(stat_dim=4,hidden_dim=48,num_classes=2,use_stat=True,use_seq=False,
      use_adapter=True,use_se=True,use_cosface=True,use_logit_adj=False,class_counts=[1,1])


def load_data(out,split):
 with np.load(out/'data'/f'{split}.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}


def prepare(root,out):
 if (out/'FROZEN_PLAN.json').exists():
  verify(out);return
 if (out/'runs').exists():raise RuntimeError('Runs exist without a frozen plan')
 code=out/'code';code.mkdir(exist_ok=True,parents=True)
 shutil.copy2(root/'fcl_models_v4.py',code/'fcl_models_v4.py')
 prior=root/'supplementary_data/e4_source_acquisition_20261111_v1'
 manifest=read(prior/'external_source_manifest.json')
 specifications=[next(r for r in manifest if r['capture_id']==cap) for cap in CAPTURES.values()]
 # New, explicitly authorized Rbot-only protocol: capture42 is not relabelled
 # as accepted and none of its rows are used in these arrays.
 configurations=[];references={};tables=[];filters=[]
 for split,cap in CAPTURES.items():
  spec=next(r for r in specifications if r['capture_id']==cap)
  assert spec['family'].lower()=='rbot'
  raw=prior/spec['path'];assert sha(raw)==spec['sha256']
  readme=prior/'official_metadata'/f'{cap}_README.md'
  assert 'Probable Name: Rbot' in readme.read_text() or 'Probable Name: RBot' in readme.read_text()
  for filename in [f'{cap}_argus.conf',f'{cap}_ra.conf']:
   q=prior/'official_metadata'/filename
   effective='\n'.join(s.strip() for s in q.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#'))
   configurations.append(dict(capture=cap,file=filename.split('_',1)[1],effective_sha256=hashlib.sha256(effective.encode()).hexdigest()))
   references[str(q.relative_to(root))]=sha(q)
  references[str(readme.relative_to(root))]=sha(readme);references[str(raw.relative_to(root))]=spec['sha256']
  pieces=[];offset=0;counts=Counter()
  for chunk in pd.read_csv(raw,dtype=str,keep_default_na=False,chunksize=200000):
   labels=chunk.Label.str.strip();normal=labels.str.startswith('flow=From-Normal');attack=labels.str.startswith('flow=From-Botnet')
   accepted=normal|attack
   counts['raw_rows']+=len(chunk);counts['excluded_label']+=int((~accepted).sum())
   frame=chunk.loc[accepted].copy();frame['row_1based']=np.flatnonzero(accepted.to_numpy())+offset+1
   a=frame[FEATURES].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
   stamp=pd.to_datetime(frame.StartTime,format='%Y/%m/%d %H:%M:%S.%f',errors='coerce')
   good=np.isfinite(a).all(1)&(a>=0).all(1)&(a[:,1]>=1)&(a[:,2]>=a[:,3])&stamp.notna().to_numpy()
   counts['excluded_invalid']+=int((~good).sum())
   frame=frame.iloc[np.flatnonzero(good)].copy()
   frame[FEATURES]=a[good];frame['label']=frame.Label.str.strip().str.startswith('flow=From-Botnet').astype(int)
   frame['time']=stamp.iloc[np.flatnonzero(good)].to_numpy();frame['capture']=cap;frame['split']=split
   pieces.append(frame[['row_1based','StartTime','Label','label','time','capture','split']+FEATURES]);offset+=len(chunk)
  table=pd.concat(pieces,ignore_index=True);tables.append(table);filters.append(dict(split=split,capture=cap,**dict(counts)))
  print('PARSED',split,cap,len(table),flush=True)
 # Only the already defined three sources are candidates; quarantine ambiguous
 # raw vectors before sampling. Input identity deliberately excludes labels.
 df=pd.concat(tables,ignore_index=True)
 _,group=np.unique(df[FEATURES].to_numpy(dtype=np.float64),axis=0,return_inverse=True)
 df['raw_group']=group
 bad_groups=set(df.groupby('raw_group').split.nunique().loc[lambda x:x>1].index)
 conflicts=set(df.groupby('raw_group').label.nunique().loc[lambda x:x>1].index)
 quarantine=df.raw_group.isin(bad_groups|conflicts)
 quarantine_count=int(quarantine.sum())
 kept=df.loc[~quarantine].sort_values(['time','capture','row_1based']).drop_duplicates('raw_group',keep='first').copy()
 # Chronological thirds are formed separately per class. Their time ranges may
 # overlap across classes; do not describe this as a strict online time stream.
 selected=[];support=[]
 for split,cap in CAPTURES.items():
  for label in [0,1]:
   cohort=kept.loc[(kept.split==split)&(kept.label==label)].sort_values(['time','row_1based'])
   groups=np.array_split(np.arange(len(cohort)),3)
   per_task={'train':600,'val':200,'test':500}[split]
   for task,indices in enumerate(groups,1):
    source=cohort.iloc[indices].copy()
    assert len(source)>=per_task,(split,label,task,len(source),per_task)
    source['rank']=source.row_1based.map(lambda r:hashlib.sha256(f'20261113|{cap}|{r}'.encode()).hexdigest())
    sample=source.sort_values('rank').iloc[:per_task].sort_values(['time','row_1based']).copy()
    sample['task']=task;sample['client']=np.arange(len(sample))%3 if split=='train' else -1
    sample['sample_id']=sample.row_1based.map(lambda r:f'cap{cap}:row{r}')
    selected.append(sample)
    support.append(dict(split=split,capture=cap,task=task,label=label,eligible_before_sampling=len(source),selected=len(sample),
                        earliest=str(sample.time.min()),latest=str(sample.time.max())))
 selected=pd.concat(selected,ignore_index=True)
 # Fit preprocessing only on Task1/train. If float32 transformation merges
 # vectors across roles, quarantine members and refit, never inspect scores.
 float_quarantined=[]
 for iteration in range(10):
  raw=selected[FEATURES].to_numpy(dtype=np.float64);logged=np.sign(raw)*np.log1p(np.abs(raw))
  fit=(selected.split=='train')&(selected.task==1)
  mu=logged[fit].mean(0);scale=logged[fit].std(0);scale[scale==0]=1
  x=((logged-mu)/scale).astype(np.float32)
  _,g=np.unique(x,axis=0,return_inverse=True);selected['model_group']=g
  cross=set(selected.groupby('model_group').split.nunique().loc[lambda a:a>1].index)
  if not cross:break
  remove=selected.model_group.isin(cross);float_quarantined.extend(selected.loc[remove,'sample_id'].tolist())
  selected=selected.loc[~remove].copy().reset_index(drop=True)
 else:raise RuntimeError('Preprocessing dedup did not converge')
 data=out/'data';data.mkdir(exist_ok=True)
 final_support=[]
 for split in CAPTURES:
  mask=(selected.split==split).to_numpy();r=selected.loc[mask]
  for task in [1,2,3]:
   for label in [0,1]:
    sub=r[(r.task==task)&(r.label==label)]
    assert len(sub)>=100,(split,task,label,len(sub))
    for client in ([0,1,2] if split=='train' else [-1]):
     final_support.append(dict(split=split,capture=CAPTURES[split],task=task,label=label,client=client,
                                n=int((sub.client==client).sum())))
  np.savez_compressed(data/f'{split}.npz',x=x[mask],raw=r[FEATURES].to_numpy(dtype=np.float64),
       y=r.label.to_numpy(dtype=np.int64),task=r.task.to_numpy(dtype=np.int64),client=r.client.to_numpy(dtype=np.int64),
       sample_id=r.sample_id.to_numpy(dtype=str))
 np.savez(data/'preprocessing.npz',mean=mu,scale=scale)
 selected.drop(columns=['rank']).to_csv(data/'sample_manifest.csv',index=False)
 write(data/'class_task_client_support.csv',final_support);write(data/'candidate_support.csv',support);write(data/'filter_counts.csv',filters)
 write(data/'configuration_identity.csv',configurations)
 assert len({r['effective_sha256'] for r in configurations if r['file']=='argus.conf'})==1
 assert len({r['effective_sha256'] for r in configurations if r['file']=='ra.conf'})==1
 assert selected.sample_id.nunique()==len(selected)
 plan=dict(protocol='CTU_RBOT_BINARY_CAPTURE_HOLDOUT_CLASS_STRATIFIED_CHRONOLOGICAL_TASKS',
    source_roles=CAPTURES,label_map=LABELS,features=FEATURES,stat_dim=4,width=48,full_update=True,
    task_definition='Three class-stratified chronological thirds; BENIGN and Rbot recur throughout; current records never repeated',
    task_order_limitation='Per-class time thirds can overlap; not new-class learning or a strict online timeline',
    source_scope='Three captures in one CTU environment with host reuse; simulated clients within capture44',
    dataset_usage='Unused for prior model development; eligibility and sampling never use model scores',
    schedules=SCHEDULES,seeds=list(SEEDS),lr=.001737670720958496,weight_decay=1e-4,batch_size=64,
    local_epochs=3,replay_capacity=240,replay_weight=.5,optimizer_restart='each client-round',
    replay_sampling='shuffle memory each current batch, take min(64,len(memory)); same ordinary ER rule',
    replay_storage='latest floor(240/seen_classes) per class after task; indices remain local',
    preprocessing='sign*log1p(abs), population mean/SD fitted Task1 train only; frozen',
    score='sum float64 non-BENIGN probabilities',threshold_rule='max val TPR at empirical FPR<=.01; tie lower FPR, then higher threshold',
    calibration='seen validation tasks pooled per checkpoint; never test-selected',test_policy='all six validation runs complete before any test inference',
    metrics='fixed-task macro-F1, class recall, argmax binary counts, calibrated counts, unique pooled terminal F1',
    selection_seed=20261113,raw_quarantine_records=quarantine_count,raw_cross_split_groups=len(bad_groups),raw_conflicting_label_groups=len(conflicts),
    float32_quarantine_ids=float_quarantined,planned_full_runs=6,independent_initializations=True)
 import torch
 torch.set_num_threads(2);seed_all(42);model=make_model()
 state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 size=len(pickle.dumps(state,protocol=4));plan['parameter_count']=sum(p.numel() for p in model.parameters())
 plan['reference_state_bytes']=size
 # A new model-specific common upper bound, including all 19 rounds and both
 # directions. The small serialization reserve is explicit, not traffic padding.
 plan['cap_bytes']=3*19*2*(size+512)
 plan['byte_scope']='actual pickle protocol4 CPU full state uploads and downloads, Task1 included; no wire overhead/search cost'
 acceptance=dict(status='PASS_RBOT_BINARY_CAPTURE_HOLDOUT',captures=[44,45,51],capture42_used=False,
    source_roles_disjoint=True,published_configurations_match=True,raw_vectors_cross_split_disjoint=True,
    final_float32_vectors_cross_split_disjoint=True,all_classes_supported=True,performance_used=False,
    scope=plan['source_scope'],label_limit='provider From-Normal and From-Botnet labels only, not per-flow independent attack certification')
 save(data/'acceptance.json',acceptance);save(data/'feature_schema.json',dict(features=FEATURES,dim=4,labels=LABELS))
 save(data/'source_hashes.json',references)
 artifacts={str(q.relative_to(out)):sha(q) for q in data.iterdir() if q.is_file()}
 plan['artifact_hashes']=artifacts;plan['code_hashes']={str(q.relative_to(out)):sha(q) for q in code.glob('*.py')}
 plan['frozen_utc']=datetime.now(timezone.utc).isoformat()
 save(out/'FROZEN_PLAN.json',plan)
 print('FROZEN',plan['parameter_count'],'params',plan['cap_bytes']/2**20,'MiB cap',flush=True)


def verify(out):
 plan=read(out/'FROZEN_PLAN.json')
 for k,v in {**plan['artifact_hashes'],**plan['code_hashes']}.items():assert sha(out/k)==v,k
 assert read(out/'data/acceptance.json')['status']=='PASS_RBOT_BINARY_CAPTURE_HOLDOUT'
 return plan


def predict(model,x,device):
 import torch
 model.eval();logits=[]
 with torch.no_grad():
  for i in range(0,len(x),256):logits.append(model(torch.from_numpy(x[i:i+256]).to(device)).cpu())
 logits=torch.cat(logits);return logits.numpy(),logits.softmax(1).numpy()


def score(p):return p[:,1:].astype(np.float64).sum(1)


def counts(y,p,threshold=None):
 truth=y!=0;alarm=p.argmax(1)!=0 if threshold is None else score(p)>=threshold
 tp=int((truth&alarm).sum());fp=int((~truth&alarm).sum());tn=int((~truth&~alarm).sum());fn=int((truth&~alarm).sum())
 return dict(TP=tp,FP=fp,TN=tn,FN=fn,TPR_percent=100*tp/(tp+fn),FPR_percent=100*fp/(fp+tn))


def threshold(y,p):
 from sklearn.metrics import roc_curve
 fp,tp,ts=roc_curve(y!=0,score(p),drop_intermediate=False)
 ids=np.flatnonzero(fp<=.01);best=ids[tp[ids]==tp[ids].max()];best=best[fp[best]==fp[best].min()]
 t=float(ts[best[0]])
 return t


def evaluate(y,p):
 from sklearn.metrics import f1_score,recall_score
 pred=p.argmax(1)
 return dict(macro_f1_percent=100*float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
    class_recalls_percent=(100*recall_score(y,pred,labels=[0,1],average=None,zero_division=0)).tolist(),argmax=counts(y,p))


def state_hash(state):
 h=hashlib.sha256()
 for k,v in state.items():
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
 return h.hexdigest()


def train(out,device_choice):
 import torch
 plan=verify(out);torch.set_num_threads(2)
 device=torch.device('cuda:0' if device_choice=='auto' and torch.cuda.is_available() else 'cpu' if device_choice=='auto' else device_choice)
 train_data=load_data(out,'train');val=load_data(out,'val');x=torch.from_numpy(train_data['x']);y=torch.from_numpy(train_data['y'])
 for seed in SEEDS:
  for arm,schedule in SCHEDULES.items():
   run=out/'runs'/arm/f'seed_{seed}';run.mkdir(parents=True,exist_ok=True)
   if (run/'validation_complete.json').exists():continue
   if list(run.glob('task_*/model.pth')):raise RuntimeError('Incomplete attempt exists; preserve and use a new attempt directory')
   seed_all(seed);net=make_model().to(device);memory=[[] for _ in range(3)];records=[];stages=[];start=time.monotonic()
   init_hash=state_hash(net.state_dict())
   for task in [1,2,3]:
    available=[np.flatnonzero((train_data['task']==task)&(train_data['client']==c)) for c in range(3)]
    for round_no in range(1,schedule[task-1]+1):
     updates=[];weights=[]
     for client,indices in enumerate(available):
      callseed=seed*100000+task*10000+client*100+round_no;seed_all(callseed)
      local=copy.deepcopy(net).train()
      opt=torch.optim.AdamW(local.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
      current_rng=torch.Generator().manual_seed(callseed+10000000);replay_rng=torch.Generator().manual_seed(callseed+20000000)
      down={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
      down_bytes=len(pickle.dumps(down,protocol=4));n_steps=current_n=replay_n=0;losses=[];current_hash=hashlib.sha256();replay_hash=hashlib.sha256();by_class=Counter()
      for epoch in range(plan['local_epochs']):
       order=torch.randperm(len(indices),generator=current_rng).numpy()
       for begin in range(0,len(order),plan['batch_size']):
        ids=indices[order[begin:begin+plan['batch_size']]];batch=x[ids].to(device);target=y[ids].to(device)
        current_hash.update(ids.astype('<i8').tobytes());current_n+=len(ids)
        loss=torch.nn.functional.cross_entropy(local(batch,targets=target),target)
        if memory[client]:
         selection=torch.randperm(len(memory[client]),generator=replay_rng)[:min(plan['batch_size'],len(memory[client]))].tolist()
         ridx=np.array([memory[client][j] for j in selection],dtype=np.int64)
         replay_hash.update(ridx.astype('<i8').tobytes());replay_n+=len(ridx);by_class.update(train_data['y'][ridx].tolist())
         loss=loss+plan['replay_weight']*torch.nn.functional.cross_entropy(local(x[ridx].to(device),targets=y[ridx].to(device)),y[ridx].to(device))
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();n_steps+=1;losses.append(float(loss.detach()))
      state={k:v.detach().cpu().clone() for k,v in local.state_dict().items()};up_bytes=len(pickle.dumps(state,protocol=4))
      updates.append(state);weights.append(len(indices))
      records.append(dict(task=task,round=round_no,client=client,updates=n_steps,current_presentations=current_n,
        replay_presentations=replay_n,replay_class_counts=dict(by_class),current_ids_sha256=current_hash.hexdigest(),
        replay_ids_sha256=replay_hash.hexdigest(),upload_bytes=up_bytes,download_bytes=down_bytes,mean_loss=float(np.mean(losses))))
     total=sum(weights);merged={k:sum(s[k].to(device)*(n/total) for s,n in zip(updates,weights)) for k in updates[0]}
     net.load_state_dict(merged)
     if task==1 and round_no==5:
      save(run/'task1_round5.json',dict(state_sha256=state_hash(net.state_dict()),initial_state_sha256=init_hash,
             calls_sha256=hashlib.sha256(json.dumps([{k:v for k,v in r.items() if k not in ['upload_bytes','download_bytes']} for r in records],sort_keys=True).encode()).hexdigest()))
    for client,indices in enumerate(available):
     combined=memory[client]+indices.tolist();groups={}
     for i in combined:groups.setdefault(int(train_data['y'][i]),[]).append(i)
     quota=plan['replay_capacity']//len(groups);memory[client]=[i for g in groups.values() for i in g[-quota:]]
    stage_dir=run/f'task_{task}';stage_dir.mkdir(exist_ok=True)
    state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};torch.save(state,stage_dir/'model.pth')
    mask=val['task']<=task;logits,p=predict(net,val['x'][mask],device);t=threshold(val['y'][mask],p)
    np.savez_compressed(stage_dir/'validation_predictions.npz',sample_id=val['sample_id'][mask],task=val['task'][mask],y=val['y'][mask],logits=logits,p=p,score=score(p),pred=p.argmax(1))
    _,fitted=predict(net,train_data['x'][train_data['task']<=task],device)
    stage=dict(task=task,threshold=t,validation=evaluate(val['y'][mask],p),validation_alarm=counts(val['y'][mask],p,t),
       validation_normal=int((val['y'][mask]==0).sum()),threshold_comparison='>=',
       train_fit=evaluate(train_data['y'][train_data['task']<=task],fitted),
       cumulative_duplex_bytes=sum(r['upload_bytes']+r['download_bytes'] for r in records),
       model_state_sha256=state_hash(state))
    save(stage_dir/'validation.json',stage);stages.append(stage)
    print('TRAIN',arm,seed,'task',task,'valF1',stage['validation']['macro_f1_percent'],flush=True)
   assert stages[-1]['cumulative_duplex_bytes']<=plan['cap_bytes']
   save(run/'client_round_records.json',records)
   save(run/'validation_complete.json',dict(arm=arm,seed=seed,schedule=schedule,initial_state_sha256=init_hash,
        device=str(device),seconds=time.monotonic()-start,stages=stages,plan_sha256=sha(out/'FROZEN_PLAN.json')))
 paths=[out/'runs'/arm/f'seed_{seed}/validation_complete.json' for seed in SEEDS for arm in SCHEDULES]
 assert all(p.exists() for p in paths)
 equality=[]
 for seed in SEEDS:
  a=read(out/'runs/front'/f'seed_{seed}/task1_round5.json');b=read(out/'runs/late'/f'seed_{seed}/task1_round5.json')
  assert a==b,(seed,'first five rounds diverged')
  equality.append(dict(seed=seed,initial_weights_identical=True,first5_weights_identical=True,first5_calls_identical=True))
 save(out/'VALIDATION_COMPLETE.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),six_runs_complete=True,
       results_sha256={str(p.relative_to(out)):sha(p) for p in paths},paired_first5=equality,test_inference_started=False))


def test(out):
 import torch
 plan=verify(out);torch.set_num_threads(2);done=read(out/'VALIDATION_COMPLETE.json')
 for k,v in done['results_sha256'].items():assert sha(out/k)==v
 test_data=load_data(out,'test');rows=[];fixed=[]
 for seed in SEEDS:
  for arm in SCHEDULES:
   run=out/'runs'/arm/f'seed_{seed}'
   for task in [1,2,3]:
    model=make_model();model.load_state_dict(torch.load(run/f'task_{task}/model.pth',map_location='cpu'));model.eval()
    mask=test_data['task']<=task;logits,p=predict(model,test_data['x'][mask],torch.device('cpu'));v=read(run/f'task_{task}/validation.json');t=v['threshold']
    y=test_data['y'][mask];tasks=test_data['task'][mask]
    np.savez_compressed(run/f'task_{task}/test_predictions.npz',sample_id=test_data['sample_id'][mask],task=tasks,y=y,logits=logits,p=p,score=score(p),pred=p.argmax(1),alarm=score(p)>=t,threshold=t)
    result=dict(arm=arm,seed=seed,checkpoint=task,**evaluate(y,p),alarm=counts(y,p,t),threshold=t,
      cumulative_duplex_bytes=v['cumulative_duplex_bytes'],normal_test=int((y==0).sum()))
    save(run/f'task_{task}/test.json',result);rows.append(result)
    for test_task in range(1,task+1):
     member=tasks==test_task;ev=evaluate(y[member],p[member]);al=counts(y[member],p[member],t)
     fixed.append(dict(arm=arm,seed=seed,checkpoint=task,test_task=test_task,n=int(member.sum()),
         macro_f1_percent=ev['macro_f1_percent'],rbot_correct_class_recall_percent=ev['class_recalls_percent'][1],
         alarm_TPR_percent=al['TPR_percent'],alarm_FPR_percent=al['FPR_percent']))
 save(out/'test_stage_results.json',rows);write(out/'fixed_task_results.csv',fixed)
 main=[];paired=[]
 for seed in SEEDS:
  for arm in SCHEDULES:
   early=next(r for r in rows if r['arm']==arm and r['seed']==seed and r['checkpoint']==1)
   final=next(r for r in rows if r['arm']==arm and r['seed']==seed and r['checkpoint']==3)
   acquisition=next(r for r in fixed if r['arm']==arm and r['seed']==seed and r['checkpoint']==2 and r['test_task']==2)
   records=read(out/'runs'/arm/f'seed_{seed}/client_round_records.json')
   main.append(dict(arm=arm,schedule='/'.join(map(str,SCHEDULES[arm])),seed=seed,
       task1_macro_f1_percent=early['macro_f1_percent'],task1_rbot_correct_recall_percent=early['class_recalls_percent'][1],
       task1_alarm_TPR_percent=early['alarm']['TPR_percent'],task1_alarm_FPR_percent=early['alarm']['FPR_percent'],
       task2_acquisition_f1_percent=acquisition['macro_f1_percent'],final_pooled_f1_percent=final['macro_f1_percent'],
       task1_duplex_bytes=early['cumulative_duplex_bytes'],full_duplex_bytes=final['cumulative_duplex_bytes'],
       current_presentations=sum(r['current_presentations'] for r in records),replay_presentations=sum(r['replay_presentations'] for r in records),
       optimizer_updates=sum(r['updates'] for r in records)))
  a=next(r for r in main if r['seed']==seed and r['arm']=='front');b=next(r for r in main if r['seed']==seed and r['arm']=='late')
  paired.append(dict(seed=seed,direction='9/5/5 minus 5/7/7',**{'delta_'+k:a[k]-b[k] for k in a if k not in ['arm','schedule','seed']}))
 write(out/'main_per_seed.csv',main);write(out/'paired_differences.csv',paired)
 save(out/'TEST_COMPLETE.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),six_runs_complete=True,
     stage_evaluations=len(rows),fixed_task_cells=len(fixed),validation_complete_sha256=sha(out/'VALIDATION_COMPLETE.json')))
 print('TEST COMPLETE',len(rows),'stages',len(fixed),'fixed cells',flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','train','test'])
 parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[3])
 parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--device',default='auto')
 a=parser.parse_args();root=a.project_root.resolve();out=a.output.resolve()
 if a.action=='prepare':prepare(root,out)
 elif a.action=='train':train(out,a.device)
 else:test(out)
