"""Four frozen statistical-only baselines on capture-day held-out flow records.

Binary domain-incremental supplement; no dummy bytes and no claim of original
multimodal FedDIA identity. Test is a separate action after all validation runs.
"""
import argparse,copy,csv,hashlib,json,os,pickle,random,shutil,subprocess,sys,threading,time
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/'supplementary_data/independent_stat_binary_v1_20261110'
METHODS={'Small5':(48,[5,5,5],False),'Small9':(48,[9,5,5],False),'FullReplay':(128,[5,5,5],False),'FedDIAStat':(128,[5,5,5],True)}
SEEDS=(42,43,123)

def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def csvout(p,rows):
 if not rows:return
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
def seed_all(seed):
 import torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def model(width,dim):
 from fcl_models_v4 import V4DualBackbone
 return V4DualBackbone(stat_dim=dim,hidden_dim=width,num_classes=2,use_stat=True,use_seq=False,use_adapter=True,use_se=True,use_cosface=True,use_logit_adj=False,class_counts=[1,1])

def load_data(out,split):
 with np.load(out/'data'/f'{split}.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
 with np.load(out/'data/preprocessing.npz') as f:mu=f['mean'];scale=f['scale']
 raw=a['stat_raw'].astype(np.float64);a['x']=((np.sign(raw)*np.log1p(np.abs(raw))-mu)/scale).astype(np.float32)
 assert np.isfinite(a['x']).all();return a

def prepare(out):
 if (out/'training_plan.json').exists():return
 import torch
 gate=read(out/'data/audit_report.json');assert gate['status']=='PASS_FLOW_RECORD_CAPTURE_DAY_BINARY_STAT_ONLY'
 assert gate['source_groups_disjoint'] and gate['selected_model_vectors_cross_split_disjoint']
 assert all(x['row_recomputed'] for x in gate['independent_csv_recomputations'])
 for name,h in read(out/'data/artifact_manifest.json').items():assert sha(out/'data'/name)==h,name
 dim=read(out/'data/feature_schema.json')['dim'];wide=model(128,dim)
 state={k:v.detach().cpu().clone() for k,v in wide.state_dict().items()};size=len(pickle.dumps(state))
 cap=3*15*(2*size+512)
 plan={'status_scope':'BINARY_STAT_ONLY_CAPTURE_DAY_HELDOUT_SUPPLEMENT','methods':{k:{'width':v[0],'rounds':v[1],'freeze_after_task1':v[2]} for k,v in METHODS.items()},'seeds':list(SEEDS),'lr':.001737670720958496,'weight_decay':1e-4,'batch_size':64,'local_epochs':3,'replay_capacity':240,'replay_weight':.5,'optimizer':'AdamW restarted each client-round','replay':'per binary class latest samples, total240/client; may replace old attack subtype when both binary labels recur','client_ownership':'three simulated clients; sources shared within train, no site independence claim','feature_dim':dim,'cap_bytes':cap,'reference_state_bytes':size,'cap_rule':'Full128 full-state down/up,3 clients15 rounds;512 metadata bytes/client-round reserved','no_hyperparameter_search':True,'test_policy':'Only after all12 validation runs; no test-driven selection or retraining','source_plan_sha256':sha(out/'EXPERIMENT_PLAN.md'),'data_manifest_sha256':sha(out/'data/artifact_manifest.json')}
 code=out/'code';code.mkdir(exist_ok=True)
 for name in ('run_source_stat_core.py','fcl_models_v4.py'):
  shutil.copy2(ROOT/name,code/name)
 shutil.copy2(ROOT/'data_pipeline/build_stat_source_binary.py',code/'build_stat_source_binary.py')
 save(out/'code_manifest.json',{f.name:sha(f) for f in code.iterdir() if f.is_file()})
 save(out/'training_plan.json',plan)
 print('Locked',12,'runs; cap MiB',cap/2**20,flush=True)

def thresholds(y,p):
 from sklearn.metrics import roc_curve
 truth=y==1;score=1-p[:,0];fpr,tpr,t=roc_curve(truth,score,drop_intermediate=False);result=[]
 for target in (.01,.001):
  eligible=np.flatnonzero(fpr<=target);best=eligible[tpr[eligible]==tpr[eligible].max()];i=best[np.argmin(fpr[best])];value=float(t[i])
  result.append({'target_fpr':target,'threshold':value if np.isfinite(value) else None,'threshold_positive_infinity':bool(np.isposinf(value)),'comparison':'>=','score':'1-P(BENIGN)','selection':'validation maxTPR under empirical FPR; ties lower FPR','val_counts':counts(y,p,value)})
 return result

def counts(y,p,threshold=None):
 positive=p.argmax(1)==1 if threshold is None else (1-p[:,0])>=threshold;truth=y==1
 tp=int((truth&positive).sum());fp=int((~truth&positive).sum());tn=int((~truth&~positive).sum());fn=int((truth&~positive).sum())
 return {'TP':tp,'FP':fp,'TN':tn,'FN':fn,'TPR':tp/(tp+fn) if tp+fn else None,'FPR':fp/(fp+tn) if fp+tn else None}

def predict(net,x,device):
 import torch
 net.eval();pieces=[]
 with torch.no_grad():
  for i in range(0,len(x),256):pieces.append(net(torch.as_tensor(x[i:i+256],device=device)).softmax(1).cpu().numpy())
 return np.concatenate(pieces)

def metrics(y,p):
 from sklearn.metrics import f1_score,confusion_matrix
 return {'pooled_macro_f1':100*float(f1_score(y,p.argmax(1),labels=[0,1],average='macro',zero_division=0)),'argmax':counts(y,p),'confusion_matrix':confusion_matrix(y,p.argmax(1),labels=[0,1]).tolist()}

def tensor_hash(h,tensor):
 a=tensor.detach().cpu().contiguous().numpy();h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())

def worker(out,method,seed):
 import torch
 from torch.utils.data import TensorDataset,DataLoader
 assert torch.cuda.is_available(),'CUDA unavailable; refusing silent CPU fallback'
 torch.set_num_threads(2);device=torch.device('cuda:0')
 run=out/'runs'/method/f'seed_{seed}';run.mkdir(parents=True,exist_ok=True)
 if (run/'validation_result.json').exists():return
 plan=read(out/'training_plan.json')
 assert sha(out/'data/artifact_manifest.json')==plan['data_manifest_sha256'] and sha(out/'EXPERIMENT_PLAN.md')==plan['source_plan_sha256']
 for name,h in read(out/'code_manifest.json').items():assert sha(out/'code'/name)==h,name
 # Only train/val inputs loaded during development; test arrays are never read here.
 train=load_data(out,'train');val=load_data(out,'val')
 x=torch.from_numpy(train['x']);y=torch.from_numpy(train['y']);width,rounds,freeze=METHODS[method]
 seed_all(seed);net=model(width,plan['feature_dim']).to(device)
 memory=[[] for _ in range(3)];records=[];stages=[];begin=time.monotonic()
 for task in range(3):
  available=[np.flatnonzero((train['task']==task)&(train['client']==c)) for c in range(3)]
  assert all(len(a)>0 for a in available)
  for roundid in range(1,rounds[task]+1):
   states=[];weights=[];keys=None
   for client,indices in enumerate(available):
    callseed=seed*100000+task*10000+client*100+roundid;seed_all(callseed)
    local=copy.deepcopy(net).train();is_frozen=freeze and task>0
    for name,param in local.named_parameters():param.requires_grad=not is_frozen or 'adapter' in name or 'classifier' in name
    optimizer=torch.optim.AdamW([v for v in local.parameters() if v.requires_grad],lr=plan['lr'],weight_decay=plan['weight_decay'])
    current_generator=torch.Generator().manual_seed(callseed+10000000);replay_generator=torch.Generator().manual_seed(callseed+20000000)
    loader=DataLoader(TensorDataset(x[indices],y[indices]),batch_size=plan['batch_size'],shuffle=True,generator=current_generator)
    down=len(pickle.dumps({k:v.detach().cpu().clone() for k,v in net.state_dict().items()}))
    current_h=hashlib.sha256();replay_h=hashlib.sha256();buffer_h=hashlib.sha256();updates=presentations=replay_samples=0;replay_classes=Counter();losses=[]
    for idx in memory[client]:tensor_hash(buffer_h,x[idx]);tensor_hash(buffer_h,y[idx])
    for epoch in range(plan['local_epochs']):
     for batch,target in loader:
      tensor_hash(current_h,batch);tensor_hash(current_h,target);presentations+=len(target)
      loss=torch.nn.functional.cross_entropy(local(batch.to(device),targets=target.to(device)),target.to(device))
      if memory[client]:
       order=torch.randperm(len(memory[client]),generator=replay_generator)[:min(plan['batch_size'],len(memory[client]))].tolist()
       ridx=[memory[client][i] for i in order];rx=x[ridx];ry=y[ridx];tensor_hash(replay_h,rx);tensor_hash(replay_h,ry)
       loss=loss+plan['replay_weight']*torch.nn.functional.cross_entropy(local(rx.to(device),targets=ry.to(device)),ry.to(device))
       replay_samples+=len(ry);replay_classes.update(ry.tolist())
      assert torch.isfinite(loss),'Non-finite loss'
      optimizer.zero_grad();loss.backward();optimizer.step();updates+=1;losses.append(float(loss.detach()))
    state={k:v.detach().cpu().clone() for k,v in local.state_dict().items()};keys=[k for k in state if not is_frozen or 'adapter' in k or 'classifier' in k]
    up=len(pickle.dumps({k:state[k] for k in keys}));states.append(state);weights.append(len(indices))
    records.append({'task':task+1,'round':roundid,'client':client,'optimizer_updates':updates,'current_presentations':presentations,'replay_samples':replay_samples,'replay_class_presentations':dict(replay_classes),'current_batches_sha256':current_h.hexdigest(),'replay_batches_sha256':replay_h.hexdigest(),'initial_buffer_sha256':buffer_h.hexdigest(),'uplink_bytes':up,'downlink_bytes':down,'backbone_frozen':is_frozen,'mean_joint_loss':float(np.mean(losses))})
   merged=net.state_dict()
   for k in keys:merged[k]=sum(s[k].to(device)*(w/sum(weights)) for s,w in zip(states,weights))
   net.load_state_dict(merged)
  # Original binary-label-balanced latest-sample memory; same across all four arms.
  for client,indices in enumerate(available):
   combined=memory[client]+indices.tolist();by_label={}
   for idx in combined:by_label.setdefault(int(y[idx]),[]).append(idx)
   quota=max(1,plan['replay_capacity']//len(by_label));memory[client]=[idx for values in by_label.values() for idx in values[-quota:]][-plan['replay_capacity']:]
  p=predict(net,val['x'],device);m=metrics(val['y'],p);workpoints=thresholds(val['y'],p)
  stage_dir=run/f'task_{task+1}';stage_dir.mkdir(exist_ok=True);torch.save({k:v.detach().cpu().clone() for k,v in net.state_dict().items()},stage_dir/'model.pth')
  np.savez_compressed(stage_dir/'validation_scores.npz',y=val['y'],p=p)
  duplex=sum(r['uplink_bytes']+r['downlink_bytes'] for r in records)
  stage={'task_end':task+1,'validation':m,'thresholds':workpoints,'cumulative_duplex_bytes':duplex,'optimizer_updates':sum(r['optimizer_updates'] for r in records),'current_presentations':sum(r['current_presentations'] for r in records),'replay_samples':sum(r['replay_samples'] for r in records),'buffer_supports':[{str(label):sum(int(y[i])==label for i in values) for label in (0,1)} for values in memory]}
  stages.append(stage);save(stage_dir/'validation.json',stage)
  print(method,seed,'task',task+1,'val F1',m['pooled_macro_f1'],'MiB',duplex/2**20,flush=True)
 assert stages[-1]['cumulative_duplex_bytes']<=plan['cap_bytes'],'Actual communication cap exceeded'
 save(run/'client_records.json',records);save(run/'validation_result.json',{'method':method,'seed':seed,'width':width,'rounds':rounds,'freeze_after_task1':freeze,'feature_dim':plan['feature_dim'],'parameter_count':sum(p.numel() for p in net.parameters()),'stages':stages,'full_duplex_bytes':stages[-1]['cumulative_duplex_bytes'],'cap_satisfied':True,'wall_seconds':time.monotonic()-begin,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_name':torch.cuda.get_device_name(0),'test_evaluated':False,'data_manifest_sha256':plan['data_manifest_sha256']})

def launch(out):
 prepare(out)
 if (out/'validation_complete.json').exists():
  for relative,h in read(out/'validation_complete.json')['results_sha256'].items():assert sha(out/relative)==h
  for name,h in read(out/'data/artifact_manifest.json').items():assert sha(out/'data'/name)==h
  return
 errors=[]
 def gpu(slot,seeds):
  for seed in seeds:
   for method in METHODS:
    run=out/'runs'/method/f'seed_{seed}';run.mkdir(parents=True,exist_ok=True)
    if (run/'validation_result.json').exists():continue
    with (run/'console.log').open('a') as log:
     print('START',slot,method,seed,flush=True)
     result=subprocess.call([sys.executable,'-B',str(out/'code/run_source_stat_core.py'),'worker','--output',str(out),'--method',method,'--seed',str(seed)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(slot),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
    print('DONE',slot,method,seed,result,flush=True)
    if result:errors.append([method,seed]);return
 threads=[threading.Thread(target=gpu,args=(0,[42,123])),threading.Thread(target=gpu,args=(1,[43]))]
 for t in threads:t.start()
 for t in threads:t.join()
 if errors:raise RuntimeError(errors)
 paths=[out/'runs'/m/f'seed_{s}/validation_result.json' for m in METHODS for s in SEEDS]
 assert all(p.exists() for p in paths)
 save(out/'validation_complete.json',{'all12_complete':True,'results_sha256':{str(p.relative_to(out)):sha(p) for p in paths},'test_evaluated':False,'methods_frozen_before_training':True})

def test(out):
 if (out/'test_complete.json').exists():
  assert read(out/'test_complete.json')['validation_complete_sha256']==sha(out/'validation_complete.json')
  assert all((out/'runs'/m/f'seed_{seed}/test_result.json').exists() for m in METHODS for seed in SEEDS)
  return
 import torch,pandas as pd
 assert read(out/'validation_complete.json')['all12_complete']
 assert torch.cuda.is_available(), 'Test inference requires same GPU backend as validation'
 torch.set_num_threads(2);data=load_data(out,'test');plan=read(out/'training_plan.json')
 prov=pd.read_parquet(out/'data/provenance.parquet');side=prov[prov['split']=='test'].set_index('sample_id').loc[data['sample_id']]
 assert side['source_group_id'].nunique()==1
 rows=[];subtypes=[];low=[];countrows=[]
 complete_hash=sha(out/'validation_complete.json')
 for method in METHODS:
  for seed in SEEDS:
   run=out/'runs'/method/f'seed_{seed}';vr=read(run/'validation_result.json');device=torch.device('cuda:'+vr['gpu']);net=model(vr['width'],plan['feature_dim']).to(device)
   stage_results=[]
   for task in (1,2,3):
    net.load_state_dict(torch.load(run/f'task_{task}/model.pth',map_location='cpu',weights_only=False))
    p=predict(net,data['x'],device);m=metrics(data['y'],p);cal=vr['stages'][task-1]['thresholds'];alarm=[]
    for point in cal:
     t=float('inf') if point['threshold_positive_infinity'] else point['threshold'];c=counts(data['y'],p,t)
     alarm.append({'target_val_fpr':point['target_fpr'],'threshold':point['threshold'],'threshold_positive_infinity':point['threshold_positive_infinity'],'val_counts':point['val_counts'],'test_counts':c})
     countrows.append({'method':method,'seed':seed,'stage':task,'mode':f'val_threshold_{point["target_fpr"]}','source_group':'capture_day:Thursday',**c})
     if task in (1,3):
      for label in sorted(side['original_label'].unique()):
       mask=side['original_label'].to_numpy()==label
       subtypes.append({'method':method,'seed':seed,'stage':task,'target_val_fpr':point['target_fpr'],'original_label':label,'support':int(mask.sum()),'binary_alarm_recall':float(((1-p[mask,0])>=t).mean()),'source_group':'capture_day:Thursday','interpretation':'Alarm recall on original subtype; no subtype prediction head'})
    np.savez_compressed(run/f'task_{task}/test_scores.npz',y=data['y'],p=p,sample_id=data['sample_id'])
    stage_results.append({'task_end':task,'test':m,'alarm':alarm,'cumulative_duplex_bytes':vr['stages'][task-1]['cumulative_duplex_bytes']})
    countrows.append({'method':method,'seed':seed,'stage':task,'mode':'argmax','source_group':'capture_day:Thursday',**m['argmax']})
   early=next(x for x in stage_results[0]['alarm'] if x['target_val_fpr']==.01);end=next(x for x in stage_results[-1]['alarm'] if x['target_val_fpr']==.01)
   attack_sub=[x for x in subtypes if x['method']==method and x['seed']==seed and x['stage']==3 and x['target_val_fpr']==.01 and x['original_label']!='BENIGN'];worst=min(attack_sub,key=lambda x:x['binary_alarm_recall'])
   row={'method':method,'seed':seed,'task1_val_alarm_TPR':early['val_counts']['TPR'],'task1_val_alarm_FPR':early['val_counts']['FPR'],'task1_test_alarm_TPR':early['test_counts']['TPR'],'task1_test_alarm_FPR':early['test_counts']['FPR'],'task1_argmax_TPR':stage_results[0]['test']['argmax']['TPR'],'task1_argmax_FPR':stage_results[0]['test']['argmax']['FPR'],'final_pooled_macro_f1':stage_results[-1]['test']['pooled_macro_f1'],'final_test_alarm_TPR':end['test_counts']['TPR'],'final_test_alarm_FPR':end['test_counts']['FPR'],'weakest_subtype':worst['original_label'],'weakest_subtype_support':worst['support'],'weakest_subtype_alarm_recall':worst['binary_alarm_recall'],'duplex_MiB':vr['full_duplex_bytes']/2**20,'cap_satisfied':vr['cap_satisfied'],'source_group':'capture_day:Thursday'}
   rows.append(row);save(run/'test_result.json',{'summary':row,'stages':stage_results,'thresholds_not_refit_on_test':True,'validation_complete_sha256_before_test':complete_hash})
 csvout(out/'seed_results.csv',rows);csvout(out/'source_subtype_results.csv',subtypes);csvout(out/'stage_counts_by_source.csv',countrows)
 assert sha(out/'validation_complete.json')==complete_hash
 save(out/'test_complete.json',{'all12_complete':True,'validation_manifest_unchanged':True,'validation_complete_sha256':complete_hash,'test_source_groups':1,'model_training_seeds_not_independent_source_repetitions':True})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker','test']);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--method',choices=list(METHODS));p.add_argument('--seed',type=int);a=p.parse_args();out=a.output.absolute()
 {'prepare':lambda:prepare(out),'run':lambda:launch(out),'worker':lambda:worker(out,a.method,a.seed),'test':lambda:test(out)}[a.action]()
