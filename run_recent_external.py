"""Recent FCL adaptations under the frozen traffic representation.

FedCLASS follows the author-code loss (including its gradient-bearing augmented
self-distillation target). Re-Fed retains PIM + per-example gradient importance.
AF-FCL executes the author's flow/replay/teacher routines via an AST extraction
that removes unrelated image imports; its four-layer conditional flow is kept.
"""
import argparse,ast,copy,csv,hashlib,importlib.util,json,math,os,pickle,queue,shutil,subprocess,sys,threading,time
from pathlib import Path

SEEDS=(42,43,123);METHODS=('FedCLASS','ReFed','AFFCL');CAP=37208796

def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvout(p,rows):
 if not rows:return
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
def prepare(out,project,source):
 assert not out.exists(),'fresh output required';out.mkdir(parents=True)
 base=project/'supplementary_data/feddia_method_suite_20261114_v1';shutil.copytree(base/'code',out/'code',ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(__file__,out/'code/run_recent_external.py');shutil.copy2(base/'data_manifests.json',out/'data_manifests.json')
 provenance=[]
 for name in ('FedCLASS','FedCBDR','AF-FCL','NCCL'):
  src=source/name;dst=out/'official'/name;shutil.copytree(src,dst,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
  provenance.append(dict(name=name,commit=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip(),files={str(p.relative_to(dst)):sha(p) for p in dst.rglob('*') if p.is_file()}))
 save(out/'official_sources.json',provenance)
 # The used affine flow never calls UMNN. Make this optional unrelated import
 # conditional rather than installing a dependency for unused transform classes.
 vendor=out/'vendor';shutil.copytree(out/'official/AF-FCL/nflows',vendor/'nflows')
 for name in ('coupling.py','autoregressive.py'):
  p=vendor/'nflows/transforms'/name;s=p.read_text();s=s.replace('from nflows.transforms.UMNN import *','try:\n    from nflows.transforms.UMNN import *\nexcept ImportError:\n    pass  # Optional UMNN transforms are not used by the affine flow.');p.write_text(s)
 commonlr=.001737670720958496
 configs={
  'FedCLASS':[dict(lr=lr,alpha=a,T_old=5.,T_new=3.) for lr in (commonlr,.01) for a in (1.,5.,10.)],
  'ReFed':[dict(lr=lr,pim_lambda=l,pim_epochs=3) for lr in (commonlr,.01) for l in (.2,.5,.8)],
  'AFFCL':[dict(lr=lr,flow_lr=lr,flow_weight=w,flow_hidden=512,flow_layers=4,flow_history_weight=.4,feature_kd=.1,output_kd=.1,teacher_last=1.,teacher_global=1.,explore_theta=.2) for lr,w in ((commonlr,.1),(.001,.1),(.001,1.))]}
 jobs=[]
 for dataset in ('CIC_IDS','USTC','CIC_DOH'):
  for arm in METHODS:
   for index,cfg in enumerate(configs[arm]):jobs.append(dict(id=f'{dataset}_{arm}_cfg{index}_5-5-5_s42',dataset=dataset,arm=arm,seed=42,schedule=[5,5,5],role='parameter_candidate',config=cfg))
 save(out/'plan.json',dict(baseline=str(base),project=str(project),methods=list(METHODS),seeds=list(SEEDS),cap_bytes=CAP,jobs=jobs,configs=configs,
  protocol='unchanged existing diagnostic splits, immutable global vocabulary, current inputs and preprocessing shared; full runs from initialization, no shared Task1',
  selection='seed42 maximum final validation fixed-task-average macro F1; tie lower actual duplex bytes then lexicographic job ID; parameters per dataset, then CIC budget schedule',
  model='width48 original statistical/sequence encoders, SE, fusion norm and residual Adapter; external linear head, AF-FCL adds native fc2; no forced CosFace loss',
  no_new_method_selection_by_test=True,formal_scope='3 methods x 3 seeds x (CIC fixed, CIC budget if feasible, USTC fixed, DoH fixed)',
  candidates='three actual feasible round templates balanced/front/back under full initialization+model+auxiliary+control costs; infeasible minimum 1/1/1 reported without stripping modules',
  recurrence='old = union previous train labels, newly introduced = current union minus old, explicit non-contiguous index sets; empty-new stages retained',
  FedCLASS='author temperature_old5/new3 loss; CE + alpha reverse-KL with non-detached augmented student-derived target, no T^2; first-per-class memory240; train current+memory uniformly; teacher preceding global model',
  ReFed='paper Eqs3-6: task-end PIM initialized from final local model, squared L2 proximal to final global, SGD; sample gradient squared norms via torch.func.vmap, reciprocal epoch weighting over3passes; choose top240 from memory+current once; joint uniform CE; final-global broadcast charged',
  AFFCL='native AST-extracted four-layer affine conditional flow hidden512; likelihood plus last-flow replay, probability-weighted feature replay, old/global feature/output KD; classifier+flow aggregation; all auxiliary messages charged; history model copies retained',
  budget_accounting='pickle protocol default serialized CPU state; first-round initialization reception included once plus every-round upload/download; ReFed task-end classifier broadcast; method class metadata each phase; AF full flow included',
  test_gate='all validation selection and required cross-dataset training complete before any new test inference',
  limitations='adaptations under unified traffic representation, not faithful image-protocol replication; diagnostic split has development history; no claim of reproduced original benchmark scores'))
 save(out/'code_manifest.json',{str(p.relative_to(out/'code')):sha(p) for p in (out/'code').rglob('*.py')})
 save(out/'vendor_manifest.json',{str(p.relative_to(vendor)):sha(p) for p in vendor.rglob('*') if p.is_file()})


def imports(out):
 import torch
 torch.set_num_threads(2)
 sys.path.insert(0,str(out/'code'));sys.path.insert(0,str(out/'vendor'))
 import run_federated_cl_v4 as v4
 from fcl_models_v4 import V4DualBackbone
 return torch,v4,V4DualBackbone

def make_classifier(out,statdim,nclasses,arm):
 torch,v4,V4=imports(out)
 if arm!='AFFCL':return V4(stat_dim=statdim,hidden_dim=48,num_classes=nclasses,use_stat=True,use_seq=True,use_adapter=True,use_se=True,use_cosface=False,use_logit_adj=False)
 class AFClassifier(torch.nn.Module):
  def __init__(self):
   super().__init__();self.encoder=V4(stat_dim=statdim,hidden_dim=48,num_classes=nclasses,use_stat=True,use_seq=True,use_adapter=True,use_se=True,use_cosface=False);self.encoder.classifier=torch.nn.Identity();self.fc2=torch.nn.Linear(96,96);self.fc_classifier=torch.nn.Linear(96,nclasses)
  def forward_to_xa(self,x):return self.encoder.encode(x[:,:statdim],x[:,statdim:])
  def forward_from_xa(self,xa):
   logits=self.fc_classifier(torch.nn.functional.leaky_relu(self.fc2(xa)));return logits.softmax(1),logits
  def forward(self,x,seq=None,targets=None):
   # Packed interface used by native AF methods; dual interface by evaluator.
   if seq is not None:return self.forward_from_xa(self.encoder.encode(x,seq))[1]
   xa=self.forward_to_xa(x);prob,logits=self.forward_from_xa(xa);return prob,xa,logits
 return AFClassifier()

def make_af(out,model,nclasses,cfg):
 torch,v4,V4=imports(out)
 import numpy as np
 from nflows.flows.base import Flow
 from nflows.transforms.permutations import RandomPermutation,ReversePermutation
 from nflows.transforms.base import CompositeTransform
 from nflows.transforms.coupling import AffineCouplingTransform
 from nflows.nn.nets.myresnet import ResidualNet
 from nflows.distributions.normal import StandardNormal
 namespace=dict(torch=torch,nn=torch.nn,np=np,F=torch.nn.functional,Flow=Flow,ReversePermutation=ReversePermutation,RandomPermutation=RandomPermutation,CompositeTransform=CompositeTransform,AffineCouplingTransform=AffineCouplingTransform,ResidualNet=ResidualNet,StandardNormal=StandardNormal,eps=1e-30,myitem=lambda x:x.item() if hasattr(x,'item') else x)
 tree=ast.parse((out/'official/AF-FCL/FLAlgorithms/PreciseFCLNet/model.py').read_text())
 funcs=[]
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and node.name=='MultiClassCrossEntropy':funcs.append(node)
  if isinstance(node,ast.ClassDef) and node.name=='PreciseModel':
   node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('get_1d_nflow_model','sample_from_flow','probability_in_localdata','train_a_batch_classifier','train_a_batch_flow','knowledge_distillation_on_output','knowledge_distillation_on_xa_output')];node.decorator_list=[];funcs.append(node)
 core=ast.unparse(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])))
 # Only numerical guards: keep the original affine flow, trust weighting,
 # replay and both teachers. A zero class-wise variance is possible in traffic
 # minibatches and otherwise produces weights close to 1e30 in float32.
 core=core.replace('xa_u_yi_var + eps','xa_u_yi_var.clamp_min(1e-6)')
 core=core.replace('c_loss_flow.backward()', 'c_loss_flow.backward(); torch.nn.utils.clip_grad_norm_(self.classifier.parameters(),10.,error_if_nonfinite=True)')
 core=core.replace('c_loss.backward()', 'c_loss.backward(); torch.nn.utils.clip_grad_norm_(self.classifier.parameters(),10.,error_if_nonfinite=True)')
 core=core.replace('loss.backward()', 'loss.backward(); torch.nn.utils.clip_grad_norm_(self.flow.parameters(),10.,error_if_nonfinite=True)')
 exec(compile(core,'official_AF_core_with_declared_numeric_guards','exec'),namespace)
 cls=namespace['PreciseModel'];obj=cls();obj.classifier=model;obj.algorithm='PreciseFCL';obj.num_classes=nclasses;obj.xa_shape=[96]
 for k,value in dict(k_loss_flow=cfg['flow_weight'],k_kd_global_cls=cfg['teacher_global'],k_kd_last_cls=cfg['teacher_last'],k_kd_feature=cfg['feature_kd'],k_kd_output=cfg['output_kd'],k_flow_lastflow=cfg['flow_history_weight'],flow_explore_theta=cfg['explore_theta']).items():setattr(obj,k,value)
 obj.classify_criterion=torch.nn.NLLLoss();obj.classify_criterion_noreduce=torch.nn.NLLLoss(reduction='none')
 obj.flow=obj.get_1d_nflow_model(96,cfg['flow_hidden'],nclasses,cfg['flow_layers']).cuda()
 return obj

def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def aggregate_flow_state(model,client_states,weights):
 import torch
 current=state(model);parameter_keys={k for k,_ in model.named_parameters()};total=sum(weights)
 result={}
 for key,value in current.items():
  if key in parameter_keys:result[key]=sum(local[key]*(n/total) for local,n in zip(client_states,weights))
  else:
   assert all(torch.equal(value,local[key]) for local in client_states),('flow buffer mutated',key)
   result[key]=value.clone()
  if 'permutation' in key:
   assert torch.equal(torch.sort(result[key]).values,torch.arange(result[key].numel())),('invalid flow permutation',key)
 return result

def size(x):return len(pickle.dumps(x))
def pack_batch(batch,device):return [x.to(device) for x in batch]
def fedclass_loss(logits,teacher,old,seen,cfg):
 import torch
 F=torch.nn.functional
 p=F.softmax(logits[:,seen]/cfg['T_new'],dim=1);z=F.softmax(logits[:,seen]/cfg['T_old'],dim=1)
 oldpos=[seen.index(k) for k in old];newpos=[i for i,k in enumerate(seen) if k not in old]
 q=F.softmax(teacher[:,old]/cfg['T_old'],dim=1).detach();ratio=1-p[:,newpos].sum(1,keepdim=True)
 target=z.index_copy(1,torch.tensor(oldpos,device=logits.device),q*ratio)
 return cfg['alpha']*(p*torch.log(torch.clamp(1e-8+p/(1e-9+target),min=1e-9,max=1e3))).sum(1).mean()

def worker(out,j):
 torch,v4,V4=imports(out);import numpy as np
 from torch.utils.data import DataLoader,TensorDataset
 from torch.func import functional_call,grad,vmap
 assert torch.cuda.is_available(),'GPU required'
 for f,h in read(out/'code_manifest.json').items():assert sha(out/'code'/f)==h,f
 for f,h in read(out/'vendor_manifest.json').items():assert sha(out/'vendor'/f)==h,f
 manifest=read(out/'data_manifests.json')[j['dataset']];assert sha(manifest['data_path'])==manifest['data_sha256'];data=pickle.loads(Path(manifest['data_path']).read_bytes());labels={n:i for i,n in enumerate(manifest['taxonomy'])}
 run=out/'runs'/j['id'];run.mkdir(parents=True,exist_ok=True)
 if (run/'training_result.json').exists():return
 v4.set_seed(j['seed']);cfg=j['config'];device='cuda:0';arm=j['arm'];statdim=len(v4.normalize_stat_feature(v4._split(data,0,0,'train')[0]['stat_raw']));net=make_classifier(out,statdim,len(labels),arm).cuda();af=make_af(out,net,len(labels),cfg) if arm=='AFFCL' else None
 clients=[dict(buffer=[],teacher=None,last_flow=None) for _ in range(3)];calls=[];messages=[];history=[];buffers=[];valtraj=[];past=set();begin=time.monotonic()
 def message(task,client,direction,kind,obj,round=0):
  messages.append(dict(task=task,client=client,round=round,direction=direction,kind=kind,serialized_bytes=size(obj)))
 # Server initialization is not an additional message: the first actual
 # model/flow reception is included in round1 download exactly once.
 def frozen(model):
  x=copy.deepcopy(model).eval()
  for p in x.parameters():p.requires_grad_(False)
  return x
 def evaluate_val(stage):
  net.eval();values=[]
  for task in range(stage+1):
   loader=v4.make_global_loader(data,task,'val',labels,64,True);ys=[];ps=[]
   with torch.no_grad():
    for stat,seq,y in loader:ys.extend(y.tolist());ps.extend(net(stat.cuda(),seq.cuda()).argmax(1).cpu().tolist())
   from sklearn.metrics import f1_score
   value=100*f1_score(ys,ps,labels=np.unique(ys),average='macro',zero_division=0);values.append(value);valtraj.append(dict(stage=stage+1,task=task+1,macro_f1_percent=value))
  return float(np.mean(values))
 for task,rounds in enumerate(j['schedule']):
  current={labels[s['label']] for c in range(3) for s in v4._split(data,c,task,'train')};seen=sorted(past|current);old=sorted(past);new=sorted(current-past)
  for c in range(3):
   message(task+1,c,'download','class_control_metadata',dict(old=old,new=new,seen=seen))
   if task>0 and arm in ('FedCLASS','AFFCL'):
    clients[c]['teacher']=frozen(net)
    (run/'teachers').mkdir(exist_ok=True);torch.save(dict(state=state(net),old_class_ids=old,source='preceding-stage final global'),run/f'teachers/task{task+1}_client{c}.pth')
    if af:clients[c]['last_flow']=frozen(af.flow)
  for rnd in range(rounds):
   weights=[];flowweights=[];sizes=[];lastlocals=[]
   for c,client in enumerate(clients):
    callseed=j['seed']*100000+task*10000+c*100+rnd+1;v4.set_seed(callseed)
    local=copy.deepcopy(net).cuda().train();message(task+1,c,'download','classifier',state(net),rnd+1)
    if af:message(task+1,c,'download','flow',state(af.flow),rnd+1)
    current_samples=v4._split(data,c,task,'train')
    tensorized=[(torch.tensor(v4.normalize_stat_feature(s['stat_raw']),dtype=torch.float32),torch.tensor(v4.normalize_seq_feature(s['seq_feature']),dtype=torch.float32),labels[s['label']]) for s in current_samples]
    merged=tensorized+client['buffer'] if arm!='AFFCL' else tensorized
    loader=DataLoader(merged,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(callseed+10000000))
    optimizer=torch.optim.SGD(local.parameters(),lr=cfg['lr'],momentum=.9,weight_decay=1e-5) if arm=='FedCLASS' else torch.optim.Adam(local.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    updates=0;flowupdates=0;headupdates=0;seenpresent=0;replaypresent=0;distillpresent=0;ch=hashlib.sha256();losses=[]
    if af:
     localaf=make_af(out,local,len(labels),cfg);localaf.flow.load_state_dict(af.flow.state_dict());localaf.classifier_optimizer=torch.optim.Adam(local.parameters(),lr=cfg['lr'],weight_decay=1e-4)
     localaf.classifier_fb_optimizer=torch.optim.Adam([p for name,p in local.named_parameters() if 'fc2' in name],lr=cfg['lr'],weight_decay=1e-4);localaf.flow_optimizer=torch.optim.Adam(localaf.flow.parameters(),lr=cfg['flow_lr'],weight_decay=1e-4)
     globalteacher=frozen(net)
    # Loader holds original and memory tensors once per epoch, as external
    # reference implementations do. This differs from M0's weighted side loss.
    for epoch in range(3):
     for stat,seq,y in loader:
      for a in (stat,seq,y):ch.update(a.contiguous().numpy().tobytes())
      stat,seq,y=stat.cuda(),seq.cuda(),y.cuda();seenpresent+=len(y)
      if af:
       x=torch.cat((stat,seq),1);local.eval();localaf.flow.train()
       result=localaf.train_a_batch_flow(x,y,client['last_flow'],seen,old);flowupdates+=1
       local.train();localaf.flow.eval()
       result=localaf.train_a_batch_classifier(x,y,localaf.flow,client['teacher'],globalteacher,old,seen);headupdates+=1;replaypresent+=len(y);updates+=1;losses.append(float(result['c_loss']))
      else:
       logits=local(stat,seq);loss=torch.nn.functional.cross_entropy(logits,y)
       if arm=='FedCLASS' and task>0:
        with torch.no_grad():teacher=client['teacher'](stat,seq)
        loss=loss+fedclass_loss(logits,teacher,old,seen,cfg);distillpresent+=len(y)
       assert torch.isfinite(loss),'nonfinite training loss'
       optimizer.zero_grad();loss.backward();optimizer.step();updates+=1;losses.append(float(loss.detach()))
    if arm!='AFFCL':replaypresent=3*len(client['buffer'])
    currentpresent=3*len(tensorized);assert seenpresent==3*len(merged)
    weights.append(state(local));sizes.append(len(merged));lastlocals.append(copy.deepcopy(local).cpu())
    message(task+1,c,'upload','classifier',weights[-1],rnd+1)
    if af:
     flowweights.append(state(localaf.flow));message(task+1,c,'upload','flow',flowweights[-1],rnd+1);del localaf,globalteacher
    assert all(torch.isfinite(v).all() for v in weights[-1].values()),'nonfinite classifier state'
    if af:assert all(torch.isfinite(v).all() for v in flowweights[-1].values()),'nonfinite flow state'
    calls.append(dict(task=task+1,client=c,round=rnd+1,current_presentations=currentpresent,raw_replay_presentations=replaypresent if arm!='AFFCL' else 0,generated_classifier_presentations=replaypresent if af else 0,generated_flow_history_presentations=currentpresent if af and task>0 else 0,distillation_presentations=distillpresent,main_optimizer_updates=updates,head_optimizer_updates=headupdates,flow_optimizer_updates=flowupdates,batches_sha256=ch.hexdigest(),mean_loss=float(np.mean(losses))))
    del local
   total=sum(sizes)
   net.load_state_dict({k:sum(w[k]*(n/total) for w,n in zip(weights,sizes)) for k in weights[0]})
   if af:af.flow.load_state_dict(aggregate_flow_state(af.flow,flowweights,sizes))
   save(run/'progress.json',dict(task=task+1,round=rnd+1,calls=len(calls),test_accessed=False))
  (run/'task_checkpoints').mkdir(exist_ok=True);torch.save(state(net),run/f'task_checkpoints/task_{task+1}.pth')
  if af:
   (run/'flow_checkpoints').mkdir(exist_ok=True);torch.save(state(af.flow),run/f'flow_checkpoints/task_{task+1}.pth')
  score=evaluate_val(task)
  for c,client in enumerate(clients):
   samples=v4._split(data,c,task,'train');currentitems=[(torch.tensor(v4.normalize_stat_feature(s['stat_raw']),dtype=torch.float32),torch.tensor(v4.normalize_seq_feature(s['seq_feature']),dtype=torch.float32),labels[s['label']]) for s in samples]
   pimbytes=0;pimupdates=0;gradient_examples=0
   if arm=='FedCLASS':
    pool=client['buffer']+currentitems;classes=sorted({x[2] for x in pool});quota=240//len(classes);client['buffer']=[x for label in classes for x in [v for v in pool if v[2]==label][:quota]]
   elif arm=='ReFed':
    pool=client['buffer']+currentitems;message(task+1,c,'download','PIM_global_anchor',state(net));pim=lastlocals[c].cuda().train();anchor={k:v.detach().clone() for k,v in net.named_parameters()};scores=np.zeros(len(pool));pimoptimizer=torch.optim.SGD(pim.parameters(),lr=cfg['lr']);q=(1-cfg['pim_lambda'])/(2*cfg['pim_lambda'])
    # Per-example gradients of the actual full PIM, not just last-layer norms.
    def loss_one(params,buffs,stat,seq,y):return torch.nn.functional.cross_entropy(functional_call(pim,(params,buffs),(stat.unsqueeze(0),seq.unsqueeze(0))),y.unsqueeze(0))
    gradfunc=vmap(grad(loss_one),in_dims=(None,None,0,0,0),randomness='error')
    for epoch in range(cfg['pim_epochs']):
     for offset in range(0,len(pool),32):
      part=pool[offset:offset+32];stat=torch.stack([x[0] for x in part]).cuda();seq=torch.stack([x[1] for x in part]).cuda();y=torch.tensor([x[2] for x in part],device=device)
      grads=gradfunc(dict(pim.named_parameters()),dict(pim.named_buffers()),stat,seq,y);norm=sum(g.detach().reshape(len(y),-1).square().sum(1) for g in grads.values());scores[offset:offset+len(part)]+=norm.cpu().numpy()/(epoch+1);gradient_examples+=len(y)
      loss=torch.nn.functional.cross_entropy(pim(stat,seq),y)+q/2*sum((p-anchor[k]).square().sum() for k,p in pim.named_parameters());pimoptimizer.zero_grad();loss.backward();pimoptimizer.step();pimupdates+=1
    assert np.isfinite(scores).all();chosen=sorted(range(len(pool)),key=lambda i:(-scores[i],i))[:240];client['buffer']=[pool[i] for i in chosen];pimbytes=size(state(pim));(run/'selection').mkdir(exist_ok=True);np.savez_compressed(run/f'selection/task{task+1}_client{c}.npz',score=scores,selected=np.asarray(chosen),labels=np.asarray([x[2] for x in pool]));del pim,anchor,grads
   counts={labels_:sum(x[2]==idx for x in client['buffer']) for labels_,idx in labels.items()}
   buffers.append(dict(task=task+1,client=c,counts=counts,capacity=240,actual_records=len(client['buffer']),buffer_serialized_bytes=size(client['buffer'])))
   (run/'buffers').mkdir(exist_ok=True);torch.save(client['buffer'],run/f'buffers/task{task+1}_client{c}.pth')
   teacherbytes=size(state(client['teacher'])) if client['teacher'] else 0;flowbytes=size(state(af.flow)) if af else 0;oldflowbytes=size(state(client['last_flow'])) if client['last_flow'] else 0
   history.append(dict(task=task+1,client=c,raw_buffer_bytes=size(client['buffer']) if client['buffer'] else 0,teacher_bytes=teacherbytes,flow_bytes=flowbytes,last_flow_bytes=oldflowbytes,PIM_transient_bytes=pimbytes,PIM_retained_bytes=0,PIM_optimizer_updates=pimupdates,importance_gradient_examples=gradient_examples,retained_history_bytes=(size(client['buffer']) if client['buffer'] else 0)+teacherbytes+flowbytes+oldflowbytes))
  past|=current
 for c in clients:
  c['teacher']=None;c['last_flow']=None
 assert all(torch.isfinite(v).all() for v in state(net).values())
 torch.save(state(net),run/'final_model.pth');save(run/'client_records.json',calls);save(run/'communication_messages.json',messages);save(run/'history_state.json',history);save(run/'buffer_distribution.json',buffers);save(run/'validation_trajectories.json',valtraj)
 up=sum(x['serialized_bytes'] for x in messages if x['direction']=='upload');down=sum(x['serialized_bytes'] for x in messages if x['direction']=='download')
 save(run/'training_result.json',dict(job=j,metrics=dict(val_macro_f1=score),duplex_bytes=up+down,upload_bytes=up,download_bytes=down,parameter_count=sum(p.numel() for p in net.parameters()),auxiliary_parameter_count=sum(p.numel() for p in af.flow.parameters()) if af else 0,label_map=labels,data_sha256=manifest['data_sha256'],code_sha256=sha(out/'code/run_recent_external.py'),seconds=time.monotonic()-begin,status='VALIDATION_COMPLETE_NO_NEW_TEST_INFERENCE'))
 print('TRAINED',j['id'],score,up+down,flush=True)


def batch(out,jobs):
 q=queue.Queue();fail=[]
 for j in jobs:
  run=out/'runs'/j['id'];run.mkdir(parents=True,exist_ok=True);save(run/'job.json',j)
  if not (run/'training_result.json').exists():q.put(j)
 def consume(gpu):
  while True:
   try:j=q.get_nowait()
   except queue.Empty:return
   run=out/'runs'/j['id'];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
   print('START',gpu,j['id'],flush=True)
   with (run/'console.log').open('a') as log:rc=subprocess.call([sys.executable,'-B',str(out/'code/run_recent_external.py'),'worker','--output',str(out),'--job',str(run/'job.json')],cwd=out/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
   print('DONE',gpu,j['id'],rc,flush=True)
   if rc:fail.append(j['id'])
   save(out/'execution_status.json',dict(status='FAILED' if fail else 'VALIDATION_TRAINING',completed=len(list((out/'runs').glob('*/training_result.json'))),pending=q.qsize(),failures=fail))
 ts=[threading.Thread(target=consume,args=(gpu,)) for gpu in (0,1)]
 for t in ts:t.start()
 for t in ts:t.join()
 assert not fail,fail

def choose(out,jobs):
 rows=[(read(out/'runs'/j['id']/'training_result.json'),j) for j in jobs]
 return min(rows,key=lambda x:(-x[0]['metrics']['val_macro_f1'],x[0]['duplex_bytes'],x[1]['id']))[1]

def run(out):
 plan=read(out/'plan.json');dev=plan['jobs'];batch(out,dev);selected=[];params={}
 for dataset in ('CIC_IDS','USTC','CIC_DOH'):
  for arm in METHODS:
   winner=choose(out,[j for j in dev if j['dataset']==dataset and j['arm']==arm]);params[dataset+'_'+arm]=winner;selected.append(dict(winner,role='fixed'))
   selected.extend(dict(winner,id=f'{dataset}_{arm}_fixed_s{s}',seed=s,role='fixed') for s in (43,123))
 save(out/'parameter_selection.json',params)
 # Costs are derived from observed message sizes in the frozen fixed run.
 # Bound small pickle variations conservatively, including initialization/control.
 candidates=[];infeasible=[];budgetcost=[]
 for arm in METHODS:
  w=params['CIC_IDS_'+arm];messages=read(out/'runs'/w['id']/'communication_messages.json');variable=[m for m in messages if m['round']>0];fixed=[m for m in messages if m['round']==0];fixed_cost=sum(m['serialized_bytes'] for m in fixed)+4096
  perround=max(sum(m['serialized_bytes'] for m in variable if m['task']==t and m['round']==r) for t in (1,2,3) for r in (1,2,3,4,5))+4096
  budgetcost.append(dict(arm=arm,fixed_message_bound=fixed_cost,per_round_bound=perround,minimum_1_1_1_bound=fixed_cost+3*perround,cap_bytes=CAP))
  schedules=[]
  for name,template in [('balanced',(1,1,1)),('front',(2,1,1)),('back',(1,1,2))]:
   n=1;best=None
   while fixed_cost+perround*sum(k*n for k in template)<=CAP:best=[k*n for k in template];n+=1
   if best is not None:schedules.append((name,best))
  if not schedules:
   infeasible.append(dict(arm=arm,reason='necessary full auxiliary-model traffic exceeds cap even at one round per task',**budgetcost[-1]));continue
  # If only 1/1/1 fits, it is the sole feasible schedule; do not invent others.
  unique={tuple(schedule):name for name,schedule in schedules}
  if len(unique)<3:
   # Enumerate nearby permutations when rounding templates yields duplicates.
   maxround=(CAP-fixed_cost)//perround
   for schedule in [(2,1,1),(1,2,1),(1,1,2),(1,1,1)]:
    if sum(schedule)<=maxround and len(unique)<3:unique.setdefault(tuple(schedule),'nearby_feasible')
  for schedule,name in unique.items():candidates.append(dict(w,id=f'CIC_IDS_{arm}_budget_'+ '-'.join(map(str,schedule))+'_s42',schedule=list(schedule),role='budget_candidate',policy=name))
 save(out/'budget_feasibility.json',dict(costs=budgetcost,infeasible=infeasible,candidates=candidates));batch(out,candidates)
 for arm in METHODS:
  group=[j for j in candidates if j['arm']==arm]
  if not group:continue
  group=[j for j in group if read(out/'runs'/j['id']/'training_result.json')['duplex_bytes']<=CAP];assert group,'cost bound violated'
  winner=choose(out,group);selected.append(dict(winner,role='budget_final'));selected.extend(dict(winner,id=f'CIC_IDS_{arm}_budget_final_s{s}',seed=s,role='budget_final') for s in (43,123))
 save(out/'selected_runs.json',selected);save(out/'selection.json',dict(parameters=params,budget=[j for j in selected if j['role']=='budget_final'],rule=plan['selection']))
 batch(out,selected)
 for j in selected:
  result=read(out/'runs'/j['id']/'training_result.json');assert j['role']!='budget_final' or result['duplex_bytes']<=CAP
 save(out/'TEST_GATE.json',dict(status='OPEN',all_training_and_selection_complete=True,selected_runs_sha256=sha(out/'selected_runs.json'),selection_sha256=sha(out/'selection.json')))
 for dataset in ('CIC_IDS','USTC','CIC_DOH'):
  print('EVALUATE',dataset,flush=True);evaluate(out,dataset)
 save(out/'execution_status.json',dict(status='ALL_TRAINING_AND_EVALUATION_COMPLETE_REPORT_PENDING',formal_evaluations=len(selected),training_jobs=len(list((out/'runs').glob('*/training_result.json'))),infeasible_budget_methods=[x['arm'] for x in infeasible]))

def evaluate(out,dataset):
 torch,v4,V4=imports(out);import run_method_suite as base
 for arm in METHODS:base.ARMS[arm]=dict(width=48,freeze=False,replay=False)
 def setup(v4,arm):
  # evaluate() sets the chosen arm but build(args) lacks that name; inspect arm
  # definitions carrying a unique explicit method marker.
  name=arm['method'];manifest=read(out/'data_manifests.json')[dataset];data=pickle.loads(Path(manifest['data_path']).read_bytes());dim=len(v4.normalize_stat_feature(v4._split(data,0,0,'train')[0]['stat_raw']))
  v4.build_model=lambda args,labels,counts,components,device:make_classifier(out,dim,len(labels),name).to(device)
 for arm in METHODS:base.ARMS[arm]['method']=arm
 base.model_setup=setup;base.evaluate(out,dataset)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','run','evaluate']);p.add_argument('--output',type=Path,required=True);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--sources',type=Path,default=Path('/tmp/feddia_recent_external_20261114'));p.add_argument('--job',type=Path);p.add_argument('--dataset',default='CIC_IDS');a=p.parse_args();out=a.output.resolve()
 if a.action=='prepare':prepare(out,a.project.resolve(),a.sources)
 elif a.action=='worker':worker(out,read(a.job))
 elif a.action=='evaluate':evaluate(out,a.dataset)
 else:run(out)
