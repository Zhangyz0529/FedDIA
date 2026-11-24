"""Same Protocol C populations, with a disjoint source-within-capture diagnosis.

This is a post-development extension, not a new independent confirmation.
The existing FULL+ER runs and frozen preprocessing remain immutable.
"""
import argparse,copy,hashlib,importlib.util,json,pickle,shutil,sys,textwrap,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def csvout(p,rows):
    import csv
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def setup(project):
    origin=project/'supplementary_data/independent_schedule_20261113_v1'
    sys.path.insert(0,str(origin/'code'))
    spec=importlib.util.spec_from_file_location('original_protocol_C',origin/'code/independent_schedule.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(2)
    return origin,module


def model(original,arm):
    if arm!='PLAIN':return original.make_model()
    from fcl_models_v4 import V4DualBackbone
    return V4DualBackbone(stat_dim=4,hidden_dim=48,num_classes=2,use_stat=True,use_seq=False,
        use_adapter=False,use_se=False,use_cosface=False,use_logit_adj=False,class_counts=[1,1])


def load(origin,out,split):
    p=(out/'data/source_unused.npz') if split=='source_unused' else origin/f'data/{split}.npz'
    with np.load(p) as a:return {k:a[k] for k in a.files}


def prepare(project,out):
    origin,original=setup(project)
    if (out/'DATA_ACCEPTANCE.json').exists():return
    assert not (out/'runs').exists()
    out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True);(out/'code').mkdir(exist_ok=True)
    original.verify(origin)
    save(out/'PLAN.json',dict(protocol='unchanged Protocol C Rbot binary statistics capture holdout',
        schedule=[5,7,7],seeds=[42,43,123],methods=['FULL','FROZEN','PLAIN','LOGISTIC_REFERENCE'],
        preprocessing='reuse frozen original Task1-training mean/SD; no refit',
        hyperparameters='NN unchanged original AdamW lr 0.001737670720958496, 3 local epochs, batch64, ER240 weight0.5',
        freeze='after shared Task1: freeze stat_encoder, SE and fusion_norm; train Adapter and classifier',
        plain='same width48 statistical encoder and fusion normalization; no SE/Adapter/CosFace; linear head',
        full='reuse three original late-schedule runs',frozen='reuse each same-seed original Task1; continue Tasks2/3',
        logistic_C_candidates=[.1,1.,10.],logistic_selection='seed-independent final validation pooled macroF1; tie smaller C',
        logistic_scope='centralized cumulative-seen training reference, not a federated continual method or communication competitor',
        source_unused='200 records per task/class from unchanged eligible chronological thirds; rank by SHA256 20261121; exclude all existing transformed inputs',
        source_unused_not_used_for_selection=True,new_test_policy='after all NN training and logistic validation selection',
        independent_confirmation=False,source_roles={'source_train':44,'source_unused':44,'validation':45,'test':51}))
    # Re-execute ONLY the original read/filter/quarantine segment. It contains
    # no writes to the old archive and reproduces the eligible source universe.
    text=(origin/'code/independent_schedule.py').read_text()
    begin=text.index(" prior=root/'supplementary_data/e4_source_acquisition_20261111_v1'")
    end=text.index(' selected=[];support=[]',begin)
    namespace=dict(vars(original),root=project)
    exec(compile(textwrap.dedent(text[begin:end]),'original_read_filter_quarantine','exec'),namespace)
    kept=namespace['kept']
    known=[load(origin,out,s) for s in ('train','val','test')]
    forbidden={x.tobytes() for a in known for x in a['x']}
    selected_ids={i for a in known for i in a['sample_id']}
    with np.load(origin/'data/preprocessing.npz') as a:mu=a['mean'];scale=a['scale']
    records=[];support=[]
    expected=pd.read_csv(origin/'data/candidate_support.csv')
    for label in (0,1):
        cohort=kept.loc[(kept.capture==44)&(kept.label==label)].sort_values(['time','row_1based'])
        for task,indices in enumerate(np.array_split(np.arange(len(cohort)),3),1):
            part=cohort.iloc[indices].copy()
            n=int(expected.loc[(expected.split=='train')&(expected.task==task)&(expected.label==label),'eligible_before_sampling'].iloc[0])
            assert len(part)==n
            original_ids={f'cap44:row{int(r)}' for r in part.row_1based}
            train=known[0];assert set(train['sample_id'][(train['task']==task)&(train['y']==label)])<=original_ids
            part['rank']=part.row_1based.map(lambda r:hashlib.sha256(f'20261121|44|{r}'.encode()).hexdigest())
            chosen=[]
            for row in part.sort_values('rank').itertuples():
                sid=f'cap44:row{row.row_1based}'
                if sid in selected_ids:continue
                raw=np.array([getattr(row,k) for k in original.FEATURES],dtype=np.float64)
                x=((np.log1p(raw)-mu)/scale).astype(np.float32)
                if x.tobytes() in forbidden:continue
                forbidden.add(x.tobytes());chosen.append(dict(sample_id=sid,task=task,y=label,raw=raw,x=x))
                if len(chosen)==200:break
            assert len(chosen)==200
            records.extend(chosen);support.append(dict(task=task,label=label,n=len(chosen),capture=44))
    np.savez_compressed(out/'data/source_unused.npz',x=np.stack([r['x'] for r in records]),
        raw=np.stack([r['raw'] for r in records]),y=np.array([r['y'] for r in records]),
        task=np.array([r['task'] for r in records]),sample_id=np.array([r['sample_id'] for r in records]))
    csvout(out/'data/source_unused_support.csv',support)
    shutil.copy2(__file__,out/'code/run_protocolC_bridge.py')
    shutil.copy2(origin/'code/fcl_models_v4.py',out/'code/fcl_models_v4.py')
    save(out/'DATA_ACCEPTANCE.json',dict(status='PASS',n_source_unused=len(records),
        original_eligible_cohort_counts_reproduced=True,unchanged_task_boundaries=True,
        source_unused_IDs_and_float32_vectors_disjoint=True,no_model_scores_used=True,
        original_plan_sha256=sha(origin/'FROZEN_PLAN.json'),
        original_input_hashes={s:sha(origin/f'data/{s}.npz') for s in ('train','val','test')},
        original_source_hashes=namespace['references'],new_source_unused_sha256=sha(out/'data/source_unused.npz')))
    print('SOURCE_WITHIN_CAPTURE_READY',len(records),flush=True)


def train(project,out):
    origin,original=setup(project)
    train=load(origin,out,'train');val=load(origin,out,'val');x=torch.from_numpy(train['x']);y=torch.from_numpy(train['y'])
    for arm in ('FULL','FROZEN','PLAIN'):
        for seed in (42,43,123):
            run=out/'runs'/f'{arm}_s{seed}';run.mkdir(parents=True,exist_ok=True)
            if (run/'training_complete.json').exists():continue
            assert not list(run.glob('task_*.pth')),'Preserve an incomplete attempt separately before resuming'
            oldrun=origin/'runs/late'/f'seed_{seed}'
            if arm=='FULL':
                for task in (1,2,3):shutil.copy2(oldrun/f'task_{task}/model.pth',run/f'task_{task}.pth')
                save(run/'training_complete.json',dict(arm=arm,seed=seed,reused=True,
                    source=str(oldrun),cumulative_duplex_bytes=read(oldrun/'validation_complete.json')['stages'][-1]['cumulative_duplex_bytes']))
                continue
            original.seed_all(seed);net=model(original,arm);memory=[[] for _ in range(3)];calls=[];beg=time.monotonic()
            for task,rounds in enumerate([5,7,7],1):
                current=[np.flatnonzero((train['task']==task)&(train['client']==c)) for c in range(3)]
                if arm=='FROZEN' and task==1:
                    net.load_state_dict(torch.load(oldrun/'task_1/model.pth',map_location='cpu'))
                    oldcalls=read(oldrun/'client_round_records.json')
                    calls.extend(dict(r,shared_prefix=True) for r in oldcalls if r['task']==1)
                else:
                    if arm=='FROZEN':
                        for name,p in net.named_parameters():p.requires_grad_(name.startswith(('adapter.','classifier.')))
                    for rnd in range(1,rounds+1):
                        states=[];sizes=[]
                        for client,indices in enumerate(current):
                            callseed=seed*100000+task*10000+client*100+rnd;original.seed_all(callseed)
                            local=copy.deepcopy(net).train()
                            optimizer=torch.optim.AdamW([p for p in local.parameters() if p.requires_grad],lr=.001737670720958496,weight_decay=1e-4)
                            rng=torch.Generator().manual_seed(callseed+10000000)
                            rrng=torch.Generator().manual_seed(callseed+20000000)
                            down={k:v.detach().clone() for k,v in net.state_dict().items()}
                            steps=now=replay=0;batchhash=hashlib.sha256()
                            for epoch in range(3):
                                order=torch.randperm(len(indices),generator=rng).numpy()
                                for offset in range(0,len(order),64):
                                    ids=indices[order[offset:offset+64]];batchhash.update(ids.astype('<i8').tobytes())
                                    target=y[ids];loss=torch.nn.functional.cross_entropy(local(x[ids],targets=target),target);now+=len(ids)
                                    if memory[client]:
                                        selection=torch.randperm(len(memory[client]),generator=rrng)[:min(64,len(memory[client]))].tolist()
                                        ids2=np.array([memory[client][j] for j in selection],dtype=np.int64);replay+=len(ids2)
                                        loss+=.5*torch.nn.functional.cross_entropy(local(x[ids2],targets=y[ids2]),y[ids2])
                                    assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();optimizer.step();steps+=1
                            state={k:v.detach().clone() for k,v in local.state_dict().items()};states.append(state);sizes.append(len(indices))
                            calls.append(dict(task=task,round=rnd,client=client,current_presentations=now,replay_presentations=replay,
                                updates=steps,current_ids_sha256=batchhash.hexdigest(),upload_bytes=len(pickle.dumps(state,protocol=4)),download_bytes=len(pickle.dumps(down,protocol=4))))
                        merged={k:(net.state_dict()[k].detach().clone() if arm=='FROZEN' and task>1 and k.startswith(('stat_encoder.','se.','fusion_norm.')) else sum(s[k]*(n/sum(sizes)) for s,n in zip(states,sizes))) for k in states[0]}
                        net.load_state_dict(merged)
                for client,indices in enumerate(current):
                    pool=memory[client]+indices.tolist();classes={}
                    for i in pool:classes.setdefault(int(train['y'][i]),[]).append(i)
                    memory[client]=[i for group in classes.values() for i in group[-(240//len(classes)):]]
                torch.save({k:v.detach().clone() for k,v in net.state_dict().items()},run/f'task_{task}.pth')
                np.savez_compressed(run/f'buffer_task{task}.npz',**{f'client{c}':np.array(v) for c,v in enumerate(memory)})
                _,p=original.predict(net,val['x'][val['task']<=task],torch.device('cpu'))
                save(run/f'validation_task{task}.json',original.evaluate(val['y'][val['task']<=task],p))
                print('BRIDGE_TRAIN',arm,seed,task,flush=True)
            if arm=='FROZEN':
                first=torch.load(run/'task_1.pth',map_location='cpu')
                for task in (2,3):
                    later=torch.load(run/f'task_{task}.pth',map_location='cpu')
                    assert all(torch.equal(v,later[k]) for k,v in first.items() if k.startswith(('stat_encoder.','se.','fusion_norm.')))
            csvout(run/'client_round_records.csv',calls)
            save(run/'training_complete.json',dict(arm=arm,seed=seed,reused=False,
                shared_Task1=arm=='FROZEN',seconds=time.monotonic()-beg,
                cumulative_duplex_bytes=sum(r['upload_bytes']+r['download_bytes'] for r in calls)))
    candidates=[]
    for c in (.1,1.,10.):
        for task in (1,2,3):
            fitted=LogisticRegression(C=c,max_iter=2000,solver='lbfgs',random_state=42).fit(train['x'][train['task']<=task],train['y'][train['task']<=task])
            mask=val['task']<=task;p=fitted.predict_proba(val['x'][mask])
            candidates.append(dict(C=c,stage=task,validation_macroF1=100*f1_score(val['y'][mask],p.argmax(1),average='macro')))
            (out/'logistic').mkdir(exist_ok=True);(out/'logistic'/f'C{c}_task{task}.pkl').write_bytes(pickle.dumps(fitted))
    chosen=min((r for r in candidates if r['stage']==3),key=lambda r:(-r['validation_macroF1'],r['C']))
    csvout(out/'logistic_validation_candidates.csv',candidates)
    save(out/'SELECTION.json',dict(neural='unchanged original settings, no new neural tuning',logistic=chosen,
        source_unused_not_used=True,logistic_not_FCL=True))
    save(out/'TEST_GATE.json',dict(status='OPEN',all_training_complete=True,selection_sha256=sha(out/'SELECTION.json')))


def evaluate(project,out):
    origin,original=setup(project);assert read(out/'TEST_GATE.json')['status']=='OPEN'
    domains={s:load(origin,out,s) for s in ('train','source_unused','val','test')};rows=[];trajectories=[]
    jobs=[(arm,seed) for arm in ('FULL','FROZEN','PLAIN') for seed in (42,43,123)]+[('LOGISTIC_REFERENCE',0)]
    selected_C=read(out/'SELECTION.json')['logistic']['C']
    for arm,seed in jobs:
        pred_dir=out/'runs'/f'{arm}_s{seed}'/'predictions';pred_dir.mkdir(parents=True,exist_ok=True)
        for stage in (1,2,3):
            if arm=='LOGISTIC_REFERENCE':
                fitted=pickle.loads((out/'logistic'/f'C{selected_C}_task{stage}.pkl').read_bytes())
                predict=lambda x:(np.log(np.maximum(fitted.predict_proba(x),1e-300)),fitted.predict_proba(x))
            else:
                net=model(original,arm);net.load_state_dict(torch.load(pred_dir.parent/f'task_{stage}.pth',map_location='cpu'))
                predict=lambda x:original.predict(net,x,torch.device('cpu'))
            a=domains['val'];mask=a['task']<=stage;_,vp=predict(a['x'][mask]);threshold=original.threshold(a['y'][mask],vp)
            for domain,a in domains.items():
                mask=a['task']<=stage;z,p=predict(a['x'][mask]);y=a['y'][mask]
                assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
                np.savez_compressed(pred_dir/f'stage{stage}_{domain}.npz',sample_id=a['sample_id'][mask],task=a['task'][mask],y=y,logits=z,p=p,pred=p.argmax(1),threshold=threshold)
                m=original.evaluate(y,p);counts=original.counts(y,p,threshold)
                rows.append(dict(arm=arm,seed=seed,stage=stage,domain=domain,n=len(y),normal=int((y==0).sum()),attack=int((y==1).sum()),
                    macroF1_percent=m['macro_f1_percent'],BENIGN_recall_percent=m['class_recalls_percent'][0],Rbot_recall_percent=m['class_recalls_percent'][1],
                    threshold=threshold,TP=counts['TP'],FP=counts['FP'],TN=counts['TN'],FN=counts['FN'],
                    calibrated_TPR_percent=counts['TPR_percent'],actual_FPR_percent=counts['FPR_percent']))
                for task in range(1,stage+1):
                    member=a['task'][mask]==task
                    trajectories.append(dict(arm=arm,seed=seed,stage=stage,domain=domain,task=task,
                        macroF1_percent=original.evaluate(y[member],p[member])['macro_f1_percent']))
    csvout(out/'domain_per_seed.csv',rows);csvout(out/'fixed_task_trajectories.csv',trajectories)
    summaries=[]
    for arm in ('FULL','FROZEN','PLAIN','LOGISTIC_REFERENCE'):
        for stage in (1,2,3):
            for domain in domains:
                group=[r for r in rows if (r['arm'],r['stage'],r['domain'])==(arm,stage,domain)]
                r=dict(arm=arm,stage=stage,domain=domain,n_seeds=len(group))
                for metric in ('macroF1_percent','BENIGN_recall_percent','Rbot_recall_percent','calibrated_TPR_percent','actual_FPR_percent'):
                    values=[x[metric] for x in group];r[metric+'_mean']=float(np.mean(values));r[metric+'_sample_SD']=float(np.std(values,ddof=1)) if len(group)>1 else None
                summaries.append(r)
    csvout(out/'domain_summary.csv',summaries)
    pair=[]
    for other in ('FROZEN','PLAIN'):
        for domain in domains:
            for seed in (42,43,123):
                a=next(r for r in rows if (r['arm'],r['seed'],r['stage'],r['domain'])==('FULL',seed,3,domain))
                b=next(r for r in rows if (r['arm'],r['seed'],r['stage'],r['domain'])==(other,seed,3,domain))
                pair.append(dict(comparison='FULL-'+other,domain=domain,seed=seed,final_macroF1_difference_pp=a['macroF1_percent']-b['macroF1_percent']))
    csvout(out/'paired_differences.csv',pair)
    save(out/'VERIFICATION.json',dict(status='PASS',domain_records=len(rows),predictions_saved=True,
        fixed_task_records=len(trajectories),methods=4,neural_evaluations=9,
        full_runs_reused=3,new_frozen_continuations=3,new_plain_full_runs=3,
        source_unused=1200,original_train_val_test_unchanged=True,source_unused_never_for_selection=True,
        logistic_reference_cumulative_centralized=True,logistic_n_seeds=1))
    print('PROTOCOL_C_BRIDGE_COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path);parser.add_argument('action',choices=['prepare','train','evaluate','all'])
    a=parser.parse_args();project=a.project.resolve();out=(a.output or project/'supplementary_data/evidence_bridge_20261121_v1/protocolC').resolve()
    if a.action in ('prepare','all'):prepare(project,out)
    if a.action in ('train','all'):train(project,out)
    if a.action in ('evaluate','all'):evaluate(project,out)
