"""Isolated method-paper suite: shared settings, validation selection, test gate.

Reuses the archived V4 implementation, not independently tuned historical scores.
Each trained checkpoint supplies main comparisons, ablations and trajectories.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import queue
import shutil
import subprocess
import sys
import threading
import time


SEEDS = (42, 43, 123)
CAP = 37208796
COMMON = dict(lr=0.001737670720958496, batch_size=64, local_epochs=3)
ARMS = {
    'M0': dict(width=48, freeze=False, replay=True),
    'M1': dict(width=48, freeze=False, replay=False),
    'M2': dict(width=48, freeze=True, replay=True),
    'M3': dict(width=48, freeze=True, replay=False),
    'M4': dict(width=48, freeze=False, replay=True, plain=True),
    'M5': dict(width=48, freeze=False, replay=True, derpp=True),
    'M6': dict(width=128, freeze=False, replay=True),
}


def read(p):
    return json.loads(Path(p).read_text())


def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def csvout(p, rows):
    if not rows: return
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)


def environment(dataset, arm, schedule):
    os.environ.update(TARGET_DATASET=dataset, RUN_MODE='Ours', V4_VARIANT='FULL',
        V4_STAGE='replay', V4_PRIOR_MODE='none', V4_REPLAY_CAPACITY='240',
        V4_REPLAY_METHOD='derpp' if arm.get('derpp') else 'er',
        V4_COMPONENTS=('' if arm.get('plain') else 'adapter,se,cosface') + (',replay' if arm['replay'] else ''),
        V4_FREEZE_BACKBONE=str(int(arm['freeze'])), V4_START_TASK='0',
        V4_INITIAL_CHECKPOINT='', V4_FIXED_UPDATES='0', V4_RECORD_TASK1_CURVE='0',
        V4_FROZEN_TRAINABLE='adapter_classifier', V4_ROUNDS_PER_TASK=','.join(map(str, schedule)),
        NUM_CLIENTS='3', NUM_TASKS='3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')


def model_setup(v4, arm):
    """The plain alternative shares encoders but uses direct concat + linear head."""
    original = v4.build_model
    def build(args, labels, counts, components, device):
        if arm.get('plain'):
            components = dict(components, adapter=False, se=False, cosface=False)
        model = original(args, labels, counts, components, device)
        if arm.get('plain'):
            import torch
            model.fusion_norm = torch.nn.Identity()
        return model
    v4.build_model = build


def prepare(out, project):
    if (out / 'plan.json').exists(): return
    out.mkdir(parents=True, exist_ok=True)
    manifests = {}
    paths = {'CIC_IDS': 'experiment_logs/reviewer_repair_20260926_v1',
        'USTC': 'experiment_logs/reviewer_followups_20260926_v1/cross_dataset/USTC',
        'CIC_DOH': 'experiment_logs/reviewer_followups_20260926_v1/cross_dataset/CIC_DOH'}
    for dataset, relative in paths.items():
        m = read(project / relative / 'data_manifest.json')
        assert sha(m['data_path']) == m['data_sha256']
        normal=[name for name in m['taxonomy'] if name.casefold()=='benign']
        assert len(normal)==1
        m.setdefault('normal_label',normal[0])
        manifests[dataset] = m
    code = out / 'code'; code.mkdir()
    for name in ('run_method_suite.py', 'run_federated_cl_v4.py', 'review_repair_experiments.py',
                 'fcl_derpp_reference.py', 'fcl_models_v4.py', 'config.py', 'fcl_data_utils.py', 'fcl_protocol.py'):
        shutil.copy2(project / name, code / name)
    shutil.copytree(project / 'data_pipeline', code / 'data_pipeline', ignore=shutil.ignore_patterns('__pycache__'))
    save(out / 'code_manifest.json', {str(p.relative_to(code)): sha(p) for p in code.rglob('*.py')})
    save(out / 'data_manifests.json', manifests)
    archive = project / 'supplementary_data/e1_e4_evidence_20261111_v1/core_run_configuration.csv'
    rows = list(csv.DictReader(archive.open()))
    for r in rows:
        reasons = []
        if r['data_hash'] != manifests['CIC_IDS']['data_sha256']: reasons.append('different data')
        if r['schedule'] != '[5, 5, 5]': reasons.append('different schedule')
        if abs(float(r['learning_rate']) - COMMON['lr']) > 1e-14: reasons.append('different learning rate')
        if int(r['local_epochs']) != 3: reasons.append('different local epochs')
        reasons.append('archived sampling does not certify independent current/replay RNG for new four-cell design')
        r['new_suite_reuse'] = 'SUPPORT_ONLY_NO_RETRAIN_OF_ARCHIVE'
        r['reuse_reason'] = '; '.join(reasons)
    csvout(out / 'archive_audit.csv', rows)
    jobs = []
    def job(dataset, arm, seed, schedule, role, **kw):
        return dict(id=f'{dataset}_{arm}_{"-".join(map(str,schedule))}_s{seed}' + kw.pop('suffix', ''),
            dataset=dataset, arm=arm, seed=seed, schedule=schedule, role=role, **kw)
    # M5 coefficients are selected with seed42 validation only; all other settings fixed.
    for arm in ARMS:
        if arm != 'M5':
            jobs += [job('CIC_IDS', arm, seed, [5,5,5], 'fixed') for seed in SEEDS]
    for i, (alpha, beta) in enumerate(((.1,.5), (.5,.5), (1.,.5), (.5,1.))):
        jobs.append(job('CIC_IDS','M5',42,[5,5,5],'der_candidate',alpha=alpha,beta=beta,suffix=f'_c{i}'))
    # Costs depend only on state shapes; schedules frozen without looking at scores.
    environment('CIC_IDS', ARMS['M0'], [5,5,5])
    import torch
    import run_federated_cl_v4 as v4
    budget = []
    labels = {name:i for i,name in enumerate(manifests['CIC_IDS']['taxonomy'])}
    original_build = v4.build_model
    for arm_id in ('M0','M4','M2','M6','M5'):
        arm = ARMS[arm_id]
        v4.build_model = original_build; model_setup(v4,arm)
        args = v4.V4Args(); args.hidden_dim = arm['width']
        model = v4.build_model(args, labels, [1]*len(labels), v4.component_flags(), torch.device('cpu'))
        state = model.state_dict()
        full = len(pickle.dumps(state))
        head = len(pickle.dumps({k:v for k,v in state.items() if 'adapter' in k or 'classifier' in k}))
        def cost(s):
            return 3*sum(n*(full + (head if arm['freeze'] and t>0 else full) + 2048) for t,n in enumerate(s))
        # Three equal-opportunity allocation templates: balanced/front/back.
        for policy, ratio in [('balanced',(1,1,1)),('front',(2,1,1)),('back',(1,1,2))]:
            n = 1
            while cost([r*(n+1) for r in ratio]) <= CAP: n += 1
            schedule = [r*n for r in ratio]
            assert min(schedule)>=2 and cost(schedule)<=CAP
            j = job('CIC_IDS',arm_id,42,schedule,'budget_candidate',policy=policy,cap_bytes=CAP)
            budget.append(dict(**j, conservative_bytes=cost(schedule),parameter_count=sum(p.numel() for p in model.parameters())))
            if not any(x['id']==j['id'] for x in jobs): jobs.append(j)
    v4.build_model = original_build
    save(out / 'plan.json', dict(project=str(project), seeds=SEEDS, common=COMMON, arms=ARMS,
        cap_bytes=CAP, jobs=jobs, budget_candidates=budget,
        cross_dataset={'datasets':['USTC','CIC_DOH'],'arms':['M0','M4','M5'],'seeds':SEEDS,
            'coefficients':'transfer CIC validation-selected DER++ coefficients without extra tuning'},
        selection='maximize final validation task-average macro F1, then fewer actual bytes, then id; seed42 only',
        current_rng='per-client/task/round seed, separate current sampler and replay sampler on every call',
        ablation='M1/M2/M3 resume same-seed M0 Task1 checkpoint, restore deterministic ER buffers and include copied prefix costs; verify shared tensors, current-batch hashes and M0/M2 ER hashes',
        derpp='reference online reservoir over presented examples, pre-update margin-free logits, independent MSE/CE draws; within-Task1 online replay allowed, no historical prior-task buffer',
        communication='full serialized state download + actual selected-key upload, all three tasks; simulated bytes, not wire traffic',
        evaluation='fixed per-task populations, macro over labels present in that population; also final pooled; probabilities sum over attack classes, >= threshold',
        threshold='validation only maximum TPR subject FPR<=1% or0.1%; tie minimumFPR then highest threshold; explicit +inf candidate',
        limitations='legacy reconstructed diagnostic input pairs; no verified flow/session alignment or physical capture separation',
        test_gate='all planned validation training and selections complete before test inference; no candidate-loser test inference',
        optional_not_scheduled=['width96','buffer120','new independent protocol','Protocol C retraining']))
    (out/'README.md').write_text('新版FedDIA方法实验独立输出。\n\nplan.json冻结配置、预算候选与选择规则；archive_audit.csv记录归档复用判断。\n全部训练先只评价验证集。完成选择后统一推理测试集。Protocol B/C及旧阶段预算均不重训。\n')


def worker(out, job):
    arm = ARMS[job['arm']]
    environment(job['dataset'], arm, job['schedule'])
    import torch
    import numpy as np
    import run_federated_cl_v4 as v4
    import review_repair_experiments as repair
    assert torch.cuda.is_available(), 'GPU unavailable; rerun outside sandbox'
    torch.set_num_threads(2)
    for f,h in read(out/'code_manifest.json').items(): assert sha(out/'code'/f)==h, f
    manifest = read(out/'data_manifests.json')[job['dataset']]
    assert sha(manifest['data_path']) == manifest['data_sha256']
    data = pickle.loads(Path(manifest['data_path']).read_bytes())
    labels = {name:i for i,name in enumerate(manifest['taxonomy'])}
    run = out/'runs'/job['id']; run.mkdir(parents=True, exist_ok=True)
    if (run/'training_result.json').exists(): return
    args = v4.V4Args()
    for k,v in COMMON.items(): setattr(args,k,v)
    args.hidden_dim = arm['width']; args.der_alpha = job.get('alpha',.5); args.der_beta = job.get('beta',.5)
    model_setup(v4, arm)
    records=[]; diags=[]
    # Copy the actual shared prefix, avoiding numerical differences between GPUs.
    # ER memory is deterministically reconstructed by the archived continuation
    # loader; the optimizer is fresh per call, so no optimizer state is omitted.
    prefix = None
    if job['dataset']=='CIC_IDS' and job['arm'] in ('M1','M2','M3') and job['schedule']==[5,5,5]:
        prefix=out/'runs'/f'CIC_IDS_M0_5-5-5_s{job["seed"]}'
        args.start_task=1;args.initial_checkpoint=str(prefix/'task_checkpoints/task_1.pth')
        records.extend(r for r in read(prefix/'client_records.json') if r['task']==1)
        (run/'task_checkpoints').mkdir(exist_ok=True)
        shutil.copy2(prefix/'task_checkpoints/task_1.pth',run/'task_checkpoints/task_1.pth')
        save(run/'shared_prefix.json',dict(source=str(prefix),checkpoint_sha256=sha(args.initial_checkpoint),
            task1_cost_included=True,buffer_restoration='deterministic task-end class-balanced latest records',optimizer='fresh each client-round'))
    repair.instrument(v4, 'DERPP_reference' if arm.get('derpp') else job['arm'], records, diags)
    base = v4.V4Client
    def update(h,batch):
        for tensor in batch:
            a=torch.as_tensor(tensor).detach().cpu().contiguous().numpy()
            h.update(str((a.shape,str(a.dtype))).encode()); h.update(a.tobytes())
    class MatchedClient(base):
        def store_replay(self,*a,**kw):
            super().store_replay(*a,**kw)
            task=1+getattr(self,'buffer_task',0);self.buffer_task=task
            path=run/'buffers'/f'task{task}_client{self.client_id}.pth';path.parent.mkdir(exist_ok=True)
            torch.save(dict(buffer=self.replay_buffer,seen_examples=getattr(self,'seen_examples',None),
                reconstruction='online reservoir' if arm.get('derpp') else 'task-end latest per class'),path)
        def train(self, model, loader, task_id, logger):
            round_id=1+sum(r['client']==self.client_id and r['task']==task_id+1 for r in records)
            self.call_seed=job['seed']*100000+task_id*10000+self.client_id*100+round_id
            v4.set_seed(self.call_seed); self.replay_call=0
            loader.generator=torch.Generator().manual_seed(self.call_seed+10000000)
            loader.sampler.generator=torch.Generator().manual_seed(self.call_seed+20000000)
            self.ch=hashlib.sha256(); self.rh=hashlib.sha256(); self.rc={}
            client=self
            class Loader:
                def __iter__(self):
                    for batch in loader: update(client.ch,batch); yield batch
                def __getattr__(self,k): return getattr(loader,k)
            result=super().train(model, Loader(), task_id, logger)
            records[-1].update(current_batches_sha256=self.ch.hexdigest(),replay_batches_sha256=self.rh.hexdigest(),replay_class_presentations=self.rc)
            save(run/'progress.json',dict(status='TRAINING_VALIDATION_ONLY',last_call=records[-1],calls=len(records)))
            return result
        def _replay_loader(self, current_loader, model):
            loader=super()._replay_loader(current_loader,model)
            if loader is None: return None
            self.replay_call+=1; s=self.call_seed+30000000+self.replay_call
            loader.generator=torch.Generator().manual_seed(s)
            loader.sampler.generator=torch.Generator().manual_seed(s+1000000)
            client=self
            class Replay:
                def __iter__(self):
                    for batch in loader:
                        update(client.rh,batch)
                        for label,count in zip(*torch.unique(batch[2],return_counts=True)):
                            k=str(int(label));client.rc[k]=client.rc.get(k,0)+int(count)
                        yield batch
            return Replay()
        def draw(self,size):
            batch=super().draw(size)
            update(self.rh,batch)
            for label,count in zip(*torch.unique(batch[2],return_counts=True)):
                k=str(int(label));self.rc[k]=self.rc.get(k,0)+int(count)
            return batch
    v4.V4Client=MatchedClient
    v4.set_seed(job['seed'])
    os.environ['V4_TASK_CHECKPOINT_ROOT']=str(run/'task_checkpoints')
    begin=time.monotonic(); device=torch.device('cuda:0')
    metrics,state=v4.run_fcl_pipeline(args,data,labels,device,v4.setup_logger(str(run/'training.log')),'val')
    duplex=sum(r['upload_serialized_state_bytes']+r['downlink_full_state_bytes'] for r in records)
    assert not job.get('cap_bytes') or duplex<=job['cap_bytes'], 'cap exceeded'
    model=v4.build_model(args,labels,[1]*len(labels),v4.component_flags(),device)
    torch.save(state,run/'final_model.pth')
    save(run/'client_records.json',records); save(run/'diagnostics.json',diags)
    save(run/'training_result.json',dict(job=job,metrics=metrics,duplex_bytes=duplex,
        task1_bytes=sum(r['upload_serialized_state_bytes']+r['downlink_full_state_bytes'] for r in records if r['task']==1),
        parameter_count=sum(p.numel() for p in model.parameters()),label_map=labels,
        seconds=time.monotonic()-begin,data_sha256=manifest['data_sha256'],
        gpu=torch.cuda.get_device_name(0),code_sha256=sha(out/'code/run_method_suite.py'),
        status='VALIDATION_TRAINING_COMPLETE_TEST_NOT_ACCESSED'))


def batch(out,jobs):
    q=queue.Queue(); failures=[]
    for j in {j['id']:j for j in jobs}.values():
        if not (out/'runs'/j['id']/'training_result.json').exists(): q.put(j)
    def consumer(gpu):
        while True:
            try: j=q.get_nowait()
            except queue.Empty: return
            run=out/'runs'/j['id'];run.mkdir(parents=True,exist_ok=True)
            save(run/'job.json',j)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            print('START',gpu,j['id'],flush=True)
            with (run/'console.log').open('a') as log:
                rc=subprocess.call([sys.executable,'-B',str(out/'code/run_method_suite.py'),'worker','--output',str(out),'--job',str(run/'job.json')],env=env,cwd=out/'code',stdout=log,stderr=subprocess.STDOUT)
            print('DONE',gpu,j['id'],rc,flush=True)
            if rc: failures.append(j['id'])
            save(out/'execution_status.json',dict(status='FAILED' if failures else 'RUNNING_VALIDATION',completed=len(list((out/'runs').glob('*/training_result.json'))),pending=q.qsize(),failures=failures))
            q.task_done()
    ts=[threading.Thread(target=consumer,args=(gpu,)) for gpu in (0,1)]
    for t in ts:t.start()
    for t in ts:t.join()
    if failures: raise RuntimeError(str(failures))


def choose(out,jobs):
    rs=[read(out/'runs'/j['id']/'training_result.json') for j in jobs]
    return min(rs,key=lambda r:(-r['metrics']['val_macro_f1'],r['duplex_bytes'],r['job']['id']))


def run_suite(out):
    plan=read(out/'plan.json')
    # First priority: M0/M4 and the four-cell ablation, then DER++ selection.
    first=sorted([j for j in plan['jobs'] if not (j['arm']=='M5' and j['role']=='budget_candidate')],
        key=lambda j:(0 if j['role']=='fixed' and j['arm'] in ('M0','M4','M1','M2','M3') else 1,
                      ('M0','M4','M1','M2','M3','M6','M5').index(j['arm']),j['seed']))
    priority=[j for j in first if j['role']=='fixed' and j['arm'] in ('M0','M4')]
    batch(out,priority)
    batch(out,[j for j in first if j not in priority])
    best=choose(out,[j for j in plan['jobs'] if j['role']=='der_candidate'])
    coefficients={k:best['job'][k] for k in ('alpha','beta')}
    selected=[j for j in plan['jobs'] if j['role']=='fixed']
    selected.append(dict(best['job'],role='fixed'))
    extra=[]
    for seed in (43,123):
        j=dict(best['job'],seed=seed,id=f'CIC_IDS_M5_5-5-5_s{seed}',role='fixed');extra.append(j);selected.append(j)
    # Budget M5 needs coefficients selected above. Its initial placeholders must
    # never be trained before that choice; runner separates those candidates below.
    budgetjobs=[]
    for j in plan['budget_candidates']:
        j={k:v for k,v in j.items() if k not in ('conservative_bytes','parameter_count')}
        if j['arm']=='M5':j.update(coefficients)
        budgetjobs.append(j)
    batch(out,budgetjobs)
    selections={}
    for arm in ('M0','M4','M2','M6','M5'):
        winner=choose(out,[j for j in budgetjobs if j['arm']==arm])
        selections[arm]=winner['job']
        selected.append(dict(winner['job'],role='budget_final'))
        for seed in (43,123):
            j=dict(winner['job'],seed=seed,id=winner['job']['id'].rsplit('_s',1)[0]+f'_s{seed}',role='budget_final')
            extra.append(j);selected.append(j)
    for dataset in ('USTC','CIC_DOH'):
        for arm in ('M0','M4','M5'):
            for seed in SEEDS:
                j=dict(id=f'{dataset}_{arm}_5-5-5_s{seed}',dataset=dataset,arm=arm,seed=seed,schedule=[5,5,5],role='cross_dataset')
                if arm=='M5':j.update(coefficients)
                extra.append(j);selected.append(j)
    save(out/'selection.json',dict(derpp=best['job'],budget=selections,criterion=plan['selection']))
    save(out/'selected_runs.json',selected)
    batch(out,extra)
    verify_branches(out)
    save(out/'TEST_GATE.json',dict(status='OPEN',all_validation_training_complete=True,selection_sha256=sha(out/'selection.json'),selected_runs_sha256=sha(out/'selected_runs.json')))
    # Evaluation uses subprocesses to keep dataset-specific CFG imports isolated.
    for dataset in ('CIC_IDS','USTC','CIC_DOH'):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        subprocess.run([sys.executable,'-B',str(out/'code/run_method_suite.py'),'evaluate','--output',str(out),'--dataset',dataset],env=env,cwd=out/'code',check=True)
    summarize(out)


def verify_branches(out):
    import torch
    checks=[]
    for seed in SEEDS:
        roots={arm:out/'runs'/f'CIC_IDS_{arm}_5-5-5_s{seed}' for arm in ('M0','M1','M2','M3')}
        states={arm:torch.load(p/'task_checkpoints/task_1.pth',map_location='cpu') for arm,p in roots.items()}
        records={arm:read(p/'client_records.json') for arm,p in roots.items()}
        equal=all(all(torch.equal(v,states[arm][k]) for k,v in states['M0'].items()) for arm in ('M1','M2','M3'))
        current=all([r['current_batches_sha256'] for r in records['M0']]==[r['current_batches_sha256'] for r in records[a]] for a in ('M1','M2','M3'))
        replay=[r['replay_batches_sha256'] for r in records['M0']]==[r['replay_batches_sha256'] for r in records['M2']]
        checks.append(dict(seed=seed,task1_state_tensor_equal=equal,all_current_batches_equal=current,er_draws_equal=replay))
    save(out/'ablation_matching_checks.json',checks)
    assert all(all(r[k] for k in ('task1_state_tensor_equal','all_current_batches_equal','er_draws_equal')) for r in checks)


def evaluate(out,dataset):
    assert read(out/'TEST_GATE.json')['status']=='OPEN'
    environment(dataset,ARMS['M0'],[5,5,5])
    import torch
    import numpy as np
    from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_curve
    import run_federated_cl_v4 as v4
    torch.set_num_threads(2);device=torch.device('cuda:0')
    manifest=read(out/'data_manifests.json')[dataset]
    data=pickle.loads(Path(manifest['data_path']).read_bytes())
    labels={n:i for i,n in enumerate(manifest['taxonomy'])};normal=labels[manifest['normal_label']]
    # Population IDs encode the immutable pickle position, not a claimed flow ID.
    populations={}
    for t in range(3):
        for split in ('val','test'):
            samples=[];ids=[]
            for c in range(3):
                ss=v4._split(data,c,t,split);samples.extend(ss)
                ids.extend(f'{dataset}/Client_{c}/Task_{t+1}/{split}/{i}' for i in range(len(ss)))
            populations[t,split]=(samples,np.asarray(ids))
    original=v4.build_model
    for job in {j['id']:j for j in read(out/'selected_runs.json') if j['dataset']==dataset}.values():
        run=out/'runs'/job['id']
        if (run/'evaluation.json').exists():continue
        arm=ARMS[job['arm']]; v4.build_model=original;model_setup(v4,arm)
        args=v4.V4Args();args.hidden_dim=arm['width']
        model=v4.build_model(args,labels,[1]*len(labels),v4.component_flags(),device)
        rows=[];alerts=[]
        for stage in range(1,4):
            model.load_state_dict(torch.load(run/f'task_checkpoints/task_{stage}.pth',map_location=device));model.eval()
            collected={}
            for t in range(stage):
                for split in ('val','test'):
                    samples,ids=populations[t,split];ys=[];ls=[]
                    with torch.no_grad():
                        for stat,seq,y in v4.make_loader(samples,labels,64,True,False):
                            ls.append(model(stat.to(device),seq.to(device)).cpu());ys.append(y)
                    logits=torch.cat(ls);prob=logits.softmax(1).numpy();y=torch.cat(ys).numpy()
                    collected[t,split]=(y,prob)
                    np.savez_compressed(run/f'stage{stage}_task{t+1}_{split}.npz',id=ids,y=y,logits=logits.numpy(),probabilities=prob,prediction=prob.argmax(1))
                    if split=='test':
                        pred=prob.argmax(1)
                        rows.append(dict(stage=stage,task=t+1,macro_f1_percent=100*f1_score(y,pred,labels=np.unique(y),average='macro',zero_division=0),
                            per_class_recall=recall_score(y,pred,labels=list(range(len(labels))),average=None,zero_division=0).tolist(),
                            confusion_matrix=confusion_matrix(y,pred,labels=list(range(len(labels)))).tolist(),support=len(y)))
            for population in list(range(stage))+['pooled']:
                arr={}
                for split in ('val','test'):
                    group=list(range(stage)) if population=='pooled' else [population]
                    y=np.concatenate([collected[t,split][0] for t in group]);p=np.concatenate([collected[t,split][1] for t in group])
                    arr[split]=(y,p[:,[i for i in range(len(labels)) if i!=normal]].sum(1))
                yv,sv=arr['val'];yt,st=arr['test']
                fpr,tpr,thresholds=roc_curve(yv!=normal,sv,drop_intermediate=False)
                # sklearn1.2 uses max(score)+1 for reject-all; enforce literal inf.
                thresholds[0]=np.inf
                for cap in (.01,.001):
                    eligible=np.flatnonzero(fpr<=cap)
                    idx=min(eligible,key=lambda i:(-tpr[i],fpr[i],-thresholds[i]))
                    threshold=thresholds[idx];a=st>=threshold;b=yt!=normal
                    tp=int((a&b).sum());fp=int((a&~b).sum());tn=int((~a&~b).sum());fn=int((~a&b).sum())
                    alerts.append(dict(stage=stage,population='pooled' if population=='pooled' else population+1,target_fpr=cap,
                        threshold=float(threshold) if np.isfinite(threshold) else None,threshold_positive_infinity=bool(np.isposinf(threshold)),
                        val_tpr=float(tpr[idx]),val_fpr=float(fpr[idx]),TP=tp,FP=fp,TN=tn,FN=fn,test_tpr=tp/(tp+fn),test_fpr=fp/(fp+tn)))
            if stage==3:
                y=np.concatenate([collected[t,'test'][0] for t in range(3)])
                p=np.concatenate([collected[t,'test'][1] for t in range(3)])
                pooled=100*f1_score(y,p.argmax(1),labels=np.unique(y),average='macro',zero_division=0)
        save(run/'evaluation.json',dict(job=job,trajectory=rows,alerts=alerts,final_pooled_f1_percent=pooled,
            final_task_average_f1_percent=sum(r['macro_f1_percent'] for r in rows if r['stage']==3)/3))


def summarize(out):
    import numpy as np
    selected=read(out/'selected_runs.json');rows=[];trajectories=[];alerts=[]
    for j in selected:
        run=out/'runs'/j['id'];r=read(run/'training_result.json');e=read(run/'evaluation.json')
        records=read(run/'client_records.json');diags=read(run/'diagnostics.json')
        row=dict(id=j['id'],dataset=j['dataset'],arm=j['arm'],role=j['role'],seed=j['seed'],schedule=j['schedule'],
            task1_f1_percent=e['trajectory'][0]['macro_f1_percent'],final_pooled_f1_percent=e['final_pooled_f1_percent'],
            final_task_average_f1_percent=e['final_task_average_f1_percent'],duplex_bytes=r['duplex_bytes'],task1_bytes=r['task1_bytes'],parameter_count=r['parameter_count'],
            optimizer_updates=sum(x['optimizer_updates'] for x in records),current_presentations=sum(x['current_sample_presentations'] for x in records),
            replay_presentations=sum(x['replay_samples'] for x in records),
            peak_buffer_serialized_bytes_per_client=max([x['buffer_serialized_bytes_at_train_end'] for x in records]+[x['serialized_bytes'] for x in diags if x['event']=='buffer_update']))
        rows.append(row)
        trajectories.extend(dict(id=j['id'],role=j['role'],dataset=j['dataset'],arm=j['arm'],seed=j['seed'],**x) for x in e['trajectory'])
        alerts.extend(dict(id=j['id'],role=j['role'],dataset=j['dataset'],arm=j['arm'],seed=j['seed'],**x) for x in e['alerts'])
    csvout(out/'main_per_seed.csv',rows);csvout(out/'fixed_task_trajectories.csv',trajectories);csvout(out/'alerts.csv',alerts)
    summary=[]
    for dataset,role,arm in sorted({(r['dataset'],r['role'],r['arm']) for r in rows}):
        rr=[r for r in rows if (r['dataset'],r['role'],r['arm'])==(dataset,role,arm)]
        s=dict(dataset=dataset,role=role,arm=arm,n=len(rr))
        for key in ('task1_f1_percent','final_pooled_f1_percent','final_task_average_f1_percent','duplex_bytes','peak_buffer_serialized_bytes_per_client'):
            a=[r[key] for r in rr];s[key+'_mean']=float(np.mean(a));s[key+'_sd']=float(np.std(a,ddof=1))
        summary.append(s)
    csvout(out/'summary.csv',summary)
    effects=[]
    for seed in SEEDS:
        rr={r['arm']:r for r in rows if r['role']=='fixed' and r['dataset']=='CIC_IDS' and r['seed']==seed}
        for metric in ('final_pooled_f1_percent','final_task_average_f1_percent'):
            effects.append(dict(seed=seed,metric=metric,er_when_full=rr['M0'][metric]-rr['M1'][metric],
                full_when_er=rr['M0'][metric]-rr['M2'][metric],er_when_frozen=rr['M2'][metric]-rr['M3'][metric],
                interaction=(rr['M0'][metric]-rr['M1'][metric])-(rr['M2'][metric]-rr['M3'][metric]),
                custom_minus_plain=rr['M0'][metric]-rr['M4'][metric]))
    csvout(out/'ablation_paired_effects.csv',effects)
    save(out/'execution_status.json',dict(status='COMPLETE',selected_table_entries=len(rows),unique_training_runs=len(list((out/'runs').glob('*/training_result.json'))),test_gate='PASSED',optional_experiments='NOT_AUTOMATICALLY_STARTED'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker','evaluate','summarize']);p.add_argument('--output',type=Path,required=True);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--job',type=Path);p.add_argument('--dataset')
    a=p.parse_args();out=a.output.resolve()
    if a.action=='prepare':prepare(out,a.project.resolve())
    elif a.action=='worker':worker(out,read(a.job))
    elif a.action=='evaluate':evaluate(out,a.dataset)
    elif a.action=='summarize':summarize(out)
    else:run_suite(out)
