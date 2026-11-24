"""Controlled FIFO and transparently adapted Fed-LwF extension.

All input transforms, architecture, client aggregation, evaluation and sampling
come from the previously frozen method suite. No historical run is edited.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import queue
import shutil
import subprocess
import sys
import threading


def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvout(p,rows):
    if not rows:return
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def extract_existing(out,baseline):
    """Reconstruct selection and stage tables directly from frozen validation runs."""
    plan=read(baseline/'plan.json');selection=read(baseline/'selection.json');candidates=[]
    for job in plan['budget_candidates']:
        r=read(baseline/'runs'/job['id']/'training_result.json')
        candidates.append(dict(arm=job['arm'],policy=job['policy'],schedule=job['schedule'],
            validation_task_average_f1_percent=r['metrics']['val_macro_f1'],
            actual_duplex_bytes=r['duplex_bytes'],cap_bytes=plan['cap_bytes'],
            feasible=r['duplex_bytes']<=plan['cap_bytes'],selected=selection['budget'][job['arm']]['id']==job['id'],
            source=str(baseline/'runs'/job['id']/'training_result.json'),source_sha256=sha(baseline/'runs'/job['id']/'training_result.json')))
    for arm in selection['budget']:
        group=[r for r in candidates if r['arm']==arm and r['feasible']]
        winner=min(group,key=lambda r:(-r['validation_task_average_f1_percent'],r['actual_duplex_bytes'],r['source']))
        assert winner['selected'],arm
    csvout(out/'existing_budget_candidates.csv',candidates)
    save(out/'existing_budget_selection.json',dict(cap_bytes=plan['cap_bytes'],cap_MiB=plan['cap_bytes']/2**20,
        criterion=plan['selection'],tie_rule='lower actual bytes, then job id',
        selection_reconstructed_matches=True,derpp=selection['derpp'],M6_reason=[r for r in candidates if r['arm']=='M6'],
        budget_underuse='unspent budget alone does not invalidate a feasible predeclared candidate',baseline=str(baseline)))
    import numpy as np
    trajectory=list(csv.DictReader((baseline/'fixed_task_trajectories.csv').open()));raw=[];summary=[]
    metrics=[('task2_acquisition',2,2),('task3_acquisition',3,3),('final_task1',3,1),('final_task2',3,2)]
    for arm in ('M0','M1','M2','M3'):
        row=dict(arm=arm,n=3)
        for name,stage,task in metrics:
            values=[]
            for seed in (42,43,123):
                r=next(r for r in trajectory if r['dataset']=='CIC_IDS' and r['role']=='fixed' and r['arm']==arm and int(r['seed'])==seed and int(r['stage'])==stage and int(r['task'])==task)
                value=float(r['macro_f1_percent']);values.append(value)
                raw.append(dict(arm=arm,seed=seed,metric=name,macro_f1_percent=value))
            row[name+'_mean_percent']=float(np.mean(values));row[name+'_sd_percent']=float(np.std(values,ddof=1))
        summary.append(row)
    csvout(out/'existing_ablation_stage_absolute.csv',summary);csvout(out/'existing_ablation_stage_per_seed.csv',raw)
    old=baseline.parent/'label_vocabulary_audit_20261112_v1'
    save(out/'existing_vocab_audit_index.json',dict(path=str(old),files={p.name:sha(p) for p in old.iterdir() if p.is_file()},action='reuse historical equivalence audit; no retraining'))


def prepare(out,baseline,project,official):
    if (out/'plan.json').exists():return
    out.mkdir(parents=True,exist_ok=True);extract_existing(out,baseline)
    code=out/'code';shutil.copytree(baseline/'code',code,ignore=shutil.ignore_patterns('__pycache__'))
    # Only add the new arms to the exact previously frozen prefix hook.
    p=code/'run_method_suite.py';source=p.read_text()
    assert "job['arm'] in ('M1','M2','M3')" in source
    source=source.replace("job['arm'] in ('M1','M2','M3')","job['arm'] in ('FIFO','LWF')")
    source=source.replace("prefix=out/'runs'/f'CIC_IDS_M0_5-5-5_s{job[\"seed\"]}'", "prefix=Path(read(out/'plan.json')['baseline'])/'runs'/f'CIC_IDS_M0_5-5-5_s{job[\"seed\"]}'")
    p.write_text(source);shutil.copy2(project/'run_external_fifo.py',code/'run_external_fifo.py')
    shutil.copy2(baseline/'data_manifests.json',out/'data_manifests.json')
    shutil.copytree(official,out/'official_GLFC',ignore=shutil.ignore_patterns('.git','__pycache__'))
    commit=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()
    save(out/'official_source.json',dict(repository='https://github.com/conditionWang/FCIL',commit=commit,
        files={str(p.relative_to(official)):sha(p) for p in official.rglob('*.py')}))
    jobs=[]
    for value in (.1,.5,1.):jobs.append(dict(id=f'LWF_lambda{value}_5-5-5_s42',dataset='CIC_IDS',arm='LWF',seed=42,schedule=[5,5,5],role='lambda_candidate',kd_lambda=value))
    jobs += [dict(id=f'FIFO_5-5-5_s{s}',dataset='CIC_IDS',arm='FIFO',seed=s,schedule=[5,5,5],role='fifo') for s in (42,43,123)]
    save(out/'plan.json',dict(baseline=str(baseline),project=str(project),seeds=[42,43,123],cap_bytes=37208796,
        jobs=jobs,kd_temperature=2.,lambda_candidates=[.1,.5,1.],budget_candidates=[[7,7,7],[10,5,5],[5,5,10]],
        selection='maximum final validation task-average macro F1, lower actual duplex bytes, lexicographic id; seed42 only',
        architecture='same width48 custom dual input, full updates, same fixed global vocabulary',
        FIFO='task-end append in frozen client/task pickle row order, retain last240, no class quota; same ER loss0.5, shuffle, epochs, batch and current RNG',
        LWF='FedAvg + previous-phase global teacher cloned from first current-model download per phase; no replay; CE plus lambda*T^2*KL(teacher old-class softmax || student old-class softmax); old IDs from past train only',
        LWF_naming='Federated LwF adaptation, not GLFC/EdgeFedCIL faithful reproduction',
        teacher_communication='teacher exactly equals first current-model download of each new phase; cache locally, no separate teacher transmission, no prototypes or auxiliary uploads',
        local_history='one active frozen full model + old-class ID metadata per client; report serialized and tensor bytes; no old input samples for LWF',
        test_gate='all validation selection frozen and all new training finished before new test inference',
        fifo_matching='shared Task1 checkpoint, prefix communication included; all current batches, steps and replay sample counts compared to M0',
        severely_underrepresented_rule='count < 0.25 * floor(240 / historically_seen_client_class_count)',
        limitation='same development-used diagnostic split and uncertified legacy input pairing; not new external-data generalization'))
    save(out/'code_manifest.json',{str(p.relative_to(code)):sha(p) for p in code.rglob('*.py')})
    (out/'ADAPTATION_AUDIT.md').write_text('''# 外部方法适配核查

GLFC官方代码已按commit冻结，见official_source.json。核心包括本地梯度补偿、历史知识蒸馏、样本herding、原型梯度共享、服务器代理图像重建和代理教师选择。不能删除后两项仍称完整GLFC。

| 项目 | GLFC官方实现及适配判断 | 本轮Fed-LwF适配 |
|---|---|---|
| 输入 | RGB图像、PIL增强，代理重建固定1×3×32×32；双输入需要重新定义有效代理分布与重建器 | 沿用相同width48统计—序列骨架 |
| 类别空间 | task_size递增、每阶段抽六类及100类代理数组；本协议首次四类、后续一新类及再现，需重写对应逻辑 | 固定六类词表，蒸馏索引仅为过去训练出现类别，允许再现 |
| 推理信息 | 官方输出分类模型，当前协议不增加推理任务身份 | 单一全局头，推理不使用任务身份 |
| 历史知识 | 梯度补偿＋蒸馏＋herding＋代理选教师；仅复制本地损失会丢失核心模块 | 保留冻结教师、旧类软目标蒸馏，无历史样本缓存 |
| 额外资源 | 原型梯度上传、辅助编码器及代理重建数据、候选教师均须计入 | 每客户端一份冻结教师；从阶段首轮现有下行复制，不另传；记录实际本地历史状态 |
| 命名 | 未进行完整双输入代理机制适配，本轮不声称GLFC复现 | Fed-LwF联邦化适配，不是独立提出完整FCL框架的复现 |

GLFC直接移植需要新定义代理样本语义和验证方法，超出不重构现有数据协议的本轮范围。按用户明确允许的替代路径选择Fed-LwF。它补充一个透明蒸馏机制参照，不能据此声称已经证明优于GLFC等完整框架。

EdgeFedCIL出版页本轮可访问，其数据声明列数据集地址，未在检索页找到可冻结的完整官方训练实现；不据此断言代码不存在，也不删减压缩/误差反馈/保护头后冒称复现。

来源：
- GLFC: https://github.com/conditionWang/FCIL
- ProxyServer: https://github.com/conditionWang/FCIL/blob/main/src/ProxyServer.py
- GLFC local mechanisms: https://github.com/conditionWang/FCIL/blob/main/src/GLFC.py
- EdgeFedCIL: https://www.mdpi.com/1424-8220/26/14/4630
- LwF作者代码: https://github.com/lizhitwo/LearningWithoutForgetting
- LwF作者论文页: https://zhizhongli.vision/projects/learning-without-forgetting/

LwF适配损失为当前数据全局分类CE（沿用共同CosFace训练间隔）与旧类条件软分布KL；原始多任务/图像架构改为单头、双输入和FedAvg。lambda仅以验证集选择，T固定2，未使用ER。温度平方缩放和单头旧类KL均为本适配明确选择，不声称逐行复现作者MatConvNet实现。
''')


def setup_method(out,job,base):
    import torch
    import torch.nn.functional as F
    import run_federated_cl_v4 as v4
    original=v4.V4Client
    manifest=read(out/'data_manifests.json')['CIC_IDS'];data=pickle.loads(Path(manifest['data_path']).read_bytes())
    labels={n:i for i,n in enumerate(manifest['taxonomy'])};run=out/'runs'/job['id']
    if job['arm']=='FIFO':
        class FIFOClient(original):
            def store_replay(self,samples,label_map,model=None):
                candidates=[(torch.tensor(v4.normalize_stat_feature(s['stat_raw']),dtype=torch.float32),
                    torch.tensor(v4.normalize_seq_feature(s['seq_feature']),dtype=torch.float32),label_map[s['label']]) for s in samples]
                self.replay_buffer=(self.replay_buffer+candidates)[-self.args.replay_capacity:]
        v4.V4Client=FIFOClient
    else:
        old={t:sorted({labels[s['label']] for c in range(3) for p in range(t) for s in v4._split(data,c,p,'train')}) for t in range(3)}
        class LwFClient(original):
            def train(self,model,loader,task_id,logger):
                model.to(self.device).train()
                if task_id>0 and getattr(self,'teacher_task',None)!=task_id:
                    self.teacher=copy.deepcopy(model).eval()
                    for p in self.teacher.parameters():p.requires_grad_(False)
                    self.teacher_task=task_id
                    state={k:v.detach().cpu().clone() for k,v in self.teacher.state_dict().items()}
                    root=run/'teachers';root.mkdir(exist_ok=True)
                    torch.save(dict(state=state,old_classes=old[task_id],source='first current model download; identical tensors; no extra wire payload'),root/f'task{task_id+1}_client{self.client_id}.pth')
                    self.teacher_bytes=len(pickle.dumps(dict(state=state,old_classes=old[task_id])))
                    self.teacher_tensor_bytes=sum(x.numel()*x.element_size() for x in state.values())
                def teacher_hash():
                    h=hashlib.sha256()
                    for k,v in self.teacher.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
                    return h.hexdigest()
                before=teacher_hash() if task_id>0 else None
                optimizer=torch.optim.AdamW(self._trainable_parameters(model,False),lr=self.args.lr,weight_decay=1e-4)
                resources=dict(optimizer_updates=0,replay_batches=0,replay_samples=0,distillation_samples=0,
                    teacher_serialized_bytes=getattr(self,'teacher_bytes',0),teacher_tensor_bytes=getattr(self,'teacher_tensor_bytes',0),teacher_extra_download_bytes=0)
                for _ in range(self.args.local_epochs):
                    for stat,seq,y in loader:
                        stat,seq,y=stat.to(self.device),seq.to(self.device),y.to(self.device)
                        loss=F.cross_entropy(model(stat,seq,targets=y),y)
                        if task_id>0:
                            with torch.no_grad():q=(self.teacher(stat,seq)[:,old[task_id]]/2.).softmax(1)
                            logp=(model(stat,seq)[:,old[task_id]]/2.).log_softmax(1)
                            loss=loss+job['kd_lambda']*4.*F.kl_div(logp,q,reduction='batchmean')
                            resources['distillation_samples']+=len(y)
                        optimizer.zero_grad();loss.backward();optimizer.step();resources['optimizer_updates']+=1
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if task_id>0:
                    assert teacher_hash()==before,'teacher changed during local training'
                    resources.update(teacher_state_sha256=before,old_class_ids=old[task_id],teacher_immutable=True)
                return state,list(state),resources
        v4.V4Client=LwFClient


def worker(out,job,base):
    base.environment('CIC_IDS',base.ARMS[job['arm']],job['schedule'])
    setup_method(out,job,base)
    base.worker(out,job)
    r=read(out/'runs'/job['id']/'training_result.json');r['extension_code_sha256']=sha(out/'code/run_external_fifo.py')
    save(out/'runs'/job['id']/'training_result.json',r)


def batch(out,jobs):
    q=queue.Queue();failures=[]
    for j in jobs:
        if not (out/'runs'/j['id']/'training_result.json').exists():q.put(j)
    def consume(gpu):
        while True:
            try:j=q.get_nowait()
            except queue.Empty:return
            run=out/'runs'/j['id'];run.mkdir(parents=True,exist_ok=True);save(run/'job.json',j)
            print('START',gpu,j['id'],flush=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            with (run/'console.log').open('a') as log:
                rc=subprocess.call([sys.executable,'-B',str(out/'code/run_external_fifo.py'),'worker','--output',str(out),'--job',str(run/'job.json')],cwd=out/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
            print('DONE',gpu,j['id'],rc,flush=True)
            if rc:failures.append(j['id'])
            save(out/'execution_status.json',dict(status='FAILED' if failures else 'VALIDATION_TRAINING',completed=len(list((out/'runs').glob('*/training_result.json'))),pending=q.qsize(),failures=failures))
    threads=[threading.Thread(target=consume,args=(g,)) for g in (0,1)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert not failures,failures


def run(out,base):
    plan=read(out/'plan.json');baseline=Path(plan['baseline'])
    dev=[j for j in plan['jobs'] if j['arm']=='LWF'];batch(out,dev)
    best=base.choose(out,dev);chosen=best['job'];fixed=[dict(chosen,role='external_fixed')]
    final=[]
    for seed in (43,123):
        j=dict(chosen,id=f'LWF_fixed_s{seed}',seed=seed,role='external_fixed');final.append(j);fixed.append(j)
    batch(out,final)
    fifo=[j for j in plan['jobs'] if j['arm']=='FIFO'];batch(out,fifo)
    candidates=[dict(id='LWF_budget_'+ '-'.join(map(str,s))+'_s42',dataset='CIC_IDS',arm='LWF',seed=42,schedule=s,role='budget_candidate',kd_lambda=chosen['kd_lambda'],cap_bytes=plan['cap_bytes']) for s in plan['budget_candidates']]
    batch(out,candidates);win=base.choose(out,candidates);budget=[dict(win['job'],role='external_budget')];extra=[]
    for seed in (43,123):
        j=dict(win['job'],id=f'LWF_budget_final_s{seed}',seed=seed,role='external_budget');extra.append(j);budget.append(j)
    save(out/'selection.json',dict(lambda_selection=chosen,budget_selection=win['job'],criterion=plan['selection']))
    selected=fixed+fifo+budget;save(out/'selected_runs.json',selected);batch(out,extra)
    checks=[]
    import torch
    for seed in (42,43,123):
        a=baseline/'runs'/f'CIC_IDS_M0_5-5-5_s{seed}';b=out/'runs'/f'FIFO_5-5-5_s{seed}'
        ac=read(a/'client_records.json');bc=read(b/'client_records.json')
        keys=['task','client','round','optimizer_updates','current_sample_presentations','current_batches_sha256','replay_samples']
        matching=len(ac)==len(bc) and all(all(x[k]==y[k] for k in keys) for x,y in zip(ac,bc))
        sa=torch.load(a/'task_checkpoints/task_1.pth',map_location='cpu');sb=torch.load(b/'task_checkpoints/task_1.pth',map_location='cpu')
        equal=all(torch.equal(v,sb[k]) for k,v in sa.items());checks.append(dict(seed=seed,current_and_exposure_match=matching,task1_tensor_equal=equal))
    save(out/'fifo_matching_checks.json',checks);assert all(r['current_and_exposure_match'] and r['task1_tensor_equal'] for r in checks)
    save(out/'TEST_GATE.json',dict(status='OPEN',all_validation_training_complete=True,selection_sha256=sha(out/'selection.json'),selected_runs_sha256=sha(out/'selected_runs.json')))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable,'-B',str(out/'code/run_external_fifo.py'),'evaluate','--output',str(out)],env=env,cwd=out/'code',check=True)
    save(out/'execution_status.json',dict(status='TRAINING_AND_TEST_COMPLETE_REPORT_PENDING',completed=len(list((out/'runs').glob('*/training_result.json'))),formal_runs=len(selected)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker','evaluate']);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--official',type=Path);p.add_argument('--job',type=Path);a=p.parse_args();out=a.output.resolve()
    if a.action=='prepare':prepare(out,a.baseline.resolve(),a.project.resolve(),a.official.resolve())
    else:
        import run_method_suite as base
        base.ARMS.update(FIFO=dict(width=48,freeze=False,replay=True),LWF=dict(width=48,freeze=False,replay=False))
        if a.action=='worker':worker(out,read(a.job),base)
        elif a.action=='evaluate':base.evaluate(out,'CIC_IDS')
        else:run(out,base)
