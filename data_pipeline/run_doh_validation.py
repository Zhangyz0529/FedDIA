#!/usr/bin/env python3
"""Gated DoH Source25 pilot, validation-only tuning, and paired 3-seed tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
METHODS=['FedDIA','NoFreezeReplay','SmallNoFreezeReplay','DERPPNoFreeze']
SEEDS=[42,43,44]
PARAMS=[dict(lr=.001,batch_size=64,local_epochs=2),dict(lr=.0003,batch_size=64,local_epochs=2)]


def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(out,data):
    if out.exists():raise ValueError('Use fresh output')
    out.mkdir(parents=True);code=out/'code';(code/'data_pipeline').mkdir(parents=True)
    for name in ['config.py','run_federated_cl_v4.py','fcl_models_v4.py','fcl_protocol.py','fcl_data_utils.py','fcl_derpp_reference.py']:
        shutil.copy2(PROJECT/name,code/name)
    for name in ['federated_contract.py','run_doh_validation.py']:
        shutil.copy2(PROJECT/'data_pipeline'/name,code/'data_pipeline'/name)
    save(out/'code_manifest.json',{str(p.relative_to(code)):sha(p) for p in sorted(code.rglob('*.py'))})
    save(out/'plan.json',dict(data=str(data.resolve()),data_sha256=sha(data),methods=METHODS,seeds=SEEDS,validation_candidates=PARAMS,
        pilot=dict(seed=42,rounds=[1,1,1],local_epochs=1,lr=.001,batch_size=64),rounds=[5,5,5],replay_capacity_samples=240,
        preprocessing='log1p statistics then fixed z-score fit ONLY Task_1/train; sequence /255; no future-task, val or test fit',
        validation='two matched candidates per method at seed42, selection by seen-task mean val Macro-F1; then fixed for all three final seeds',
        test_policy='test excluded from tuning and pilot promotion; final test only after parameters locked',
        comparison='FedDIA and NoFreezeReplay share architecture; SmallNoFreezeReplay hidden64 vs128, same rounds, not budget-optimized; adapted DER++ no freeze with reservoir online logits',
        communication='CPU tensor state serialized using pickle default protocol, full downlink each client-round, actual selected upload tensors; excludes framing and control metadata; includes Task1',
        pilot_gate='finite loss/logits, positive updates/replay, all 3 tasks completed, correct frozen parameter invariance and positive backbone changes when unfrozen; no test-score gate'))


def worker(out,method,seed,phase,index):
    plan=json.loads((out/'plan.json').read_text());data_path=Path(plan['data'])
    if sha(data_path)!=plan['data_sha256']:raise ValueError('Data changed')
    code=out/'code'
    for rel,h in json.loads((out/'code_manifest.json').read_text()).items():
        if sha(code/rel)!=h:raise ValueError('Frozen source changed')
    sys.path.insert(0,str(code))
    os.environ.update(TARGET_DATASET='CIC_DOH_SOURCE25',FCL_DATA_PATH=str(data_path),V4_VARIANT='FULL',V4_PRIOR_MODE='none',
        V4_COMPONENTS='adapter,se,cosface,replay',V4_FREEZE_BACKBONE='1' if method=='FedDIA' else '0',
        V4_REPLAY_METHOD='derpp' if method=='DERPPNoFreeze' else 'er',V4_REPLAY_CAPACITY='240',
        V4_ROUNDS_PER_TASK='1,1,1' if phase=='pilot' else '5,5,5',V4_TASK_ORDER='1,2,3',V4_START_TASK='0',V4_FIXED_UPDATES='0',
        V4_RECORD_TASK1_CURVE='0',NUM_CLIENTS='3',NUM_TASKS='3',SEED=str(seed))
    import torch
    from sklearn.metrics import f1_score,recall_score,confusion_matrix
    import run_federated_cl_v4 as v
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    torch.set_num_threads(2);device=torch.device('cuda:0')
    target=out/phase/method/(f'trial_{index}' if phase=='tune' else f'seed_{seed}');target.mkdir(parents=True,exist_ok=True)
    if (target/'result.json').exists():return
    data,labels=v.load_data_and_labels()
    first=[s['stat_raw'] for c in range(3) for s in v._split(data,c,0,'train')]
    x=np.log1p(np.asarray(first,dtype=np.float64));mean=x.mean(0);scale=x.std(0);scale=np.where(scale<1e-6,1.,scale)
    save(target/'preprocessing.json',dict(stat='(log1p(x)-mean)/scale',fit_split='Task_1/train',fit_samples=len(first),mean=mean.tolist(),scale=scale.tolist(),seq='uint8 / 255'))
    v.normalize_stat_feature=lambda x:((np.log1p(np.asarray(x,dtype=np.float64))-mean)/scale).astype(np.float32)
    base_seq=v.normalize_seq_feature;v.normalize_seq_feature=lambda x:base_seq(x)/255.
    if phase=='pilot':params=dict(lr=.001,batch_size=64,local_epochs=1)
    elif phase=='tune':params=plan['validation_candidates'][index]
    else:params=json.loads((out/'selected_params.json').read_text())[method]['params']
    args=v.V4Args()
    for k,value in params.items():setattr(args,k,value)
    args.hidden_dim=64 if method=='SmallNoFreezeReplay' else 128
    base=v.V4Client
    if method=='DERPPNoFreeze':
        from fcl_derpp_reference import ReferenceDERPPClient
        base=ReferenceDERPPClient
    records=[];losses=[]
    class Measured(base):
        def train(self,model,loader,task_id,logger):
            before={k:p.detach().cpu().clone() for k,p in model.named_parameters() if 'adapter' not in k and 'classifier' not in k}
            down=len(pickle.dumps({k:x.detach().cpu() for k,x in model.state_dict().items()}))
            old=torch.Tensor.backward;local_losses=[]
            def backward(t,*a,**kw):
                if t.numel()==1:local_losses.append(float(t.detach().cpu()))
                return old(t,*a,**kw)
            torch.Tensor.backward=backward
            try:state,keys,resources=super().train(model,loader,task_id,logger)
            finally:torch.Tensor.backward=old
            changed=any(not torch.equal(before[k],p.detach().cpu()) for k,p in model.named_parameters() if k in before)
            frozen=task_id>0 and method=='FedDIA'
            if frozen and changed:raise ValueError('Frozen backbone parameters changed')
            if not frozen and not changed:raise ValueError('Unfrozen backbone failed to update')
            if not local_losses or not np.isfinite(local_losses).all():raise ValueError('Nonfinite or missing training loss')
            losses.extend(local_losses)
            records.append(dict(task=task_id+1,client=self.client_id,backbone_frozen=frozen,backbone_changed=changed,
                upload_bytes=len(pickle.dumps({k:state[k] for k in keys})),download_bytes=down,buffer_samples=len(self.replay_buffer),
                buffer_serialized_bytes=len(pickle.dumps(self.replay_buffer)),loss_mean=float(np.mean(local_losses)),**resources))
            return state,keys,resources
    v.V4Client=Measured;v.set_seed(seed);started=time.monotonic()
    metrics,state=v.run_fcl_pipeline(args,data,labels,device,v.setup_logger(str(target/'training.log')),'val')
    torch.save(state,target/'model.pth')
    model=v.build_model(args,labels,[1]*len(labels),v.component_flags(),device);model.load_state_dict(state);model.eval()
    stage_test={};pooled=None;recall=None
    if phase=='final':
        # Evaluate saved endpoints; test scores never select a checkpoint or hyperparameter.
        paths=sorted((target/'checkpoints').glob('task_*.pth'))
        if len(paths)!=3:raise ValueError('Missing stage endpoint checkpoints')
        for step,path in enumerate(paths):
            model.load_state_dict(torch.load(path,map_location=device,weights_only=False))
            stage_test[str(step+1)]={str(t+1):v.evaluate(model,v.make_global_loader(data,t,'test',labels,args.batch_size),device) for t in range(step+1)}
        model.load_state_dict(state);ys=[];ps=[];probs=[]
        with torch.no_grad():
            for task in range(3):
                for sx,qx,y in v.make_global_loader(data,task,'test',labels,args.batch_size):
                    logits=model(sx.to(device),qx.to(device));assert torch.isfinite(logits).all()
                    ys.extend(y.tolist());ps.extend(logits.argmax(1).cpu().tolist());probs.extend(torch.softmax(logits,1).cpu().tolist())
        pooled=100*float(f1_score(ys,ps,labels=list(labels.values()),average='macro',zero_division=0))
        recall={k:float(recall_score(ys,ps,labels=[i],average=None,zero_division=0)[0]) for k,i in labels.items()}
        np.savez_compressed(target/'test_predictions.npz',y=np.array(ys),pred=np.array(ps),prob=np.array(probs))
        save(target/'confusion_matrix.json',confusion_matrix(ys,ps,labels=list(labels.values())).tolist())
    r=metrics['training_resources']
    if r['total_optimizer_updates']<=0 or r['total_replay_samples']<=0:raise ValueError('Missing updates or replay')
    save(target/'client_rounds.json',records)
    save(target/'result.json',dict(method=method,seed=seed,phase=phase,params=params,hidden_dim=args.hidden_dim,metrics=metrics,
        test_stage_endpoints=stage_test,pooled_test_macro_f1_percent=pooled,test_recall_fraction=recall,
        upload_bytes=sum(x['upload_bytes'] for x in records),download_bytes=sum(x['download_bytes'] for x in records),
        finite_loss=True,first_loss=float(np.mean(losses[:min(10,len(losses))])),last_loss=float(np.mean(losses[-min(10,len(losses)):])),
        seconds=time.monotonic()-started,training_device=torch.cuda.get_device_name(0),data_sha256=plan['data_sha256']))


def controller(out):
    code=out/'code';script=code/'data_pipeline/run_doh_validation.py'
    def queue(gpu,jobs):
        for phase,method,seed,index in jobs:
            target=out/phase/method/(f'trial_{index}' if phase=='tune' else f'seed_{seed}');target.mkdir(parents=True,exist_ok=True)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
                V4_TASK_CHECKPOINT_ROOT=str(target/'checkpoints'))
            with (target/'console.log').open('a') as log:
                subprocess.run([sys.executable,'-B',str(script),'worker','--output',str(out),'--method',method,'--seed',str(seed),'--phase',phase,'--index',str(index)],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    def both(jobs):
        with ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(queue,gpu,jobs[gpu::2]) for gpu in range(2)]
            for f in fs:f.result()
    save(out/'status.json',dict(stage='pilot',training_started=True))
    both([('pilot',m,42,0) for m in METHODS[:2]])
    save(out/'pilot_gate.json',dict(passed=True,basis='Both complete validation-only pilots passed resource, finite-loss, replay and frozen/unfrozen parameter checks; no test used'))
    save(out/'status.json',dict(stage='validation_search',pilot_passed=True))
    both([('tune',m,42,i) for m in METHODS for i in range(2)])
    selected={}
    for m in METHODS:
        results=[json.loads((out/'tune'/m/f'trial_{i}/result.json').read_text()) for i in range(2)]
        index=max(range(2),key=lambda i:results[i]['metrics']['val_macro_f1'])
        selected[m]=dict(params=PARAMS[index],trial=index,val_macro_f1_percent=results[index]['metrics']['val_macro_f1'])
    save(out/'selected_params.json',selected);save(out/'status.json',dict(stage='multi_seed_final',pilot_passed=True))
    both([('final',m,s,0) for m in METHODS for s in SEEDS])
    report(out);save(out/'status.json',dict(stage='complete',final_runs=12,pilot_passed=True))


def report(out):
    rows=[];text=['# 新DoH四类 V4复验','','新25维、固定来源split、模拟客户端；与旧190次结果分表。数值为百分数，误差为3个seed的样本标准差。','',
        '| 方法 | Pooled Macro-F1 | 最终任务平均F1 | 上传+下载MB |','| --- | ---: | ---: | ---: |']
    for m in METHODS:
        runs=[json.loads((out/'final'/m/f'seed_{s}/result.json').read_text()) for s in SEEDS]
        f=[r['pooled_test_macro_f1_percent'] for r in runs];b=[(r['upload_bytes']+r['download_bytes'])/1e6 for r in runs]
        row=dict(method=m,seeds=SEEDS,pooled_macro_f1_mean=float(np.mean(f)),pooled_macro_f1_std=float(np.std(f,ddof=1)),
            final_task_average_f1_mean=float(np.mean([np.mean([x['macro_f1'] for x in r['test_stage_endpoints']['3'].values()]) for r in runs])),
            duplex_MB_mean=float(np.mean(b)))
        rows.append(row);text.append(f'| {m} | {row["pooled_macro_f1_mean"]:.2f} ± {row["pooled_macro_f1_std"]:.2f} | {row["final_task_average_f1_mean"]:.2f} | {row["duplex_MB_mean"]:.2f} |')
    text+=['','首轮两个pilot只读取验证集；每方法两组相同超参候选按验证集选定，再锁参执行42/43/44三个seed。最终阶段逐任务测试指标、逐类召回、预测分数、实际通信/回放/更新记录见final/。',
        '', '预处理仅拟合Task_1/train的log1p统计均值/标准差，全程固定；序列/255。DP关闭。FedDIA固定Task1后冻结主干；NoFreezeReplay其余设置相同。SmallNoFreezeReplay为hidden64完整模型+回放，同轮数而非等通信预算搜索；DERPPNoFreeze为V4适配的在线logits/蓄水池DER++，保留V4 AdamW及余弦头。',
        '', '通信为序列化模型状态的逻辑上传/下载字节，包含初始阶段及每轮完整下发，不含网络协议开销；调参成本另列tune/，表格为单次最终训练成本。不同内存语义不代表等内存字节。',
        '', '来源组仅在train/val/test之间隔离，客户端共享采集来源；正常test仅41个，不推断Low-FPR优势。阶段任务样本量不均，任务平均F1与pooled F1不同。无Pareto前沿或预算最优声明。']
    save(out/'summary.json',rows);(out/'RESULTS.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['start','run','worker','report']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--method',choices=METHODS);p.add_argument('--seed',type=int,default=42);p.add_argument('--phase',choices=['pilot','tune','final']);p.add_argument('--index',type=int,default=0)
    a=p.parse_args();out=a.output.resolve()
    if a.action=='start':prepare(out,a.data);controller(out)
    elif a.action=='run':controller(out)
    elif a.action=='worker':worker(out,a.method,a.seed,a.phase,a.index)
    else:report(out)
