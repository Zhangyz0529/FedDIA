#!/usr/bin/env python3
"""Read-only diagnosis: class exposure, replay memory, source shift and saved models.

ER class exposure is an expectation from exact reconstructed memory counts, not
an observed per-class replay trace. DER++ logs cannot recover reservoir labels.
No parameter updates or model fitting are performed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from federated_contract import sha


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def diagnose(parent,runs,federated,out):
    if out.exists():raise ValueError('Fresh diagnostic directory required')
    out.mkdir(parents=True);assignment=pd.read_parquet(federated/'assignment.parquet');provenance=pd.read_parquet(parent/'selected/provenance.parquet')
    fields=json.loads((parent/'feature_schema.json').read_text())['fields']
    assignment.groupby(['task','split','label']).size().rename('samples').reset_index().to_csv(out/'class_support.csv',index=False)
    assignment.groupby(['client','task','split','label']).size().rename('samples').reset_index().to_csv(out/'client_support.csv',index=False)
    current=[];replay=[];replay_gaps=[]
    for method in ['FedDIA','NoFreezeReplay','SmallNoFreezeReplay','DERPPNoFreeze']:
        for seed in [42,43,44]:
            records=json.loads((runs/'final'/method/f'seed_{seed}/client_rounds.json').read_text())
            result=json.loads((runs/'final'/method/f'seed_{seed}/result.json').read_text())
            epochs=result['params']['local_epochs']
            for client in range(3):
                memory=Counter()
                for task in range(1,4):
                    name=f'Client_{chr(65+client)}';taskname=f'Task_{task}'
                    subset=assignment[(assignment.client==name)&(assignment.task==taskname)&(assignment.split=='train')]
                    counts=Counter(subset.label);calls=[x for x in records if x['task']==task and x['client']==client]
                    rounds=len(calls)
                    for label,n in counts.items():current.append(dict(method=method,seed=seed,client=name,task=task,label=label,unique_samples=n,
                        rounds=rounds,epochs=epochs,exact_current_presentations=n*epochs*rounds))
                    if method=='DERPPNoFreeze':
                        replay_gaps.append(dict(method=method,seed=seed,client=name,task=task,observed_replay_samples=sum(x['replay_samples'] for x in calls),
                            reason='Online reservoir label contents and each replay draw were not logged; exact per-class exposure cannot be recovered'))
                        continue
                    observed=sum(x['replay_samples'] for x in calls)
                    assert all(x['buffer_samples']==sum(memory.values()) for x in calls)
                    for label,n in memory.items():replay.append(dict(method=method,seed=seed,client=name,task=task,label=label,memory_class_count=n,
                        memory_total=sum(memory.values()),observed_total_replay_draws=observed,
                        expected_class_replay_draws=observed*n/sum(memory.values()),class_fraction=n/sum(memory.values()),
                        evidence='Exact buffer label counts; expected random draws, not observed per-class replay'))
                    memory.update(counts);quota=max(1,240//len(memory));memory=Counter({k:min(v,quota) for k,v in memory.items()})
    pd.DataFrame(current).to_csv(out/'current_exposure.csv',index=False)
    pd.DataFrame(replay).to_csv(out/'er_replay_expectations.csv',index=False)
    save(out/'missing_replay_trace.json',replay_gaps)
    # Source conditional distributions: log1p stats, robust scale from same-class train.
    inputs={};source_rows=[];shifts=[];sequence=[]
    for split in ['train','val','test']:
        with np.load(parent/'selected'/f'{split}.npz',allow_pickle=False) as a:inputs[split]={k:a[k].copy() for k in a.files}
    for label in ['Benign','dns2tcp','dnscat2','iodine']:
        train=inputs['train'];mask=train['label']==label;x=np.log1p(train['stat_raw'][mask].astype(np.float64));iqr=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
        scale=np.where(iqr>1e-6,iqr,1.);train_signatures={v.tobytes() for v in train['seq_feature'][mask]}
        for split in ['train','val','test']:
            f=provenance[(provenance.label==label)&(provenance.split==split)];a=inputs[split];m=a['label']==label;y=np.log1p(a['stat_raw'][m].astype(np.float64))
            z=a['stat_raw'][m];s=a['seq_feature'][m]
            source_rows.append(dict(label=label,split=split,n=len(f),source_groups=f.source_group_id.nunique(),captures=f.capture_hash.nunique(),
                start_utc=pd.to_datetime(f.start_timestamp.min(),unit='s',utc=True).isoformat(),end_utc=pd.to_datetime(f.end_timestamp.max(),unit='s',utc=True).isoformat(),
                packet_count_median=float(f.packet_count.median()),duration_s_median=float(np.median(z[:,0])),
                sequence_valid_length_median=float(f.sequence_valid_length.median()),missing_initial_syn_fraction=float(f.missing_initial_syn.mean())))
            sequence.append(dict(label=label,split=split,unique_sequence_prefixes=len({v.tobytes() for v in s}),samples=len(s),
                exact_prefix_seen_in_same_class_train_fraction=float(np.mean([v.tobytes() in train_signatures for v in s]))))
            if split!='train':
                for j,field in enumerate(fields):shifts.append(dict(label=label,split=split,feature=field,
                    log1p_wasserstein_train_IQR=wasserstein_distance(x[:,j],y[:,j])/scale[j],
                    log1p_train_IQR=float(iqr[j]),fallback_unit_scale=bool(iqr[j]<=1e-6),
                    outside_same_class_train_range_fraction=float(np.mean((y[:,j]<x[:,j].min())|(y[:,j]>x[:,j].max()))),
                    train_median=float(np.median(train['stat_raw'][mask,j])),target_median=float(np.median(z[:,j]))))
    pd.DataFrame(source_rows).to_csv(out/'source_support_and_times.csv',index=False)
    pd.DataFrame(shifts).to_csv(out/'conditional_stat_shift.csv',index=False)
    pd.DataFrame(sequence).to_csv(out/'sequence_overlap.csv',index=False)
    # Stage1-only normalization outlier audit, distinct from label-conditional shift.
    prep=json.loads((runs/'final/FedDIA/seed_42/preprocessing.json').read_text());mu=np.array(prep['mean']);sd=np.array(prep['scale']);normalization=[]
    for split,a in inputs.items():
        for label in ['Benign','dns2tcp','dnscat2','iodine']:
            z=(np.log1p(a['stat_raw'][a['label']==label].astype(np.float64))-mu)/sd
            normalization.append(dict(label=label,split=split,mean_feature_fraction_abs_z_gt5=float(np.mean(np.abs(z)>5)),
                sample_fraction_any_abs_z_gt5=float(np.mean((np.abs(z)>5).any(1))),maximum_abs_z=float(np.abs(z).max())))
    pd.DataFrame(normalization).to_csv(out/'normalization_shift.csv',index=False)
    save(out/'input_hashes.json',{str(p.resolve()):sha(p) for p in [federated/'dataset.pkl',federated/'assignment.parquet',parent/'selected/provenance.parquet',*[parent/'selected'/f'{s}.npz' for s in ['train','val','test']]]})
    header_profiles(inputs,out)
    infer(runs,out)


def header_profiles(inputs,out):
    """Inspect observable preserved fields; no inference of OS/application truth."""
    import dpkt
    rows=[]
    for split,a in inputs.items():
        for sid,label,seq in zip(a['sample_id'],a['label'],a['seq_feature']):
            b=bytes(seq);r=dict(sample_id=sid,label=label,split=split,version=int(b[0]>>4),parsed_ipv4_tcp=False)
            if b[0]>>4==4 and b[9]==6:
                ihl=(b[0]&15)*4;offset=(b[ihl+12]>>4)*4
                if ihl>=20 and offset>=20 and ihl+offset<=512:
                    opts=dpkt.tcp.parse_opts(b[ihl+20:ihl+offset])
                    r.update(parsed_ipv4_tcp=True,ip_header_bytes=ihl,tcp_header_bytes=offset,
                             tcp_timestamp_option=any(x and x[0]==8 for x in opts),ttl=int(b[8]),
                             tcp_window=int.from_bytes(b[ihl+14:ihl+16],'big'))
            rows.append(r)
    f=pd.DataFrame(rows);f.to_parquet(out/'first_packet_headers.parquet',index=False)
    f.groupby(['label','split']).agg(samples=('sample_id','size'),ipv4_tcp_parsed=('parsed_ipv4_tcp','sum'),
        median_tcp_header_bytes=('tcp_header_bytes','median'),timestamp_option_fraction=('tcp_timestamp_option','mean'),
        median_ttl=('ttl','median'),median_tcp_window=('tcp_window','median')).reset_index().to_csv(out/'first_packet_header_summary.csv',index=False)


def infer(runs,out):
    import os
    os.environ.update(TARGET_DATASET='CIC_DOH_SOURCE25',V4_VARIANT='FULL',V4_COMPONENTS='adapter,se,cosface,replay',V4_PRIOR_MODE='none')
    sys.path.insert(0,str(runs/'code'))
    import torch
    from sklearn.metrics import recall_score,f1_score
    import run_federated_cl_v4 as v
    plan=json.loads((runs/'plan.json').read_text());v.CFG['fcl_data_path']=plan['data']
    torch.set_num_threads(2);data,labels=v.load_data_and_labels();rows=[]
    for method in ['FedDIA','NoFreezeReplay']:
        for seed in [42,43,44]:
            folder=runs/'final'/method/f'seed_{seed}';p=json.loads((folder/'preprocessing.json').read_text());mu=np.array(p['mean']);sd=np.array(p['scale'])
            v.normalize_stat_feature=lambda x:((np.log1p(np.asarray(x,dtype=np.float64))-mu)/sd).astype(np.float32)
            v.normalize_seq_feature=lambda x:np.asarray(x,dtype=np.float32).reshape(512)/255
            args=v.V4Args();model=v.build_model(args,labels,[1]*4,v.component_flags(),torch.device('cpu'));model.eval()
            for stage in range(1,4):
                model.load_state_dict(torch.load(folder/'checkpoints'/f'task_{stage}.pth',map_location='cpu',weights_only=False))
                for task in sorted({stage,1} if stage!=3 else {1,2,3}):
                    for split in ['train','val','test']:
                        ys=[];ps=[];prob=[]
                        with torch.no_grad():
                            for sx,qx,y in v.make_global_loader(data,task-1,split,labels,64):
                                output=model(sx,qx);assert torch.isfinite(output).all()
                                ys.extend(y.tolist());ps.extend(output.argmax(1).tolist());prob.extend(torch.softmax(output,1).tolist())
                        ys=np.array(ys);ps=np.array(ps);prob=np.array(prob)
                        for name,lab in labels.items():
                            mask=ys==lab
                            if not mask.any():continue
                            rows.append(dict(method=method,seed=seed,stage=stage,evaluated_task=task,split=split,label=name,support=int(mask.sum()),
                                recall=float(np.mean(ps[mask]==lab)),mean_true_class_probability=float(prob[mask,lab].mean()),
                                predicted_as={k:int(np.sum(ps[mask]==i)) for k,i in labels.items()}))
            print('inference complete',method,seed,flush=True)
    save(out/'checkpoint_class_diagnostics.json',rows)
    table=pd.DataFrame([{k:v for k,v in x.items() if k!='predicted_as'} for x in rows]);table.to_csv(out/'checkpoint_class_diagnostics.csv',index=False)
    summary=table.groupby(['method','stage','evaluated_task','split','label'])[['recall','mean_true_class_probability']].mean().reset_index()
    summary.to_csv(out/'checkpoint_class_mean.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','runs','federated','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();diagnose(a.parent.resolve(),a.runs.resolve(),a.federated.resolve(),a.output.resolve())
