#!/usr/bin/env python3
"""Export a reproducible simulated-client DoH stream without changing source splits.

Normal samples are partitioned over all client/task cells without duplication.
Each attack enters exactly one global stage. Clients share acquisition groups
within a split: this is a controlled logical federation, not independent sites.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import numpy as np
import pandas as pd
from federated_contract import FIELDS,SCHEMA,sha,validate_contract

SPLITS=('train','val','test')
KEY='CIC_DOH_SOURCE25'


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def ordered(indices,ids,seed,split,label):
    return sorted(indices,key=lambda i:hashlib.sha256(f'{seed}:{split}:{label}:{ids[i]}'.encode()).hexdigest())


def export(source,out,attacks=('dns2tcp','dnscat2','iodine'),clients=3,seed=42):
    if len(attacks)!=3 or len(set(attacks))!=3:raise ValueError('Exactly three distinct attack stages required')
    if clients<1:raise ValueError('clients must be positive')
    if out.exists():raise ValueError('Use a fresh output directory')
    schema=json.loads((source/'feature_schema.json').read_text())
    if schema['version']!=SCHEMA or schema['fields']!=FIELDS:raise ValueError('Unexpected ordered feature schema')
    selection=json.loads((source/'class_selection.json').read_text())
    if selection['export_status']!='EXPORTED_SOURCE_GROUP_SPLITS':raise ValueError('Parent export not certified')
    frame=pd.read_parquet(source/'selected/provenance.parquet')
    if not frame.sample_id.is_unique or frame.groupby('source_group_id').split.nunique().max()!=1:
        raise ValueError('Invalid parent identity/source splits')
    required={'Benign',*attacks}
    if set(frame.label)!=required:raise ValueError('Parent must contain exactly the requested four classes')
    out.mkdir(parents=True)
    support=frame.groupby(['label','split','source_group_id']).size().rename('samples').reset_index()
    support.to_csv(out/'source_support.csv',index=False)
    names=[f'Client_{chr(65+i)}' for i in range(clients)];tasks=[f'Task_{i+1}' for i in range(3)]
    data={c:{t:{s:[] for s in SPLITS} for t in tasks} for c in names};rows=[];parents={}
    for split in SPLITS:
        path=source/'selected'/f'{split}.npz';parents[str(path.resolve())]=sha(path)
        with np.load(path,allow_pickle=False) as a:
            ids=a['sample_id'];labels=a['label'];part=frame[frame.split==split].set_index('sample_id')
            if set(ids)!=set(part.index) or any(part.loc[sid,'label']!=label for sid,label in zip(ids,labels)):
                raise ValueError('Parent NPZ/provenance mismatch')
            allocation=[]
            for label in ['Benign',*attacks]:
                indices=ordered(np.flatnonzero(labels==label),ids,seed,split,label)
                cells=[(c,t) for t in tasks for c in names] if label=='Benign' else [(c,tasks[attacks.index(label)]) for c in names]
                if len(indices)<len(cells):raise ValueError(f'Insufficient {split}/{label} for nonempty cells')
                for n,i in enumerate(indices):allocation.append((i,*cells[n%len(cells)]))
            for i,c,t in allocation:
                sid=str(ids[i]);r=part.loc[sid]
                sample=dict(stat_raw=a['stat_raw'][i].copy(),seq_feature=a['seq_feature'][i].copy(),label=str(labels[i]),sample_id=sid)
                data[c][t][split].append(sample)
                rows.append(dict(sample_id=sid,label=str(labels[i]),split=split,client=c,task=t,source_group_id=r.source_group_id))
    assigned=pd.DataFrame(rows)
    if not assigned.sample_id.is_unique or set(assigned.sample_id)!=set(frame.sample_id):raise ValueError('Lost or repeated samples')
    checked=assigned.merge(frame[['sample_id','split','source_group_id']],on='sample_id',suffixes=('','_parent'),validate='one_to_one')
    assert (checked.split==checked.split_parent).all() and (checked.source_group_id==checked.source_group_id_parent).all()
    counts=assigned.groupby(['client','task','split','label']).size().rename('samples').reset_index()
    if len(counts)!=clients*3*3*2:raise ValueError('Missing required client/task/split/class support')
    assigned.to_parquet(out/'assignment.parquet',index=False);counts.to_csv(out/'cell_support.csv',index=False)
    assigned.groupby(['source_group_id','split','client','task','label']).size().rename('samples').reset_index().to_csv(out/'source_ownership.csv',index=False)
    shutil.copy2(source/'feature_schema.json',out/'feature_schema.json')
    path=out/'dataset.pkl';path.write_bytes(pickle.dumps(data,protocol=4))
    parents[str((source/'selected/provenance.parquet').resolve())]=sha(source/'selected/provenance.parquet')
    parents[str((source/'feature_schema.json').resolve())]=sha(source/'feature_schema.json')
    view=source.parent/'views/doh_min500/selected'
    overlap={}
    if view.exists():
        for split in SPLITS:
            with np.load(view/f'{split}.npz',allow_pickle=False) as a:
                other=set(a['sample_id']);ours=set(assigned.loc[assigned.split==split,'sample_id'])
                overlap[split]=dict(two_class_samples=len(other),shared_with_four_class=len(other&ours),two_class_subset=other<=ours)
    manifest=dict(dataset_registry_key=KEY,feature_schema=SCHEMA,schema_sha256=sha(out/'feature_schema.json'),dataset_sha256=sha(path),samples=len(assigned),
                  clients=names,tasks={t:['Benign',attack] for t,attack in zip(tasks,attacks)},seed=seed,parent_hashes=parents,
                  split_policy='parent splits unchanged',client_policy='hash-balanced sample partition within parent split; simulated clients share source groups',
                  normal_policy='disjoint normal samples across tasks and clients; no repeat exposure',training_started=False,
                  preprocessing='raw statistics and byte values preserved; no standardization fitted',two_class_overlap=overlap)
    save(out/'dataset_manifest.json',manifest)
    cfg=dict(stat_dim=25,feature_schema=SCHEMA)
    validate_contract(path,cfg,KEY,pickle.loads(path.read_bytes()))
    for wrong in [dict(stat_dim=25),dict(stat_dim=29,feature_schema=SCHEMA)]:
        try:validate_contract(path,wrong,'USTC')
        except ValueError:pass
        else:raise AssertionError('Legacy schema accepted Source25')
    audit=dict(status='EXPORTED_SIMULATED_3TASK_PROTOCOL',samples=len(assigned),minimum_cell_class_support=int(counts.samples.min()),
               all_parent_samples_assigned_once=True,parent_splits_preserved=True,source_groups_cross_split=0,
               source_groups_shared_between_clients=int((assigned.groupby('source_group_id').client.nunique()>1).sum()),
               source_exclusive_full_class_clients_possible=False,
               reason='Each class has only one source group within each parent split; three full-support independent-source clients cannot be constructed.',
               schema_contract_and_negative_checks_passed=True,training_started=False)
    save(out/'audit_report.json',audit)
    text=['# DoH 四类联邦分配与支持审计','',f'状态：{audit["status"]}；{len(assigned)}个样本，{clients}个模拟客户端、3个全局攻击增量阶段，seed={seed}。',
          '', '任务顺序预先固定：'+ ' → '.join(attacks)+'；Benign贯穿三个任务，正常样本不重复。客户端只在原split内部划分，不改变原来源/划分。',
          '', '| 类别 | train | val | test |','| --- | ---: | ---: | ---: |']
    for label in ['Benign',*attacks]:text.append('| '+label+' | '+' | '.join(str(len(frame[(frame.label==label)&(frame.split==s)])) for s in SPLITS)+' |')
    text+=['', '| 任务 | 新引入攻击 | train | val | test |', '| --- | --- | ---: | ---: | ---: |']
    for task,attack in zip(tasks,attacks):
        text.append('| '+task+' | '+attack+' | '+' | '.join(str(len(assigned[(assigned.task==task)&(assigned.split==s)])) for s in SPLITS)+' |')
    text+=['',f'每个client×task×split都有正常和该阶段攻击，最小单类支持{audit["minimum_cell_class_support"]}。',
           '', '**来源边界：** 每类每split仅一个来源组。客户端是共享采集来源的逻辑分区，不是独立组织/主机；正常来源组也跨任务。仅保证来源组不跨train/val/test，不声称客户端来源独立或真实时间演进。',
           '', 'source_support.csv：类别×split×来源组；cell_support.csv：客户端×任务×split×类别；source_ownership.csv：来源组在客户端/任务中的分布；assignment.parquet：逐sample_id唯一归属。',
           '', 'dataset.pkl供V4读取；dataset_manifest.json将其哈希、新特征schema及字段顺序绑定到CIC_DOH_SOURCE25。不得按旧USTC25或旧DoH29加载。原始统计值与序列字节保持不变，尚未拟合标准化。',
           '', 'V4沿用固定四类分类头，标签集合来自训练split；不是未知类别数量的开放集识别。分配策略依据支持和固定哈希，不依据验证/测试性能。',
           '', 'DoH两类视图与本四类数据重叠，详情见manifest；不能视为独立复验。正常val=59/test=41，单任务更少，不支撑低误报率结论。旧190次结果与新版本应分表。',
           '', '已完成导出及V4加载/模型前向检查时见v4_preflight.json；未训练。后续先FedDIA、NoFreezeReplay，再关键预算对照，报告阶段指标、pooled Macro-F1、逐类召回及上传/下载累计字节。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(audit,ensure_ascii=False))
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--clients',type=int,default=3);p.add_argument('--seed',type=int,default=42);p.add_argument('--attacks',default='dns2tcp,dnscat2,iodine')
    a=p.parse_args();export(a.input.resolve(),a.output.resolve(),tuple(a.attacks.split(',')),a.clients,a.seed)
