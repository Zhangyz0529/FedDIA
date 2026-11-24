#!/usr/bin/env python3
"""Audit candidate source capacity before fixing a matched-coverage experiment.

A family count is an upper bound: shared-day, duplicate and overlap constraints
can merge families. Filename dates are only provisional evidence, never a
replacement for PCAP timestamps, accepted labels or source provenance checks.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
import pandas as pd
from federated_contract import sha


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def provisional_components(families):
    parent={k:k for k in families}
    def root(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    keys=sorted(families);edges=[]
    for n,a in enumerate(keys):
        for b in keys[:n]:
            common=families[a]&families[b]
            if common:parent[root(b)]=root(a);edges.append(dict(left=a,right=b,filename_days=sorted(common)))
    return {k:root(k) for k in keys},edges


def audit(raw,parent,runs,out):
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir(parents=True)
    meta=json.loads((parent/'file_manifest.json').read_text());used={x['acquisition_family'] for x in meta}
    records=[];archives=[]
    for archive in sorted((raw/'PCAPs').rglob('*.zip')):
        if 'extracted_pcaps' in archive.parts:continue
        with zipfile.ZipFile(archive) as z:
            entries=[m for m in z.infolist() if m.filename.lower().endswith(('.pcap','.pcapng'))]
            central=[dict(member=m.filename,bytes=m.file_size,crc32=f'{m.CRC:08x}') for m in entries]
            archives.append(dict(path=str(archive.resolve()),pcap_members=len(entries),uncompressed_bytes=sum(m.file_size for m in entries),
                central_metadata_sha256=hashlib.sha256(json.dumps(central,sort_keys=True).encode()).hexdigest()))
            for m in entries:
                stem=Path(m.filename).stem;family=re.split(r'_doh\d+_',stem)[0]
                if archive.name.startswith('BenignDoH_NonDoH-'):
                    label='Benign';family='release_capture_family:'+archive.name
                else:
                    label='dns2tcp' if stem.startswith('dns2tcp_') else 'dnscat2' if stem.startswith('dsncat2_') else 'iodine' if stem.startswith('iodine_') else None
                    if label is None or family==stem:continue
                    family='scenario:'+family
                named_day=re.search(r'(20\d\d-\d\d-\d\d)T',stem)
                records.append(dict(label=label,family=family,archive=str(archive.relative_to(raw)),member=m.filename,bytes=m.file_size,crc32=f'{m.CRC:08x}',
                    filename_day=named_day.group(1) if named_day else None,old_version_family=family in used,
                    verification='ZIP central metadata only; raw CRC/PCAP window/accepted support not yet certified'))
    inventory=pd.DataFrame(records)
    inventory.to_parquet(out/'archive_member_inventory.parquet',index=False);save(out/'archive_catalog.json',archives)
    families=inventory.groupby(['label','family']).agg(archive_members=('member','size'),uncompressed_bytes=('bytes','sum'),
        old_version_family=('old_version_family','max'),archives=('archive','nunique')).reset_index()
    # Different ZIP members can be copies; count source families, never members.
    families.to_csv(out/'family_capacity.csv',index=False)
    capacity=[];graphs=[]
    for label in ['Benign','dns2tcp','dnscat2','iodine']:
        f=families[families.label==label];n=len(f);used_count=int(f.old_version_family.sum());fresh=n-used_count
        capacity.append(dict(label=label,total_family_upper_bound=n,old_used_families=used_count,unused_family_upper_bound=fresh,
            minimum_total_families_for_two_train_two_val_one_test=5,minimum_fresh_eval_families_for_two_val_one_test=3,
            upper_bound_can_meet_minimum=(n>=5 and fresh>=3),
            certification='Upper bound only; no independent source support certified by this inventory'))
        known=inventory[(inventory.label==label)&inventory.filename_day.notna()]
        days={k:set(g.filename_day) for k,g in known.groupby('family')}
        components,edges=provisional_components(days)
        graphs.append(dict(label=label,families_with_filename_days=len(days),provisional_components=len(set(components.values())),
            family_component=components,shared_filename_day_edges=edges,
            caveat='Filename-day graph is provisional. PCAP windows required; duplicates/packet overlaps can further merge. Not an accepted split.'))
    save(out/'capacity.json',capacity);save(out/'provisional_day_graph.json',graphs)
    # Independent recomputation of final pooled scores from saved predictions.
    import numpy as np
    from sklearn.metrics import f1_score
    checks=[]
    for method in ['FedDIA','NoFreezeReplay','SmallNoFreezeReplay','DERPPNoFreeze']:
        for seed in [42,43,44]:
            folder=runs/'final'/method/f'seed_{seed}';d=json.loads((folder/'result.json').read_text())
            with np.load(folder/'test_predictions.npz',allow_pickle=False) as a:
                value=100*float(f1_score(a['y'],a['pred'],labels=[0,1,2,3],average='macro',zero_division=0))
            if not np.isclose(value,d['pooled_test_macro_f1_percent']):raise ValueError('Result recomputation mismatch')
            checks.append(dict(method=method,seed=seed,recomputed_pooled_f1=value,matched=True))
    save(out/'independent_prediction_recheck.json',checks)
    blocked=[x['label'] for x in capacity if not x['upper_bound_can_meet_minimum']]
    policy=dict(status='BLOCKED_FULL_FOUR_CLASS_FRESH_EVALUATION' if blocked else 'CANDIDATE_SOURCE_AUDIT_REQUIRED',blocked_classes=blocked,
        question='At fixed new-class sample count and optimizer updates, does source coverage improve held-out-source recall?',
        method='NoFreezeReplay',normal_policy='same normal samples in both arms',new_class_train_samples_per_arm=60,
        arms=dict(single_source='60 unique samples from one training group; rotate eligible groups',multi_source='60 unique samples balanced over 2 or 3 training groups'),
        source_count_gate='At least 2 training groups, 2 fresh validation groups and 1 fresh test group per evaluated class; every training group must support the matched single-source arm',
        evaluation='Shared grouped validation and test; primary new-class recall at introduction, by source, then final retention; no aggregate-only selection',
        training_controls=['same client assignment rule','same task order','same actual optimizer-update budget','same Task1-only preprocessing fit on identical Task1 samples','same replay rule and capacity'],
        instrumentation=['actual current/replay labels per optimizer step','task-end buffer label counts','per-source metrics','sample_id/source/split hashes'],
        weighted_CE_followup='Only after source control; same multi-source training data, alter current-task CE only, replay term unchanged',
        holdout='Old val/test are diagnosed sources. No new evaluation groups have been frozen by this blocked proposal; candidate metadata are not new test data.',
        no_training=True)
    save(out/'protocol_candidate.json',policy)
    text=['# 来源覆盖对照的可行性盘点','',
        '本次已读取发布ZIP目录、旧来源manifest并重算12次最终预测。尚未生成或训练新对照；来源上界已发现完整四类方案的硬性不足。',
        '', '| 类别 | 发布场景族上界 | 旧版已用 | 未用场景族上界 | 满足新评价最低来源数的上界条件 |',
        '| --- | ---: | ---: | ---: | --- |']
    for x in capacity:text.append(f'| {x["label"]} | {x["total_family_upper_bound"]} | {x["old_used_families"]} | {x["unused_family_upper_bound"]} | {"可能，待核验" if x["upper_bound_can_meet_minimum"] else "不足"} |')
    text+=['','“多来源训练≥2＋多来源验证≥2＋新测试≥1”至少需要5个互不跨split的来源组。dns2tcp发布场景族只有4个，旧版已用3个，最多只剩1个新族。即使允许旧诊断来源转入训练，也不能同时留下2个全新验证组和1个全新测试组。文件再多不改变此来源上界。',
        '', '上表按旧规则保持整个场景族/正常发布采集队列完整；尚未核验全部新候选CRC、原始时间范围、包重叠或接受标签支持，因此不是“确认独立来源”的数量。同一天、重复及重叠约束只能进一步减少组数。',
        '', '另一个风险：按所有发布文件名日期构造的候选图，会将部分旧版分开的场景族连通。它提示旧组ID不能直接沿用到扩展全集；必须对拟使用的完整来源宇宙重新计算分组并保存跨版本关联。文件名日期只是初筛，不能代替PCAP时间窗。',
        '', '可保留的下一步：优先核验dnscat2和iodine的未用场景，在相同新类N=60下构造单来源/多来源候选；正常来源需同步补足。若保留dns2tcp作初始参考任务，其旧评估只能作为诊断背景，不能将整个四类pooled结果称作全新来源确认性评价。另一条路线是新增可追溯的独立dns2tcp采集。两者属于不同评价范围，应明确选择后再冻结实际source_id清单。',
        '', 'protocol_candidate.json给出样本量、更新数、预处理、回放、评价与真实逐类抽样记录约束；状态为blocked，不是已冻结的新split。不通过来源验收时不启动训练，也不靠重复样本/拆同族文件凑来源。',
        '', '科学结论边界：重算的12次pooled分数与已有结果一致；旧三seed仅反映同一划分的训练波动。进一步来源实验需要来源层面的评价支持。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    save(out/'audit_inputs.json',{str(p.resolve()):sha(p) for p in [parent/'file_manifest.json',runs/'plan.json',runs/'summary.json']})
    code=out/'code';code.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('federated_contract.py')]:shutil.copy2(p,code/p.name)
    save(out/'code_manifest.json',{p.name:sha(p) for p in code.iterdir()})
    print(json.dumps(dict(status=policy['status'],capacity=capacity,prediction_rechecks=len(checks)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['raw','parent','runs','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.raw.resolve(),a.parent.resolve(),a.runs.resolve(),a.output.resolve())
