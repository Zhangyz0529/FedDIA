#!/usr/bin/env python3
"""Distinguish zero-packet source families from potential evaluation support."""
import argparse
import json
from pathlib import Path
import shutil
import pandas as pd
from audit_doh_full_sources import build_graph
from federated_contract import sha


def refine(out,parent):
    dest=out/'support_checks'
    if dest.exists():raise ValueError('Fresh support check directory required')
    dest.mkdir();f=pd.read_parquet(out/'members.parquet');records=f.to_dict('records')
    for r in records:
        if r['parsed_packets']==0:r['member_sha256']='unique_empty:'+r['archive']+'/'+r['member']
        if pd.isna(r['start']):r['start']=None
        if pd.isna(r['end']):r['end']=None
    ids,_,touched,_,_=build_graph(records,json.loads((parent/'file_manifest.json').read_text()),pd.read_parquet(parent/'selected/provenance.parquet'),json.loads((parent/'source_group_edges.json').read_text()))
    f['sensitivity_component']=f.family.map(ids);rows=[]
    for label,g in f.groupby('label'):
        fresh=set(g.sensitivity_component)-touched;nonempty=set(g[g.parsed_packets>0].sensitivity_component)
        rows.append(dict(label=label,untouched_components_in_sensitivity=len(fresh),untouched_components_with_parsed_packets=len(fresh&nonempty),
            untouched_empty_only_components=len(fresh-nonempty),accepted_label_support='NOT_EXTRACTED',official_status='BLOCKED_SOURCE_GROUP_REAUDIT'))
    pd.DataFrame(rows).to_csv(dest/'empty_duplicate_sensitivity_support.csv',index=False)
    families=f.groupby(['label','family']).agg(members=('member','size'),parsed_packets=('parsed_packets','sum')).reset_index()
    families[families.parsed_packets==0].to_csv(dest/'empty_only_families.csv',index=False)
    feasibility=pd.read_csv(out/'split_feasibility.csv')
    feasibility['optimistic_additional_component_lower_bound']=feasibility.apply(lambda r:max(5-r.components_with_observed_edges,3-r.untouched_components_before_new_overlap_checks),axis=1)
    feasibility['budget_caveat']='Component lower bound only; not file count, acquisition guarantee, or accepted label support; rebuild global graph after acquisition'
    feasibility.to_csv(dest/'component_budget_lower_bounds.csv',index=False)
    note=['','对空重复连接的进一步核验：敏感性图中的3个“未触及旧诊断”的dnscat2分量全部只有空文件，对应txt-baseline_99911、txt-baseline_dnsadguardcom、txt-baseline_dnsgoogle三族，各60个空PCAP、零个包。它们不是新增可评价来源。即便仅在诊断中忽略空PCAP重复连接，有包的全新攻击分量仍为0。',
        '', '| 类别 | 敏感性图未触及分量 | 其中有包 | 其中仅空文件 |','| --- | ---: | ---: | ---: | ---: |']
    for r in rows:note.append(f'| {r["label"]} | {r["untouched_components_in_sensitivity"]} | {r["untouched_components_with_parsed_packets"]} | {r["untouched_empty_only_components"]} |')
    note+=['', '正常类暂未触及旧诊断的1个候选分量包含Firefox-AdGuard/CloudFlare/Quad9三个族、6个发布文件；它不是三个独立组，其中3个文件包尾截断。正式接受样本数与新增逐包重叠仍未核验。',
        '', '按现行官方图及至少2训练+2全新验证+1全新测试的必要条件，正常类至少还缺2个独立分量、各攻击类至少还缺4个；这是未计标签支持和后续合并的乐观下界，不能换算成同数量文件。多类别可共享同一分量，但该分量只能整体进入一个split，预算不得简单按类别独立相加。明细见support_checks/component_budget_lower_bounds.csv。']
    q=out/'REPORT.md';q.write_text(q.read_text()+'\n'.join(note)+'\n')
    code=dest/'code';code.mkdir()
    for name in ['refine_doh_graph_support.py','audit_doh_full_sources.py','federated_contract.py']:shutil.copy2(Path(__file__).with_name(name),code/name)
    (dest/'code_manifest.json').write_text(json.dumps({p.name:sha(p) for p in code.iterdir()},indent=2)+'\n')
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    a=p.parse_args();refine(a.output.resolve(),a.parent.resolve())
