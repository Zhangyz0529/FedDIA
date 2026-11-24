#!/usr/bin/env python3
"""Reconcile global components with diagnosed samples and classify duplicate evidence.

The empty-file sensitivity table is diagnostic only. It does not change the
maintained grouping rule, blocked status, or any existing data split.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import pandas as pd
from audit_doh_full_sources import build_graph
from federated_contract import sha


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def summarize(out,parent):
    f=pd.read_parquet(out/'members.parquet');plan=pd.read_parquet(out/'member_plan.parquet')
    assert len(f)==len(plan)==13154 and f.crc_size_verified.all()
    assert not f.duplicated(['archive','member']).any()
    # Catalog-stage text is retained under its true temporal meaning.
    if 'verification' in f:
        f=f.rename(columns={'verification':'catalog_verification_before_scan'})
        f.to_parquet(out/'members.parquet',index=False)
    for p in (out/'archives').glob('*.json'):
        m=json.loads(p.read_text());raw=Path(json.loads((out/'plan.json').read_text())['raw'])/m['archive']
        assert (raw.stat().st_size,raw.stat().st_mtime_ns)==(m['archive_size'],m['archive_mtime_ns'])
    frame=pd.read_parquet(parent/'selected/provenance.parquet');oldmeta=json.loads((parent/'file_manifest.json').read_text());oldedges=json.loads((parent/'source_group_edges.json').read_text())
    hashcomp=f[['member_sha256','component']].drop_duplicates()
    assert hashcomp.member_sha256.is_unique
    linked=frame.merge(hashcomp,left_on='capture_hash',right_on='member_sha256',validate='many_to_one',how='left')
    assert not linked.component.isna().any()
    migration=linked.groupby(['source_group_id','label','split','component']).size().rename('accepted_samples').reset_index()
    migration.to_csv(out/'old_to_global_group_map.csv',index=False)
    cross=[]
    for comp,g in linked.groupby('component'):
        cross.append(dict(component=comp,old_source_groups=g.source_group_id.nunique(),old_splits=sorted(g.split.unique()),old_samples=len(g)))
    save(out/'diagnosed_component_history.json',cross)
    duplicates=[]
    for h,g in f.groupby('member_sha256'):
        if len(g)<2:continue
        duplicates.append(dict(sha256=h,members=len(g),families=g.family.nunique(),labels=sorted(g.label.unique()),
            only_empty_captures=bool((g.parsed_packets.fillna(-1)==0).all()),
            captures=g[['archive','member','label','family','timestamp_status','parsed_packets']].to_dict('records')))
    save(out/'duplicate_file_groups.json',duplicates)
    # Diagnose how much merging relies on identical zero-packet headers.
    sensitivity_records=f.to_dict('records')
    for r in sensitivity_records:
        if r['parsed_packets']==0:r['member_sha256']='unique_empty:'+r['archive']+'/'+r['member']
        if pd.isna(r['start']):r['start']=None
        if pd.isna(r['end']):r['end']=None
    ids,_,touched,_,_=build_graph(sensitivity_records,oldmeta,frame,oldedges)
    sensitivities=[]
    for label,g in f.groupby('label'):
        components={ids[x] for x in g.family}
        sensitivities.append(dict(label=label,nonempty_duplicate_sensitivity_components=len(components),
            nonempty_duplicate_sensitivity_untouched_components=len(components-touched),
            official_grouping_changed=False))
    save(out/'empty_duplicate_sensitivity.json',sensitivities)
    emptygroups=[x for x in duplicates if x['only_empty_captures']]
    # A failed parse is retained as unknown rather than silently excluded from the universe.
    f[f.timestamp_status!='FULL_SCAN_OK'].to_parquet(out/'empty_and_scan_anomalies.parquet',index=False)
    verification=dict(status='BLOCKED_SOURCE_GROUP_REAUDIT',all_13154_planned_members_scanned=True,all_member_CRC_size_verified=True,
        original_archives_stat_unchanged=True,accepted_old_samples_reconciled=len(linked),new_accepted_samples='NOT_EXTRACTED',
        new_packet_overlap='NOT_COMPLETED',identical_empty_file_groups=len(emptygroups),
        empty_duplicate_policy='Official literal duplicate-file rule retained; separate sensitivity excludes zero-packet duplicate links for interpretation only',
        global_graph_regression_tests=5,training_started=False,new_split_frozen=False)
    save(out/'final_verification.json',verification)
    text=['','补充核验：旧版4,409个已接受样本均已通过capture SHA映射到全局图，映射见old_to_global_group_map.csv；各分量的旧split历史见diagnosed_component_history.json。这直接记录了哪些旧局部分组在扩展后合并，不能继承旧source_group_id作为新来源身份。',
        '', f'重复文件证据中有{len(emptygroups)}组仅包含零包文件。相同空PCAP头的SHA相等不提供共享流量证据。按现行字面重复文件规则，官方图保留这些连接并维持阻塞；另存empty_duplicate_sensitivity.json，仅用于检查空头重复的影响，不自动变更协议或解锁实验。',
        '', '| 类别 | 排除零包重复边的诊断分量数 | 其中未触及旧诊断的分量 |','| --- | ---: | ---: |']
    for r in sensitivities:text.append(f'| {r["label"]} | {r["nonempty_duplicate_sensitivity_components"]} | {r["nonempty_duplicate_sensitivity_untouched_components"]} |')
    text+=['', '空采集、包尾截断/解析异常、新样本未提取、逐包重叠未核验是不同状态，不能互相替代。原ZIP CRC通过而文件包尾截断，表示发布成员的字节与归档相符但结构不完整；没有静默截断修复。已读完整包的时间可作为正向连通见证，不能据此认证整文件可训练。',
        '', '可行性判断仍维持BLOCKED_SOURCE_GROUP_REAUDIT：对任何拟缩小的类别集合，逐来源获取/保持结论须对应真正留出的评价分量；当前没有冻结新的评价source_id，也没有以四类pooled诊断指标替代确认性结果。']
    q=out/'REPORT.md';q.write_text(q.read_text()+'\n'.join(text)+'\n')
    code=out/'analysis_code';code.mkdir()
    for name in ['summarize_doh_full_graph.py','audit_doh_full_sources.py','federated_contract.py','test_full_source_graph.py']:
        shutil.copy2(Path(__file__).with_name(name),code/name)
    save(out/'analysis_code_manifest.json',{p.name:sha(p) for p in code.iterdir()})
    print(json.dumps(dict(verification=verification,empty_file_sensitivity=sensitivities),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    a=p.parse_args();summarize(a.output.resolve(),a.parent.resolve())
