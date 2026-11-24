#!/usr/bin/env python3
"""Verify pilot inputs and expose representation ambiguity without fitting models."""
from collections import Counter,defaultdict
import argparse,hashlib,json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import doh_pilot as pilot


def finalize(out,raw,project):
    dest=out/'input_checks'
    if dest.exists():raise ValueError('Fresh input check stage required')
    dest.mkdir();manifest=json.loads((out/'file_manifest.json').read_text());groups=defaultdict(list);summary=[];identities=[];seen=set()
    for i,m in enumerate(manifest):
        folder=out/'captures'/f'{i:02d}';p=pd.read_parquet(folder/'provenance.parquet')
        assert pilot.sha(m['capture'])==m['sha256']
        with np.load(folder/'model_inputs.npz',allow_pickle=False) as a:
            assert list(a['sample_id'])==list(p.sample_id)
            assert a['stat_raw'].shape==(len(p),25) and a['seq_feature'].shape==(len(p),512)
            assert np.isfinite(a['stat_raw']).all() and not seen.intersection(a['sample_id']);seen.update(a['sample_id'])
            signatures=[]
            for sid,label,x,q in zip(a['sample_id'],a['label'],a['stat_raw'],a['seq_feature']):
                signature=hashlib.sha256(x.tobytes()+q.tobytes()).hexdigest();signatures.append(signature)
                groups[signature].append(dict(sample_id=str(sid),label=str(label)))
            summary.append(dict(label=m['label'],saved_samples=len(p),unique_model_inputs=len(set(signatures)),
                sampled_singletons=m['pilot_segment_support'].get('one_packet',0),
                sampled_segments_ge3=m['pilot_segment_support'].get('at_least_three_packets',0),
                sampled_candidate_segments=m['pilot_candidate_segments']))
        with Path(m['capture']).open('rb') as f:
            for row in p.itertuples():
                f.seek(int(row.packet_offsets[0]));decoded=pilot.decode(f.read(int(row.packet_caplens[0])),m['link_type'])
                identities.append(dict(sample_id=row.sample_id,protocol=decoded['proto'],first_ip=decoded['src'][0],first_port=decoded['src'][1],
                    peer_ip=decoded['dst'][0],peer_port=decoded['dst'][1],cohort_label=row.label))
    pd.DataFrame(identities).to_parquet(dest/'pilot_flow_identity.parquet',index=False)
    conflict=[dict(input_sha256=h,label_counts=dict(Counter(x['label'] for x in values)),members=values) for h,values in groups.items() if len({x['label'] for x in values})>1]
    pilot.save(dest/'cross_label_identical_inputs.json',conflict);pd.DataFrame(summary).to_csv(dest/'pilot_support_and_uniqueness.csv',index=False)
    joint=json.loads((out/'joint_split.json').read_text());assert joint['full_five_classes']['feasible'] is False
    assert len(joint['all_class_subsets'])==26 and all(x['feasible'] is False for x in joint['all_class_subsets'])
    relations=json.loads((out/'source_relations.json').read_text());assert len(relations)==10
    checks=dict(status='BLOCKED_SOURCE_GROUP_REAUDIT',source_hashes_verified=True,pilot_samples=len(seen),unique_model_input_groups=len(groups),
        cross_label_identical_input_groups=len(conflict),cross_label_participating_samples=sum(len(x['members']) for x in conflict),
        label_ambiguity_interpretation='Identical anonymized model inputs can represent valid traffic in different labeled scenarios; not proof that raw labels or captures are corrupt',
        joint_all_class_and_26_subset_checks_passed=True,regression_tests_passed=7,independent_packet_recomputations=25,
        no_new_split_or_training=True,acquisition_batch_count='UNKNOWN',full_population_sample_count='NOT_COMPUTED')
    pilot.save(dest/'verification.json',checks)
    evidence=out/'evidence';evidence.mkdir()
    docs=[raw/'README.pdf',raw/'CSV/README.pdf',raw/'CSV/README_PCAP.pdf']
    pilot.save(evidence/'local_document_hashes.json',{str(p):pilot.sha(p) for p in docs})
    previous=project/'experiment_logs/raw_dataset_usability_v1_20261109/documents'
    for name,target in [('iot_csv.txt','release_csv_readme.txt'),('iot_pcap.txt','release_pcap_readme.txt')]:shutil.copy2(previous/name,evidence/target)
    main=Path('/tmp/ciciot2023_local_readme.txt')
    if main.exists():shutil.copy2(main,evidence/'release_main_readme.txt')
    available=[]
    for p in raw.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.json','.py','.md','.txt','.log','.pdf'}:available.append(str(p.relative_to(raw)))
    pilot.save(evidence/'local_lineage_material_inventory.json',dict(files=available,original_capture_to_batch_manifest_found=False,merge_inputs_manifest_found=False,
        statement='Inspected local materials; absence in this directory is not proof the publisher never retained such records'))
    pilot.save(evidence/'publisher_notes.json',dict(landing='https://www.unb.ca/cic/datasets/iotdataset-2023.html',
        paper='https://www.mdpi.com/1424-8220/23/13/5941',
        primary_summary=['Publisher describes merging two monitoring streams for each experiment.',
            'The complete captured attack scenario is assigned its scenario label.',
            'CSV uses packet-window summaries, not our bounded per-flow pilot unit.'],
        local_capture_original_release_hash_list='NOT_AVAILABLE',supplementary_materials_retrieved=False,
        download_entry_observation='Official link opened a registration form with server error; no form submitted',
        inference='None of these descriptions maps individual local packet records to distinct acquisition batches.'))
    # Source manifest and CSV association inventories carry hashes/paths without changing raw files.
    class_csv=[]
    for m in manifest:
        folder='Benign_Final' if m['label']=='BenignTraffic' else m['label']
        for p in sorted((raw/'CSV/extracted/CSV'/folder).glob('*.csv')):
            class_csv.append(dict(label=m['label'],csv=str(p),bytes=p.stat().st_size,header=pd.read_csv(p,nrows=0).columns.tolist(),
                same_basename_as_local_pcap=p.name==m['label']+'.pcap.csv',packet_to_row_mapping_verified=False))
    pilot.save(evidence/'related_csv_schemas.json',class_csv)
    text=['','补充：五个材料的实际时间及重提取检查','', '| 类别 | UTC起点 | UTC终点 | 实际有包UTC日数 | 时间回退次数 |','| --- | --- | --- | ---: | ---: |']
    for m in manifest:text.append(f'| {m["label"]} | {m["start_utc"]} | {m["end_utc"]} | {len(m["actual_UTC_day_packet_counts"])} | {m["timestamp_regressions"]} |')
    text+=['', '日期数仅为观察值：例如连续良性采集跨午夜、攻击文件跨日期，都不能自动变成多个独立采集组。DDoS存在78次全局时间回退，未裁剪修正；试点中同一流段发生时间回退时隔离。时间间隙与回退证据保留在file_manifest.json。',
        '', '| 类别 | 检查输入数 | 唯一模型输入 | 抽查流段数 | 其中>=3包 |','| --- | ---: | ---: | ---: | ---: |']
    for r in summary:text.append(f'| {r["label"]} | {r["saved_samples"]} | {r["unique_model_inputs"]} | {r["sampled_candidate_segments"]} | {r["sampled_segments_ge3"]} |')
    syn=next(r for r in summary if r['label']=='DDoS-SYN_Flood')
    text += ['', f'SYN类抽查流段中单包占{100*syn["sampled_singletons"]/syn["sampled_candidate_segments"]:.2f}%，旧>=3包门槛仅留下{100*syn["sampled_segments_ge3"]/syn["sampled_candidate_segments"]:.2f}%。这说明样本单位与排除规则必须在正式生成前单独论证，不能照搬DoH的TLS/多包门槛。此比例只描述预先抽查窗口，非全语料总体估计。',
        '', f'检查集还有{len(conflict)}组完全相同模型输入跨场景标签出现，涉及{checks["cross_label_participating_samples"]}个检查样本。匿名化、单包流段及场景级标签可能使不同场景产生相同可观察输入；这不是原始数据错标的证据，但足以阻止将本试点直接认证为正式单流攻击检测集。冲突未被静默删除或用于挑选表现更好的类别。',
        '', '当前否决的是“本地五个PCAP已足以支撑目标确认性协议”，不是判定整个CICIoT2023数据集不可用。若未来恢复批次映射，仍须重新全局分组与联合求解；若采用新的包/主机窗口单位，须定义新特征与标签范围，保留同包可追溯性，不能把它描述为原CSV行或旧25维流样本。',
        '', '验收汇总：7项回归检查通过；25个样本从原始包偏移独立重算一致；所有保存数组形状/有限性与原始SHA复核通过。没有性能选参或测试成绩参与类别选择。']
    q=out/'REPORT.md';q.write_text(q.read_text()+'\n'.join(text)+'\n')
    code=dest/'code';code.mkdir()
    for name in ['finalize_ciciot_audit.py','doh_pilot.py','test_ciciot_source_audit.py','test_doh_pilot.py','audit_ciciot_sources.py']:
        shutil.copy2(Path(__file__).with_name(name),code/name)
    pilot.save(dest/'code_manifest.json',{p.name:pilot.sha(p) for p in code.iterdir()})
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['output','raw','project']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();finalize(a.output.resolve(),a.raw.resolve(),a.project.resolve())
