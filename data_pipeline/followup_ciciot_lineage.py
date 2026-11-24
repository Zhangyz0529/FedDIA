#!/usr/bin/env python3
"""Targeted lineage followup; inspect existing artifacts, never regenerate samples."""
import argparse,hashlib,json
from pathlib import Path
import shutil,zipfile
from federated_contract import sha


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def followup(project,out):
    if out.exists():raise ValueError('Fresh followup version required')
    out.mkdir(parents=True)
    previous=project/'experiment_logs/ciciot2023_five_class_source_audit_v1_20261109/file_manifest.json'
    previous_by_name={Path(r['capture']).name:r for r in json.loads(previous.read_text())}
    legacy=project.parent/'CIC_loV';manifest_path=legacy/'manifest_pcap_stat39_v1.json';m=json.loads(manifest_path.read_text())
    checks=[]
    for s in m['source_pcaps']:
        name=Path(s['pcap']).name;current=previous_by_name[name];actual=sha(s['pcap'])
        if actual!=s['pcap_sha256'] or actual!=current['sha256']:raise ValueError('Existing construction refers to different raw input')
        checks.append(dict(capture=name,current_raw_sha256=actual,legacy_manifest_sha256_matches=True,
            source_packet_count=s['raw_packets'],existing_window_count=s['samples'],raw_block_count=s['groups'],
            new_original_acquisition_identity_found=False))
    materials=[
        (legacy/'README_pcap_multimodal.md','documents100-packet units and raw-block split, explicitly not independent capture/session','NO_NEW_BATCH_IDENTITY'),
        (manifest_path,'existing Stat39 window-to-parent-PCAP lineage for same five inputs','SAME_RAW_INPUTS_ONLY'),
        (legacy/'RAW_WINDOW_REPLAY_VALIDATION.json','saved window reconstruction evidence','PACKET_CONSTRUCTION_ONLY'),
        (legacy/'sample_construction_audit_20261102/REPORT_CN.md','previous construction audit states scenario labels and block split limitations','NO_NEW_BATCH_IDENTITY'),
        (project.parents[2]/'reproducibility/AdaFuse-Net-Repro/docs/ciciot2023_construction.md','internal subset documents same five PCAP and same-parent blocks','SAME_RAW_INPUTS_ONLY'),
        (project.parent/'frozen_cross_dataset_v1/runs/seed42_v1/SOURCE_AUDIT.json','native-flow scenario labels, five-tuple role splits, not capture independence','NO_NEW_BATCH_IDENTITY')]
    rows=[]
    for p,description,result in materials:
        rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),relevance=description,result=result))
    for p in sorted((legacy/'raw_packet_index').glob('*.json')):
        data=json.loads(p.read_text());rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),
            relevance='raw record offsets for existing PCAP',result='OFFSET_INDEX_NOT_ORIGINAL_BATCH_MAPPING',
            source_pcap=data.get('pcap'),original_batch_id=None))
    raw=project.parents[2]/'data/CICloV2023';zpath=raw/'CSV/CSV.zip'
    with zipfile.ZipFile(zpath) as z:
        extras=[i for i in z.infolist() if not i.is_dir() and not i.filename.lower().endswith('.csv')]
        zip_extra=[dict(member=i.filename,bytes=i.file_size,crc32=f'{i.CRC:08x}') for i in extras]
    save(out/'local_materials_review.json',rows);save(out/'same_input_checks.json',checks)
    save(out/'csv_archive_nondata_members.json',zip_extra)
    save(out/'status.json',dict(role='DATA_CONSTRUCTION_AUDIT_ONLY',source_gate='BLOCKED_SOURCE_GROUP_REAUDIT',
        existing_windows_from_same_five_inputs=m['sample_count'],new_capture_activities_recovered=0,new_original_batch_mappings_recovered=0,
        source_packet_hashes_rechecked=True,regenerated_samples=False,training_started=False,
        online_scope='UNB landing, primary paper and public source search reviewed; official supplementary entry previously opened registration/error page; no restricted material retrieved',
        absence_claim='No such mapping found in reviewed local/accessible public materials; not a claim it never existed'))
    text=['# CICIoT2023：仅追查可改变来源结论的材料','',
        '角色调整为 **数据构造审计**，退出当前确认性主实验候选；状态保持BLOCKED_SOURCE_GROUP_REAUDIT。没有继续生成流段或训练。',
        '', f'发现工作区已有{m["sample_count"]}个Stat39报文窗口、原包偏移索引和构造核验记录。五个源PCAP的现时SHA与该manifest及本项目五类审计完全一致；这不是新增原始录制。',
        '', '已有构造采用100包窗口、10000原始包块；这有助于区分场景识别与单流检测，并保留同包生成证据。但这些块仍出自五个已知父文件，不能替代攻击运行ID、监测流原始文件或mergecap输入映射。未重新加载大型PKL或执行任何旧训练。',
        '', '| 材料 | 能证明什么 | 是否改变来源结论 |','| --- | --- | --- |']
    for _,description,result in materials:text.append('| '+description+' | 见local_materials_review.json中的原路径与SHA | '+result+' |')
    text+=['', 'raw_packet_index/*.json是原文件的报文块offset索引，不是原始采集批次索引。CSV.zip除CSV外仅有README_CSV.pdf，未发现合并前PCAP、merge manifest或攻击执行日志。',
        '', '公开查阅范围为UNB发布页、原始论文及面向原始补充资料的检索。没有取得可关联本地报文的新增批次清单；登记入口/受限材料未获取，不提交个人信息或联系第三方。该结果不意味着发布方从未保留这些材料。',
        '', '只有取得原始输入PCAP（或可信标识）、合并前后映射、攻击活动时间/角色/执行记录，才重开来源与标签粒度审计。仅取得更多同父文件窗口、CSV分片、流程示意图或现有模型成绩不触发重开。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    code=out/'code';code.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('federated_contract.py')]:shutil.copy2(p,code/p.name)
    save(out/'code_manifest.json',{p.name:sha(p) for p in code.iterdir()});print((out/'status.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();followup(a.project.resolve(),a.output.resolve())
