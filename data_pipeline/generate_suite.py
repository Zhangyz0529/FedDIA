"""Configurable multi-dataset source generation; no training side effects.

Prepare freezes effective class/threshold settings and code. Run uses separate
subprocesses for independent datasets; source inventories and rejected classes
remain visible. USTC capture cohorts are NOT reinterpreted as per-flow attack
truth and a one-capture class cannot pass a three-source classification gate.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import doh_pilot as pilot
import doh_expand as expand
from dataset_filters import validate_selection,filter_dataset
import additional_datasets as additional

PROJECT=Path(__file__).resolve().parent.parent
USTC_MALWARE=['Cridex','Geodo','Htbot','Miuref','Neris','Nsis-ay','Shifu','Tinba','Virut','Zeus']
USTC_BENIGN=['BitTorrent','Facetime','FTP','Gmail','MySQL','Outlook','Skype','SMB','Weibo','WorldOfWarcraft']
USTC_PUBLISHER='https://github.com/davidyslu/USTC-TFC2016'


def save(path,v):pilot.save(path,v)


def read(path):return json.loads(Path(path).read_text())


def freeze(out,config):
    code=out/'code';code.mkdir(exist_ok=True)
    for name in ['generate_suite.py','dataset_filters.py','doh_pilot.py','doh_expand.py','additional_datasets.py']:
        source=Path(__file__).resolve().parent/name
        if source.resolve()!=(code/name).resolve():shutil.copy2(source,code/name)
    save(code/'effective_config.json',config)
    save(out/'code_manifest.json',{p.name:pilot.sha(p) for p in code.iterdir() if p.is_file()})


def prepare_dataset(name,config,data_root,out,thresholds,project):
    allowed={'CIC_DOH':pilot.LABELS,'USTC':['Benign',*USTC_MALWARE],'CIC_IDS':additional.CIC_CLASSES,'ISCX':additional.ISCX_CLASSES}[name]
    validate_selection(config['classes'],allowed,thresholds)
    raw=(data_root/config['raw_directory']).resolve()
    if config['adapter'] in ('cic_strict_csv_window','iscx_release_activity'):
        plan=additional.prepare(name,raw,out,config,thresholds)
    elif config['adapter']=='doh_archive':
        expand.prepare(raw,out,config['per_family'],config['classes'],config.get('normal_files'))
        plan=read(out/'plan.json');plan.update(dataset=name,effective_config=config,thresholds=thresholds)
        plan['reuse_roots']=[str((project/p).resolve()) for p in config.get('reuse_roots',[])]
    elif config['adapter']=='ustc_capture_cohort':
        if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing started output')
        out.mkdir(parents=True);jobs=[];excluded=[]
        for app in config['benign_apps']:
            if app not in USTC_BENIGN:raise ValueError('Unknown benign application '+app)
        candidates=[]
        if 'Benign' in config['classes']:
            candidates.extend((raw/'Benign'/f'{app}.pcap','Benign',app) for app in config['benign_apps'])
        candidates.extend((raw/'Malware'/f'{label}.pcap',label,label) for label in config['classes'] if label!='Benign')
        for p,label,app in candidates:
            if not p.exists():excluded.append(dict(path=str(p),label=label,reason='missing_capture'));continue
            if p.stat().st_size>config['max_capture_bytes']:
                excluded.append(dict(path=str(p),label=label,reason='fixed_resource_size_limit'));continue
            jobs.append(dict(capture=str(p.resolve()),label=label,application=app,acquisition_family='USTC_capture:'+app,
                             original_sha256=pilot.sha(p),bytes=p.stat().st_size))
        plan=dict(dataset=name,version='source_suite_v3_20261108',raw_root=str(raw),jobs=jobs,excluded_files=excluded,
                  requested_classes=config['classes'],effective_config=config,thresholds=thresholds,
                  source_group_policy='capture identity, duplicate hashes and exact overlapping packets; no sample-random fallback',
                  label_policy=config['label_scope'],rules=pilot.RULES,training_started=False)
    else:raise ValueError('Unsupported/blocked adapter')
    if config.get('reuse_roots') and 'reuse_roots' not in plan: plan['reuse_roots']=[str((project/p).resolve()) for p in config['reuse_roots']]
    save(out/'plan.json',plan);freeze(out,dict(dataset=name,settings=config,thresholds=thresholds,raw_root=str(raw)))
    return out


def reuse_doh_caches(out,plan):
    lookup={}
    for rootname in plan.get('reuse_roots',[]):
        root=Path(rootname)
        if not (root/'plan.json').exists():continue
        old=read(root/'plan.json')
        if old['rules']!=pilot.RULES:raise ValueError('Refusing cache reuse under different packet/feature/label rules')
        for name,h in read(root/'code_manifest.json').items():
            if pilot.sha(root/'code'/name)!=h:raise ValueError('Parent frozen code changed')
        for i,job in enumerate(old['jobs']):lookup[job['capture']]=(root,i,job)
    reused=[]
    for i,job in enumerate(plan['jobs']):
        if job['capture'] not in lookup:continue
        root,old_index,old_job=lookup[job['capture']]
        if old_job['group']!=job['group'] or old_job['acquisition_family']!=job['acquisition_family']:continue
        old_cache=root/'captures'/f'{old_index:03d}';target=out/'captures'/f'{i:03d}'
        if target.exists():continue
        metadata=read(old_cache/'complete.json')
        if pilot.sha(Path(job['capture']))!=metadata['sha256'] or pilot.sha(old_cache/'capture.pcap')!=metadata['sha256']:
            raise ValueError('Parent raw hash mismatch')
        target.mkdir(parents=True)
        hashes={}
        for name in ['provenance.parquet','quarantine.parquet','model_inputs.npz','packet_fingerprints.npy']:
            hashes[name]=pilot.sha(old_cache/name);shutil.copy2(old_cache/name,target/name)
            if pilot.sha(target/name)!=hashes[name]:raise ValueError('Artifact copy mismatch')
        for name in ['provenance.parquet','quarantine.parquet']:
            f=pd.read_parquet(target/name)
            if not f.empty:f['capture_index']=i;f.to_parquet(target/name,index=False)
        if (old_cache/'related.csv').exists():shutil.copy2(old_cache/'related.csv',target/'related.csv')
        # Share immutable working bytes, keep original source hash and dependency explicit.
        (target/'capture.pcap').symlink_to((old_cache/'capture.pcap').resolve())
        metadata.update(index=i,working_path=str((target/'capture.pcap').resolve()),reused_from=str(old_cache.resolve()))
        save(target/'complete.json',metadata)
        entry=dict(new_index=i,parent=str(old_cache.resolve()),original_sha256=metadata['sha256'],parent_artifact_hashes=hashes)
        save(target/'reuse_manifest.json',entry);reused.append(entry)
    save(out/'cache_reuse.json',dict(reused_captures=len(reused),entries=reused,
                                   note='No model-dependent selection; rules equal, parent code/raw hashes checked, independent packet-index verification repeated'))
    print('Verified cached captures reused:',len(reused),flush=True)


def reuse_cohort_caches(out,plan):
    lookup={}
    for rootname in plan.get('reuse_roots',[]):
        root=Path(rootname)
        if not (root/'generation_result.json').exists():raise RuntimeError('Wait for parent generation completion before cache reuse')
        previous=read(root/'plan.json')
        for name,h in read(root/'code_manifest.json').items():
            if pilot.sha(root/'code'/name)!=h:raise ValueError('Parent code changed')
        for i,job in enumerate(previous['jobs']):lookup[job['capture']]=(root,i,job)
    reused=[]
    for i,job in enumerate(plan['jobs']):
        if job['capture'] not in lookup:continue
        root,j,previous=lookup[job['capture']]
        if previous['label']!=job['label'] or previous['acquisition_family']!=job['acquisition_family']:continue
        src=root/'captures'/f'{j:03d}';dst=out/'captures'/f'{i:03d}'
        if dst.exists():continue
        m=read(src/'complete.json')
        if pilot.sha(Path(job['capture']))!=m['sha256'] or pilot.sha(src/'capture.pcap')!=m['sha256']:raise ValueError('Parent source bytes changed')
        dst.mkdir(parents=True);hashes={}
        for name in ['provenance.parquet','quarantine.parquet','model_inputs.npz','packet_fingerprints.npy']:
            hashes[name]=pilot.sha(src/name);shutil.copy2(src/name,dst/name)
            if pilot.sha(dst/name)!=hashes[name]:raise ValueError('Copied cache mismatch')
        for name in ['provenance.parquet','quarantine.parquet']:
            f=pd.read_parquet(dst/name)
            if not f.empty:f['capture_index']=i;f.to_parquet(dst/name,index=False)
        (dst/'capture.pcap').symlink_to((src/'capture.pcap').resolve())
        m.update(index=i,working_path=str((dst/'capture.pcap').resolve()),archive_members=job['archive_members'])
        save(dst/'complete.json',m);reused.append(dict(index=i,parent=str(src),parent_hashes=hashes))
    save(out/'cache_reuse.json',dict(reused_captures=len(reused),entries=reused))


def cohort_label(src,dst,proto,group):
    return group,'release_capture_application_cohort_not_per_flow_attack_truth'


def ustc_work_one(out,i,job,dataset='USTC',publisher=USTC_PUBLISHER):
    cache=out/'captures'/f'{i:03d}'
    if (cache/'complete.json').exists():return read(cache/'complete.json')
    if cache.exists():raise RuntimeError('Incomplete cache; inspect before resuming')
    cache.mkdir(parents=True);source=Path(job['capture']);working=cache/'capture.pcap'
    if pilot.sha(source)!=job['original_sha256']:raise ValueError('Source changed since plan')
    shutil.copyfile(source,working)
    if pilot.sha(working)!=job['original_sha256']:raise ValueError('Working copy mismatch')
    rejected=[]
    records,arrays,meta,fps=pilot.inspect_capture(working,job['label'],job['original_sha256'],rejected,
                                               label_resolver=cohort_label,require_tls=False)
    counts=Counter(k for row in records+rejected for k in row['packet_indices'])
    if len(counts)!=meta['total_packets'] or any(n!=1 for n in counts.values()) or set(counts)!=set(range(meta['total_packets'])):
        raise ValueError('Packet coverage failure')
    for row in records:
        row.update(sample_id=hashlib.sha256(json.dumps([dataset,job['original_sha256'],pilot.RULES['version'],row['packet_indices']]).encode()).hexdigest(),
                   dataset=dataset,publisher=publisher,extractor_version=dataset.lower()+'_ip_segment_v4_1',label_scope='capture_cohort_only',
                   capture_index=i,acquisition_family=job['acquisition_family'])
        row['session_id']=row['sample_id']
    for row in rejected:row.update(capture_index=i,capture_hash=job['original_sha256'],acquisition_family=job['acquisition_family'])
    pd.DataFrame(records).to_parquet(cache/'provenance.parquet',index=False)
    pd.DataFrame(rejected).to_parquet(cache/'quarantine.parquet',index=False)
    np.savez_compressed(cache/'model_inputs.npz',stat_raw=np.stack([x for x,s in arrays]) if arrays else np.empty((0,25),np.float32),
                        seq_feature=np.stack([s for x,s in arrays]) if arrays else np.empty((0,512),np.uint8),
                        sample_id=np.array([r['sample_id'] for r in records]),label=np.array([r['label'] for r in records]))
    np.save(cache/'packet_fingerprints.npy',np.frombuffer(b''.join(sorted(fps)),dtype='V32'))
    if pilot.sha(source)!=job['original_sha256']:raise ValueError('Original mutated')
    meta.update(index=i,sha256=job['original_sha256'],file_size=job['bytes'],relative_path=str(source),
                acquisition_family=job['acquisition_family'],group=job['label'],application=job['application'],
                publisher=publisher,archive_members=job.get('archive_members'),restored_source=job.get('restored_source'),label_scope='capture_cohort_not_independent_per_flow_truth',working_path=str(working),packet_membership_exact=True)
    save(cache/'complete.json',meta)
    print(f'{dataset} {i+1}: {job["application"]}, accepted={len(records)}, packets={meta["total_packets"]}',flush=True)
    return meta


def combine_ustc(out,metadata,plan):
    # Match duplicates/actual raw overlaps, do not split a capture into fake sources.
    parent=list(range(len(metadata)));edges=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(metadata):
        for j,b in enumerate(metadata[:i]):
            same=a['sha256']==b['sha256'];same_family=a['acquisition_family']==b['acquisition_family'];overlap=0
            if max(a['capture_start_time'],b['capture_start_time'])<=min(a['capture_end_time'],b['capture_end_time']):
                x=np.load(out/'captures'/f'{i:03d}'/'packet_fingerprints.npy',mmap_mode='r');y=np.load(out/'captures'/f'{j:03d}'/'packet_fingerprints.npy',mmap_mode='r')
                overlap=len(np.intersect1d(x,y,assume_unique=True))
            if same or same_family or overlap:
                parent[find(j)]=find(i);edges.append(dict(left=i,right=j,duplicate_hash=same,same_recording_family=same_family,exact_overlap_packets=overlap))
    frames=[];xs=[];ss=[];ids=[];labels=[];qs=[]
    for i,m in enumerate(metadata):
        cache=out/'captures'/f'{i:03d}';f=pd.read_parquet(cache/'provenance.parquet')
        if not f.empty:f['source_group_id']='capture_group_'+str(find(i));frames.append(f)
        q=pd.read_parquet(cache/'quarantine.parquet')
        if not q.empty:q['source_group_id']='capture_group_'+str(find(i));qs.append(q)
        with np.load(cache/'model_inputs.npz',allow_pickle=False) as a:
            xs.append(a['stat_raw']);ss.append(a['seq_feature']);ids.extend(a['sample_id']);labels.extend(a['label'])
    frame=pd.concat(frames,ignore_index=True);assignment={}
    group_labels={g:set(f.label) for g,f in frame.groupby('source_group_id')}
    for label in plan['requested_classes']:
        groups=[g for g,ls in sorted(group_labels.items()) if ls=={label}]
        for n,g in enumerate(groups):assignment[g]=('train','val','test')[min(n,2)]
    if plan['dataset']=='ISCX':
        from sklearn.model_selection import StratifiedGroupKFold
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42)
        assignment={}
        for fold,(_,indices) in enumerate(splitter.split(np.zeros((len(frame),1)),frame.label,groups=frame.source_group_id)):
            for group in frame.iloc[indices].source_group_id.unique():assignment[group]=('train','val','test')[fold]
    frame['split']=frame.source_group_id.map(assignment).fillna('unassigned')
    assert len(ids)==len(set(ids))
    frame.to_parquet(out/'provenance.parquet',index=False);pd.concat(qs,ignore_index=True).to_parquet(out/'quarantine.parquet',index=False)
    np.savez_compressed(out/'model_inputs.npz',stat_raw=np.concatenate(xs),seq_feature=np.concatenate(ss),sample_id=np.array(ids),label=np.array(labels))
    save(out/'file_manifest.json',metadata);save(out/'source_group_edges.json',edges)
    save(out/'feature_schema.json',dict(version=pilot.RULES['schema'],fields=pilot.FEATURES,legacy_compatible=False,
                                      rules=pilot.RULES,require_tls=False,direction='first observed packet',label_scope=plan['label_policy']))
    save(out/'audit_report.json',dict(dataset=plan['dataset'],status='CANDIDATE_INPUTS_SOURCE_GATE_PENDING',samples=len(frame),labels=dict(Counter(frame.label)),
                                     captures=len(metadata),source_groups=frame.source_group_id.nunique(),packet_coverage_exact=True,
                                     label_scope=plan['label_policy'],training_started=False,
                                     limitation=('Malware classes have one local capture each; no capture-independent train/val/test can be manufactured' if plan['dataset']=='USTC' else 'Capture activity cohorts, including background flows; VPN does not mean attack. Capture-family grouped three-fold split.')))
    return frame


def run_dataset(out):
    plan=read(out/'plan.json')
    for name,h in read(out/'code_manifest.json').items():
        if pilot.sha(out/'code'/name)!=h:raise ValueError('Frozen code/config changed')
    if plan['dataset']=='CIC_IDS':
        additional.run_cic(out,plan)
    elif plan['dataset']=='ISCX':
        reuse_cohort_caches(out,plan)
        metadata=[]
        for i,job in enumerate(plan['jobs']):
            metadata.append(ustc_work_one(out,i,job,dataset='ISCX',publisher=additional.ISCX_URL))
            save(out/'progress.json',dict(completed=i+1,total=len(plan['jobs']),training_started=False))
        combine_ustc(out,metadata,plan);expand.verify(out,write_report=False)
    elif plan['dataset']=='CIC_DOH':
        reuse_doh_caches(out,plan);expand.run(out)
    else:
        metadata=[]
        for i,job in enumerate(plan['jobs']):
            metadata.append(ustc_work_one(out,i,job));save(out/'progress.json',dict(completed=i+1,total=len(plan['jobs']),training_started=False))
        combine_ustc(out,metadata,plan)
        # Generic packet-index verifier only reconstructs inputs; it does not apply DoH label rules.
        expand.verify(out,write_report=False)
    selected=filter_dataset(out,plan['requested_classes'],plan['thresholds'],normal=plan['effective_config'].get('normal_class','Benign'))
    save(out/'generation_result.json',dict(dataset=plan['dataset'],class_selection=selected,
                                          candidate_report=read(out/'audit_report.json'),training_started=False))
    text=['# '+plan['dataset']+' 可选类别生成结果','',
          '未训练。类别选择和最低支持规则在生成前固定；全部候选、原始来源和排除记录保留。','',
          '| 类别 | 候选样本 | 来源组 | 保留 | 原因 |','| --- | ---: | ---: | --- | --- |']
    for row in selected['class_ledger']:
        text.append(f"| {row['label']} | {row['samples']} | {row['source_groups']} | {row['selected']} | {', '.join(row['reasons']) or '通过'} |")
    text+=['',f"导出状态：{selected['export_status']}。保留类别：{selected['selected_classes']}。",'',
           'selected/ 只在至少两个类别、配置要求的正常类、各划分支持和来源隔离均通过时生成NPZ；不指定联邦客户端或任务，不输出dataset.pkl。',
           'USTC/ISCX使用采集队列标签；ISCX是应用/VPN活动分类，不是攻击识别。CIC使用严格CSV记录对应；DoH使用原拓扑角色/TLS规则。新25维不等于旧25/29/77维。',
           '来源组是所选本地采集/场景单位，不代表独立主机/组织；低误报评价还需检查正常支持数。',
           '', 'class_selection.json 为实际门槛与排除理由，plan.json 为冻结后的有效配置。']
    (out/'GENERATION.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(dataset=plan['dataset'],selected=selected['selected_classes'],status=selected['export_status']),ensure_ascii=False),flush=True)


def summarize(root):
    plan=read(root/'suite_plan.json');rows=[]
    for name,path in plan['outputs'].items():
        p=Path(path)/'generation_result.json'
        if p.exists():
            d=read(p);s=d['class_selection'];rows.append(dict(dataset=name,status=s['export_status'],selected=s['selected_classes'],
                                                          candidates=d['candidate_report'].get('accepted_samples',d['candidate_report'].get('samples')),
                                                          output=path,blockers=s['blockers']))
        else:rows.append(dict(dataset=name,status='INCOMPLETE',output=path))
    views=[]
    if (root/'views').exists():
        for path in sorted((root/'views').rglob('class_selection.json')):
            d=read(path);n=0
            for split in ('train','val','test'):
                p=path.parent/'selected'/f'{split}.npz'
                if p.exists():
                    with np.load(p,allow_pickle=False) as a:n+=len(a['sample_id'])
            views.append(dict(path=str(path.parent),samples=n,selected_classes=d['selected_classes'],thresholds=d['thresholds'],status=d['export_status']))
    save(root/'summary.json',dict(results=rows,class_views=views,pending_datasets=plan['pending_datasets'],training_started=False))
    text=['# 多数据集生成汇总','','| 数据集 | 候选样本 | 保留类别 | 状态 |','| --- | ---: | --- | --- |']
    for r in rows:text.append(f"| {r['dataset']} | {r.get('candidates','')} | {', '.join(r.get('selected',[]))} | {r['status']} |")
    text+=['','类别门槛、排除清单见各目录 GENERATION.md 和 class_selection.json；无训练或模型性能结果。',
           '各数据集按各自标签语义验收；应用/VPN活动不解释成网络攻击。']
    if views:
        text+=['','可选类别视图：']
        for view in views:text.append('- '+str(Path(view['path']).relative_to(root))+': '+', '.join(view['selected_classes'])+'，'+str(view['samples'])+' 个样本；门槛 '+str(view['thresholds']))
    for row in rows:
        if row['dataset']=='USTC':text+=['','USTC恶意类各一个来源，不能用单文件随机拆分替代来源隔离。']
        if row['dataset']=='CIC_IDS':text+=['','CIC-IDS仅输出通过严格匹配的有界候选；攻击类别集中单个采集日时，不能强制获得三来源划分。']
        if row['dataset']=='ISCX':text+=['','ISCX为采集活动/VPN分类，背景流量仍按采集队列解释，不代表每个流的应用真值。']
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n')
    return rows


def select_view(parent,out,classes,thresholds):
    """Create a new taxonomy view; parent arrays/provenance stay immutable."""
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing existing view output')
    plan=read(parent/'plan.json')
    allowed=plan.get('requested_classes',list(pd.read_parquet(parent/'provenance.parquet').label.unique()))
    validate_selection(classes,allowed,thresholds)
    out.mkdir(parents=True,exist_ok=True)
    dependencies={}
    for name in ['provenance.parquet','model_inputs.npz']:
        source=(parent/name).resolve();dependencies[name]=dict(path=str(source),sha256=pilot.sha(source))
        (out/name).symlink_to(source)
    save(out/'plan.json',dict(dataset=plan.get('dataset','CIC_DOH'),requested_classes=classes,thresholds=thresholds,
                              parent=str(parent),selection_basis='counts/source support only; no model scores'))
    freeze(out,dict(classes=classes,thresholds=thresholds,parent=str(parent)))
    result=filter_dataset(out,classes,thresholds,normal=plan.get('effective_config',{}).get('normal_class',None if plan.get('dataset')=='ISCX' else 'BENIGN' if plan.get('dataset')=='CIC_IDS' else 'Benign'))
    for name,v in dependencies.items():
        if pilot.sha(Path(v['path']))!=v['sha256']:raise ValueError('Parent changed during selection')
    save(out/'lineage.json',dict(parent=str(parent),dependencies=dependencies,parent_modified=False,training_started=False))
    (out/'README.md').write_text('# 可选类别视图\n\n请求类别：'+', '.join(classes)+'。\n\n状态：'+result['export_status']+'。实际保留：'+', '.join(result['selected_classes'])+'。\n\nselected/ 为新导出；原始候选及来源旁表通过只读用途符号链接引用父版本，保留父版本才能重现。阈值、标签映射与排除理由见class_selection.json。未训练。\n')
    print(json.dumps(dict(output=str(out),classes=result['selected_classes'],status=result['export_status']),ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','worker','report','select'])
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'suite_config.json');p.add_argument('--dataset',default='all')
    p.add_argument('--classes');p.add_argument('--output',type=Path)
    p.add_argument('--input',type=Path)
    p.add_argument('--min-total-samples',type=int)
    p.add_argument('--min-per-split',type=int)
    p.add_argument('--min-source-groups',type=int)
    args=p.parse_args()
    if args.action=='select':
        if not args.input or not args.output or not args.classes:raise ValueError('select requires --input, --output and --classes')
        parent=args.input.resolve();thresholds=dict(read(parent/'plan.json')['thresholds'])
        for key in ['min_total_samples','min_per_split','min_source_groups']:
            value=getattr(args,key)
            if value is not None:thresholds[key]=value
        return select_view(parent,args.output.resolve(),[s.strip() for s in args.classes.split(',') if s.strip()],thresholds)
    if args.action=='worker':return run_dataset(args.output.resolve())
    if args.action in ('run','report'):
        root=args.output.resolve()
        if args.action=='report':print(summarize(root));return
        suite=read(root/'suite_plan.json');processes=[]
        for name,path in suite['outputs'].items():
            out=Path(path);log=(out/'generation.log').open('a')
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            proc=subprocess.Popen([sys.executable,'-B',str(out/'code/generate_suite.py'),'worker','--output',str(out)],stdout=log,stderr=subprocess.STDOUT,env=env)
            processes.append((name,proc,log))
        failed=[]
        for name,proc,log in processes:
            status=proc.wait();log.close()
            if status:failed.append(dict(dataset=name,exit_code=status))
        summarize(root)
        if failed:raise SystemExit(json.dumps(failed))
        return
    config=read(args.config);project=PROJECT;data_root=(project/config['data_root']).resolve()
    root=args.output.resolve() if args.output else (project/config['output_root']).resolve()
    if root.exists() and any(root.iterdir()):raise RuntimeError('Refusing started suite output')
    root.mkdir(parents=True);outputs={};pending={}
    for name,cfg in config['datasets'].items():
        if not cfg.get('enabled') or (args.dataset!='all' and name!=args.dataset):
            pending[name]=cfg.get('blocked_reason','not_selected');continue
        cfg=deepcopy(cfg)
        if args.classes:
            if args.dataset=='all':raise ValueError('--classes requires a single --dataset')
            cfg['classes']=[x.strip() for x in args.classes.split(',') if x.strip()]
        out=root/name;prepare_dataset(name,cfg,data_root,out,config['thresholds'],project);outputs[name]=str(out)
    if not outputs:raise ValueError('No enabled requested dataset')
    save(root/'suite_plan.json',dict(version=config['version'],config_path=str(args.config.resolve()),config_sha256=pilot.sha(args.config),
                                   thresholds=config['thresholds'],outputs=outputs,pending_datasets=pending,training_started=False))
    print(json.dumps(dict(outputs=outputs,pending=pending),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
