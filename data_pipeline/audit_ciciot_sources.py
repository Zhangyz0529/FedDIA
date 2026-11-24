#!/usr/bin/env python3
"""Five-class CICIoT2023 lineage, same-packet pilot and joint source-split gate.

PCAP clock gaps and extraction windows are observations, never acquisition IDs.
Labels follow the release scenario cohort; no claim of per-flow attack truth.
The pilot permits singleton TCP/UDP segments to expose SYN-flood support loss
under a >=3-packet filter. No full dataset, federated assignment or training.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import combinations
import json
import mmap
from pathlib import Path
import shutil
import struct
import subprocess
import time
import dpkt
import numpy as np
import pandas as pd
import doh_pilot as pilot

URL='https://www.unb.ca/cic/datasets/iotdataset-2023.html'
PAPER='https://www.mdpi.com/1424-8220/23/13/5941'
LABELS=['BenignTraffic','DDoS-SYN_Flood','DictionaryBruteForce','Mirai-udpplain','Recon-PortScan']


def save(p,v):pilot.save(p,v)


def joint_split(components,labels,requirements=None,max_states=1000000):
    """Whole-component assignment with a necessary-count gate and exact search."""
    requirements=requirements or {'train':2,'val':2,'test':1}
    keys=sorted(components,key=lambda k:(-len(set(components[k])&set(labels)),k));needed=sum(requirements.values())
    available={l:sum(l in components[k] for k in keys) for l in labels}
    if any(v<needed for v in available.values()):return dict(feasible=False,status='PROVEN_INFEASIBLE_COMPONENT_COUNTS',available=available,required=needed,assignment=None,states=0)
    splits=list(requirements);counts={s:Counter() for s in splits};assignment={};states=0;exceeded=False
    def search(i):
        nonlocal states,exceeded
        states+=1
        if states>max_states:exceeded=True;return False
        for l in labels:
            deficit=sum(max(0,requirements[s]-counts[s][l]) for s in splits)
            if sum(l in components[k] for k in keys[i:])<deficit:return False
        if i==len(keys):return True
        k=keys[i];present=set(components[k])&set(labels)
        for s in splits:
            assignment[k]=s
            for l in present:counts[s][l]+=1
            if search(i+1):return True
            for l in present:counts[s][l]-=1
            assignment.pop(k)
            if exceeded:return False
        return False
    found=search(0)
    return dict(feasible=True if found else None if exceeded else False,status='FEASIBLE' if found else 'SEARCH_LIMIT' if exceeded else 'PROVEN_INFEASIBLE_JOINT_ASSIGNMENT',available=available,required=needed,assignment=dict(assignment) if found else None,states=states)


def packet_layout(mm):
    modes={b'\xd4\xc3\xb2\xa1':('<',1000),b'\xa1\xb2\xc3\xd4':('>',1000),b'\x4d\x3c\xb2\xa1':('<',1),b'\xa1\xb2\x3c\x4d':('>',1)}
    if bytes(mm[:4]) not in modes:raise ValueError('Only observed classic PCAP supported; no invented interfaces')
    endian,ns=modes[bytes(mm[:4])];major,minor=struct.unpack_from(endian+'HH',mm,4)
    if (major,minor)!=(2,4):raise ValueError('Unexpected PCAP version')
    return endian,ns,int(struct.unpack_from(endian+'I',mm,20)[0]),int(struct.unpack_from(endian+'I',mm,16)[0])


def extract_windows(frames,capture_hash,label,link):
    candidates=[];exclusions=Counter();segment_support=Counter()
    for window_id,packets in sorted(frames.items()):
        active={};finished=[];lookup={p['index']:p for p in packets}
        for packet in packets:
            try:p=pilot.decode(packet['raw'],link)
            except (ValueError,dpkt.UnpackError,IndexError,AttributeError) as e:
                exclusions[str(e) if isinstance(e,ValueError) else 'decode_error']+=1;continue
            key=(p['proto'],*sorted((p['src'],p['dst'])));prior=active.get(key)
            idle=120. if p['proto']==6 else 30.;syn=bool(p['flags']&dpkt.tcp.TH_SYN and not p['flags']&dpkt.tcp.TH_ACK)
            new_syn=prior is not None and syn and (prior.initial_isn!=p['isn'] or not prior.initial_flags&dpkt.tcp.TH_SYN)
            if prior is None or prior.ended or new_syn or packet['ts']-prior.times[-1]>idle or packet['ts']-prior.times[0]>300.:
                if prior:finished.append(prior)
                active[key]=pilot.Segment(p['src'],p['proto'],0,p['flags'],p['isn'])
            active[key].add(packet['index'],packet['ts'],p)
        finished.extend(active.values())
        for s in finished:
            segment_support['all_observed_flow_window_segments']+=1
            if s.bad_time:exclusions['segment_timestamp_regression']+=1;continue
            n=len(s.indices);segment_support['one_packet' if n==1 else 'two_packets' if n==2 else 'at_least_three_packets']+=1
            feature=s.features();sequence=np.frombuffer(bytes(s.prefix).ljust(512,b'\0'),dtype=np.uint8).copy()
            if not np.isfinite(feature).all():exclusions['nonfinite_features']+=1;continue
            sid=__import__('hashlib').sha256(json.dumps(['ciciot_pilot_flow_window_v1',capture_hash,window_id,s.indices],separators=(',',':')).encode()).hexdigest()
            raw=[lookup[i] for i in s.indices]
            record=dict(sample_id=sid,label=label,capture_hash=capture_hash,window_id=window_id,packet_indices=s.indices,
                packet_offsets=[p['offset'] for p in raw],packet_caplens=[p['caplen'] for p in raw],packet_timestamps=[p['ts'] for p in raw],
                packet_directions=s.directions,packet_count=n,start_timestamp=s.times[0],end_timestamp=s.times[-1],
                sequence_valid_length=len(s.prefix),sequence_boundaries=s.prefix_boundaries,
                label_source='publisher_release_scenario_cohort',label_scope='capture_scenario_not_certified_per_flow_attack',
                source_group_id=None,original_acquisition_id=None,extractor_version='ciciot_pilot_flow_window_v1',
                sample_unit='bidirectional_endpoint_segment_within_a_bounded_observation_window',window_boundary_censoring_possible=True)
            candidates.append((sid,record,feature,sequence))
    return candidates,dict(exclusions),dict(segment_support)


def one_capture(out,index,path,label,expected,windows,window_packets,sample_cap):
    target=out/'captures'/f'{index:02d}';target.mkdir(parents=True)
    before=path.stat();h=pilot.sha(path);frames=defaultdict(list);days=Counter();first=last=previous=None;regressions=0;max_gap=0;gap_examples=[];n=0;selected=0
    starts=sorted({expected*i//windows for i in range(windows)})
    if len(starts)>1 and min(np.diff(starts))<window_packets:raise ValueError('Pilot windows overlap; change prespecified plan')
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        endian,ns,link,snaplen=packet_layout(mm);unpack=struct.Struct(endian+'IIII').unpack_from;pos=24;w=0
        while pos<len(mm):
            if pos+16>len(mm):raise ValueError('Truncated PCAP record header')
            sec,frac,caplen,wirelen=unpack(mm,pos);offset=pos+16
            if frac*ns>=1000000000 or offset+caplen>len(mm):raise ValueError('Invalid timestamp or truncated frame')
            timestamp_ns=sec*1000000000+frac*ns;ts=timestamp_ns/1e9
            first=timestamp_ns if first is None else min(first,timestamp_ns);last=timestamp_ns if last is None else max(last,timestamp_ns)
            days[int(timestamp_ns//86400000000000)]+=1
            if previous is not None:
                if timestamp_ns<previous:regressions+=1
                gap=timestamp_ns-previous
                max_gap=max(max_gap,gap)
                if gap>600000000000 and len(gap_examples)<20:gap_examples.append(dict(packet_index=n,gap_seconds=gap/1e9))
            previous=timestamp_ns
            while w+1<len(starts) and n>=starts[w+1]:w+=1
            if starts[w]<=n<starts[w]+window_packets:
                frames[w].append(dict(index=n,offset=offset,caplen=caplen,ts=ts,raw=bytes(mm[offset:offset+caplen])));selected+=1
            n+=1;pos=offset+caplen
            if n%5000000==0:print(label,'scanned_packets',n,flush=True)
    if n!=expected:raise ValueError('Packet count differs from independent capinfos scan')
    all_candidates,exclusions,support=extract_windows(frames,h,label,link)
    all_candidates.sort(key=lambda x:x[0]);chosen=all_candidates[:sample_cap]
    records=[r for _,r,_,_ in chosen]
    pd.DataFrame(records).to_parquet(target/'provenance.parquet',index=False)
    np.savez_compressed(target/'model_inputs.npz',stat_raw=np.stack([x for _,_,x,_ in chosen]) if chosen else np.empty((0,25),np.float32),
        seq_feature=np.stack([q for _,_,_,q in chosen]) if chosen else np.empty((0,512),np.uint8),
        sample_id=np.array([s for s,_,_,_ in chosen],dtype=str),label=np.array([label]*len(chosen),dtype=str))
    # Independently seek original frame offsets and reconstruct selected inputs.
    verified=[]
    for _,r,x,q in chosen[:5]:
        segment=None
        with path.open('rb') as f:
            for ordinal,offset,cap,ts in zip(r['packet_indices'],r['packet_offsets'],r['packet_caplens'],r['packet_timestamps']):
                f.seek(offset);raw=f.read(cap);p=pilot.decode(raw,link)
                if segment is None:segment=pilot.Segment(p['src'],p['proto'],0,p['flags'],p['isn'])
                segment.add(ordinal,ts,p)
        recomputed=np.frombuffer(bytes(segment.prefix).ljust(512,b'\0'),dtype=np.uint8)
        if not np.array_equal(segment.features(),x) or not np.array_equal(recomputed,q):raise ValueError('Independent input recomputation failed')
        verified.append(r['sample_id'])
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or pilot.sha(path)!=h:raise ValueError('Raw capture changed')
    meta=dict(label=label,capture=str(path),sha256=h,file_size=before.st_size,link_type=link,snaplen=snaplen,pcap_format='classic_pcap_2_4',
        raw_packets=n,start_ns=first,end_ns=last,start_utc=pd.to_datetime(first,unit='ns',utc=True).isoformat(),end_utc=pd.to_datetime(last,unit='ns',utc=True).isoformat(),
        actual_UTC_day_packet_counts={pd.to_datetime(d*86400,unit='s',utc=True).date().isoformat():v for d,v in sorted(days.items())},
        timestamp_regressions=regressions,max_forward_gap_seconds=max_gap/1e9,gap_examples=gap_examples,
        original_recording_id=None,attack_batch_id=None,interface_identity_retained=False,lineage_status='MERGED_RELEASE_FILE_OR_SLICE_UNRESOLVED',
        source_identity_policy='Entire release class file remains indivisible; days, gaps and pilot windows are not new acquisition IDs',
        pilot_windows=len(frames),pilot_packets=selected,pilot_candidate_segments=len(all_candidates),pilot_exported_samples=len(chosen),
        pilot_segment_support=support,pilot_exclusions=exclusions,independent_recomputed_samples=len(verified),source_hash_unchanged=True,
        label_scope='publisher_scenario_cohort; not individual-flow maliciousness certification',training_allowed=False)
    save(target/'independent_recompute_checks.json',verified);save(target/'audit.json',meta);print(label,'done',n,'pilot',len(all_candidates),'saved',len(chosen),flush=True)
    return meta


def summarize(out,metadata,raw):
    parents=list(range(len(metadata)));edges=[];relations=[]
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,a in enumerate(metadata):
        for j,b in enumerate(metadata[:i]):
            duplicate=a['sha256']==b['sha256'];days=set(a['actual_UTC_day_packet_counts'])&set(b['actual_UTC_day_packet_counts'])
            intersects=max(a['start_ns'],b['start_ns'])<=min(a['end_ns'],b['end_ns'])
            if duplicate or days:
                parents[root(j)]=root(i);edges.append(dict(left=i,right=j,reasons=['duplicate_file'] if duplicate else ['shared_actual_UTC_day'],witness=dict(left=a['capture'],right=b['capture'],days=sorted(days))))
            relations.append(dict(left=a['label'],right=b['label'],duplicate_SHA256=duplicate,shared_actual_UTC_days=sorted(days),time_windows_intersect=intersects,
                exact_timestamp_raw_packet_overlap='PROVEN_ZERO_BY_DISJOINT_COMPLETE_TIME_WINDOWS' if not intersects else 'NOT_FULLY_CHECKED',
                retimed_or_raw_only_overlap='NOT_CHECKED; byte similarities alone would not certify shared acquisition'))
    components=defaultdict(set);class_rows=[]
    for i,a in enumerate(metadata):
        comp='ciciot_material_'+str(root(i));components[comp].add(a['label'])
        target=out/'captures'/f'{i:02d}';f=pd.read_parquet(target/'provenance.parquet');f['source_group_id']=comp;f.to_parquet(target/'provenance.parquet',index=False)
        folder='Benign_Final' if a['label']=='BenignTraffic' else a['label']
        csvs=sorted((raw/'CSV/extracted/CSV'/folder).glob('*.csv'))
        class_rows.append(dict(label=a['label'],material_component=comp,traceable_material_components=1,certified_original_acquisition_batches='UNKNOWN',
            pilot_windows=a['pilot_windows'],sampled_packets=a['pilot_packets'],pilot_same_packet_cohort_candidates=a['pilot_candidate_segments'],
            pilot_saved_samples=a['pilot_exported_samples'],source_component_train_val_test_allocation='INFEASIBLE_NO_SPLIT',
            minimum_required_components=5,label_basis='publisher entire-scenario labeling + exact local release basename; per-flow truth not claimed',
            related_csv_files=len(csvs),exact_basename_csv_present=any(p.name==a['label']+'.pcap.csv' for p in csvs),
            original_recording_or_batch_mapping='NOT_FOUND_LOCALLY',sample_support_scope='bounded pilot lower bound; not a full-population count'))
    save(out/'source_relations.json',relations);save(out/'merge_witness_edges.json',edges);pd.DataFrame(class_rows).to_csv(out/'class_source_split_feasibility.csv',index=False)
    feasibility=joint_split(components,LABELS)
    subsets=[]
    for count in range(2,len(LABELS)+1):
        for labels in combinations(LABELS,count):
            result=joint_split(components,list(labels));subsets.append(dict(classes=list(labels),**result))
    save(out/'joint_split.json',dict(full_five_classes=feasibility,all_class_subsets=subsets,
        selected_classes=[],selection_uses_model_performance=False,meaningful_continual_protocol_ready=False))
    stats=dict(status='BLOCKED_SOURCE_GROUP_REAUDIT',dataset='CICIoT2023',local_directory=str(raw),file_count=len(metadata),
        global_material_components=len(components),certified_acquisition_batches='UNKNOWN',joint_split_feasible=feasibility['feasible'],
        no_two_or_more_class_subset_feasible=all(x['feasible'] is False for x in subsets),new_split_frozen=False,training_started=False,
        source_hashes_unchanged=all(x['source_hash_unchanged'] for x in metadata),
        independent_recomputed_samples=sum(x['independent_recomputed_samples'] for x in metadata),
        pilot_saved_samples=sum(x['pilot_exported_samples'] for x in metadata),pilot_is_trainable_dataset=False,
        source_gate_reason='One indivisible release material component per class; original acquisition identities lost/unavailable; cannot create 2train+2val+1test by splitting files/days/windows')
    save(out/'audit_status.json',stats);save(out/'file_manifest.json',metadata)
    save(out/'feature_schema.json',dict(version='ip_packet_stats25_v1',fields=pilot.FEATURES,sequence='anonymized IP prefix512; not payload',
        pilot_unit='bidirectional segment within sampled observation window',min_packets=1,TLS_filter=False,statistics_definition=pilot.RULES['stats'],
        sampling='10 uniformly spaced 2000-packet windows/file; select up to500 valid segments/file by stable sample_id order; no model scores'))
    text=['# CICIoT2023 五类原始包来源可行性审计','',
        '**结论：BLOCKED_SOURCE_GROUP_REAUDIT。当前五类无法作为新来源确认性的FedDIA主数据集；未冻结split、未分配客户端/任务、未训练。**',
        '', '| 类别 | 原始包数 | 可追溯材料分量 | 已证原始批次 | 试提取同源候选 | 保存输入 | train/val/test |',
        '| --- | ---: | ---: | --- | ---: | ---: | --- |']
    for a in metadata:text.append(f'| {a["label"]} | {a["raw_packets"]} | 1 | 未恢复 | {a["pilot_candidate_segments"]} | {a["pilot_exported_samples"]} | 整组约束不可行 |')
    text+=['', '每个文件先完整扫描原始记录和实际时间，计算前后SHA，统计实际UTC日、时间回退及间隙。经典PCAP只保留单一链路类型，未提供原始监测接口ID、输入录制路径或攻击批次映射；本地没有找到合并manifest、采集运行日志或原始批次旁表。材料分量数不等于真实物理采集次数，后者保持未知。',
        '', '联合划分按全部五类统一构图，整组分配；同UTC日、重复SHA及时间重叠状态见source_relations.json和merge_witness_edges.json。完整时间窗不相交的文件不可能共享“相同时间戳+相同原始包”事件；改时间后的重复或原始字节近似未认证，未据此宣称独立。没有把文件内不同日期、间隙或试提取窗口当成独立采集。',
        '', '每类只有一个可追溯、不可拆的发布材料分量，无法同时提供2训练+2验证+1测试；普通整组train/val/test三分也无法覆盖同一类。联合求解及所有至少两类子集均失败，因而没有按模型成绩挑类，也没有足以构成来源隔离持续学习任务的保留集合。',
        '', '标签证据：发布论文说明双监测流按实验合并，并对捕获的整个攻击场景赋该场景标签。本地同名PCAP/CSV支持场景队列关联，不提供逐流攻击真值或原始批次身份。试提取采用场景队列标签，不把每条后台连接认定为实际攻击；正常类同样只限发布良性场景。',
        '', '提取验收范围：每文件预先固定10个均匀位置、每窗最多2000包；每窗按双向端点/TCP边界/120秒TCP空闲/30秒UDP空闲/300秒活跃上限形成观察流段。从同一原始包集合重算25维IP统计和匿名IP字节前缀512，保存源offset、packet_index、时间和边界。所有来源窗口仍归整个发布文件，不用于split。',
        '', '试点显式允许单包流段，避免SYN类被旧>=3包门槛静默排除；one/two/>=3包支持与排除原因见captures/*/audit.json。单位是受观察窗截断的流段，不能冒称完整会话，也不等于原CSV的10/100包统计窗口；尚未冻结正式样本协议。保存输入是按sample_id固定排序最多500/类的提取检查集，候选数仅为被抽查窗口的下界，不是全文件有效样本总数。',
        '', f'已从原包偏移独立重算{stats["independent_recomputed_samples"]}个保存输入，统计和字节完全一致；检查集不能替代来源与标签验收。没有导出正式selected/或联邦dataset.pkl。',
        '', '下一路线：CICIoT2023暂不升级主实验。要继续，需恢复带原始录制/批次身份的PCAP与合并关系，或获得新可追溯采集；不能用更多CSV分片代替。DoH保留补采路线和既有12次诊断结果；ISCX三类仅作为另行的应用分类流程验证候选，本轮没有自动切换或训练。',
        '', '依据：[UNB发布说明]('+URL+')；[原始论文]('+PAPER+')；本地README文本及材料清单见evidence/。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n');print(json.dumps(stats,ensure_ascii=False),flush=True)


def run(raw,out,workers):
    if out.exists():raise ValueError('Fresh audit version required')
    out.mkdir(parents=True);code=out/'code';code.mkdir()
    for p in [Path(__file__),Path(pilot.__file__)]:shutil.copy2(p,code/p.name)
    save(out/'code_manifest.json',{p.name:pilot.sha(p) for p in code.iterdir()})
    jobs=[]
    for label in LABELS:
        p=raw/'PCAP'/f'{label}.pcap'
        if not p.is_file():raise ValueError('Missing explicit candidate '+label)
        result=subprocess.run(['capinfos','-T','-m','-Q','-c',str(p)],capture_output=True,text=True,check=True)
        import csv,io
        expected=int(next(csv.DictReader(io.StringIO(result.stdout)))['Number of packets'])
        jobs.append((p,label,expected))
    save(out/'plan.json',dict(dataset='CICIoT2023',classes=LABELS,source_definition='release parent material indivisible until original acquisition mapping recovered',
        requirements={'train_components':2,'fresh_val_components':2,'fresh_test_components':1},windows=10,packets_per_window=2000,saved_segment_cap_per_class=500,
        sampling_uses_labels_or_performance=False,cohort_labels_from_release_filename=True,workers=workers,training_started=False))
    metadata=[]
    with ProcessPoolExecutor(workers) as pool:
        futures={pool.submit(one_capture,out,i,p,l,n,10,2000,500):i for i,(p,l,n) in enumerate(jobs)}
        for future in as_completed(futures):
            i=futures[future];metadata.append((i,future.result()));save(out/'progress.json',dict(completed=len(metadata),total=5,training_started=False))
    summarize(out,[m for _,m in sorted(metadata)],raw)
    for name,h in json.loads((out/'code_manifest.json').read_text()).items():
        if pilot.sha(Path(__file__).parent/name)!=h:raise ValueError('Audit source changed while running')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();run(a.raw.resolve(),a.output.resolve(),a.workers)
