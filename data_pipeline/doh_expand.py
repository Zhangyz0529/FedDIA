"""Expand source-audited DoH inputs without training or rewriting prior versions.

Archive membership + CRC/size verifies local file provenance. Acquisition-family,
UTC-day, duplicate-file and exact packet-overlap constraints define conservative
source groups. This is acquisition/scenario separation, NOT population-level
independence of hosts, sites or organizations. Feature/session rules are reused
unchanged from the frozen pilot and saved alongside this script.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import zipfile
import zlib

import numpy as np
import pandas as pd
import doh_pilot as pilot

VERSION='doh_source_groups_v2_2'
SPLITS=('train','val','test')


def save(path,value):pilot.save(path,value)


def read(path):return json.loads(Path(path).read_text())


def day(ts):return datetime.fromtimestamp(ts,timezone.utc).strftime('%Y-%m-%d')


def window(path):
    first=None;last=None;n=0
    with path.open('rb') as f:
        r=pilot.reader(f);link=r.datalink()
        for ts,raw in r:
            t=float(ts);n+=1;first=t if first is None else min(first,t);last=t if last is None else max(last,t)
    if first is None:raise ValueError('empty capture')
    return dict(start=first,end=last,packets=n,link=link)


def archive_index(raw):
    index=defaultdict(list);catalog=[]
    for archive in sorted((raw/'PCAPs').rglob('*.zip')):
        if 'extracted_pcaps' in archive.parts:continue
        try:
            with zipfile.ZipFile(archive) as z:
                entries=[i for i in z.infolist() if i.filename.lower().endswith(('.pcap','.pcapng'))]
                description=[dict(member=i.filename,bytes=i.file_size,crc32=f'{i.CRC:08x}') for i in entries]
                central_sha=hashlib.sha256(json.dumps(description,sort_keys=True).encode()).hexdigest()
                catalog.append(dict(archive=str(archive.relative_to(raw)),members=len(entries),central_metadata_sha256=central_sha))
                for i in entries:
                    candidate=raw/'PCAPs/extracted_pcaps'/i.filename
                    if not candidate.exists():candidate=raw/'PCAPs/extracted_pcaps/malicuous'/Path(i.filename).name
                    if candidate.is_file():
                        index[str(candidate.resolve())].append(dict(archive=str(archive.relative_to(raw)),member=i.filename,
                                                                    bytes=i.file_size,crc32=f'{i.CRC:08x}',central_metadata_sha256=central_sha))
        except zipfile.BadZipFile:
            catalog.append(dict(archive=str(archive.relative_to(raw)),error='bad_zip'))
    return index,catalog


def prepare(raw,out,per_family,classes=None,normal_files=None):
    classes=list(classes or pilot.LABELS)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing to replace a started dataset version')
    out.mkdir(parents=True);(out/'code').mkdir()
    for p in (Path(__file__),Path(pilot.__file__)):
        shutil.copy2(p,out/'code'/p.name)
    index,catalog=archive_index(raw);save(out/'archive_catalog.json',catalog)
    inventory=[];families=defaultdict(list);missing=Counter()
    pcaps=raw/'PCAPs/extracted_pcaps/malicuous'
    for tool in [x for x in classes if x != "Benign"]:
        for c in sorted((raw/'CSVs/CSVs'/tool/'Separate').glob('*.csv')):
            p=pcaps/(c.stem+'.pcap')
            if not p.exists():missing[tool]+=1;continue
            family=re.split(r'_doh\d+_',c.stem)[0]
            if family==c.stem:continue
            entries=index.get(str(p.resolve()),[])
            inventory.append(dict(label_group=tool,capture=str(p.relative_to(raw)),csv=str(c.relative_to(raw)),
                                  family=family,bytes=p.stat().st_size,archive_candidates=len(entries)))
            if 100000<p.stat().st_size<3000000 and entries:
                families[(tool,family)].append((p.stat().st_size,p,c))
    pd.DataFrame(inventory).to_parquet(out/'malicious_source_inventory.parquet',index=False)
    save(out/'missing_csv_capture_counts.json',dict(missing))
    jobs=[];selection=[]
    # Cover three known acquisition families and distinct UTC days per attack.
    # Size bounds are fixed resource limits, not model-driven sample selection.
    for tool in [x for x in classes if x != "Benign"]:
        selected_days=set();accepted_families=0
        for (label,family),values in sorted(families.items()):
            if label!=tool:continue
            values=sorted(values,key=lambda x:(x[0],str(x[1])))
            day_bins=defaultdict(list);chosen=[];d=None
            for _,p,c in values:
                w=window(p);candidate_day=day(w['start'])
                if candidate_day in selected_days or day(w['end'])!=candidate_day:continue
                day_bins[candidate_day].append((p,c,w))
                if len(day_bins[candidate_day])>=per_family:
                    d=candidate_day
                    for p,c,w in day_bins[d][:per_family]:
                        chosen.append(dict(capture=str(p.resolve()),csv=str(c.resolve()),group=tool,
                                           acquisition_family='scenario:'+family,planned_window=w,
                                           archive_members=index[str(p.resolve())],csv_association='exact stem within release tool group'))
                    break
            if len(chosen)<per_family:continue
            jobs.extend(chosen);selected_days.add(d);accepted_families+=1
            selection.append(dict(label=tool,family=family,day=d,files=len(chosen)))
            if accepted_families==3:break
        if accepted_families<3:raise RuntimeError(f'Insufficient three-family/day coverage for {tool}')
    normal=normal_files if normal_files is not None else [('Google/small/dump.pcap','Firefox','Google'),
            ('Google/dump_00005_20200113142226.pcap','Chrome','Google'),
            ('Quad9/dump_00001_20200111222621.pcap','Chrome','Quad9')]
    if 'Benign' not in classes: normal=[]
    for relative,browser,provider in normal:
        p=raw/'PCAPs/extracted_pcaps'/relative;entries=index.get(str(p.resolve()),[])
        if not entries:raise RuntimeError('No release-archive provenance: '+str(p))
        expected=f'BenignDoH_NonDoH-{browser}-{provider}.zip'
        selected=[e for e in entries if Path(e['archive']).name==expected]
        if not selected:raise RuntimeError('Browser/provider archive association mismatch')
        related=[c for c in (raw/'CSVs/CSVs'/browser/'Separate').glob('*.csv')
                 if re.sub(r'\.[0-9a-f]{6}$','',c.stem)==p.stem]
        # Multiple dump.* CSVs cannot be resolved by basename; retain candidates only.
        c=related[0] if len(related)==1 else None
        w=window(p)
        jobs.append(dict(capture=str(p.resolve()),csv=str(c.resolve()) if c else None,group='browser',
                         acquisition_family='release_capture_family:'+expected,planned_window=w,archive_members=selected,
                         csv_association='unique normalized basename' if c else 'unresolved multiple release CSV candidates',
                         csv_candidates=[str(x.resolve()) for x in related]))
        selection.append(dict(label='Benign',family=expected,day=day(w['start']),files=1))
    save(out/'plan.json',dict(version=VERSION,raw_root=str(raw),per_family=per_family,requested_classes=classes,jobs=jobs,selection=selection,
                              rules=pilot.RULES,source_group_policy='same acquisition family OR shared UTC capture day OR duplicate capture OR exact timestamp+raw packet overlap',
                              group_split_policy='chronological group order within each single-label class, first group train, second val, remainder test; mixed groups fail closed',
                              target='held-out local acquisition/scenario groups; not independent hosts/sites',
                              label_policy='publisher endpoint roles+TLS evidence; archive must agree with browser/tool identity',
                              training_allowed=False))
    save(out/'code_manifest.json',{p.name:pilot.sha(p) for p in (out/'code').glob('*.py')})
    print(json.dumps(dict(prepared_jobs=len(jobs),selection=selection),ensure_ascii=False),flush=True)


def verified_copy(source,target,entries):
    h=hashlib.sha256();crc=0
    with source.open('rb') as src,target.open('wb') as dst:
        for chunk in iter(lambda:src.read(1<<20),b''):
            dst.write(chunk);h.update(chunk);crc=zlib.crc32(chunk,crc)
    crc=f'{crc&0xffffffff:08x}';matching=[e for e in entries if e['bytes']==source.stat().st_size and e['crc32']==crc]
    if not matching:raise ValueError('Local capture differs from release archive size/CRC: '+str(source))
    if pilot.sha(target)!=h.hexdigest():raise ValueError('Working copy mismatch')
    return h.hexdigest(),crc,matching


def work_one(raw,out,index,job):
    cache=out/'captures'/f'{index:03d}'
    if (cache/'complete.json').exists():
        meta=read(cache/'complete.json')
        if pilot.sha(Path(job['capture']))!=meta['sha256']:raise ValueError('Resume input changed')
        return meta
    if cache.exists():raise RuntimeError('Incomplete capture output, inspect before resuming: '+str(cache))
    cache.mkdir(parents=True);source=Path(job['capture']);target=cache/'capture.pcap'
    h,crc,entries=verified_copy(source,target,job['archive_members'])
    rejection=[]
    records,arrays,meta,fp=pilot.inspect_capture(target,job['group'],h,rejection)
    all_counts=Counter(i for row in records+rejection for i in row['packet_indices'])
    if len(all_counts)!=meta['total_packets'] or set(all_counts)!=set(range(meta['total_packets'])) or any(n!=1 for n in all_counts.values()):
        raise ValueError('Packet membership coverage failed')
    for row in records+rejection:row.update(capture_index=index,capture_hash=h,acquisition_family=job['acquisition_family'])
    # Check archive-declared browser/tool source against accepted endpoint roles.
    if any(r['label'] not in ({'Benign'} if job['group']=='browser' else {job['group']}) for r in records):raise ValueError('Label/source conflict')
    pd.DataFrame(records).to_parquet(cache/'provenance.parquet',index=False)
    pd.DataFrame(rejection).to_parquet(cache/'quarantine.parquet',index=False)
    np.savez_compressed(cache/'model_inputs.npz',stat_raw=np.stack([a for a,b in arrays]) if arrays else np.empty((0,25),dtype=np.float32),
                        seq_feature=np.stack([b for a,b in arrays]) if arrays else np.empty((0,512),dtype=np.uint8),
                        sample_id=np.array([r['sample_id'] for r in records]),label=np.array([r['label'] for r in records]))
    np.save(cache/'packet_fingerprints.npy',np.frombuffer(b''.join(sorted(fp)),dtype='V32'))
    csvmeta=None
    if job['csv']:
        c=Path(job['csv']);ct=cache/'related.csv';shutil.copyfile(c,ct)
        csvmeta=dict(path=str(c.relative_to(raw)),sha256=pilot.sha(c),working_sha256=pilot.sha(ct),file_relation=job['csv_association'],
                     use='route A audit only; not used as old CSV statistics or label override')
    if pilot.sha(source)!=h:raise ValueError('Original changed')
    meta.update(index=index,dataset='CIC_DOH',relative_path=str(source.relative_to(raw)),sha256=h,file_size=source.stat().st_size,
                crc32=crc,archive_members=entries,acquisition_family=job['acquisition_family'],group=job['group'],csv=csvmeta,
                source_original_unchanged=True,packet_membership_exact=True,accepted=len(records),working_path=str(target.resolve()))
    save(cache/'complete.json',meta)
    print(f'{index+1}: {job["group"]} accepted={len(records)} packets={meta["total_packets"]} CRC/source hash OK',flush=True)
    return meta


def source_groups(metadata,out):
    parent=list(range(len(metadata)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(j)]=find(i)
    edges=[]
    for i,a in enumerate(metadata):
        days=set(range(int(a['capture_start_time']//86400),int(a['capture_end_time']//86400)+1))
        for j,b in enumerate(metadata[:i]):
            other=set(range(int(b['capture_start_time']//86400),int(b['capture_end_time']//86400)+1));reasons=[]
            if a['acquisition_family']==b['acquisition_family']:reasons.append('same_acquisition_family')
            if days&other:reasons.append('shared_utc_capture_day')
            if a['sha256']==b['sha256']:reasons.append('duplicate_capture')
            overlap=0
            if max(a['capture_start_time'],b['capture_start_time'])<=min(a['capture_end_time'],b['capture_end_time']):
                x=np.load(out/'captures'/f'{i:03d}'/'packet_fingerprints.npy',mmap_mode='r')
                y=np.load(out/'captures'/f'{j:03d}'/'packet_fingerprints.npy',mmap_mode='r')
                overlap=len(np.intersect1d(x,y,assume_unique=True))
                if overlap:reasons.append('exact_timestamp_raw_overlap')
            if reasons:
                union(i,j);edges.append(dict(left=i,right=j,reasons=reasons,exact_overlap_packets=overlap))
    ids={i:f'acquisition_group_{find(i):03d}' for i in range(len(metadata))}
    return ids,edges


def combine(out,metadata):
    ids,edges=source_groups(metadata,out);frames=[];stat=[];seq=[];sampleids=[];labels=[];rejects=[]
    for i,m in enumerate(metadata):
        c=out/'captures'/f'{i:03d}';f=pd.read_parquet(c/'provenance.parquet')
        if not f.empty:f['source_group_id']=ids[i];frames.append(f)
        q=pd.read_parquet(c/'quarantine.parquet')
        if not q.empty:q['source_group_id']=ids[i];rejects.append(q)
        with np.load(c/'model_inputs.npz',allow_pickle=False) as a:
            stat.append(a['stat_raw']);seq.append(a['seq_feature']);sampleids.extend(a['sample_id']);labels.extend(a['label'])
    data=pd.concat(frames,ignore_index=True);X=np.concatenate(stat);S=np.concatenate(seq)
    if len(set(sampleids))!=len(sampleids):raise ValueError('Duplicate sample IDs: resolve raw duplicates before export')
    if not np.isfinite(X).all():raise ValueError('Nonfinite statistics')
    support={g:dict(Counter(f.label)) for g,f in data.groupby('source_group_id')};assignment={};blockers=[]
    groups_by_label=defaultdict(list)
    for g,counts in support.items():
        if len(counts)!=1:blockers.append('mixed-label source group requires explicitly planned grouped allocation: '+g);continue
        group_start=min(m['capture_start_time'] for i,m in enumerate(metadata) if ids[i]==g)
        groups_by_label[next(iter(counts))].append((group_start,g))
    for label in read(out/'plan.json').get('requested_classes',pilot.LABELS):
        groups=sorted(groups_by_label[label])
        if len(groups)<3:blockers.append('fewer than 3 nonoverlapping acquisition/scenario groups: '+label)
        for i,(_,g) in enumerate(groups):assignment[g]=SPLITS[min(i,2)]
    data['split']=data.source_group_id.map(assignment).fillna('unassigned')
    duplicates=defaultdict(list)
    for i,(x,s) in enumerate(zip(X,S)):duplicates[hashlib.sha256(x.tobytes()+s.tobytes()).hexdigest()].append(i)
    cross=[v for v in duplicates.values() if len({data.iloc[i]['split'] for i in v})>1]
    if cross:blockers.append('cross-split identical model inputs require provenance review')
    split_support={s:dict(Counter(data.loc[data.split==s,'label'])) for s in SPLITS}
    missing={s:[l for l in read(out/'plan.json').get('requested_classes',pilot.LABELS) if not split_support[s].get(l)] for s in SPLITS}
    if any(missing.values()):blockers.append('missing class support in proposed splits')
    data.to_parquet(out/'provenance.parquet',index=False)
    data[['sample_id','label','source_group_id','split','capture_index','packet_count','sequence_valid_length']].to_parquet(out/'samples.parquet',index=False)
    pd.concat(rejects,ignore_index=True).to_parquet(out/'quarantine.parquet',index=False)
    np.savez_compressed(out/'model_inputs.npz',stat_raw=X,seq_feature=S,sample_id=np.array(sampleids),label=np.array(labels))
    save(out/'file_manifest.json',metadata);save(out/'source_group_edges.json',edges)
    save(out/'split_manifest.json',dict(version=VERSION,source_group_support=support,assignment=assignment,support=split_support,missing_class_support=missing,
                                      independent_unit='local release acquisition family + UTC day + observed packet overlap closure',
                                      claim='held-out acquisition/scenario groups among selected files; no host/site independence claim',training_allowed=False))
    report=dict(version=VERSION,status='SOURCE_AUDIT_COMPLETE' if not blockers else 'SOURCE_AUDIT_BLOCKED',accepted_samples=len(data),selected_captures=len(metadata),
                labels=dict(Counter(data.label)),source_groups=len(support),source_groups_per_class={l:len(v) for l,v in groups_by_label.items()},
                split_support=split_support,cross_split_identical_inputs=len(cross),duplicate_input_groups=sum(len(v)>1 for v in duplicates.values()),
                raw_hashes_unchanged=True,archive_crc_size_verified=True,all_packets_accounted_once=True,
                blockers=blockers,training_allowed=False,training_started=False,
                limits=['acquisition/scenario separation does not certify independent hosts/sites','selected captures, not all local dataset files','route A new25 features; CSV old29 not reused'])
    save(out/'audit_report.json',report)
    # Save explicit split arrays only after source-level gates; no federated task schedule is invented here.
    if not blockers:
        (out/'splits').mkdir(exist_ok=True)
        for split in SPLITS:
            mask=(data.split==split).to_numpy()
            np.savez_compressed(out/'splits'/f'{split}.npz',stat_raw=X[mask],seq_feature=S[mask],sample_id=np.array(sampleids)[mask],label=np.array(labels)[mask])
    save(out/'feature_schema.json',read(out.parent/'doh_pilot_v2_20261108/feature_schema.json') if (out.parent/'doh_pilot_v2_20261108/feature_schema.json').exists()
         else dict(version=pilot.RULES['schema'],fields=pilot.FEATURES,rules=pilot.RULES))
    print(json.dumps(report,ensure_ascii=False),flush=True)


def verify(out,write_report=True):
    data=pd.read_parquet(out/'provenance.parquet');metadata=read(out/'file_manifest.json');checks=[]
    with np.load(out/'model_inputs.npz',allow_pickle=False) as saved:
        idmap={x:i for i,x in enumerate(saved['sample_id'])}
        for i,m in enumerate(metadata):
            rows=data[data.capture_index==i].sort_values('sample_id').head(3)
            if rows.empty:continue
            path=out/'captures'/f'{i:03d}'/'capture.pcap'
            if pilot.sha(path)!=m['sha256']:raise ValueError('Working capture changed')
            wanted={int(v) for a in rows.packet_indices for v in a};packets={}
            with path.open('rb') as f:
                r=pilot.reader(f);link=r.datalink()
                for index,(ts,raw) in enumerate(r):
                    if index in wanted:packets[index]=(float(ts),pilot.decode(raw,link))
                    if len(packets)==len(wanted):break
            for row in rows.itertuples():
                unit=[packets[int(x)] for x in row.packet_indices];origin=unit[0][1]['src']
                t=np.array([ts for ts,p in unit]);length=np.array([p['size'] for ts,p in unit],dtype=float)
                direction=np.array([int(p['src']!=origin) for ts,p in unit]);forward=length[direction==0];back=length[direction==1]
                def moments(v):return [float(v.mean()),float(v.std()),float(v.max()),float(v.min())] if len(v) else [0.]*4
                duration=t[-1]-t[0];rates=[len(t)/duration,length.sum()/duration,len(forward)/duration,len(back)/duration] if duration else [0.]*4
                vector=np.array([duration,len(forward),len(back),forward.sum(),back.sum(),*moments(forward),*moments(back),*moments(length),*moments(np.diff(t)),*rates],dtype=np.float32)
                sequence=np.frombuffer(b''.join(p['network'] for ts,p in unit)[:512].ljust(512,b'\0'),dtype=np.uint8)
                k=idmap[row.sample_id];ok=np.array_equal(vector,saved['stat_raw'][k]) and np.array_equal(sequence,saved['seq_feature'][k])
                checks.append(dict(sample_id=row.sample_id,capture_index=i,recomputed_equal=bool(ok)))
                if not ok:raise ValueError('Independent recomputation mismatch')
    save(out/'independent_recompute_checks.json',checks)
    if not checks:raise ValueError('No recomputation checks')
    report=read(out/'audit_report.json');report['independent_recomputed_samples']=len(checks)
    save(out/'audit_report.json',report)
    if not write_report: return
    lines=['# DoH 来源扩充与生成验收','',f'状态：{report["status"]}。未训练，未导出联邦任务 dataset.pkl。',
           '',f'核验 {len(metadata)} 个采集，生成 {len(data)} 个同包集合候选输入；按本地发布采集族、UTC采集日、重复文件和精确包重叠形成 {report["source_groups"]} 个保守来源组。',
           '', '| split | Benign | dns2tcp | dnscat2 | iodine |','| --- | ---: | ---: | ---: | ---: |']
    for split in SPLITS:lines.append('| '+split+' | '+' | '.join(str(report['split_support'][split].get(l,0)) for l in pilot.LABELS)+' |')
    lines+=['',f'逐文件原始SHA256、工作副本SHA256、原始ZIP成员CRC/大小一致；每个包索引在接受/隔离记录中恰好出现一次。按包索引独立重算 {len(checks)} 个样本通过。',
            '',f'跨split完全相同模型输入组：{report["cross_split_identical_inputs"]}。',
            '', '## 证据范围','',
            '正常来源依据本地原始ZIP中浏览器/服务商采集批次；同批切片归一组。攻击依据采集文件名中完整场景族（去除工作进程号和时间后缀），不同工作进程或相邻小文件不当作独立来源。',
            '时间重叠按UTC采集日进一步保守合并；实际时间区间重叠的文件还检查原始时间戳+原始包精确重复。只对本轮选中文件有完整包级证据。',
            '这验证的是选定本地采集/场景组的隔离，不是不同组织、不同主机、不同采集系统的统计独立性；同一实验平台背景和固定客户端地址仍是局限。',
            '采用路线A，特征仍为新25维，序列为匿名化IP包字节前缀。CSV仅为来源旁证，不声称已认证旧29维CSV逐流对齐。',
            '文件大小范围和每族文件数是预先固定的资源约束；不是代表性抽样，需披露类别/场景选择偏差。没有拟合任何标准化。',
            '', '## 尚需处理','']
    lines.extend('- '+x for x in report['blockers'])
    lines+=['- 即使来源门槛通过，也不自动启动训练；先确认当前采集组隔离就是论文要检验的泛化定义，再指定联邦任务协议。',
            '', '## 产物','',
            'plan.json、archive_catalog.json、file_manifest.json：固定选择、档案出处、校验与时间范围。',
            'source_group_edges.json、split_manifest.json：不能跨split的关系与各类支持。',
            'provenance.parquet、quarantine.parquet、model_inputs.npz：可追溯输入及排除记录。',
            'splits/*.npz：仅当来源门槛通过时导出；training_allowed仍为false，尚无联邦任务分配。',
            'captures/：逐文件工作副本和可恢复缓存；code/：冻结生成代码。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(out):
    plan=read(out/'plan.json');raw=Path(plan['raw_root']);code=read(out/'code_manifest.json')
    for name,h in code.items():
        if pilot.sha(out/'code'/name)!=h:raise ValueError('Frozen code changed')
    save(out/'label_rules.json',dict(publisher=pilot.PUBLISHER,resolvers=sorted(pilot.RESOLVERS),browser_clients=sorted(pilot.BROWSERS),
                                   tunnel_clients=sorted(pilot.TUNNEL_CLIENTS),policy=plan['label_policy'],limitations='publisher-topology labels, not independent payload-content proof'))
    metadata=[]
    for i,job in enumerate(plan['jobs']):
        metadata.append(work_one(raw,out,i,job))
        save(out/'progress.json',dict(completed_captures=i+1,total=len(plan['jobs']),training_started=False))
    combine(out,metadata);verify(out)
    save(out/'extraction_manifest.json',dict(version=VERSION,code_manifest=code,rules=pilot.RULES,
                                            dependencies={k:importlib.metadata.version(k) for k in ['numpy','dpkt','pandas','pyarrow']},training_started=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','verify'])
    parser.add_argument('--raw-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--per-family',type=int,default=5)
    args=parser.parse_args();out=args.output.resolve()
    if args.action=='prepare':prepare(args.raw_root.resolve(),out,args.per_family)
    elif args.action=='run':run(out)
    else:verify(out)
