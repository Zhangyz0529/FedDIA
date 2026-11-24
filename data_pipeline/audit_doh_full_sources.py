#!/usr/bin/env python3
"""Stream release members without extraction; audit one global source graph.

CRC/SHA and full capinfos timestamp scans are separate from accepted sample and
label support. Known packet overlap is inherited; new overlap checks remain
explicitly incomplete. No data split, sample extraction or training is started.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
import zlib
import pandas as pd
from federated_contract import sha


def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def scan_archive(raw,relative,rows,out):
    source=raw/relative;before=source.stat();index=hashlib.sha256(relative.encode()).hexdigest()[:16]
    target=out/'archives'/f'{index}.jsonl';metadata=out/'archives'/f'{index}.json'
    if metadata.exists():
        old=json.loads(metadata.read_text())
        if (old['archive_size'],old['archive_mtime_ns'])!=(before.st_size,before.st_mtime_ns):raise ValueError('Archive changed')
        return old
    if target.exists():raise ValueError('Incomplete scan must be inspected')
    counts=Counter();total=0;started=time.monotonic()
    with zipfile.ZipFile(source) as archive,target.open('w') as log:
        for r in rows:
            member=archive.getinfo(r['member'])
            if member.file_size!=r['bytes'] or f'{member.CRC:08x}'!=r['crc32']:raise ValueError('ZIP directory changed')
            h=hashlib.sha256();crc=0;n=0;broken=False
            proc=subprocess.Popen(['capinfos','-T','-m','-Q','-t','-E','-c','-a','-e','-S','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                with archive.open(member) as f:
                    for b in iter(lambda:f.read(1<<20),b''):
                        n+=len(b);h.update(b);crc=zlib.crc32(b,crc)
                        if not broken:
                            try:proc.stdin.write(b)
                            except BrokenPipeError:broken=True
                try:proc.stdin.close()
                except BrokenPipeError:broken=True
                proc.stdin=None;stdout,stderr=proc.communicate(timeout=120)
            except BaseException:
                proc.kill();proc.wait();raise
            if n!=member.file_size or f'{crc&0xffffffff:08x}'!=r['crc32']:raise ValueError('Member content CRC/length mismatch')
            item={**r,'member_sha256':h.hexdigest(),'verified_bytes':n,'crc_size_verified':True,
                'capinfos_returncode':proc.returncode,'capinfos_stderr':stderr.decode(errors='replace')[:2000],
                'parsed_packets':None,'start':None,'end':None,'link_type':None,'timestamp_status':'PARSE_FAILED',
                'new_sample_support':'NOT_EXTRACTED','new_packet_overlap':'NOT_CHECKED'}
            try:
                records=list(csv.DictReader(io.StringIO(stdout.decode())))
                if len(records)!=1:raise ValueError('Invalid capinfos table')
                info=records[0];item['parsed_packets']=int(info['Number of packets']);item['link_type']=info['File encapsulation'];item['file_type']=info['File type']
                if item['parsed_packets']==0:item['timestamp_status']='EMPTY_CAPTURE' if proc.returncode==0 else 'INVALID_CAPTURE'
                else:
                    item['start']=float(info['Start time']);item['end']=float(info['End time'])
                    item['timestamp_status']='FULL_SCAN_OK' if proc.returncode==0 else 'PARTIAL_SCAN_ERROR'
            except (ValueError,KeyError,UnicodeError):pass
            counts[item['timestamp_status']]+=1;total+=n
            log.write(json.dumps(item,ensure_ascii=False)+'\n');log.flush()
    after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Archive changed during scan')
    result=dict(archive=relative,archive_size=before.st_size,archive_mtime_ns=before.st_mtime_ns,members=len(rows),uncompressed_bytes=total,
        statuses=dict(counts),seconds=time.monotonic()-started,member_log=target.name,original_unchanged=True)
    save(metadata,result);print(relative,dict(counts),'GB',round(total/1e9,3),flush=True)
    return result


def build_graph(records,oldmeta,oldprovenance,oldedges):
    """Family nodes, cross-label day/hash/known-overlap edges, stable component IDs."""
    families=sorted({r['family'] for r in records});parent={f:f for f in families};edges=[]
    def root(f):
        while parent[f]!=f:parent[f]=parent[parent[f]];f=parent[f]
        return f
    def link(a,b,reason,evidence):
        if a==b:return
        parent[root(b)]=root(a);edges.append(dict(left=a,right=b,reason=reason,evidence=evidence))
    days=defaultdict(dict);hashes=defaultdict(dict)
    for r in records:
        family=r['family'];witness=dict(archive=r['archive'],member=r['member'],sha256=r['member_sha256'],status=r['timestamp_status'])
        # Windows contribute conservative positive links even if the scan failed later.
        if r['start'] is not None and r['end'] is not None:
            for day in range(int(r['start']//86400),int(r['end']//86400)+1):days[day].setdefault(family,witness)
        hashes[r['member_sha256']].setdefault(family,witness)
    for day,seen in days.items():
        fs=sorted(seen)
        for f in fs[1:]:link(fs[0],f,'shared_UTC_window_day',dict(utc_day=pd.to_datetime(day*86400,unit='s',utc=True).date().isoformat(),left_capture=seen[fs[0]],right_capture=seen[f]))
    for h,seen in hashes.items():
        fs=sorted(seen)
        for f in fs[1:]:link(fs[0],f,'duplicate_capture_SHA256',dict(sha256=h,left_capture=seen[fs[0]],right_capture=seen[f]))
    byhash=defaultdict(set)
    for r in records:byhash[r['member_sha256']].add(r['family'])
    for e in oldedges:
        if e['exact_overlap_packets']<=0:continue
        a,b=oldmeta[e['left']],oldmeta[e['right']]
        for fa in byhash[a['sha256']]:
            for fb in byhash[b['sha256']]:link(fa,fb,'known_exact_timestamp_raw_packet_overlap',dict(old_left=e['left'],old_right=e['right'],packets=e['exact_overlap_packets']))
    groups=defaultdict(list)
    for f in families:groups[root(f)].append(f)
    ids={f:'global_'+hashlib.sha256('|'.join(sorted(fs)).encode()).hexdigest()[:16] for fs in groups.values() for f in fs}
    old_hashes={x['sha256'] for x in oldmeta};old_family={x['acquisition_family'] for x in oldmeta}
    # A family previously used in diagnosis taints the entire current component.
    touched={ids[f] for f in old_family if f in ids}
    accepted=Counter();old_samples=Counter()
    for r in oldprovenance.to_dict('records'):
        fs=byhash.get(r['capture_hash'],set())
        components={ids[f] for f in fs}
        if len(components)!=1:raise ValueError('Old accepted capture missing or split in release graph')
        comp=next(iter(components));accepted[(comp,r['label'])]+=1;old_samples[(comp,r['label'],r['split'])]+=1
    for r in records:r.update(component=ids[r['family']],component_used_in_diagnosis=ids[r['family']] in touched,exact_old_capture=r['member_sha256'] in old_hashes)
    return ids,edges,touched,accepted,old_samples


def report(out,parent):
    allrecords=[]
    for p in sorted((out/'archives').glob('*.jsonl')):allrecords.extend(json.loads(line) for line in p.read_text().splitlines())
    catalog=pd.read_parquet(out/'member_plan.parquet')
    if len(allrecords)!=len(catalog):raise ValueError('Unscanned members remain; no completed report')
    oldmeta=json.loads((parent/'file_manifest.json').read_text());oldprov=pd.read_parquet(parent/'selected/provenance.parquet')
    oldedges=json.loads((parent/'source_group_edges.json').read_text())
    ids,edges,touched,accepted,old_samples=build_graph(allrecords,oldmeta,oldprov,oldedges)
    frame=pd.DataFrame(allrecords);frame.to_parquet(out/'members.parquet',index=False);save(out/'merge_witness_edges.json',edges)
    rows=[]
    for (comp,label),g in frame.groupby(['component','label']):
        rows.append(dict(component=comp,label=label,families=g.family.nunique(),members=len(g),full_time_scan_ok=int((g.timestamp_status=='FULL_SCAN_OK').sum()),
            empty_captures=int((g.timestamp_status=='EMPTY_CAPTURE').sum()),partial_or_failed=int(g.timestamp_status.isin(['PARTIAL_SCAN_ERROR','PARSE_FAILED','INVALID_CAPTURE']).sum()),
            unscanned_files=0,component_used_in_diagnosis=comp in touched,old_accepted_samples=accepted[(comp,label)],
            old_train_samples=old_samples[(comp,label,'train')],old_val_samples=old_samples[(comp,label,'val')],old_test_samples=old_samples[(comp,label,'test')],
            new_accepted_samples=None,new_accepted_support_status='NOT_EXTRACTED',new_packet_overlap_status='NOT_CHECKED'))
    supports=pd.DataFrame(rows);supports.to_csv(out/'component_class_support.csv',index=False)
    ledger=[]
    for label in ['Benign','dns2tcp','dnscat2','iodine']:
        g=frame[frame.label==label];comps=set(g.component);fresh=comps-touched
        # Raw windows/hash grouping is sufficient for a negative gate, not positive certification.
        ledger.append(dict(label=label,family_upper_bound=g.family.nunique(),planned_members=len(g),scanned_members=len(g),
            full_time_scan_ok=int((g.timestamp_status=='FULL_SCAN_OK').sum()),empty_captures=int((g.timestamp_status=='EMPTY_CAPTURE').sum()),
            partial_or_failed=int(g.timestamp_status.isin(['PARTIAL_SCAN_ERROR','PARSE_FAILED','INVALID_CAPTURE']).sum()),unscanned_files=0,
            components_with_observed_edges=len(comps),components_used_in_diagnosis=len(comps&touched),untouched_components_before_new_overlap_checks=len(fresh),
            old_accepted_samples=int(supports[supports.label==label].old_accepted_samples.sum()),new_accepted_support='UNKNOWN_NOT_EXTRACTED',
            minimum_training_groups=2,minimum_fresh_validation_groups=2,minimum_fresh_test_groups=1,
            necessary_group_count_conditions_met=len(comps)>=5 and len(fresh)>=3,
            assignment_status='BLOCKED_SOURCE_GROUP_REAUDIT'))
    pd.DataFrame(ledger).to_csv(out/'split_feasibility.csv',index=False)
    save(out/'component_membership.json',ids)
    save(out/'audit_status.json',dict(status='BLOCKED_SOURCE_GROUP_REAUDIT',members=len(frame),unscanned_files=0,
        CRC_verified_members=int(frame.crc_size_verified.sum()),timestamp_status_counts=frame.timestamp_status.value_counts().to_dict(),
        global_components=frame.component.nunique(),mixed_label_components=int((frame.groupby('component').label.nunique()>1).sum()),
        old_accepted_support_reconciled=int(supports.old_accepted_samples.sum())==len(oldprov),
        new_packet_overlap_scan='NOT_COMPLETED',new_accepted_label_support='NOT_EXTRACTED',new_split_frozen=False,training_started=False,
        positive_feasibility_claim=False,raw_archives_unchanged=True))
    text=['# 全局来源分组与划分可行性','',
        '维持 **BLOCKED_SOURCE_GROUP_REAUDIT**。所有类别在同一张图中按既有同族/UTC日规则分组，并加入重复文件SHA和已知包重叠见证。没有逐类独立分组后拼表，没有生成新split或训练。',
        '', '| 类别 | 全部成员 | 完整时间扫描 | 空采集 | 时间扫描异常 | 分量数 | 已诊断分量 | 暂未触及旧诊断的分量 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in ledger:text.append(f'| {r["label"]} | {r["planned_members"]} | {r["full_time_scan_ok"]} | {r["empty_captures"]} | {r["partial_or_failed"]} | {r["components_with_observed_edges"]} | {r["components_used_in_diagnosis"]} | {r["untouched_components_before_new_overlap_checks"]} |')
    text+=['', '“暂未触及”仅表示在已完成的族/时间窗/文件哈希/旧包重叠图中未与旧来源相连；新增文件的逐包重叠尚未全量核验，不是已认证独立来源。扫描异常和空采集不能当作有标签样本支持。',
        '', '全量范围固定为发布ZIP清单中的33个正常成员、2800个dns2tcp成员、4800个dnscat2成员、5521个iodine成员。各成员从原ZIP流式读取，校验CRC/长度并计算SHA；capinfos完整读取包记录和时间边界，无解压副本。原归档mtime/大小前后核对。',
        '', 'component_class_support.csv分别列出每个全局分量各类的原始文件支持、已验收旧样本及其旧split支持。新材料没有提取正式样本，因此新接受样本数为未知，不能用原始包数或文件数代替。split_feasibility.csv是全局图上的必要条件表，不是可分配方案；混合标签分量必须整体进入一个split。',
        '', '同一场景族的全体成员归属同一节点；merge_witness_edges.json保存跨族同UTC时间窗日、重复SHA及已知精确包重叠的原始文件见证。UTC日是保守代理，不等于采集独立性的实证判定；新增包重叠未知项保留，不声称“无泄漏”。',
        '', '完整四类方案仍受dns2tcp最多4个场景族的上界阻断，达不到2训练+2新验证+1新测试。对任何较小类别集合，也必须先检查全局图、正式标签支持和未完成重叠检查，不能凭场景族数量宣布可用。',
        '', '新图若将旧train/val/test的来源合并，只表示扩大文件宇宙后旧局部分组不能直接继承，不构成已发现同一样本泄漏的证据。已有12次结果保留为原划分上的探索性检测失败与训练波动，不能转写为新来源确认性结果。',
        '', '下一步：保留分组规则时补采能形成新连通分量的可追溯材料，预算按最终全局分量而非文件数计；若研究场景泛化，先依据采集设计论证不同研究单位，再单列协议版本。旧测试已参与诊断，新方案仅依赖训练/验证选择，确认性测试应提前冻结新的未用来源。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n');print(json.dumps(ledger,ensure_ascii=False),flush=True)


def run(raw,parent,catalog,out,workers):
    if out.exists():raise ValueError('Fresh audit output required')
    (out/'archives').mkdir(parents=True)
    inventory=pd.read_parquet(catalog/'archive_member_inventory.parquet');inventory.to_parquet(out/'member_plan.parquet',index=False)
    save(out/'plan.json',dict(status='BLOCKED_SOURCE_GROUP_REAUDIT',raw=str(raw),parent=str(parent),catalog=str(catalog),workers=workers,
        planned_members=len(inventory),uncompressed_bytes=int(inventory.bytes.sum()),group_policy='global across all labels: same family OR shared UTC window day OR duplicate SHA OR known exact packet overlap',
        new_overlap_policy='not asserted absent; retain NOT_CHECKED',no_training=True,capinfos_version=subprocess.check_output(['capinfos','-v'],text=True).splitlines()[0]))
    jobs=[(a,g.to_dict('records')) for a,g in inventory.groupby('archive')];jobs.sort(key=lambda x:(not x[0].split('/')[-1].startswith('Benign'),-sum(r['bytes'] for r in x[1])))
    completed=[]
    with ThreadPoolExecutor(workers) as pool:
        futures=[pool.submit(scan_archive,raw,a,rows,out) for a,rows in jobs]
        for f in as_completed(futures):
            completed.append(f.result());save(out/'progress.json',dict(completed_archives=len(completed),total_archives=len(jobs),scanned_members=sum(x['members'] for x in completed),training_started=False))
    report(out,parent)
    code=out/'code';code.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('federated_contract.py')]:shutil.copy2(p,code/p.name)
    save(out/'code_manifest.json',{p.name:sha(p) for p in code.iterdir()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['raw','parent','catalog','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();run(a.raw.resolve(),a.parent.resolve(),a.catalog.resolve(),a.output.resolve(),a.workers)
