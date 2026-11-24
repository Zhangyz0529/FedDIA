#!/usr/bin/env python3
"""Check bounded raw capture anchors for family/day links across the source universe.

Family/day connected components are the established conservative grouping rule.
Witnesses prove links, not independent sites or sample-level train/test leakage.
Unscanned members/packet overlaps can add links; no feature extraction/training.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zlib
import pandas as pd
import doh_expand as expand
from federated_contract import sha


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def verify(raw,parent,catalog,out):
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir(parents=True);inventory=pd.read_parquet(catalog/'archive_member_inventory.parquet')
    inventory=inventory[(inventory.label!='Benign')&inventory.filename_day.notna()]
    windows=[];excluded=[];budget=512_000_000;used_bytes=0
    for (label,family,day),g in inventory.groupby(['label','family','filename_day']):
        available=g[(g.bytes>100_000)&(g.bytes<3_000_000)].copy()
        if available.empty:
            excluded.append(dict(label=label,family=family,filename_day=day,reason='no_anchor_in_prespecified_100KB_3MB_resource_window'));continue
        available=available.sort_values(['bytes','member']);r=available.iloc[0];p=raw/'PCAPs/extracted_pcaps/malicuous'/Path(r.member).name
        if not p.is_file():excluded.append(dict(family=family,filename_day=day,reason='missing_local_capture'));continue
        if used_bytes+p.stat().st_size>budget:
            excluded.append(dict(family=family,filename_day=day,reason='prespecified_512MB_total_anchor_cap'));continue
        h=hashlib.sha256();crc=0;before=p.stat()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b);crc=zlib.crc32(b,crc)
        if before.st_size!=int(r.bytes) or f'{crc&0xffffffff:08x}'!=r.crc32:
            excluded.append(dict(family=family,filename_day=day,reason='local_archive_size_or_CRC_mismatch'));continue
        w=expand.window(p)
        if sha(p)!=h.hexdigest() or p.stat().st_mtime_ns!=before.st_mtime_ns:raise ValueError('Original changed')
        used_bytes+=before.st_size
        windows.append(dict(label=label,family=family,capture=str(p.resolve()),sha256=h.hexdigest(),archive=r.archive,member=r.member,crc32=r.crc32,
            archive_size_crc_verified=True,filename_day=day,**w,origin='new_anchor'))
    old=json.loads((parent/'file_manifest.json').read_text())
    for r in old:
        windows.append(dict(label='Benign' if r['group']=='browser' else r['group'],family=r['acquisition_family'],
            capture=r['relative_path'],sha256=r['sha256'],start=r['capture_start_time'],end=r['capture_end_time'],origin='old_verified_manifest'))
    save(out/'capture_windows.json',windows);save(out/'excluded_anchors.json',excluded)
    family_days=defaultdict(set);family_labels={}
    for r in windows:
        family=r['family'];family_labels[family]=r['label']
        for day in range(int(r['start']//86400),int(r['end']//86400)+1):family_days[family].add(day)
    parentmap={f:f for f in family_days}
    def root(f):
        while parentmap[f]!=f:parentmap[f]=parentmap[parentmap[f]];f=parentmap[f]
        return f
    families=sorted(family_days);edges=[]
    for i,a in enumerate(families):
        for b in families[:i]:
            days=family_days[a]&family_days[b]
            if days:
                parentmap[root(b)]=root(a)
                edges.append(dict(left=a,right=b,shared_actual_UTC_days=[datetime.fromtimestamp(d*86400,timezone.utc).date().isoformat() for d in sorted(days)]))
    roots={f:root(f) for f in families};used={r['acquisition_family'] for r in old};oldroots={roots[f] for f in used}
    rows=[]
    for label in ['Benign','dns2tcp','dnscat2','iodine']:
        fs=[f for f,l in family_labels.items() if l==label];rs={roots[f] for f in fs}
        rows.append(dict(label=label,witnessed_families=len(fs),components_in_witness_graph=len(rs),components_untouched_by_old_used_families=len(rs-oldroots),
            caveat='Witness graph only; additional anchors and packet overlaps may merge components; unscanned families not certified independent'))
    result=dict(new_verified_anchors=sum(r['origin']=='new_anchor' for r in windows),anchor_bytes=used_bytes,rows=rows,
        family_component=roots,actual_day_edges=edges,no_training=True,raw_files_unchanged=True,
        interpretation='Retaining both whole-family and whole-UTC-day constraints requires recomputing full-universe linkage. Links are not proof of duplicate samples or label leakage.')
    save(out/'verified_day_links.json',result)
    code=out/'code';code.mkdir()
    for p in [Path(__file__),Path(expand.__file__),Path(expand.pilot.__file__),Path(__file__).with_name('federated_contract.py')]:shutil.copy2(p,code/p.name)
    save(out/'code_manifest.json',{p.name:sha(p) for p in code.iterdir()})
    print(json.dumps({k:v for k,v in result.items() if k not in {'family_component','actual_day_edges'}},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['raw','parent','catalog','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.raw.resolve(),a.parent.resolve(),a.catalog.resolve(),a.output.resolve())
