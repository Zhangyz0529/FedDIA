#!/usr/bin/env python3
"""Bounded full capinfos probes of primary local captures; no source rewriting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv,io,json
from pathlib import Path
import shutil,subprocess
import pandas as pd
from federated_contract import sha


def probe(job):
    dataset,path=job;before=path.stat()
    r=subprocess.run(['capinfos','-T','-m','-Q','-t','-E','-c','-a','-e','-S',str(path)],capture_output=True,text=True)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed')
    result=dict(dataset=dataset,path=str(path.resolve()),bytes=before.st_size,returncode=r.returncode,stderr=r.stderr[:2000],scope='FULL_CAPTURE_METADATA_SCAN',original_stat_unchanged=True)
    try:result.update(next(csv.DictReader(io.StringIO(r.stdout))))
    except (StopIteration,ValueError):pass
    return result


def main(raw,out):
    jobs=[]
    for dataset,folders in [('USTC-TFC2016-master',['Benign','Malware']),('ISCX',['pcap']),('UNSW_NB15',['Pcap']),('CICloV2023',['PCAP'])]:
        for folder in folders:
            for p in sorted((raw/dataset/folder).iterdir()):
                if p.suffix.lower() in {'.pcap','.pcapng'}:jobs.append((dataset,p))
    for p in sorted((raw/'CIC-DDoS-2019').glob('*/PCAPs/*_0.pcap')):jobs.append(('CIC-DDoS-2019',p))
    inventory=pd.read_parquet(out/'file_inventory.parquet')
    g=inventory[(inventory.dataset=='CipherSpectrum')&(inventory.extension=='.pcap')]
    for container,f in g.groupby('level1'):
        for browser in ['firefox','chromium']:
            candidates=f[f.relative_path.str.contains('_'+browser+'_')&~f.relative_path.str.contains('__MACOSX')]
            p=raw/'CipherSpectrum'/sorted(candidates.relative_path)[0];jobs.append(('CipherSpectrum',p))
    total=sum(p.stat().st_size for _,p in jobs)
    if total>25_000_000_000:raise ValueError('Prespecified 25GB probe cap exceeded')
    (out/'capture_probe_plan.json').write_text(json.dumps(dict(jobs=[dict(dataset=d,path=str(p.resolve())) for d,p in jobs],bytes=total,CRC_archive_identity_not_verified_here=True),indent=2)+'\n')
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(probe,jobs))
    (out/'capture_probes.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    pd.DataFrame(results).to_csv(out/'capture_probes.csv',index=False)
    print('captured metadata',len(results),'bytes',total,'nonzero_returns',sum(r['returncode']!=0 for r in results))
    code=out/'probe_code';code.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('federated_contract.py')]:shutil.copy2(p,code/p.name)
    (out/'probe_code_manifest.json').write_text(json.dumps({p.name:sha(p) for p in code.iterdir()},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.raw.resolve(),a.output.resolve())
