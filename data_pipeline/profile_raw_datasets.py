#!/usr/bin/env python3
"""Count canonical CSV rows/labels and inspect recording identity without training.

Counts describe parsed records, not verified paired samples or independent
sources. Related representations and augmented captures are not new datasets.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import re
import shutil
import pandas as pd
from federated_contract import sha


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def select_jobs(raw):
    jobs=[]
    def add(dataset,files,source_role,label=None):
        for p in sorted(files):jobs.append(dict(dataset=dataset,path=str(p.resolve()),role=source_role,folder_label=label(p) if label else None))
    add('CIC-DDoS-2019',list((raw/'CIC-DDoS-2019').glob('*/CSVs/*/*.csv')),'release_labelled_flows')
    add('CICIoT2023',list((raw/'CICloV2023/CSV/extracted').rglob('*.csv')),'release_attack_folder_features',lambda p:p.parent.name)
    add('CIC_IDS',list((raw/'CIC_IDS/GeneratedLabelledFlows/TrafficLabelling').glob('*.csv')),'release_labelled_flows')
    add('UNSW_NB15',[raw/'UNSW_NB15/train.csv',raw/'UNSW_NB15/test.csv'],'canonical_table_train_test')
    add('CICIoV2024',list((raw/'CICIoV2024').glob('*/*.csv')),'equivalent_CAN_representations')
    add('USTC-TFC2016',list((raw/'USTC-TFC2016-master/flow_csv').glob('*/*.csv')),'locally_generated_flow_features')
    add('DoH_merged',[raw/'cicdohbrw2020/L1-DoH-NonDoH.csv',raw/'cicdohbrw2020/L2-BenignDoH-MaliciousDoH.csv'],'alternative_tasks_same_DoH_corpus')
    return jobs


def profile(job):
    path=Path(job['path']);before=path.stat();header=pd.read_csv(path,nrows=0,encoding='cp1252').columns.tolist()
    labels=[x for x in header if x.strip().lower() in {'label','attack_cat','category','specific_class'}]
    cols=labels or header[:1];counts={x:Counter() for x in labels};n=0
    for chunk in pd.read_csv(path,usecols=cols,dtype=str,chunksize=500_000,encoding='cp1252',on_bad_lines='error'):
        n+=len(chunk)
        for col in labels:counts[col].update(chunk[col].fillna('<MISSING>').str.strip())
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed')
    names={x.strip().lower() for x in header}
    return {**job,'bytes':before.st_size,'rows':n,'columns':header,'label_counts':{k:dict(v) for k,v in counts.items()},
        'has_explicit_label':bool(labels),'has_folder_class':job['folder_label'] is not None,
        'has_endpoint_fields':bool(names&{'source ip','sourceip','srcip','src_ip','destination ip','destinationip','dstip','dst_ip'}),
        'has_timestamp_field':bool(names&{'timestamp','time stamp','stime','ltime','starttime','start_time','end_time'}),
        'count_status':'FULL_PARSED_SELECTED_COLUMNS','sample_pairing_certified':False,'original_stat_unchanged':True}


def cipher_inventory(inventory,out):
    f=inventory[(inventory.dataset=='CipherSpectrum')&(inventory.extension=='.pcap')].copy()
    metadata=f.relative_path.map(lambda x:'__MACOSX' in Path(x).parts or Path(x).name.startswith('._'))
    f=f[~metadata];rows=[]
    pattern=re.compile(r'^(traffic_(\d{4}-\d{2}-\d{2})_(.+)_(aes-128|aes-256|chacha20)_(chromium|firefox)_(\d+)\.pcap)\.')
    for r in f.to_dict('records'):
        parts=Path(r['relative_path']).parts;m=pattern.match(parts[-1])
        rows.append(dict(relative_path=r['relative_path'],bytes=r['bytes'],container=r['level1'],folder_site=parts[-2],
            parent_recording_name=m.group(1) if m else None,named_day=m.group(2) if m else None,
            filename_site=m.group(3) if m else None,cipher_hint=m.group(4) if m else None,browser_hint=m.group(5) if m else None,
            label_certification='filename/folder hints only; not verified application or cipher truth'))
    frame=pd.DataFrame(rows);frame.to_parquet(out/'cipher_recording_hints.parquet',index=False)
    frame.groupby('container').agg(pcaps=('relative_path','size'),bytes=('bytes','sum'),named_days=('named_day','nunique'),
        named_parent_recordings=('parent_recording_name','nunique'),folder_sites=('folder_site','nunique'),filename_sites=('filename_site','nunique')).reset_index().to_csv(out/'cipher_source_summary.csv',index=False)
    frame.groupby(['container','cipher_hint'],dropna=False).size().rename('pcaps').reset_index().to_csv(out/'cipher_container_vs_hint.csv',index=False)
    save(out/'cipher_caveats.json',dict(nonmetadata_pcaps=len(frame),metadata_pcap_suffix_files=int(metadata.sum()),
        unparsed_filenames=int(frame.parent_recording_name.isna().sum()),folder_site_differs_from_filename_site=int((frame.folder_site!=frame.filename_site).sum()),
        no_new_dataset_count_for_mix=True,independent_sources_certified=False))


def run(raw,out,workers):
    out.mkdir(parents=True,exist_ok=True)
    jobs=select_jobs(raw);save(out/'csv_profile_plan.json',jobs);rows=[];errors=[]
    with ThreadPoolExecutor(workers) as pool:
        future={pool.submit(profile,j):j for j in jobs}
        for f in as_completed(future):
            try:
                r=f.result();rows.append(r)
                if len(rows)%25==0:print('profiled',len(rows),'/',len(jobs),flush=True)
            except Exception as e:errors.append(dict(job=future[f],error=repr(e)))
            if (len(rows)+len(errors))%25==0:save(out/'csv_progress.json',dict(completed=len(rows),errors=len(errors),total=len(jobs)))
    save(out/'csv_profiles.json',rows);save(out/'csv_errors.json',errors)
    counts=[]
    for r in rows:
        if not r['label_counts'] and r['folder_label'] is not None:counts.append(dict(dataset=r['dataset'],path=r['path'],role=r['role'],label_field='PARENT_FOLDER_COHORT',label=r['folder_label'],rows=r['rows']))
        for field,values in r['label_counts'].items():
            for label,n in values.items():counts.append(dict(dataset=r['dataset'],path=r['path'],role=r['role'],label_field=field.strip(),label=label,rows=n))
    pd.DataFrame(counts).to_csv(out/'csv_label_support.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in {'columns','label_counts'}} for r in rows]).to_csv(out/'csv_file_support.csv',index=False)
    cipher_inventory(pd.read_parquet(out/'file_inventory.parquet'),out)
    code=out/'code';code.mkdir(exist_ok=True)
    for p in [Path(__file__),Path(__file__).with_name('federated_contract.py')]:shutil.copy2(p,code/p.name)
    save(out/'code_manifest.json',{p.name:sha(p) for p in code.iterdir()})
    save(out/'csv_progress.json',dict(completed=len(rows),errors=len(errors),total=len(jobs)))
    print('CSV profiles complete',len(rows),'errors',len(errors),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.raw.resolve(),a.output.resolve(),a.workers)
