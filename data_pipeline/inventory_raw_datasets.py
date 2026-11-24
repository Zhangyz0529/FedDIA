#!/usr/bin/env python3
"""Read-only filesystem census; stored bytes are not unique usable data volume."""
import argparse,json,os
from pathlib import Path
import pandas as pd


def inventory(root,out):
    if out.exists():raise ValueError('Fresh output directory required')
    out.mkdir(parents=True);rows=[];errors=[]
    for dataset in sorted(root.iterdir()):
        if not dataset.is_dir() or dataset.name.startswith('.'):continue
        for directory,dirs,files in os.walk(dataset,followlinks=False):
            dirs[:]=[d for d in dirs if d not in {'__pycache__','.git','.ipynb_checkpoints'}]
            for name in files:
                p=Path(directory)/name
                try:
                    st=p.lstat();suffix=''.join(p.suffixes[-2:]).lower() if p.name.lower().endswith(('.tar.gz','.tar.bz2')) else p.suffix.lower()
                    rel=p.relative_to(dataset);rows.append(dict(dataset=dataset.name,relative_path=str(rel),level1=rel.parts[0] if len(rel.parts)>1 else '.',extension=suffix or '(none)',bytes=st.st_size,symlink=p.is_symlink()))
                except OSError as e:errors.append(dict(path=str(p),error=str(e)))
    f=pd.DataFrame(rows);f.to_parquet(out/'file_inventory.parquet',index=False)
    f.groupby(['dataset','extension']).agg(files=('bytes','size'),bytes=('bytes','sum')).reset_index().to_csv(out/'file_types.csv',index=False)
    f.groupby(['dataset','level1']).agg(files=('bytes','size'),bytes=('bytes','sum')).reset_index().to_csv(out/'subdirectories.csv',index=False)
    (out/'scan_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2)+'\n')
    (out/'scan_scope.json').write_text(json.dumps(dict(root=str(root),follow_symlinks=False,excluded_cache_dirs=['__pycache__','.git','.ipynb_checkpoints'],files=len(f),errors=len(errors),no_training=True),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();inventory(a.raw.resolve(),a.output.resolve())
