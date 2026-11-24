#!/usr/bin/env python3
"""ZIP-directory evidence for CipherSpectrum mirrors and recording identities."""
import argparse
from collections import defaultdict
import hashlib,json
from pathlib import Path
import zipfile
import pandas as pd


def main(raw,out):
    rows=[]
    for path in sorted(raw.glob('*.zip')):
        with zipfile.ZipFile(path) as z:
            for m in z.infolist():
                if m.is_dir() or not m.filename.lower().endswith('.pcap'):continue
                if '__MACOSX' in Path(m.filename).parts or Path(m.filename).name.startswith('._'):continue
                rows.append(dict(archive=path.name,member=m.filename,basename=Path(m.filename).name,bytes=m.file_size,crc32=f'{m.CRC:08x}'))
    f=pd.DataFrame(rows);groups=f.groupby(['basename','bytes','crc32']);cross=[];checks=[]
    for key,g in groups:
        if g.archive.nunique()<2:continue
        cross.append(dict(basename=key[0],bytes=key[1],crc32=key[2],archives=sorted(g.archive.unique()),occurrences=len(g)))
        if len(checks)<20:
            digests=[]
            for r in g.to_dict('records'):
                with zipfile.ZipFile(raw/r['archive']) as z:digests.append(hashlib.sha256(z.read(r['member'])).hexdigest())
            checks.append(dict(basename=key[0],archives=sorted(g.archive.unique()),all_member_SHA256_equal=len(set(digests))==1,sha256=digests[0]))
    f.to_parquet(out/'cipher_zip_members.parquet',index=False)
    (out/'cipher_cross_archive_candidates.json').write_text(json.dumps(cross,ensure_ascii=False,indent=2)+'\n')
    (out/'cipher_duplicate_sample_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    summary=dict(nonmetadata_pcaps=len(f),cross_archive_same_basename_CRC_size_groups=len(cross),sampled_SHA_checks=len(checks),all_sampled_SHA_equal=all(x['all_member_SHA256_equal'] for x in checks) if checks else None,
        warning='CRC-size candidates are not a whole-corpus SHA audit; mix and cipher folders are not automatically independent datasets')
    (out/'cipher_zip_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.raw.resolve(),a.output.resolve())
