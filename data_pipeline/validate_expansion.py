"""Independent structural acceptance checks for an expanded acquisition-group set."""
import argparse
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd

import doh_pilot as pilot


def validate(out):
    plan=json.loads((out/'plan.json').read_text());meta=json.loads((out/'file_manifest.json').read_text())
    provenance=pd.read_parquet(out/'provenance.parquet');checks=[]
    assert provenance.sample_id.is_unique
    assert provenance.groupby('capture_hash')['split'].nunique().max()==1
    assert provenance.groupby('source_group_id')['split'].nunique().max()==1
    assert provenance.groupby('acquisition_family')['split'].nunique().max()==1
    for i,m in enumerate(meta):
        cache=out/'captures'/f'{i:03d}'
        accepted=pd.read_parquet(cache/'provenance.parquet');rejected=pd.read_parquet(cache/'quarantine.parquet')
        counts=Counter(int(index) for frame in (accepted,rejected) if not frame.empty for indices in frame.packet_indices for index in indices)
        assert len(counts)==m['total_packets'] and set(counts)==set(range(m['total_packets'])) and all(x==1 for x in counts.values())
        original=Path(plan['jobs'][i]['capture'])
        assert pilot.sha(original)==m['sha256']
        assert any(e['crc32']==m['crc32'] and e['bytes']==m['file_size'] for e in m['archive_members'])
        checks.append(dict(capture_index=i,original_sha256_matches=True,archive_crc_size_matches=True,all_packet_indices_exclusive_and_complete=True))
    with np.load(out/'model_inputs.npz',allow_pickle=False) as a:
        assert list(a['sample_id'])==list(provenance.sample_id)
        assert list(a['label'])==list(provenance.label)
        assert a['stat_raw'].shape==(len(provenance),25) and a['seq_feature'].shape==(len(provenance),512)
        assert np.isfinite(a['stat_raw']).all()
        for i,r in enumerate(provenance.itertuples()):
            ep=r.endpoints
            label,evidence=pilot.label_for((ep['first_ip'],ep['first_port']),(ep['peer_ip'],ep['peer_port']),r.protocol,plan['jobs'][r.capture_index]['group'])
            assert label==r.label and evidence==r.label_source and r.tls_record_header_observed
            assert a['stat_raw'][i,0]==np.float32(r.end_timestamp-r.start_timestamp)
            assert a['stat_raw'][i,1]+a['stat_raw'][i,2]==r.packet_count
            assert r.sequence_valid_length<=512 and list(r.packet_directions).count(0)==a['stat_raw'][i,1]
    split=json.loads((out/'split_manifest.json').read_text())
    for s in ('train','val','test'):
        assert all(split['support'][s].get(label,0)>0 for label in pilot.LABELS),f'class coverage: {s}'
    assert not (out/'dataset.pkl').exists()
    stats=provenance.groupby(['label','source_group_id','acquisition_family','split']).agg(
        samples=('sample_id','count'),capture_files=('capture_index','nunique'),first_timestamp=('start_timestamp','min'),
        last_timestamp=('end_timestamp','max'),missing_syn_fraction=('missing_initial_syn','mean')).reset_index()
    stats.to_csv(out/'source_support.csv',index=False)
    report=dict(status='STRUCTURAL_SOURCE_CHECKS_PASSED',samples=len(provenance),captures=len(meta),source_groups=provenance.source_group_id.nunique(),
                packet_checks=checks,source_family_split_disjoint=True,source_group_split_disjoint=True,capture_split_disjoint=True,
                label_rules_rechecked=True,feature_metadata_checks_passed=True,class_coverage_all_splits=True,
                unit_tests='executed separately; see batch validation report',training_started=False,
                limitation='Checks selected local acquisition/scenario grouping, not host/site statistical independence or old CSV29 compatibility')
    pilot.save(out/'structural_acceptance.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='packet_checks'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    validate(parser.parse_args().output.resolve())
