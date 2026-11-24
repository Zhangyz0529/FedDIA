"""Bind new source-audited inputs to their semantics, not just their width."""
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA='ip_packet_stats25_v1'
FIELDS=('duration_s fwd_packets bwd_packets fwd_ip_bytes bwd_ip_bytes '
        'fwd_len_mean fwd_len_std fwd_len_max fwd_len_min bwd_len_mean bwd_len_std bwd_len_max bwd_len_min '
        'all_len_mean all_len_std all_len_max all_len_min iat_mean_s iat_std_s iat_max_s iat_min_s '
        'packets_per_s ip_bytes_per_s fwd_packets_per_s bwd_packets_per_s').split()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def validate_contract(path,cfg,dataset,data=None):
    path=Path(path);manifest=path.parent/'dataset_manifest.json'
    expected=cfg.get('feature_schema')
    if not expected:
        # Unrelated historical sidecars do not opt a legacy dataset into Source25.
        if not manifest.exists():return None
        if json.loads(manifest.read_text()).get('feature_schema')!=SCHEMA:return None
    if expected!=SCHEMA or cfg['stat_dim']!=25 or not manifest.exists():
        raise ValueError('Source25 requires its own dataset registry entry and manifest; USTC/legacy shape matching is insufficient')
    m=json.loads(manifest.read_text());schema=path.parent/'feature_schema.json'
    if m['dataset_registry_key']!=dataset or m['feature_schema']!=expected or m['dataset_sha256']!=sha(path):
        raise ValueError('Dataset identity/schema/content contract mismatch')
    s=json.loads(schema.read_text())
    if m['schema_sha256']!=sha(schema) or s['version']!=SCHEMA or s['fields']!=FIELDS:
        raise ValueError('Ordered feature definitions do not match Source25')
    if data is not None:
        seen=set()
        for client,tasks in data.items():
            for task,splits in tasks.items():
                for split,samples in splits.items():
                    for sample in samples:
                        sid=sample['sample_id']
                        if sid in seen:raise ValueError('Repeated sample_id in federated data')
                        seen.add(sid)
                        stat=np.asarray(sample['stat_raw']);seq=np.asarray(sample['seq_feature'])
                        if stat.shape!=(25,) or seq.shape!=(512,) or not np.isfinite(stat).all():
                            raise ValueError('Invalid Source25 sample')
        if len(seen)!=m['samples']:raise ValueError('Manifest support mismatch')
    return m
