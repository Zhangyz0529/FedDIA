"""Prespecified taxonomy/support filters; never use model performance to pick classes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate_selection(classes,allowed,thresholds):
    if not classes or len(set(classes))!=len(classes):raise ValueError('classes must be nonempty and unique')
    unknown=set(classes)-set(allowed)
    if unknown:raise ValueError(f'Unknown classes: {sorted(unknown)}')
    for key in ('min_total_samples','min_per_split','min_source_groups'):
        if type(thresholds[key]) is not int or thresholds[key]<1:raise ValueError(f'{key} must be a positive integer')


def filter_dataset(out,classes,thresholds,normal='Benign'):
    frame=pd.read_parquet(out/'provenance.parquet');available=set(frame.label)
    ledger=[]
    for label in classes:
        f=frame[frame.label==label];counts=Counter(f['split']) if 'split' in f else Counter()
        groups=f.source_group_id.nunique();reasons=[]
        if len(f)<thresholds['min_total_samples']:reasons.append('insufficient_total_samples')
        if groups<thresholds['min_source_groups']:reasons.append('insufficient_source_groups')
        for split in ('train','val','test'):
            if counts[split]<thresholds['min_per_split']:reasons.append('insufficient_'+split+'_support')
        ledger.append(dict(label=label,samples=len(f),source_groups=groups,split_support=dict(counts),
                           selected=not reasons,reasons=reasons))
    selected=[x['label'] for x in ledger if x['selected']]
    blockers=[]
    if len(selected)<2:blockers.append('fewer_than_two_eligible_classes')
    if normal is not None and normal not in selected:blockers.append('normal_class_not_eligible')
    subset=frame[frame.label.isin(selected)].copy()
    if not subset.empty and subset.groupby('source_group_id')['split'].nunique().max()>1:blockers.append('source_group_crosses_splits')
    if not subset.empty and (subset['split']=='unassigned').any():blockers.append('unassigned_source_groups')
    with np.load(out/'model_inputs.npz',allow_pickle=False) as a:
        assert list(frame.sample_id)==list(a['sample_id'])
        all_labels={}
        input_keys=[]
        for stat,seq,label in zip(a['stat_raw'],a['seq_feature'],frame.label):
            key=hashlib.sha256(stat.tobytes()+seq.tobytes()).hexdigest();input_keys.append(key)
            all_labels.setdefault(key,set()).add(label)
        conflict_keys={k for k,v in all_labels.items() if len(v)>1}
        if any(k in conflict_keys for k,keep in zip(input_keys,frame.label.isin(selected)) if keep):
            blockers.append('conflicting_label_inputs_require_review')
        mask=frame.label.isin(selected).to_numpy();x=a['stat_raw'][mask];s=a['seq_feature'][mask]
        dup={}
        for i,(stat,seq) in enumerate(zip(x,s)):
            key=hashlib.sha256(stat.tobytes()+seq.tobytes()).hexdigest()
            dup.setdefault(key,set()).add(subset.iloc[i]['split'])
        cross=sum(len(v)>1 for v in dup.values())
        if cross:blockers.append('cross_split_identical_inputs')
        target=out/'selected';target.mkdir(exist_ok=True)
        if not blockers:
            label_map={name:i for i,name in enumerate(selected)}
            for split in ('train','val','test'):
                keep=(subset['split']==split).to_numpy()
                np.savez_compressed(target/f'{split}.npz',stat_raw=x[keep],seq_feature=s[keep],
                                    sample_id=subset.loc[subset['split']==split,'sample_id'].to_numpy(dtype=str),
                                    label=subset.loc[subset['split']==split,'label'].to_numpy(dtype=str),
                                    y=np.array([label_map[v] for v in subset.loc[subset['split']==split,'label']],dtype=np.int64))
            subset.to_parquet(target/'provenance.parquet',index=False)
        report=dict(requested_classes=classes,thresholds=thresholds,class_ledger=ledger,selected_classes=selected,
                    available_classes=sorted(available),conflicting_label_input_groups=len(conflict_keys),blockers=blockers,cross_split_identical_inputs=cross,
                    export_status='EXPORTED_SOURCE_GROUP_SPLITS' if not blockers else 'BLOCKED_NO_CLASSIFICATION_EXPORT',
                    label_map={name:i for i,name in enumerate(selected)},training_started=False,training_allowed=False,
                    note='Class filtering uses prespecified support/source rules only; excluded candidates retained; no model scores used.')
        (out/'class_selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report
