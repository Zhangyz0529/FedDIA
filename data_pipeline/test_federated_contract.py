"""Regression checks for split identity and semantic (not width-only) binding."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from export_doh_federated import export
from federated_contract import FIELDS,SCHEMA,sha,validate_contract


class FederatedTests(unittest.TestCase):
    def parent(self,root):
        source=root/'parent';(source/'selected').mkdir(parents=True)
        (source/'feature_schema.json').write_text(json.dumps(dict(version=SCHEMA,fields=FIELDS)))
        (source/'class_selection.json').write_text(json.dumps(dict(export_status='EXPORTED_SOURCE_GROUP_SPLITS')))
        rows=[]
        for split in ['train','val','test']:
            ids=[];labels=[]
            for label in ['Benign','dns2tcp','dnscat2','iodine']:
                for i in range(18 if label=='Benign' else 6):
                    sid=f'{split}:{label}:{i}';ids.append(sid);labels.append(label)
                    rows.append(dict(sample_id=sid,label=label,split=split,source_group_id=f'{split}:{label}'))
            np.savez(source/'selected'/f'{split}.npz',sample_id=np.array(ids),label=np.array(labels),
                     stat_raw=np.arange(len(ids)*25,dtype=np.float32).reshape(-1,25),seq_feature=np.zeros((len(ids),512),np.uint8))
        pd.DataFrame(rows).to_parquet(source/'selected/provenance.parquet',index=False)
        return source

    def test_export_is_deterministic_and_preserves_each_parent_sample(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=self.parent(root);before={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
            export(source,root/'a');export(source,root/'b')
            self.assertEqual(sha(root/'a/dataset.pkl'),sha(root/'b/dataset.pkl'))
            assigned=pd.read_parquet(root/'a/assignment.parquet')
            self.assertTrue(assigned.sample_id.is_unique)
            self.assertEqual(len(assigned),108)
            self.assertTrue((assigned.groupby('source_group_id').split.nunique()==1).all())
            self.assertTrue((assigned[assigned.label!='Benign'].groupby('label').task.nunique()==1).all())
            self.assertEqual(before,{str(p):sha(p) for p in source.rglob('*') if p.is_file()})

    def test_reject_same_dimension_wrong_identity_and_tampered_content(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=self.parent(root);export(source,root/'a');p=root/'a/dataset.pkl'
            with self.assertRaises(ValueError):validate_contract(p,dict(stat_dim=25),'USTC')
            p.write_bytes(p.read_bytes()+b'tampered')
            with self.assertRaises(ValueError):validate_contract(p,dict(stat_dim=25,feature_schema=SCHEMA),'CIC_DOH_SOURCE25')

    def test_insufficient_support_fails_instead_of_repeating_samples(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=self.parent(root)
            with self.assertRaises(ValueError):export(source,root/'a',clients=7)


if __name__=='__main__':unittest.main()
