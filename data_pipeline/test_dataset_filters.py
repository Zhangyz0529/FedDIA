"""Taxonomy choices must not bypass sample/source coverage gates."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from dataset_filters import validate_selection,filter_dataset
import doh_pilot as pilot


class ClassSelectionTests(unittest.TestCase):
    def thresholds(self):return dict(min_total_samples=3,min_per_split=1,min_source_groups=3)

    def fixture(self,root,counts):
        rows=[];vectors=[]
        for label,split_counts in counts.items():
            for split,n in zip(['train','val','test'],split_counts):
                for i in range(n):
                    k=len(rows);rows.append(dict(sample_id=str(k),label=label,split=split,source_group_id=label+'_'+split))
                    vectors.append(np.full(25,k,np.float32))
        pd.DataFrame(rows).to_parquet(root/'provenance.parquet',index=False)
        np.savez(root/'model_inputs.npz',stat_raw=np.stack(vectors),seq_feature=np.zeros((len(rows),512),np.uint8),
                 sample_id=np.array([r['sample_id'] for r in rows]),label=np.array([r['label'] for r in rows]))

    def test_unknown_duplicate_or_empty_whitelist_rejected(self):
        for classes in [[],['Benign','Benign'],['unknown']]:
            with self.assertRaises(ValueError):validate_selection(classes,['Benign','attack'],self.thresholds())

    def test_small_class_excluded_without_dropping_audit_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,dict(Benign=[1,1,1],attack=[1,1,1],rare=[1,0,0]))
            report=filter_dataset(root,['Benign','attack','rare'],self.thresholds())
            self.assertEqual(report['selected_classes'],['Benign','attack']);self.assertEqual(report['blockers'],[])
            self.assertIn('insufficient_total_samples',report['class_ledger'][2]['reasons'])
            self.assertEqual(len(pd.read_parquet(root/'provenance.parquet')),7)
            with np.load(root/'selected/test.npz') as a:self.assertEqual(set(a['label']),{'Benign','attack'})

    def test_one_source_class_never_gets_random_split(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,dict(Benign=[1,1,1],attack=[10,0,0]))
            report=filter_dataset(root,['Benign','attack'],self.thresholds())
            self.assertIn('fewer_than_two_eligible_classes',report['blockers'])
            self.assertFalse((root/'selected/train.npz').exists())

    def test_excluding_a_class_cannot_hide_conflicting_input_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,dict(Benign=[1,1,1],attack=[1,1,1],rare=[1,0,0]))
            with np.load(root/'model_inputs.npz',allow_pickle=False) as a:values={k:a[k] for k in a.files}
            values['stat_raw'][-1]=values['stat_raw'][0]
            np.savez(root/'model_inputs.npz',**values)
            report=filter_dataset(root,['Benign','attack','rare'],self.thresholds())
            self.assertIn('conflicting_label_inputs_require_review',report['blockers'])

    def test_class_view_exports_without_modifying_parent_inputs(self):
        import generate_suite as suite
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'parent';parent.mkdir();self.fixture(parent,dict(Benign=[1,1,1],attack=[1,1,1],rare=[1,0,0]))
            (parent/'plan.json').write_text(json.dumps(dict(dataset='fixture',requested_classes=['Benign','attack','rare'],thresholds=self.thresholds())))
            h=pilot.sha(parent/'model_inputs.npz')
            report=suite.select_view(parent,root/'view',['Benign','attack'],self.thresholds())
            self.assertEqual(report['export_status'],'EXPORTED_SOURCE_GROUP_SPLITS')
            self.assertEqual(pilot.sha(parent/'model_inputs.npz'),h)
            self.assertTrue((root/'view/model_inputs.npz').is_symlink())
            self.assertTrue((root/'view/lineage.json').exists())

    def test_single_capture_resolver_does_not_require_tls_or_claim_attack_truth(self):
        import generate_suite as suite
        label,evidence=suite.cohort_label(('10.0.0.1',1),('10.0.0.2',2),17,'Miuref')
        self.assertEqual(label,'Miuref');self.assertIn('not_per_flow_attack_truth',evidence)
        import dpkt
        from test_doh_pilot import packet
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'plain.pcap'
            with path.open('wb') as f:
                w=dpkt.pcap.Writer(f)
                for t in (1.,2.,3.):
                    eth=dpkt.ethernet.Ethernet(packet());eth.data.data.data=b'plain data';eth.data.len=20+len(eth.data.data)
                    w.writepkt(bytes(eth),ts=t)
            records,arrays,_,_=pilot.inspect_capture(path,'Miuref',pilot.sha(path),label_resolver=suite.cohort_label,require_tls=False)
            self.assertEqual(len(records),1)
            self.assertFalse(records[0]['tls_record_header_observed'])


if __name__=='__main__':unittest.main()
