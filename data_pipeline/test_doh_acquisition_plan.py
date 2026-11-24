"""Plan slots must not be counted as already certified source groups."""
import copy
import unittest
from prepare_doh_acquisition import preflight,LABELS


def fixture():
    old=[dict(component='old_attack',labels=LABELS[1:]),dict(component='old_normal1',labels=['Benign']),dict(component='old_normal2',labels=['Benign'])]
    campaigns=[dict(plan_slot_id=f'c{i}',intended_split=split,target_labels=LABELS.copy(),scene_family_id=None,scene_definition_sha256=None,
        original_campaign_lineage_id=None,deployment_identity=None,UTC_capture_days=None,assets_confirmed=False,scene_equivalence_evidence_complete=False)
        for i,split in enumerate(['train','val','val','test'])]
    return dict(resources=dict(controlled_lab_available=False),campaigns=campaigns),old


class AcquisitionPlanTests(unittest.TestCase):
    def test_joint_budget_is_conditional_not_capture_approval(self):
        p,old=fixture();r=preflight(p,old)
        self.assertTrue(r['conditional_capacity_if_all_slots_are_proven_distinct'])
        self.assertEqual(r['conditional_group_counts']['dns2tcp'],{'train':2,'val':2,'test':1})
        self.assertFalse(r['capture_ready']);self.assertFalse(r['training_ready'])
        self.assertIn('controlled_lab_unavailable',r['blockers'])

    def test_same_family_across_roles_is_rejected(self):
        p,old=fixture()
        for i in [0,1]:p['campaigns'][i]['scene_family_id']='same_real_scene'
        r=preflight(p,old);self.assertTrue(any('component_crosses_splits' in x for x in r['blockers']))

    def test_renamed_slots_with_identical_configuration_still_merge(self):
        p,old=fixture()
        for i in [0,1]:p['campaigns'][i].update(scene_family_id=f'renamed_{i}',scene_definition_sha256='same_config')
        r=preflight(p,old);self.assertTrue(any('same_scene_definition_sha256' in x['reasons'] for x in r['proposed_merge_witnesses']))

    def test_same_UTC_day_merges_cross_split_capture_slots(self):
        p,old=fixture()
        for i in [1,3]:p['campaigns'][i]['UTC_capture_days']=['2027-01-01']
        r=preflight(p,old);self.assertTrue(any('shared_UTC_day' in x['reasons'] for x in r['proposed_merge_witnesses']))

    def test_missing_test_class_fails_coverage(self):
        p,old=fixture();p['campaigns'][-1]['target_labels'].remove('iodine')
        r=preflight(p,old);self.assertFalse(r['conditional_capacity_if_all_slots_are_proven_distinct'])


if __name__=='__main__':unittest.main()
