"""Source assignment must be joint; per-class counts alone are insufficient."""
import unittest
from audit_ciciot_sources import joint_split,extract_windows
from test_doh_pilot import packet
import numpy as np


class JointSplitTests(unittest.TestCase):
    def test_singleton_segment_is_explicit_and_finite(self):
        frames={0:[dict(index=10,offset=100,caplen=len(packet()),ts=1.,raw=packet())]}
        candidates,excluded,support=extract_windows(frames,'fixture','DDoS-SYN_Flood',1)
        self.assertEqual(len(candidates),1)
        self.assertEqual(support['one_packet'],1)
        self.assertEqual(candidates[0][1]['packet_indices'],[10])
        self.assertTrue(np.isfinite(candidates[0][2]).all())
        self.assertTrue(candidates[0][1]['window_boundary_censoring_possible'])

    def test_sampling_windows_do_not_form_fabricated_complete_session(self):
        frames={i:[dict(index=i*100,offset=i*1000,caplen=len(packet()),ts=float(i+1),raw=packet())] for i in range(2)}
        candidates,_,_=extract_windows(frames,'fixture','BenignTraffic',1)
        self.assertEqual(len(candidates),2)
        self.assertNotEqual(candidates[0][0],candidates[1][0])
        self.assertTrue(all(r['packet_count']==1 for _,r,_,_ in candidates))

    def test_flow_timestamp_regression_is_quarantined(self):
        frames={0:[dict(index=i,offset=i*100,caplen=len(packet()),ts=t,raw=packet()) for i,t in enumerate([2.,1.])]}
        candidates,excluded,_=extract_windows(frames,'fixture','BenignTraffic',1)
        self.assertFalse(candidates)
        self.assertEqual(excluded['segment_timestamp_regression'],1)

    def test_one_group_per_class_cannot_form_three_splits(self):
        result=joint_split({'g1':{'normal'},'g2':{'attack'}},['normal','attack'])
        self.assertFalse(result['feasible'])
        self.assertEqual(result['status'],'PROVEN_INFEASIBLE_COMPONENT_COUNTS')

    def test_mixed_label_components_are_assigned_whole(self):
        components={f'g{i}':{'normal','attack'} for i in range(5)}
        result=joint_split(components,['normal','attack'])
        self.assertTrue(result['feasible'])
        for split,n in [('train',2),('val',2),('test',1)]:
            self.assertGreaterEqual(sum(s==split for s in result['assignment'].values()),n)

    def test_per_class_counts_pass_but_joint_coloring_is_impossible(self):
        # Each class occurs in three groups. All three triples would have to
        # contain all split colors, requiring four distinct colors on a K4.
        components={'g1':{'A','B','C'},'g2':{'A','B'},'g3':{'A','C'},'g4':{'B','C'}}
        result=joint_split(components,['A','B','C'],{'train':1,'val':1,'test':1})
        self.assertFalse(result['feasible'])
        self.assertEqual(result['status'],'PROVEN_INFEASIBLE_JOINT_ASSIGNMENT')

    def test_search_limit_is_unknown_instead_of_infeasible(self):
        result=joint_split({f'g{i}':{'normal','attack'} for i in range(5)},['normal','attack'],max_states=0)
        self.assertIsNone(result['feasible'])
        self.assertEqual(result['status'],'SEARCH_LIMIT')


if __name__=='__main__':unittest.main()
