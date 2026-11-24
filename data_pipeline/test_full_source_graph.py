"""Verify that source constraints are applied globally, including transitive links."""
import unittest
import pandas as pd
from audit_doh_full_sources import build_graph


def capture(family,label,start,end,digest):
    return dict(family=family,label=label,start=start,end=end,member_sha256=digest,archive='release.zip',member=family+'.pcap',timestamp_status='FULL_SCAN_OK')


class GlobalGraphTests(unittest.TestCase):
    def test_same_day_merges_across_classes(self):
        rows=[capture('a','Benign',1,2,'x'),capture('b','dnscat2',3,4,'y')]
        ids,edges,*_=build_graph(rows,[],pd.DataFrame(),[])
        self.assertEqual(ids['a'],ids['b'])
        self.assertEqual(edges[0]['reason'],'shared_UTC_window_day')

    def test_shared_family_days_link_transitively_and_propagate_diagnosis(self):
        rows=[capture('old','dnscat2',1,2,'x'),capture('bridge','iodine',3,86401,'y'),capture('fresh','iodine',86402,86403,'z')]
        old=[dict(acquisition_family='old',sha256='x')]
        ids,edges,touched,*_=build_graph(rows,old,pd.DataFrame(),[])
        self.assertEqual(len(set(ids.values())),1)
        self.assertIn(ids['fresh'],touched)

    def test_duplicate_hash_links_different_days(self):
        rows=[capture('a','dns2tcp',1,2,'same'),capture('b','dnscat2',86401,86402,'same')]
        ids,edges,*_=build_graph(rows,[],pd.DataFrame(),[])
        self.assertEqual(ids['a'],ids['b'])
        self.assertEqual(edges[0]['reason'],'duplicate_capture_SHA256')

    def test_known_packet_overlap_links_nonmatching_capture_hashes(self):
        rows=[capture('a','dns2tcp',1,2,'x'),capture('b','dnscat2',86401,86402,'y')]
        old=[dict(acquisition_family='a',sha256='x'),dict(acquisition_family='b',sha256='y')]
        ids,edges,*_=build_graph(rows,old,pd.DataFrame(),[dict(left=0,right=1,exact_overlap_packets=2)])
        self.assertEqual(ids['a'],ids['b'])
        self.assertEqual(edges[0]['reason'],'known_exact_timestamp_raw_packet_overlap')

    def test_unmeasured_timestamps_do_not_invent_day_link(self):
        rows=[capture('a','dns2tcp',None,None,'x'),capture('b','dnscat2',1,2,'y')]
        ids,edges,*_=build_graph(rows,[],pd.DataFrame(),[])
        self.assertNotEqual(ids['a'],ids['b'])
        self.assertFalse(edges)


if __name__=='__main__':unittest.main()
