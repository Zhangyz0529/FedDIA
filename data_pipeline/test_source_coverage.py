"""Preserve conservative transitive day grouping in the coverage inventory."""
import unittest
from audit_doh_coverage import provisional_components


class SourceGraphTests(unittest.TestCase):
    def test_transitive_day_bridge_merges_families(self):
        groups,edges=provisional_components({'a':{'day1'},'bridge':{'day1','day2'},'b':{'day2'}})
        self.assertEqual(len(set(groups.values())),1)
        self.assertEqual(len(edges),2)

    def test_distinct_days_do_not_create_links(self):
        groups,edges=provisional_components({'a':{'day1'},'b':{'day2'}})
        self.assertEqual(len(set(groups.values())),2)
        self.assertFalse(edges)

    def test_unknown_dates_do_not_certify_a_split(self):
        groups,edges=provisional_components({})
        self.assertEqual(groups,{})
        self.assertEqual(edges,[])


if __name__=='__main__':unittest.main()
