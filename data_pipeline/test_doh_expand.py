"""Fail-closed provenance checks for source expansion, independent of real data."""
from pathlib import Path
import tempfile
import unittest
import zlib

import numpy as np

import doh_expand as e


class SourceExpansionTests(unittest.TestCase):
    def test_same_family_is_kept_together_across_days(self):
        meta=[dict(capture_start_time=1,capture_end_time=2,sha256='a',acquisition_family='same'),
              dict(capture_start_time=90000,capture_end_time=90001,sha256='b',acquisition_family='same'),
              dict(capture_start_time=180000,capture_end_time=180001,sha256='c',acquisition_family='other')]
        ids,edges=e.source_groups(meta,Path('/unused'))
        self.assertEqual(ids[0],ids[1]);self.assertNotEqual(ids[0],ids[2])
        self.assertIn('same_acquisition_family',edges[0]['reasons'])

    def test_actual_raw_overlap_is_recorded(self):
        meta=[dict(capture_start_time=1,capture_end_time=5,sha256='a',acquisition_family='left'),
              dict(capture_start_time=3,capture_end_time=8,sha256='b',acquisition_family='right')]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for i,content in enumerate([b'a'*32+b'b'*32,b'b'*32+b'c'*32]):
                target=root/'captures'/f'{i:03d}';target.mkdir(parents=True)
                np.save(target/'packet_fingerprints.npy',np.frombuffer(content,dtype='V32'))
            ids,edges=e.source_groups(meta,root)
            self.assertEqual(ids[0],ids[1]);self.assertEqual(edges[0]['exact_overlap_packets'],1)
            self.assertIn('exact_timestamp_raw_overlap',edges[0]['reasons'])

    def test_archive_mismatch_never_certifies_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'original';source.write_bytes(b'original data')
            before=e.pilot.sha(source)
            with self.assertRaisesRegex(ValueError,'differs from release archive'):
                e.verified_copy(source,root/'working',[dict(bytes=13,crc32='00000000')])
            self.assertEqual(before,e.pilot.sha(source))

    def test_working_copy_matches_archive_and_original_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'original';source.write_bytes(b'original data')
            expected=dict(bytes=source.stat().st_size,crc32=f'{zlib.crc32(source.read_bytes())&0xffffffff:08x}')
            h,crc,entries=e.verified_copy(source,root/'working',[expected])
            self.assertEqual(h,e.pilot.sha(source));self.assertEqual(crc,expected['crc32']);self.assertEqual(entries,[expected])


if __name__=='__main__':unittest.main()
