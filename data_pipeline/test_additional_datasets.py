"""Protect strict CSV matching and application/VPN semantics."""
import math
import mmap
from pathlib import Path
import tempfile
import unittest

import dpkt
import additional_datasets as a
from dataset_filters import filter_dataset
import test_dataset_filters as fixtures
from test_doh_pilot import packet


class AdditionalTests(unittest.TestCase):
    def test_observed_endpoints_parquet_roundtrip(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observed.parquet'
            a.observed_packet_frame([dict(src=('10.0.0.1',123),dst=('10.0.0.2',80),network=b'abc',index=1)]).to_parquet(path,index=False)
            row=pd.read_parquet(path).iloc[0]
            self.assertEqual(row.src,dict(ip='10.0.0.1',port=123))
            self.assertEqual(row.network,b'abc')

    def fixture(self):
        targets=[];rows={};packets={}
        for i in range(6):
            src=('10.0.0.'+str(i+1),1000+i);dst=('10.0.1.1',80);ts=1700000000.+i*2
            row={'Source IP':src[0],'Source Port':src[1],'Destination IP':dst[0],'Destination Port':dst[1],'Protocol':6,
                 'Total Fwd Packets':2,'Total Backward Packets':1,'Total Length of Fwd Packets':10,'Total Length of Bwd Packets':20,
                 'Flow Duration':200000,'clock':math.floor(ts/60)*60-10800,'csv_path':'fixture.csv','csv_row':i+2,'label':'BENIGN'}
            key=a.csv_key(row);values=[dict(index=3*i+j,ts=ts+.1*j,src=src if j<2 else dst,payload_len=[10,0,20][j]) for j in range(3)]
            targets.append(row);rows[key]=[row];packets[key]=values
        config=dict(duration_tolerance_us=2,min_clock_anchors=5,min_clock_consensus=.95)
        return targets,rows,packets,config

    def test_minute_clock_offset_needs_multiple_unambiguous_anchors(self):
        targets,rows,packets,cfg=self.fixture();accepted,rejected,clock=a.calibrated_matches(targets,rows,packets,cfg)
        self.assertEqual(clock['offset_seconds'],10800);self.assertEqual(len(accepted),6);self.assertEqual(rejected,[])
        cfg['min_clock_anchors']=10
        accepted,rejected,clock=a.calibrated_matches(targets,rows,packets,cfg)
        self.assertEqual(accepted,[]);self.assertEqual(clock['status'],'BLOCKED_CLOCK')

    def test_12h_missing_ampm_is_calibrated_without_using_labels(self):
        targets,rows,packets,cfg=self.fixture()
        # Three independent anchors are afternoon times written without AM/PM.
        for row in targets[3:]:row['clock']-=43200
        cfg['allow_12h_clock_ambiguity']=True
        accepted,rejected,clock=a.calibrated_matches(targets,rows,packets,cfg)
        self.assertEqual(len(accepted),6);self.assertEqual(clock['offset_options_seconds'],[10800,54000])
        self.assertEqual(rejected,[])

    def test_conflicting_csv_labels_are_not_resolved_by_attack_priority(self):
        targets,rows,packets,cfg=self.fixture();key=a.csv_key(targets[0]);other=dict(targets[0],label='FTP-Patator',csv_row=99)
        rows[key].append(other)
        accepted,rejected,clock=a.calibrated_matches(targets,rows,packets,cfg)
        self.assertEqual(len(accepted),5);self.assertEqual(rejected[0]['reason'],'ambiguous_csv_record_or_label')

    def test_direction_counts_and_payload_bytes_must_both_match(self):
        targets,rows,packets,cfg=self.fixture();row=targets[0];key=a.csv_key(row)
        self.assertEqual(a.window_candidates(row,packets[key],2),[(0,3)])
        row=dict(row);row['Total Length of Fwd Packets']=11
        self.assertEqual(a.window_candidates(row,packets[key],2),[])

    def test_classic_and_pcapng_offsets_reproduce_original_frames(self):
        for cls in (dpkt.pcap.Writer,dpkt.pcapng.Writer):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'input.pcap';raw=packet()
                with path.open('wb') as f:
                    w=cls(f);w.writepkt(raw,ts=1.25);w.writepkt(raw,ts=2.5)
                with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
                    records=list(a.indexed_packets(mm));self.assertEqual(len(records),2)
                    for n,ts,offset,cap,wire,link in records:
                        self.assertEqual(bytes(mm[offset:offset+cap]),raw);self.assertEqual(ts,[1.25,2.5][n]);self.assertEqual(link,1)
                        self.assertIsNotNone(a.fast_endpoint(mm,offset,cap,link))

    def test_vpn_is_activity_not_attack_and_ab_parts_stay_together(self):
        self.assertEqual(a.iscx_category('vpn_facebook_audio2'),'VPN_VoIP')
        self.assertEqual(a.iscx_category('facebook_audio2a'),'VoIP')
        self.assertEqual(a.iscx_family('facebook_audio2a'),a.iscx_family('facebook_audio2b'))
        self.assertIsNone(a.iscx_category('unknown'))

    def test_restoration_uses_verified_zip_without_overwriting_existing_capture(self):
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pcaps=root/'raw/pcap';pcaps.mkdir(parents=True)
            current=pcaps/'email1a.pcap';current.write_bytes(b'changed extraction copy')
            with zipfile.ZipFile(pcaps/'NonVPN-PCAPs-01.zip','w') as z:z.writestr('email1a.pcap',b'release bytes')
            cfg=dict(classes=['Email'],max_capture_bytes=1000,restore_from_archive_on_mismatch=True)
            plan=a.prepare('ISCX',root/'raw',root/'output',cfg,dict(min_total_samples=1,min_per_split=1,min_source_groups=3))
            self.assertEqual(current.read_bytes(),b'changed extraction copy')
            self.assertEqual(Path(plan['jobs'][0]['capture']).read_bytes(),b'release bytes')
            self.assertIsNotNone(plan['jobs'][0]['restored_source'])

    def test_pcap_extension_alias_is_verified_by_bytes_not_rejected(self):
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pcaps=root/'raw/pcap';pcaps.mkdir(parents=True)
            (pcaps/'facebookchat1.pcap').write_bytes(b'release bytes')
            with zipfile.ZipFile(pcaps/'NonVPN-PCAPs-01.zip','w') as z:z.writestr('facebookchat1.pcapng',b'release bytes')
            cfg=dict(classes=['Chat'],max_capture_bytes=1000,restore_from_archive_on_mismatch=True)
            plan=a.prepare('ISCX',root/'raw',root/'output',cfg,{})
            self.assertEqual(len(plan['jobs']),1);self.assertIsNone(plan['jobs'][0]['restored_source'])
            self.assertEqual(plan['jobs'][0]['archive_members'][0]['member'],'facebookchat1.pcapng')

    def test_application_classification_does_not_require_fake_benign_class(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture=fixtures.ClassSelectionTests();fixture.fixture(root,dict(Chat=[1,1,1],VoIP=[1,1,1]))
            result=filter_dataset(root,['Chat','VoIP'],fixture.thresholds(),normal=None)
            self.assertEqual(result['blockers'],[])


if __name__=='__main__':unittest.main()
