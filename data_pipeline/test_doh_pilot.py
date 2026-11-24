"""Protocol tests: packet identity, conservative labels and split leakage gates."""
import hashlib
from pathlib import Path
import socket
import tempfile
import unittest

import dpkt
import numpy as np

import doh_pilot as p


def packet(src='192.168.20.212',dst='1.1.1.1',sport=1234,dport=443,flags=dpkt.tcp.TH_ACK,seq=1):
    tcp=dpkt.tcp.TCP(sport=sport,dport=dport,flags=flags,seq=seq,data=b'\x17\x03\x03\x00\x03abc')
    ip=dpkt.ip.IP(src=socket.inet_aton(src),dst=socket.inet_aton(dst),p=6,id=321,data=tcp)
    ip.len=20+len(tcp)
    return bytes(dpkt.ethernet.Ethernet(src=b'123456',dst=b'abcdef',type=2048,data=ip))


class PilotTests(unittest.TestCase):
    def capture(self,rows):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        path=Path(tmp.name)/'test.pcap'
        with path.open('wb') as f:
            writer=dpkt.pcap.Writer(f)
            for t,raw in rows:writer.writepkt(raw,ts=t)
        return path

    def test_link_header_independence_and_anonymization(self):
        raw=packet();a=p.decode(raw,1);b=p.decode(raw[14:],101)
        self.assertEqual(a['network'],b['network'])
        ip=dpkt.ip.IP(a['network']);self.assertEqual(ip.src,bytes(4));self.assertEqual(ip.dst,bytes(4))
        self.assertEqual(ip.id,0);self.assertEqual(ip.data.sport,0);self.assertEqual(ip.data.dport,0)
        self.assertEqual(ip.data.seq,0);self.assertTrue(a['tls_seen'])

    def test_unknown_not_default_benign(self):
        self.assertIsNone(p.label_for(('10.0.0.1',1234),('1.1.1.1',443),6,'browser')[0])
        self.assertIsNone(p.label_for(('192.168.20.111',1234),('2.2.2.2',443),6,'browser')[0])
        self.assertEqual(p.label_for(('192.168.20.111',1234),('1.1.1.1',443),6,'browser')[0],'Benign')

    def test_same_packet_stats_and_source_unchanged(self):
        path=self.capture([(1.,packet()),(2.,packet()),(3.,packet())]);before=p.sha(path)
        rows,arrays,meta,_=p.inspect_capture(path,'dns2tcp',before)
        self.assertEqual(p.sha(path),before);self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['packet_indices'],[0,1,2]);self.assertEqual(rows[0]['packet_directions'],[0,0,0])
        self.assertEqual(arrays[0][0][0],2.);self.assertEqual(arrays[0][0][1],3.)
        self.assertEqual(len(arrays[0][0]),25);self.assertEqual(len(arrays[0][1]),512)
        self.assertTrue(rows[0]['missing_initial_syn'])

    def test_provenance_roundtrips_parquet_without_losing_packet_indices(self):
        path=self.capture([(1.,packet()),(2.,packet()),(3.,packet())])
        rows,_,_,_=p.inspect_capture(path,'dns2tcp',p.sha(path))
        output=path.with_suffix('.parquet')
        p.pd.DataFrame(rows).to_parquet(output,index=False)
        recovered=p.pd.read_parquet(output).iloc[0]
        self.assertEqual(list(recovered.packet_indices),[0,1,2])
        self.assertEqual(recovered.endpoints['first_port'],1234)

    def test_quarantine_preserves_rejected_packet_membership(self):
        path=self.capture([(2.,packet()),(1.,packet()),(3.,packet())])
        rejected=[]
        rows,_,_,_=p.inspect_capture(path,'dns2tcp',p.sha(path),rejected)
        self.assertEqual(rows,[])
        self.assertEqual(rejected[0]['packet_indices'],[0,1,2])
        self.assertEqual(rejected[0]['reason'],'timestamp_regression')

    def test_new_syn_segments_reused_endpoint(self):
        rows=[]
        for start,isn in [(1.,10),(10.,20)]:
            rows.extend([(start,packet(flags=dpkt.tcp.TH_SYN,seq=isn)),(start+1,packet(seq=isn+1)),(start+2,packet(seq=isn+2))])
        path=self.capture(rows);samples,_,_,_=p.inspect_capture(path,'dns2tcp',p.sha(path))
        self.assertEqual([r['packet_indices'] for r in samples],[[0,1,2],[3,4,5]])

    def test_time_regression_is_quarantined(self):
        path=self.capture([(2.,packet()),(1.,packet()),(3.,packet())])
        samples,_,meta,_=p.inspect_capture(path,'dns2tcp',p.sha(path))
        self.assertEqual(samples,[]);self.assertEqual(meta['segment_status_counts']['timestamp_regression'],1)

    def test_fragment_is_not_silently_treated_as_full_packet(self):
        raw=packet();eth=dpkt.ethernet.Ethernet(raw);eth.data.mf=1
        with self.assertRaisesRegex(ValueError,'fragment'):p.decode(bytes(eth),1)

    def test_source_day_and_exact_overlaps_merge(self):
        meta=[dict(capture_start_time=1,capture_end_time=2,sha256='a'),dict(capture_start_time=3,capture_end_time=4,sha256='b'),dict(capture_start_time=90000,capture_end_time=90001,sha256='c')]
        ids,overlap=p.component_groups(meta,[{b'a'},{b'b'},{b'a'}])
        self.assertEqual(len(set(ids.values())),1);self.assertEqual(len(overlap),1)


if __name__=='__main__':unittest.main()
