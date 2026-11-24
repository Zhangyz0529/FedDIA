#!/usr/bin/env python3
"""Summarize inventory evidence by research task, not by directory count."""
import argparse,json
from pathlib import Path
import shutil
import pandas as pd
from federated_contract import sha


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def report(out,project):
    profiles=json.loads((out/'csv_profiles.json').read_text());files=pd.read_parquet(out/'file_inventory.parquet')
    assert len(profiles)==377 and not json.loads((out/'csv_errors.json').read_text())
    def count(name,predicate=lambda x:True):return sum(p['rows'] for p in profiles if p['dataset']==name and predicate(p))
    assert count('UNSW_NB15')==257673 and count('CICIoT2023')==46776700
    for representation in ['decimal','binary','hexadecimal']:
        assert count('CICIoV2024',lambda p:'/'+representation+'/' in p['path'])==1408219
    blank=json.loads((out/'cicids_blank_row_check.json').read_text());assert blank['all_empty_comma_rows']==288602 and blank['missing_label_nonempty_rows']==0
    labelled_ids=count('CIC_IDS')-288602;assert labelled_ids==2830743
    cipher=json.loads((out/'cipher_caveats.json').read_text());assert cipher['nonmetadata_pcaps']==164205
    # A zero-length list pickle is not a populated multimodal dataset.
    import pickletools
    empty=project.parents[2]/'data/USTC-TFC2016-master/multimodal_dataset/ustc_multimodal_784.pkl'
    # project = TNSM/datasets/V_second_DSTA/Paper4_FCL; parents[2] is TNSM.
    ops=[x[0].name for x in pickletools.genops(empty.read_bytes())]
    assert ops==['PROTO','EMPTY_LIST','MEMOIZE','STOP']
    rows=[
        dict(dataset='ISCX VPN-nonVPN',directory='ISCX',raw_pcaps=37,csv_rows=None,classes='发布应用/VPN活动；当前导出3类',status='SELECTED_INPUTS_READY_FCL_ASSIGNMENT_PENDING',
             evidence='37个原始PCAP/PCAPNG；既有验收导出4034样本',use='已验收的应用流量分类输入，可继续设计联邦分配',gap='尚无联邦dataset.pkl及专用注册；非攻击检测标签；更强新来源评价未认证'),
        dict(dataset='DoHBrw-2020',directory='DoHBrw-2020; cicdohbrw2020',raw_pcaps=13154,csv_rows=None,classes='现有4类诊断版',status='DIAGNOSTIC_ONLY_SOURCE_BLOCKED',
             evidence='13154原始成员；旧4类4409样本；L1表1019117行、L2表268661行属于同一DoH母集合',use='既有划分的获取/保持/通信诊断',gap='维持BLOCKED_SOURCE_GROUP_REAUDIT；两个目录不算两套独立数据'),
        dict(dataset='CIC-IDS2017',directory='CIC_IDS',raw_pcaps=5,csv_rows=labelled_ids,classes='15个非缺失原始标签字符串',status='REBUILD_OR_TABULAR_CANDIDATE',
             evidence='5个主采集PCAP；8标签CSV非空带标签2830743行；另288602全空记录；严格配对已有1353候选',use='统计特征基线或继续严格重建双模态',gap='攻击来源覆盖、DDoS匹配及新来源划分未通过；不包括切片/扰动PCAP'),
        dict(dataset='CIC-DDoS2019',directory='CIC-DDoS-2019',raw_pcaps=965,csv_rows=count('CIC-DDoS-2019'),classes='19个原始标签字符串，含跨集合别名',status='REBUILD_OR_TABULAR_CANDIDATE',
             evidence='965个PCAP分片，2个命名采集集合；18个CSV，70427637行',use='大规模DDoS统计特征基准；端点/时间字段支持配对核验起点',gap='分片不是独立来源；别名/跨集合类别覆盖/时钟及CSV配对未认证'),
        dict(dataset='CICIoT2023',directory='CICloV2023',raw_pcaps=5,csv_rows=count('CICIoT2023'),classes='34个类别目录',status='TABULAR_CANDIDATE_RAW_SUBSET',
             evidence='309个CSV，46776700行、39列统计特征；PCAP仅5类',use='34类统计表基准；原始包可用于5类子集的同源重建',gap='本地目录误名；标签在发布目录；CSV缺端点/时间，不能直接与PCAP逐流配对'),
        dict(dataset='UNSW-NB15',directory='UNSW_NB15',raw_pcaps=6,csv_rows=count('UNSW_NB15'),classes='10类attack_cat，另有二分类label',status='TABULAR_CANDIDATE_PAIRING_MISSING',
             evidence='train/test合计257673行；Pcap中6个编号采集；CSV无端点/时间对应键',use='现有表格的分类基准',gap='PCAP与CSV样本标签未对齐；合并脚本将1-6编号当类，不能当真实攻击标签'),
        dict(dataset='USTC-TFC2016',directory='USTC-TFC2016-master',raw_pcaps=20,csv_rows=count('USTC-TFC2016'),classes='10正常应用+10恶意家族采集队列；本地flow CSV二分类',status='REBUILD_CANDIDATE_SOURCE_LIMITED',
             evidence='20个原始PCAP；20个本地flow CSV共586095行；已有200549新25维候选',use='同源统计/字节重建与采集队列诊断',gap='恶意家族各一个可追溯PCAP；时间基准混杂；来源隔离未通过；一个5字节PKL为空列表'),
        dict(dataset='CipherSpectrum',directory='CipherSpectrum',raw_pcaps=164205,csv_rows=None,classes='41站点目录；算法/浏览器/日期为文件名线索',status='ENCRYPTED_CLASSIFICATION_CANDIDATE',
             evidence='164205个非元数据PCAP；4种归档组合，非4套独立数据集',use='加密网站/流量分类、统计与字节同源重建',gap='实际任务标签未认证；顶层算法目录与文件名提示不一致；按原录制分组而非逐流随机拆分'),
        dict(dataset='CICIoV2024',directory='CICIoV2024',raw_pcaps=0,csv_rows=1408219,classes='6个specific_class',status='CAN_TASK_ADAPTER_REQUIRED',
             evidence='3种表示各6个CSV，为同一CAN数据的不同表达；单表示1408219行',use='独立CAN安全分类任务',gap='不是IP流；无时间列；需专门schema，不能套IP25维或把8字节填充后称512字节真实包'),
        dict(dataset='ZhengZhou',directory='ZhengZhou',raw_pcaps=1,csv_rows=None,classes='未发现真实标签',status='UNLABELLED_CAPTURE_ONLY',
             evidence='1个15.95GB校园PCAP，附已有模型/推理代码',use='无标签流量分布、部署与吞吐分析',gap='不能默认正常，也不能由已有模型预测生成检测真值')]
    pd.DataFrame(rows).to_csv(out/'dataset_usability.csv',index=False);save(out/'dataset_usability.json',rows)
    errors=json.loads((out/'scan_errors.json').read_text());assert not errors
    probes=json.loads((out/'capture_probes.json').read_text());assert len(probes)==78 and all(p['returncode']==0 for p in probes)
    text=['# 原始数据集可用内容盘点','',
        '结论：data下11个数据相关目录对应10套逻辑数据材料（两处DoH归并为同一套），另有字体资源目录不计入。9套有标签或类别线索，郑州校园流量尚无真实标签。可用原始材料充足，但尚不能等同于10套已验收的双模态联邦数据。',
        '', 'ISCX三类4034样本已有验收后的NPZ，但尚需联邦任务分配、dataset.pkl和专用模型输入注册。DoH四类4409样本已有V4分配与训练结果，仅保留为既有划分诊断。其余需要按任务做表格预处理、重建同源双模态或补来源证据。本轮尚未认证另一套满足DoH式“2训练+2全新验证+1全新测试”的确认性划分。',
        '', '| 数据材料 | 本地主要内容 | 当前用途 | 关键缺口 |','| --- | --- | --- | --- |']
    for x in rows:text.append('| '+x['dataset']+' | '+x['evidence']+' | '+x['use']+' | '+x['gap']+' |')
    text+=['', '路径均相对 `${WORKSPACE_ROOT}/TNSM/data/`，确切目录见dataset_usability.csv。目录CICloV2023的本地README实际标题为CICIoT2023；本轮未重命名。',
        '', '扫描范围：全目录文件元数据普查；377个选定CSV完整统计记录与标签，未扫描全部CSV数值质量；78个原始/代表性PCAP约18.59GB完整结构/时间扫描，均无capinfos错误。本轮PCAP探查不等同于全语料归档CRC认证；DoH既有13154成员全量CRC/来源图结果另行引用。未改原始文件、未训练。',
        '', '必须区分的内容：ZIP与解压副本、CAN三种数值表示、DoH两目录与L1/L2任务、jitter/padding/Walkie-Talkie扰动版本、CSV分片，都不能各自增加独立数据集或采集来源数。目录体积包含这些副本，不能当独立可用数据量相加。',
        '', 'CIC-IDS清理依据：WebAttacks文件458968行中，288602行是85字段全空的占位行；非空带标签170366行。八CSV合计非空带标签2830743行，未将占位行默认正常；原件保留。15个原始标签中Heartbleed仅11、Infiltration36等，类别选择需按支持而非只按总量。',
        '', 'CICIoT2023的34类统计表与5类原始PCAP覆盖不相等：PCAP为BenignTraffic、DDoS-SYN_Flood、DictionaryBruteForce、Mirai-udpplain、Recon-PortScan。多数CSV没有端点或时间字段；文件名/目录关联不能证明同一流配对。309个CSV可能包含分片和加工窗口，不能当309个独立采集。',
        '', 'CIC-DDoS2019保留19种原始标签字符串，未自动合并DrDoS前缀和UDP-lag/UDPLag等别名；训练/测试集合的类别支持见csv_label_support.csv，不能默认所有类别两侧都齐全。',
        '', 'UNSW原表有attack_cat与label，但缺少把表中每行回溯至六个PCAP的端点/时间字段。编号PCAP的时间相邻，不代表六个攻击类；现有3_merge_iscx.py将编号映射为类，仅能说明本地生成规则，不能作为发布攻击真值。',
        '', 'USTC本地flow CSV保存pcap_name/flow_id及二分类标签；这比无旁表输入更有线索，但新样本ID、精确包集合与家族标签依据仍需明确。部分PCAP时间戳接近零/相对时间，不能直接跨文件按UTC日分组。5字节的ustc_multimodal_784.pkl是空列表，另一个299.8MB PKL是不同文件，不能因存在前者就认为已成功生成。',
        '', 'CipherSpectrum包含31–45个文件名日期/容器及数千个命名母录制，尚未证明独立采集。名称提示显示aes_256_gcm目录含aes-128/chacha20文件，chacha20目录也含aes-256文件；站点目录与文件名域名也有差异，可能涉及第三方流量或目录整理，不能未经定义就判为错误或直接赋类。算法与网站何者为目标标签需先核实。未完成全语料SHA/包重叠审计。',
        '', 'ZhengZhou只有采集和已有UNSW模型推理材料，没有找到人工/发布方真值；不能将预测作为标签，也不能据此报告攻击召回或正常Low-FPR。',
        '', '各类数量可查csv_label_support.csv，逐文件schema与解析支持见csv_profiles.json/csv_file_support.csv，原始文件位置见file_inventory.parquet；完整列名和精确来源路径保留在这些旁表中，本文只列结论。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    save(out/'final_verification.json',dict(logical_material_collections=10,data_directories=11,excluded_font_directories=1,
        file_inventory_records=len(files),metadata_errors=0,csv_profiled=377,csv_errors=0,pcap_full_probes=78,pcap_probe_errors=0,
        canonical_CAN_representations_equal_row_counts=True,canonical_UNSW_rows=257673,canonical_IoT_rows=46776700,
        CIC_IDS_labelled_nonempty_rows=labelled_ids,all_dataset_pairing_certified=False,new_fresh_source_protocol_certified=False,training_started=False))
    code=out/'code'
    for name in ['inventory_raw_datasets.py','profile_raw_datasets.py','probe_raw_captures.py','profile_cipher_provenance.py','report_raw_usability.py','federated_contract.py']:
        shutil.copy2(Path(__file__).with_name(name),code/name)
    save(out/'code_manifest.json',{p.name:sha(p) for p in code.iterdir()})
    print('Usability report complete:',out/'REPORT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--project',type=Path,required=True)
    a=p.parse_args();report(a.output.resolve(),a.project.resolve())
