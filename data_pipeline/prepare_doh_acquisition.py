#!/usr/bin/env python3
"""Prepare a non-executing collection plan and consistency preflight.

Plan names are NOT source identities. This gate never certifies independence
from metadata alone and never installs tools, starts capture or trains a model.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import shutil,subprocess,zipfile
import pandas as pd
from federated_contract import sha

LABELS=['Benign','dns2tcp','dnscat2','iodine']


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def preflight(plan,old_components):
    campaigns=plan['campaigns'];problems=[];keys=[x['plan_slot_id'] for x in campaigns]
    if len(keys)!=len(set(keys)):raise ValueError('Repeated plan slot')
    parent=list(range(len(campaigns)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    witnesses=[]
    for i,a in enumerate(campaigns):
        for j,b in enumerate(campaigns[:i]):
            reasons=[]
            for field in ['scene_family_id','scene_definition_sha256','original_campaign_lineage_id']:
                if a.get(field) and a.get(field)==b.get(field):reasons.append('same_'+field)
            if a.get('UTC_capture_days') and set(a['UTC_capture_days'])&set(b.get('UTC_capture_days') or []):reasons.append('shared_UTC_day')
            if reasons:parent[root(j)]=root(i);witnesses.append(dict(left=a['plan_slot_id'],right=b['plan_slot_id'],reasons=reasons))
    groups=defaultdict(list)
    for i,a in enumerate(campaigns):groups[root(i)].append(a)
    for group in groups.values():
        if len({a['intended_split'] for a in group})>1:problems.append('proposed_component_crosses_splits:'+','.join(a['plan_slot_id'] for a in group))
    for a in campaigns:
        missing=[x for x in ['scene_family_id','scene_definition_sha256','original_campaign_lineage_id','deployment_identity','UTC_capture_days'] if not a.get(x)]
        if missing:problems.append(a['plan_slot_id']+':missing_identity:'+','.join(missing))
        if not a.get('scene_equivalence_evidence_complete'):problems.append(a['plan_slot_id']+':scene_equivalence_to_old_and_other_campaigns_unresolved')
        if not a.get('assets_confirmed'):problems.append(a['plan_slot_id']+':controlled_assets_unavailable')
    if not plan['resources']['controlled_lab_available']:problems.append('controlled_lab_unavailable')
    # Conditional budget assumes distinct slots; it is not a grouping result.
    counts={l:Counter() for l in LABELS}
    for g in old_components:
        for l in g['labels']:counts[l]['train']+=1
    for a in campaigns:
        for l in a['target_labels']:counts[l][a['intended_split']]+=1
    conditional={l:dict(counts[l]) for l in LABELS}
    capacity=all(counts[l]['train']>=2 and counts[l]['val']>=2 and counts[l]['test']>=1 for l in LABELS)
    if not capacity:problems.append('conditional_class_split_coverage_shortfall')
    return dict(source_gate='BLOCKED_SOURCE_GROUP_REAUDIT',capture_ready=False,training_ready=False,
        conditional_capacity_if_all_slots_are_proven_distinct=capacity,conditional_group_counts=conditional,
        proposed_merge_witnesses=witnesses,blockers=problems,
        caveat='IDs, dates and configuration hashes do not prove independence. Source/packet/label admission must be audited after acquisition.')


def create(project,out):
    if out.exists():raise ValueError('Fresh preparation version required')
    out.mkdir(parents=True)
    component_table=project/'experiment_logs/doh_source_graph_v2_20261109/component_class_support.csv'
    table=pd.read_csv(component_table);old=[]
    for comp,g in table[table.old_accepted_samples>0].groupby('component'):
        old.append(dict(component=comp,labels=sorted(g.label.unique()),intended_split='train',origin='previously_diagnosed_published_data',
            accepted_samples=dict(zip(g.label,g.old_accepted_samples))))
    tools=['dns2tcpd','dns2tcpc','dnscat2','iodine','iodined','ruby','docker','podman','tcpdump','tshark','capinfos','cloudflared','dnscrypt-proxy','dnsmasq','unbound','dig','openssl']
    caps={line.split(':')[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('CapEff:','CapBnd:','NoNewPrivs:'))}
    save(out/'environment_probe.json',dict(tools_on_PATH={n:shutil.which(n) for n in tools},uid=os.getuid(),gid=os.getgid(),sandbox_process=caps,
        host_capture_privileges='NOT_VERIFIED; sandbox capabilities are not evidence of host-level permissions',
        user_declared_lab_available=False,user_instruction='目前没有，先完成补采方案',active_network_or_capture_started=False))
    campaigns=[]
    for slot,split,purpose in [('C_T','train','new training coverage'),('C_V1','val','first fresh validation source'),('C_V2','val','second fresh validation source'),('C_TEST','test','fresh final source')]:
        campaigns.append(dict(plan_slot_id=slot,intended_split=split,purpose=purpose,target_labels=LABELS.copy(),
            source_group_id=None,scene_family_id=None,scene_definition_sha256=None,original_campaign_lineage_id=None,
            deployment_identity=None,UTC_capture_days=None,assets_confirmed=False,scene_equivalence_evidence_complete=False,
            planned_accepted_sample_target_per_label=100,planned_targets_are_not_acceptance_results=True,
            class_phases='Separate tool/browser phases and connection pools, common topology per campaign',
            requirement='Real campaign/setup evidence required; renaming IDs, changing domain/date/seed alone cannot create a source'))
    plan=dict(version='doh_acquisition_candidate_v1',source_gate='BLOCKED_SOURCE_GROUP_REAUDIT',stage='PLAN_ONLY_NO_LAB',
        resources=dict(controlled_lab_available=False,client_environments=None,controlled_DOH_endpoint=None,authoritative_DNS_zone=None,
            controlled_tunnel_server=None,capture_interface=None,clock_reference=None,tool_versions=None),
        campaigns=campaigns,existing_diagnosed_components_training_only=old,
        joint_budget='4 hypothetical new multi-class components, not 12 separate components or 16 phase captures; source merging may increase requirement',
        raw_data_root_proposal='TNSM/data/DoH_self_capture_v1 (not created)',
        rehearsal='First C_T only: Benign+dns2tcp end-to-end evidence and pairing; other labels only after this succeeds',
        comparison=dict(method='NoFreezeReplay',attack_samples_per_training_arm=60,
            single_source='rotate old-diagnostic and new-training sources when both support60',
            multi_source='30 samples from each of the same two training sources; total60',
            locks=['same normal samples','same task/client rule','same actual optimizer updates','same Task1-only preprocessing','same replay rule'],
            selection='validation only; final test never used for selecting capture configurations, thresholds or training parameters'),
        preflight_is_not_independence_certification=True)
    save(out/'campaign_plan.json',plan);save(out/'allocation_preflight.json',preflight(plan,old))
    allocation=[]
    for a in campaigns:
        for l in a['target_labels']:allocation.append(dict(plan_slot=a['plan_slot_id'],class_label=l,intended_split=a['intended_split'],
            expected_new_component='UNVERIFIED_SLOT_'+a['plan_slot_id'],actual_component=None,source_identity_confirmed=False,accepted_samples=None))
    pd.DataFrame(allocation).to_csv(out/'component_class_role_budget.csv',index=False)
    template=dict(protocol_version=plan['version'],plan_slot_id=None,campaign_id=None,original_campaign_lineage_id=None,
        scene_family_id=None,scene_definition_sha256=None,scene_equivalence_evidence_refs=[],deployment_identity=None,intended_split=None,
        roles=dict(client_id=None,DOH_gateway_id=None,authoritative_DNS_server_id=None,tunnel_server_id=None,benign_application_id=None),
        config_snapshots=[],software_versions=[],clock_alignment_records=[],capture_interfaces=[],
        captures=[dict(original_capture_id=None,path=None,sha256=None,bytes=None,start_UTC=None,end_UTC=None,link_type=None,interface_id=None,
            parent_capture_id=None,parent_sha256=None,slice_packet_range=None,uncut_original_retained=None)],
        phases=[dict(phase_id=None,class_label=None,process_event_log_ref=None,tool_session_id=None,request_id_log_ref=None,
            proxy_HTTP_connection_log_ref=None,authoritative_DNS_log_ref=None,resource_expected_payload_sha256=None,
            resource_received_payload_sha256=None,transfer_success=None,doh_request_response_evidence_ref=None,
            connection_pool_id=None,connection_tuple_map_ref=None,unexpected_background_action='quarantine',
            cross_label_connection_reuse_forbidden=True)],
        grouping_audit=dict(same_family_witnesses=[],shared_UTC_day_witnesses=[],duplicate_file_witnesses=[],exact_packet_overlap_witnesses=[],actual_global_component=None),
        sample_audit=dict(feature_schema=None,sample_ids_ref=None,source_packet_indices_ref=None,label_lineage_ref=None,
            independent_recompute_ref=None,accepted_class_counts=None,conflicting_inputs_ref=None),
        credentials='Store only external credential handles; do not put keys, tokens or TLS secrets in model inputs or this manifest')
    save(out/'capture_manifest_template.json',template)
    registry=dict(protocol_version=plan['version'],source_group_id_policy='Assigned only after global graph, never equal to plan_slot_id by default',
        family_policy='Same registered scientific scene family merges across dates; resolve aliases to historical families before counting',
        campaign_policy='Same original campaign/deployment lineage remains indivisible even if files, processes, labels or interfaces differ',
        time_policy='Any shared actual UTC capture day merges across all labels; plan campaigns on distinct UTC days and recheck actual times',
        content_policy='Retain duplicate whole-file and exact timestamp+raw overlap evidence under the existing rule; empty/invalid captures never earn support',
        novelty_policy='Date/domain/seed/name changes alone do not certify a new scene or campaign; config SHA is an identity aid, not statistical independence proof',
        taint_policy='Historical model-diagnosed components may be training only in the new version; raw integrity/label QA of heldout collections does not authorize model-based selection',
        scope='Independent local acquisition/campaign groups only; no claim of organization/population independence')
    save(out/'source_identity_rules.json',registry)
    acceptance=[
        'Before capture: controlled asset roles and private zone/owned domain are explicit; scene/campaign identities and split roles fixed; no source count from placeholder slot IDs.',
        'Pre-audit scientific family equivalence against old and planned sources. Same family/configuration identity/lineage or UTC day cannot be split across roles.',
        'dns2tcp is TCP over DNS; show actual DNS carried by HTTPS between stub and controlled DoH endpoint. Native UDP/TCP DNS alone does not qualify.',
        'First training-side rehearsal verifies expected payload hash at the controlled tunnel resource plus client/proxy/DoH/auth-server logs and captured HTTPS connection mapping.',
        'Keep one class per identified HTTPS connection pool. Shared HTTP/2 connections carrying different class activities require stream-to-packet attribution or exclusion; time-only labels are insufficient.',
        'Benign requires controlled legitimate browser/DNS activity evidence; unexpected background is unknown/quarantined, not default Benign.',
        'Freeze unmodified originals; record SHA256/size/UTC/interface and parent-to-slice packet ranges. Never append virtual trigger packets to originals.',
        'After capture: merge old and new material in one cross-class graph with family, UTC-day, duplicate and overlap witnesses, then jointly allocate all components.',
        'Only accepted source components with verified class support debit the budget. Empty/invalid data, copied files and replays do not earn new source groups.',
        'Recompute fixed-schema statistics and anonymized IP prefix from exactly the same admitted packet collection; preserve sample_id, connection, packet indices and label chain.',
        'Use planned100 qualified samples/class/component as a planning target, not evidence. Each training source needs >=60 for the matched source-coverage control; validation/test support must be audited by source and class.',
        'Count connections, observation segments and source groups separately. Do not inflate independent support by slicing one connection; audit duplicate and conflicting model inputs.',
        'Fit data transforms only on allowed historical/current training data; log actual per-class current and replay draws for the next experiment.',
        'No training until source, label, joint-split and sample-unit gates pass. Retire old diagnosed evaluation data from confirmation; choose parameters on new validation and evaluate final test after locking.'
    ]
    save(out/'acceptance_rules.json',acceptance)
    text=['# DoH补采准备：先dns2tcp，当前仅方案','',
        '状态：研究来源门槛保持 **BLOCKED_SOURCE_GROUP_REAUDIT**；执行准备为 **PLAN_ONLY_NO_LAB**。用户确认目前没有客户端/服务器/虚拟机及DNS区域等补采环境，本轮未安装、采集或训练。',
        '', '本地工具核对：PATH未找到dns2tcp客户端/服务端、dnscat2、iodine、DoH转发器或容器运行时；已有tcpdump/tshark/capinfos。当前沙箱无网络捕获能力，宿主机权限未另行核验，不能据此断言宿主机不能抓包。',
        '', '| 计划槽位 | 目标类别 | 用途 | 当前真实分量 |', '| --- | --- | --- | --- |']
    for a in campaigns:text.append(f'| {a["plan_slot_id"]} | 正常＋dns2tcp＋dnscat2＋iodine | {a["intended_split"]} | 未建立 |')
    text+=['', '四个槽位假如最终确认为四个互不合并的新分量，可联合满足三攻击类各新增1训练+2验证+1测试；正常类也同时获得3个新评价组。不是12个独立攻击分量，也不是16份分阶段PCAP就算16组。已有诊断分量只列入候选训练池；旧正常候选不作为已通过验收的评价组计入。',
        '', '原始采集/批次身份、部署与场景定义、配置/软件版本、实际UTC日期、抓包接口、父文件与切片关系、标签事件链均有模板。所有真实身份/日期/配置字段目前为空，预审明确阻塞；没有通过换名称或只换日期/域名来假定独立。',
        '', '优先拓扑（全为待配置的可控实验角色）：dns2tcpc → DNS-to-DoH stub → HTTPS观察点 → 可控DoH网关/解析器 → 私有权威DNS区域 → dns2tcpd → 可控测试资源。正常浏览器通过同组相同DoH网关访问可控正常服务。部署可使用自有虚拟环境或独立机器，具体实现/版本待基础设施落实后核定，不要求公开第三方目标。',
        '', '先在C_T做正常＋dns2tcp端到端传输及标签演练：源/目的测试负载哈希一致、DoH请求/响应与权威端日志一致、外层HTTPS抓包可关联到工具连接；确认不是明文DNS或回退流量。随后扩展同组其他工具，再执行其余已预注册场景。这个演练只验证构造，不以模型F1决定场景或holdout。',
        '', 'HTTPS连接复用是标签关键点：同一连接不混用正常与隧道活动；若HTTP/2混合类别且无法关联stream/packet，则隔离连接，不按时间粗暴赋类。标签来自过程、请求、网关、权威端和负载回执的可核查关联，不继承发布数据硬编码IP名单。',
        '', '同源提取及预算控制：每个训练来源至少60个合格单位后，比较单来源60（可轮换）与两来源各30的60样本方案，固定正常样本、任务/客户端规则、实际更新预算、历史训练预处理及回放。后续才单独改当前CE权重，再检查实际逐类回放；不在本轮实施。',
        '', '采后核销：将原件并入旧/新统一来源图，复核同族、UTC日、文件重复、包重叠和标签支持；合并导致跨split或支持不足则不核销槽位，需要补新的真实分量。配置哈希和身份记录不能代替采集独立性论证。',
        '', '文件：campaign_plan.json为待填计划；component_class_role_budget.csv为分量×类别×用途；capture_manifest_template.json为身份/过程记录；source_identity_rules.json及acceptance_rules.json为准入规则；allocation_preflight.json保留当前阻塞项。',
        '', '依据：[dns2tcp上游说明](https://github.com/alex-sector/dns2tcp)；[DoH RFC8484](https://www.rfc-editor.org/rfc/rfc8484)。工具实现与DoH封装需分别验证，未承诺当前工具链已经可运行。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    save(out/'input_refs.json',{str(component_table):sha(component_table)})
    code=out/'code';code.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('federated_contract.py')]:shutil.copy2(p,code/p.name)
    save(out/'code_manifest.json',{p.name:sha(p) for p in code.iterdir()})
    print(json.dumps(preflight(plan,old),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();create(a.project.resolve(),a.output.resolve())
