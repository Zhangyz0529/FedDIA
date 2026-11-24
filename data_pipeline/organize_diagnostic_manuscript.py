#!/usr/bin/env python3
"""Build an evidence-bound diagnostic paper brief; no training or status promotion."""
import argparse,json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from federated_contract import sha


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def organize(project,out):
    if out.exists():raise ValueError('Fresh manuscript version required')
    out.mkdir(parents=True)
    results=project/'experiment_logs/doh_source25_validation_v1_20261109'
    frame=pd.read_csv(results/'final_runs.csv');assert len(frame)==12
    methods=['FedDIA','NoFreezeReplay','SmallNoFreezeReplay','DERPPNoFreeze']
    assert set(frame.method)==set(methods) and (frame.groupby('method').seed.nunique()==3).all()
    summary=[]
    for method in methods:
        f=frame[frame.method==method]
        summary.append(dict(method=method,n_seeds=len(f),pooled_f1_mean=f.pooled_macro_f1_percent.mean(),
            pooled_f1_sample_sd=f.pooled_macro_f1_percent.std(ddof=1),final_task_mean_f1=f.final_task_average_macro_f1.mean(),
            BWT=f.test_bwt.mean(),FGT=f.test_forgetting.mean(),upload_MB=f.upload_bytes.mean()/1e6,download_MB=f.download_bytes.mean()/1e6,
            duplex_MB=f.duplex_MB.mean()))
    pd.DataFrame(summary).to_csv(out/'paper_results.csv',index=False)
    paired=frame.pivot(index='seed',columns='method',values='pooled_macro_f1_percent');delta=paired.FedDIA-paired.NoFreezeReplay
    by={x['method']:x for x in summary};saving=100*(1-by['FedDIA']['duplex_MB']/by['NoFreezeReplay']['duplex_MB'])
    diag=results/'diagnostics_v1';acquisition=pd.read_csv(diag/'checkpoint_class_mean.csv')
    acquisition=acquisition[(acquisition.stage==acquisition.evaluated_task)&acquisition.label.isin(['dnscat2','iodine'])]
    claims=[
        ('C1','固定冻结工作点减少逻辑双向通信',True,'FedDIA相对NoFreezeReplay减少约28.58%；完整下发、选定上传，含初始阶段','final_runs.csv; client_rounds.json','非网络协议层实测；非预算最优或Pareto占优'),
        ('C2','冻结方法具有稳定精度优势或与不冻结等效',False,'pooled均值差0.017pp，3个配对差标准差5.173pp','final_runs.csv; verification.json','小样本不能据均值接近证明等效'),
        ('C3','冻结限制当前训练样本上的新类拟合',True,'dnscat2引入时训练召回43.33%对100%；限当前匹配设置','diagnostics_v1/checkpoint_class_mean.csv','不推广到所有来源、任务顺序或预算'),
        ('C4','解除冻结即可解决新类跨来源表现',False,'两者dnscat2引入时val召回均0%，最终召回均0%','diagnostics_v1/checkpoint_class_mean.csv; final_runs.csv','当前split是诊断，来源独立性未认证'),
        ('C5','回放导致准确率或保持能力改善/下降',False,'四个最终方法均含回放；没有新划分上的无回放对照','plan.json; final/*/result.json','只报告回放分配和暴露关联，不单独归因'),
        ('C6','后续新类的训练暴露远少于初始攻击类',True,'dns2tcp当前呈现31280，dnscat2/iodine各600，约52.13倍','diagnostics_v1/current_exposure.csv','呈现次数不是梯度贡献；不是失衡因果实验'),
        ('C7','Task3的dnscat2回放为实测426.67次',False,'426.67是从缓存类别数和总回放推导的抽样期望','diagnostics_v1/er_replay_expectations.csv','旧日志无逐类抽取；新实验必须记录实际标签'),
        ('C8','输入分布变化与低泛化并存',True,'不冻结训练召回可达100%，新类val接近0；头部与统计分布跨split变化','diagnostics_v1/REPORT.md','来源、类别和时间混杂，不能量化独立因果贡献'),
        ('C9','当前实验认证了全新独立来源检测能力',False,'扩充文件宇宙后旧局部组会合并；确认性门槛未通过','doh_source_graph_v2_20261109/REPORT.md','旧split仅支持固定划分诊断'),
        ('C10','CICIoT构造审计等同于有效检测实验',False,'来源批次未恢复；同包复算通过不等于标签/来源通过','ciciot2023_five_class_source_audit_v1_20261109/REPORT.md','审计发现不作为FedDIA性能结果'),
        ('C11','ISCX已完成持续应用分类实验',False,'4034个三类输入已验收，联邦分配/训练尚未做','source_suite_v4_20261109/ISCX_complete/GENERATION.md','只列未来补充实验，不生成结果占位数值')]
    pd.DataFrame(claims,columns=['claim_id','claim','currently_supported','evidence','source_reference','scope_limit']).to_csv(out/'claim_evidence.csv',index=False)
    checkpoint=dict(recorded_date='2026-11-10',source_gate='BLOCKED_SOURCE_GROUP_REAUDIT',accepted_confirmatory_datasets=0,
        fixed_split_diagnostic_datasets=1,application_input_datasets_pending_federation=1,
        next_acceptance_state_review='C_T produces first traceable samples',
        trigger_is_evidence_not_calendar_date=True,
        required_evidence=['actual controlled acquisition/phase manifest and raw capture hashes','activity truth mapped to admitted connections/packets',
            'same-packet statistics/byte extraction and independent recomputation','source grouping with old/new family/day/duplicate/overlap witnesses',
            'C_T assignable to training without crossing heldout components and actual class support reported'],
        initial_CT_labels=['Benign','dns2tcp'],
        after_CT_review='Record CT label/pairing/source verdict separately; only then decide extending other tool classes and V1/V2/TEST',
        CT_pass_is_not_full_protocol_pass=True,
        full_acceptance='Each retained class jointly has >=2train, >=2fresh_val, >=1fresh_test components with qualified support',
        lab_available=False,capture_started=False,training_started=False,status_promotion_performed=False)
    save(out/'acceptance_checkpoint.json',checkpoint)
    text=['# 论文诊断版本组织稿（2026-11-10）','',
        '工作题目：既有DoH划分上固定冻结与回放的获取、保持及通信取舍。',
        '', '核心问题：在既有DoH划分上，冻结与回放如何影响新类获取、后续保持和通信成本？当前直接识别的是“有回放条件下固定冻结与不冻结”的差异；回放仅有机制暴露诊断，尚无同划分无回放对照，不能单独声称其因果效应。',
        '', '摘要工作稿：本研究对固定DoH划分上的联邦持续学习进行受控诊断，采用同包统计与字节输入，比较固定冻结、完整不冻结回放、小模型回放及V4适配DER++。我们将新类在训练样本上的拟合、引入时的既有验证/测试表现、后续保持与逻辑通信分别计量。三个seed下，FedDIA与NoFreezeReplay的最终pooled Macro-F1分别为27.39±3.61%和27.37±2.17%，固定冻结减少约28.58%的逻辑双向模型通信。然而，各方法对dnscat2最终召回均为零；不冻结能够拟合新类训练样本，但未恢复既有验证来源上的检测。结果支持该工作点的获取—保持—资源取舍分析，不能证明独立新采集来源上的检测优势。来源及CICIoT2023构造审计进一步说明采集身份、标签粒度和观察单位的限制。',
        '', '1．问题与研究范围。先提出获取与保持可能混淆的问题，再说明为什么终点F1和上传量不足以判定冻结有效。研究范围明确限定于固定DoH四类划分和模拟客户端，不将当前诊断包装成来源独立的部署验证。旧190次结果使用不同数据/特征/划分，单独列背景或附录，不并入本结果表。',
        '', '2．数据与协议。4,409个样本、三模拟客户端、三个预设阶段：Benign+dns2tcp、Benign+dnscat2、Benign+iodine。正常样本在客户端和任务之间不重复；客户端共享采集来源。输入为ip_packet_stats25_v1及匿名IP字节前缀512，不称为应用payload或旧29维/USTC25。统计log1p后固定使用Task1/train均值/标准差，序列/255；不拟合未来任务或验证/测试。分类头已知四类，任务顺序不是原始采集时间顺序。',
        '', '必须保留来源说明：原split是在所选文件宇宙中构造的局部分组；全库审计后旧组会合并，不能作为全集独立来源证明。本文使用“既有验证/测试split”，不使用“已认证独立采集测试”。原始样本/分配不因本稿重写。',
        '', '3．方法与受控比较。正式方法身份固定为V4双塔、Adapter、SE、余弦头及ER；Task1后冻结主干，后续训练Adapter与分类头。NoFreezeReplay保持其余结构和ER相同；DP关闭、类别先验修正关闭。SmallNoFreezeReplay将hidden128改为64，同轮数而非预算搜索；DERPPNoFreeze是沿用V4优化器与余弦头的在线logits/蓄水池适配，不称完全标准DER++。每客户端回放容量240样本，不宣称内存字节等额。',
        '', '四方法各用seed42的相同两候选学习率搜索，按最终任务平均验证F1选参；本轮均选择lr0.001、batch64、local_epochs2。随后固定5轮/任务，对42/43/44三个seed最终评估，记录每次45客户端轮次和实际更新/回放。选参指标与最终pooled指标分别标注，避免混用。',
        '', '4．评价。获取同时报告训练、验证和测试召回及引入任务终点；保持报告完整阶段矩阵、BWT/FGT及逐类最终召回。pooled Macro-F1和最终任务平均F1并列，阶段样本量不同不能互换。三seed的标准差只描述同一划分训练波动，不描述跨来源稳定性；低初始获取会让遗忘指标出现地板效应。',
        '', '通信为序列化模型状态的逻辑字节，含Task1、每轮完整下发及所选上传键，不含网络协议开销。超参搜索成本另列，不与单次最终训练成本混加。测试用于锁参后的结果与事后诊断，不再反过来选方案。',
        '', '5．主要结果。所有数值从逐seed记录自动汇总；百分数误差为样本标准差。','',
        '| 方法 | Pooled F1 | 最终任务平均F1 | BWT | FGT | 上传MB | 下发MB |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summary:text.append(f'| {r["method"]} | {r["pooled_f1_mean"]:.2f}±{r["pooled_f1_sample_sd"]:.2f} | {r["final_task_mean_f1"]:.2f} | {r["BWT"]:.2f} | {r["FGT"]:.2f} | {r["upload_MB"]:.2f} | {r["download_MB"]:.2f} |')
    text+=['',f'FedDIA−NoFreezeReplay的pooled配对均值为{delta.mean():.3f}个百分点，配对差标准差{delta.std(ddof=1):.3f}。不能由均值接近证明等效，也不能声称稳健精度优势。当前逻辑双向通信减少{saving:.2f}%，主要来自上传键减少；完整下发并未相应减少，不推断预算前沿占优。',
        '', '6．新类获取与后续保持。训练拟合与既有评估split结果分开。','',
        '| 方法 | 引入新类 | 训练召回 | 验证召回 | 测试召回 |','| --- | --- | ---: | ---: | ---: |']
    for method in ['FedDIA','NoFreezeReplay']:
        for label in ['dnscat2','iodine']:
            f=acquisition[(acquisition.method==method)&(acquisition.label==label)].set_index('split')
            text.append('| '+method+' | '+label+' | '+' | '.join(f'{100*f.loc[s,"recall"]:.2f}%' for s in ['train','val','test'])+' |')
    text+=['', '固定冻结在该匹配工作点限制了dnscat2训练拟合；解除冻结虽达到100%训练召回，验证召回仍为0。iodine也表现为训练拟合高而既有评估split低。两方法dnscat2最终召回均0，不能把低终点成绩全归因于后续遗忘。保持指标较好时也须同时展示新类获取，不能只凭较小FGT宣称持续学习更好。',
        '', '7．回放、失衡与来源诊断。dns2tcp当前样本呈现31280次，后两攻击各600次，约52.13倍；Task3 dnscat2占ER缓存约11.11%。426.67次是期望回放抽取，不是实测逐类日志，亦不代表梯度份额。现有数据只能说明暴露结构存在不利条件，不能区分回放、类别失衡和来源变化的独立因果贡献。后续需要的机制识别是同数据同预算的回放开/关或分配对照，当前不新增训练，也不借用旧schema的消融补证。',
        '', '统计尺度与首包头字段跨split变化，且类别/场景/时间混杂；这些证据与泛化失败一致，但尚无干预证明模型依赖哪类字段。正常测试仅41个，不支持Low-FPR优势。论文不宣称场景无关的实际攻击检测、独立组织泛化或真实时间部署。',
        '', '8．CICIoT2023作为构造边界案例。五个本地文件缺原始批次/合并映射，不能把文件日期、流段、CSV分片或10000包块当作独立采集。新试点的同包复算通过，仍有跨场景同输入和单包SYN筛选风险；已有348513个100包Stat39窗口也来自同五文件。此部分解释预测单位和标签粒度，而不是作为FedDIA主实验数据或模型排名。',
        '', '9．结论与后续。当前支持的结论是固定划分、固定调参范围和ER设置下的获取/保持/通信取舍。下一验收状态更新节点为C_T首批可追溯样本：先核对活动标签、同包输入与原包复算、全局来源归属，再判断是否扩展其余工具及验证/测试槽位。C_T局部通过不能将完整确认性数据集数从0改为1。ISCX仅在联邦分配完成后另列持续应用分类，尚无结果可写。',
        '', '证据清单：claim_evidence.csv记录可写与不可写主张；paper_results.csv为本稿结果表；input_refs.json固定引用记录的哈希；acceptance_checkpoint.json记录下一触发节点。原始训练与审计目录保持不变。']
    (out/'MANUSCRIPT_DRAFT_CN.md').write_text('\n'.join(text)+'\n')
    outline=['# 论文结构与图表清单','',
        '研究问题：在既有DoH划分上，冻结与回放如何影响新类获取、后续保持和通信成本？',
        '', '1. 问题与范围：获取和保持必须分开；当前是固定划分诊断。',
        '2. 数据与协议：四类、三客户端三阶段、同包IP25/IP512、训练侧预处理及来源边界。',
        '3. V4和对照：结构身份固定；冻结开关可比较，回放因果效应暂未识别。',
        '4. 结果：阶段矩阵、引入时召回、最终pooled/任务平均、累计逻辑双向通信。',
        '5. 机制与审计：暴露失衡、期望回放、来源混杂；CICIoT为数据构造案例。',
        '6. 限制与结论：三seed只反映训练波动，真实独立来源留待新采集。',
        '', '图表建议（现有证据即可绘制，不要求新训练）：',
        '- 表1：类×任务×split支持，附来源分组范围。',
        '- 图1：训练拟合→引入时验证/测试→最终召回，分离获取与保持。',
        '- 表2：四工作点pooled/任务平均/BWT/FGT及逐类召回。',
        '- 图2：按阶段累计上传/下发字节，分别显示Task1与冻结阶段。',
        '- 表3：当前呈现、缓存类别数、期望回放及未记录项。',
        '- 表4：DoH/CICIoT的数据构造和来源证据边界。',
        '', '旧190次实验另表；ISCX只作待开展的应用分类分支，当前不填性能结果。']
    (out/'OUTLINE.md').write_text('\n'.join(outline)+'\n')
    refs=[results/'final_runs.csv',results/'verification.json',results/'plan.json',diag/'checkpoint_class_mean.csv',diag/'findings.json',
        project/'experiment_logs/doh_source_graph_v2_20261109/audit_status.json',
        project/'experiment_logs/ciciot2023_five_class_source_audit_v1_20261109/audit_status.json',
        project/'experiment_logs/doh_acquisition_preparation_v1_20261109/verification.json']
    save(out/'input_refs.json',{str(p):sha(p) for p in refs})
    code=out/'code';code.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('federated_contract.py')]:shutil.copy2(p,code/p.name)
    save(out/'code_manifest.json',{p.name:sha(p) for p in code.iterdir()});print(out/'MANUSCRIPT_DRAFT_CN.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();organize(a.project.resolve(),a.output.resolve())
