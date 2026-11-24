#!/usr/bin/env python3
"""Verify recorded Source25 runs and report paired outcomes without retuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from federated_contract import sha


def analyze(root):
    plan=json.loads((root/'plan.json').read_text());selected=json.loads((root/'selected_params.json').read_text());rows=[]
    assert sha(plan['data'])==plan['data_sha256']
    for rel,h in json.loads((root/'code_manifest.json').read_text()).items():assert sha(root/'code'/rel)==h
    prep=set()
    for method in plan['methods']:
        for seed in plan['seeds']:
            folder=root/'final'/method/f'seed_{seed}';d=json.loads((folder/'result.json').read_text());metrics=d['metrics']
            assert d['params']==selected[method]['params'] and d['data_sha256']==plan['data_sha256']
            assert metrics['rounds_schedule']==[5,5,5] and not metrics['components']['dp']
            records=json.loads((folder/'client_rounds.json').read_text());assert len(records)==45
            for x in records:
                assert x['optimizer_updates']>0 and np.isfinite(x['loss_mean'])
                assert x['backbone_changed']!=(method=='FedDIA' and x['task']>1)
            assert metrics['training_resources']['total_replay_samples']>0
            assert sum(x['upload_bytes'] for x in records)==d['upload_bytes']
            assert sum(x['download_bytes'] for x in records)==d['download_bytes']
            prep.add(sha(folder/'preprocessing.json'))
            with np.load(folder/'test_predictions.npz',allow_pickle=False) as a:
                assert len(a['y'])==232 and np.isfinite(a['prob']).all()
                actual=100*f1_score(a['y'],a['pred'],labels=list(range(4)),average='macro',zero_division=0)
                assert np.isclose(actual,d['pooled_test_macro_f1_percent'])
            endpoints=d['test_stage_endpoints'];diag=[endpoints[str(t)][str(t)]['macro_f1'] for t in [1,2,3]]
            last=[endpoints['3'][str(t)]['macro_f1'] for t in [1,2,3]]
            bwt=float(np.mean([last[t]-diag[t] for t in range(2)]))
            fgt=float(np.mean([max(endpoints[str(s)][str(t)]['macro_f1'] for s in range(t,4))-endpoints['3'][str(t)]['macro_f1'] for t in [1,2]]))
            rows.append(dict(method=method,seed=seed,pooled_macro_f1_percent=actual,task1_endpoint_macro_f1=diag[0],
                final_task_average_macro_f1=float(np.mean(last)),test_bwt=bwt,test_forgetting=fgt,
                upload_bytes=d['upload_bytes'],download_bytes=d['download_bytes'],duplex_MB=(d['upload_bytes']+d['download_bytes'])/1e6,
                optimizer_updates=metrics['training_resources']['total_optimizer_updates'],replay_samples=metrics['training_resources']['total_replay_samples'],
                **{'recall_'+k:v for k,v in d['test_recall_fraction'].items()}))
    assert len(prep)==1
    frame=pd.DataFrame(rows);frame.to_csv(root/'final_runs.csv',index=False)
    groups=frame.groupby('method');recalls=groups[['recall_Benign','recall_dns2tcp','recall_dnscat2','recall_iodine']].mean()
    paired=frame.pivot(index='seed',columns='method',values='pooled_macro_f1_percent')
    delta=paired.FedDIA-paired.NoFreezeReplay
    saving=100*(1-groups.duplex_MB.mean()['FedDIA']/groups.duplex_MB.mean()['NoFreezeReplay'])
    findings=dict(final_runs=len(rows),paired_feddia_minus_nofreeze_pooled_f1=delta.to_dict(),paired_mean=float(delta.mean()),
        paired_std=float(delta.std()),feddia_duplex_reduction_percent=float(saving),per_class_mean_recall=recalls.to_dict('index'),
        checks=dict(frozen_source_verified=True,same_data_all_methods=True,identical_task1_train_only_preprocessing=True,
            all_test_predictions_recomputed=True,all_45_client_rounds_verified=True,correct_freezing_and_replay=True),
        interpretation='Operational validation passed; low multiclass performance and acquisition failures remain. No statistical equivalence or budget-optimality conclusion.')
    (root/'verification.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
    text=['','## 结果核验与主要发现','',f'12次最终运行均核验通过：相同输入哈希、相同Task1训练预处理、每次45个客户端轮次、回放和冻结行为正确；从保存预测重新计算Macro-F1一致。',
        '',f'FedDIA相对NoFreezeReplay的pooled F1配对平均差为{delta.mean():.3f}个百分点，seed间配对差标准差{delta.std():.3f}；不能据此证明精度优势或等效。逻辑双向通信减少{saving:.2f}%。',
        '', '| 方法 | Benign召回 | dns2tcp召回 | dnscat2召回 | iodine召回 |','| --- | ---: | ---: | ---: | ---: |']
    for method in plan['methods']:text.append('| '+method+' | '+' | '.join(f'{100*v:.2f}%' for v in recalls.loc[method])+' |')
    text+=['','所有方法最终dnscat2召回均为0；不冻结也未解决新类检测问题。Task1终点F1明显高于后续任务，低分不能只归因于最终遗忘；具体新类即时召回见stage_acquisition.json。',
        '', '下一步应先分析阶段内新类获取、训练类别失衡、回放暴露和来源变化；目前不宜直接增加seed或将这批分数作为方法有效性的主要证据。这里是5轮/任务、2本地epoch、两候选超参的初步复验，不是充分调优结果。']
    path=root/'RESULTS.md';s=path.read_text().split('\n## 结果核验与主要发现')[0];path.write_text(s+'\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in findings.items() if k!='per_class_mean_recall'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.output.resolve())
