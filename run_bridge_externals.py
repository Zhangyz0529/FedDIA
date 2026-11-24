"""Executed Re-Fed transport reuse and a budget-derived small AF-FCL flow.

Reference/optimized workers share the frozen learning code and GPU. Every
executed message is serialized to a payload file and deserialized on receipt.
"""
import argparse,copy,hashlib,importlib.util,inspect,json,os,pickle,shutil,subprocess,sys,threading
from pathlib import Path
import numpy as np

CAP=37208796


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(p)


def load_worker(out):
    spec=importlib.util.spec_from_file_location('frozen_bridge_worker',out/'code/run_recent_external.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare(project,root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'PLAN.json').exists():return
    original=project/'supplementary_data/recent_external_fcl_20261114_v1'
    text=(original/'code/run_recent_external.py').read_text()
    helper='''\ndef tensor_sha(model_state):
 import hashlib
 h=hashlib.sha256()
 for k,v in model_state.items():
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(a.tobytes())
 return h.hexdigest()
'''
    text=text.replace('\ndef worker(out,j):',helper+'\ndef worker(out,j):',1)
    old='calls=[];messages=[];history=[];buffers=[];valtraj=[];past=set();begin=time.monotonic()'
    assert old in text
    text=text.replace(old,old+';round_hashes=[];reuse_events=[]')
    old="  messages.append(dict(task=task,client=client,round=round,direction=direction,kind=kind,serialized_bytes=size(obj)))"
    new='''  payload=pickle.dumps(obj)
  folder=run/'transport_payloads';folder.mkdir(exist_ok=True)
  path=folder/f'{len(messages):05d}.pkl';path.write_bytes(payload)
  messages.append(dict(task=task,client=client,round=round,direction=direction,kind=kind,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),payload_path=str(path.relative_to(run))))
  return pickle.loads(payload)'''
    assert old in text;text=text.replace(old,new)
    old="local=copy.deepcopy(net).cuda().train();message(task+1,c,'download','classifier',state(net),rnd+1)"
    reference="local=copy.deepcopy(net).cuda().train();received=message(task+1,c,'download','classifier',state(net),rnd+1);local.load_state_dict(received)"
    assert old in text;text=text.replace(old,reference)
    old="message(task+1,c,'upload','classifier',weights[-1],rnd+1)"
    assert old in text;text=text.replace(old,"weights[-1]=message(task+1,c,'upload','classifier',weights[-1],rnd+1)")
    old="pool=client['buffer']+currentitems;message(task+1,c,'download','PIM_global_anchor',state(net));pim=lastlocals[c].cuda().train();anchor={k:v.detach().clone() for k,v in net.named_parameters()};"
    new="pool=client['buffer']+currentitems;receipt=message(task+1,c,'download','PIM_global_anchor',state(net));client['handoff']=receipt;pim=lastlocals[c].cuda().train();anchor={k:receipt[k].cuda().clone() for k,v in net.named_parameters()};"
    assert old in text;text=text.replace(old,new)
    old="save(run/'progress.json',dict(task=task+1,round=rnd+1,calls=len(calls),test_accessed=False))"
    assert old in text;text=text.replace(old,old+";round_hashes.append(dict(task=task+1,round=rnd+1,state_sha256=tensor_sha(state(net))))")
    old="torch.save(state(net),run/'final_model.pth');"
    assert old in text;text=text.replace(old,"save(run/'round_state_hashes.json',round_hashes);save(run/'reuse_events.json',reuse_events);"+old)
    optimized=text.replace("elif arm=='ReFed':","elif arm=='ReFed' and task<2:")
    new='''local=copy.deepcopy(net).cuda().train()
    if task>0 and rnd==0:
     received=client.pop('handoff')
     assert tensor_sha(received)==tensor_sha(state(net)),'phase-boundary cached server state differs'
     reuse_events.append(dict(task=task+1,client=c,round=1,received_state_sha256=tensor_sha(received),new_message_sent=False))
    else:received=message(task+1,c,'download','classifier',state(net),rnd+1)
    local.load_state_dict(received)'''
    assert reference in optimized;optimized=optimized.replace(reference,new)
    for name,code in [('reference',text),('optimized',optimized),('af_small',(original/'code/run_recent_external.py').read_text())]:
        out=root/name;out.mkdir(exist_ok=True)
        for folder in ('code','vendor'):
            shutil.copytree(original/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__'))
        if name=='af_small':shutil.copytree(original/'official',out/'official')
        (out/'code/run_recent_external.py').write_text(code)
        shutil.copy2(original/'data_manifests.json',out/'data_manifests.json')
        shutil.copy2(original/'vendor_manifest.json',out/'vendor_manifest.json')
        save(out/'code_manifest.json',{str(p.relative_to(out/'code')):sha(p) for p in (out/'code').rglob('*.py')})
    cfg=read(original/'parameter_selection.json')['CIC_IDS_ReFed']['config']
    schedules=[[8,7,7],[10,6,6],[6,6,10]]
    plan=dict(cap_bytes=CAP,refed_parameters=cfg,refed_parameter_selection='reuse old validation-selected parameters; only execution and schedule change',
        refed_candidates=schedules,refed_candidate_rule='22 rounds: balanced earliest-stage remainder, front, back; all task counts >=6; frozen before new scores',
        selection='seed42 maximum final validation fixed-task-average macroF1; tie lower executed bytes then job ID',
        formal_seeds=[42,43,123],same_GPU_reference_equivalence_required=True,
        reuse_semantics='use the received preceding endpoint PIM global anchor as next-stage first-round download; omit final PIM/cache preparation',
        transport='executed serialized payloads saved and deserialized; simulated client/server transfer, not TCP/wire measurement',
        AF_flow_selection='largest hidden power-of-two in [16,32,64,128,256,512] whose conservative full 5/5/5 payload bound fits cap; four native coupling layers retained',
        AF_parameters='reuse prior CIC validation-selected settings except resource-derived flow width; no claim of optimized hyperparameters for new width',
        AF_scope='one small-flow configuration sensitivity, not a native image-protocol reproduction',
        prior_test_used_for_development=True,new_test_after_selection_and_formal_training=True)
    save(root/'PLAN.json',plan)
    shutil.copy2(__file__,root/'run_bridge_externals.py')


def run_job(root,variant,job,gpu='0'):
    out=root/variant;run=out/'runs'/job['id'];run.mkdir(parents=True,exist_ok=True)
    save(run/'job.json',job)
    if (run/'training_result.json').exists():return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (run/'console.log').open('a') as log:
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--root',str(root),
                        '--variant',variant,'--job',str(run/'job.json')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',variant,job['id'],flush=True)


def check_equivalence(root):
    import torch
    a=root/'reference/runs/execution_equivalence_s42'
    b=root/'optimized/runs/execution_equivalence_s42'
    assert read(a/'round_state_hashes.json')==read(b/'round_state_hashes.json')
    ca=read(a/'client_records.json');cb=read(b/'client_records.json');assert ca==cb
    comparisons=[]
    for task in (1,2):
        for client in (0,1,2):
            with np.load(a/f'selection/task{task}_client{client}.npz') as x,np.load(b/f'selection/task{task}_client{client}.npz') as y:
                assert all(np.array_equal(x[k],y[k]) for k in x.files)
            x=torch.load(a/f'buffers/task{task}_client{client}.pth',map_location='cpu')
            y=torch.load(b/f'buffers/task{task}_client{client}.pth',map_location='cpu')
            assert len(x)==len(y) and all(torch.equal(p[0],q[0]) and torch.equal(p[1],q[1]) and p[2]==q[2] for p,q in zip(x,y))
            comparisons.append(dict(task=task,client=client,selection_scores_indices_buffer_exact=True))
    assert len(read(b/'reuse_events.json'))==6
    ma=read(a/'communication_messages.json');mb=read(b/'communication_messages.json')
    assert not any(m['kind']=='PIM_global_anchor' and m['task']==3 for m in mb)
    assert not any(m['kind']=='classifier' and m['direction']=='download' and m['task']>1 and m['round']==1 for m in mb)
    for folder in (a,b):
        for m in read(folder/'communication_messages.json'):
            p=folder/m['payload_path'];assert p.stat().st_size==m['serialized_bytes'] and sha(p)==m['payload_sha256']
    ra=read(a/'training_result.json');rb=read(b/'training_result.json')
    actual_saving=ra['duplex_bytes']-rb['duplex_bytes']
    assert actual_saving>0
    save(root/'REFED_EXECUTION_EQUIVALENCE.json',dict(status='PASS',rounds_exact=15,
        all_global_round_states_exact=True,all_current_batches_replay_counts_optimizer_updates_losses_exact=True,
        earlier_cache_scores_and_inputs_exact=True,comparisons=comparisons,
        cached_receipt_reused=6,terminal_PIM_steps_omitted=True,
        terminal_cache_intentionally_not_rebuilt=True,legacy_executed_bytes=ra['duplex_bytes'],optimized_executed_bytes=rb['duplex_bytes'],saving_bytes=actual_saving,
        actual_serialized_payloads_verified=True,not_real_network_measurement=True))
    archived=root.parents[1]/'recent_external_fcl_20261114_v1/runs/CIC_IDS_ReFed_fixed_s42'
    historical=[]
    for task in (1,2,3):
        x=torch.load(archived/f'task_checkpoints/task_{task}.pth',map_location='cpu')
        y=torch.load(a/f'task_checkpoints/task_{task}.pth',map_location='cpu')
        historical.append(dict(task=task,bitwise_identical=all(torch.equal(x[k],y[k]) for k in x),
            max_abs_difference=max(float((x[k]-y[k]).abs().max()) for k in x)))
    save(root/'REFERENCE_ARCHIVE_COMPARISON.json',dict(archived_reference_run='CIC_IDS_ReFed_fixed_s42',
        matched_transport_reference_run='execution_equivalence_s42',checks=historical,
        interpretation='Historical binding; same-GPU matched execution equivalence is independently required.'))
    # The maximal ordinary round already charges all six full classifier
    # transfers. Earlier anchors and boundary reuse cancel in this protocol.
    maxround=max(sum(m['serialized_bytes'] for m in mb if m['task']==t and m['round']==r)
        for t in (1,2,3) for r in range(1,6))
    bound=22*(maxround+4096)+8192
    assert bound<=CAP
    save(root/'REFED_CANDIDATE_FEASIBILITY.json',dict(status='PASS',ordinary_round_observed_max_bytes=maxround,
        conservative_22_round_bound=bound,cap_bytes=CAP,anchor_reuse_boundary_net_extra_bytes=0,
        schedules=read(root/'PLAN.json')['refed_candidates']))


def size_AF(root):
    out=root/'af_small'
    if (out/'FLOW_SELECTION.json').exists():return
    mod=load_worker(out);mod.imports(out)
    # Execute the same factory on CPU for sizing only; no loss, scores or
    # parameter updates enter the resource-derived width selection.
    code=inspect.getsource(mod.make_af).replace('.cuda()', '.cpu()')
    exec(compile(code,'AF_factory_CPU_size_only','exec'),mod.__dict__)
    import torch
    torch.manual_seed(20261121)
    original=root.parent.parent/'recent_external_fcl_20261114_v1'
    # root is supplementary_data/evidence_bridge_.../externals.
    original=Path(read(root/'PLAN.json').get('unused_path',str(root.parents[1]/'recent_external_fcl_20261114_v1')))
    cfg=read(original/'parameter_selection.json')['CIC_IDS_AFFCL']['config']
    sizes=[]
    for width in (16,32,64,128,256,512):
        selected=dict(cfg,flow_hidden=width)
        model=mod.make_classifier(out,77,6,'AFFCL');af=mod.make_af(out,model,6,selected)
        main=len(pickle.dumps(mod.state(model)));flow=len(pickle.dumps(mod.state(af.flow)))
        bound=15*6*(main+flow+1024)+8192
        sizes.append(dict(hidden=width,layers=4,classifier_bytes=main,flow_bytes=flow,
            flow_parameters=sum(p.numel() for p in af.flow.parameters()),full_5_5_5_bound_bytes=bound,feasible=bound<=CAP))
    feasible=[r for r in sizes if r['feasible']];assert feasible
    chosen=max(feasible,key=lambda r:r['hidden'])
    config=dict(cfg,flow_hidden=chosen['hidden'])
    save(out/'FLOW_SELECTION.json',dict(rule=read(root/'PLAN.json')['AF_flow_selection'],sizes=sizes,
        selected=chosen,config=config,performance_scores_used=False,original_hidden512_source='official/AF-FCL/FLAlgorithms/PreciseFCLNet/model.py lines65-82',
        new_width_is_resource_adaptation=True,not_an_author_recommended_small_configuration=True))
    print('AF_RESOURCE_WIDTH',chosen,flush=True)


def run(project,root):
    prepare(project,root);plan=read(root/'PLAN.json')
    job=dict(id='execution_equivalence_s42',dataset='CIC_IDS',arm='ReFed',seed=42,
        schedule=[5,5,5],role='execution_audit',config=plan['refed_parameters'])
    run_job(root,'reference',job);run_job(root,'optimized',job);check_equivalence(root)
    candidates=[]
    for schedule in plan['refed_candidates']:
        j=dict(job,id='candidate_'+'-'.join(map(str,schedule))+'_s42',schedule=schedule,role='budget_candidate')
        candidates.append(j);run_job(root,'optimized',j)
    out=root/'optimized'
    eligible=[]
    for j in candidates:
        r=read(out/'runs'/j['id']/'training_result.json');assert r['duplex_bytes']<=CAP
        eligible.append((r,j))
    winner=min(eligible,key=lambda x:(-x[0]['metrics']['val_macro_f1'],x[0]['duplex_bytes'],x[1]['id']))[1]
    selected=[dict(winner,id=f'optimized_refed_formal_s{seed}',seed=seed,role='budget_final') for seed in (42,43,123)]
    save(out/'selection.json',dict(rule=plan['selection'],candidates=candidates,winner=winner,formal_seed42_fresh=True))
    save(out/'selected_runs.json',selected)
    for j in selected:run_job(root,'optimized',j)
    save(out/'TEST_GATE.json',dict(status='OPEN',selection_sha256=sha(out/'selection.json'),selected_runs_sha256=sha(out/'selected_runs.json')))
    load_worker(out).evaluate(out,'CIC_IDS')
    size_AF(root);out=root/'af_small';config=read(out/'FLOW_SELECTION.json')['config']
    jobs=[dict(id=f'AF_small_formal_s{seed}',dataset='CIC_IDS',arm='AFFCL',seed=seed,schedule=[5,5,5],role='fixed',config=config) for seed in (42,43,123)]
    save(out/'selection.json',dict(rule='resource sizing only; no new performance selection',selected_config=config))
    save(out/'selected_runs.json',jobs)
    for j in jobs:run_job(root,'af_small',j)
    assert all(read(out/'runs'/j['id']/'training_result.json')['duplex_bytes']<=CAP for j in jobs)
    save(out/'TEST_GATE.json',dict(status='OPEN',selection_sha256=sha(out/'selection.json'),selected_runs_sha256=sha(out/'selected_runs.json')))
    load_worker(out).evaluate(out,'CIC_IDS')
    save(root/'EXECUTION_STATUS.json',dict(status='TRAINING_AND_EVALUATION_COMPLETE',new_ReFed_formal=3,
        new_AF_small_formal=3,ReFed_validation_candidates=3,ReFed_execution_audit_runs=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','worker','af_remaining'])
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--root',type=Path);parser.add_argument('--variant');parser.add_argument('--job',type=Path)
    a=parser.parse_args();project=a.project.resolve();root=(a.root or project/'supplementary_data/evidence_bridge_20261121_v1/externals').resolve()
    if a.action=='prepare':prepare(project,root)
    elif a.action=='run':run(project,root)
    elif a.action=='af_remaining':
        from concurrent.futures import ThreadPoolExecutor
        jobs=[j for j in read(root/'af_small/selected_runs.json') if j['seed'] in (43,123)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run_job,root,'af_small',job,'0') for job in jobs]
            for future in futures:future.result()
    else:
        # The main sequence and independent AF continuation launcher may meet
        # at one ID; an OS lock guarantees a single actual training execution.
        import fcntl
        job=read(a.job);folder=root/a.variant/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (folder/'training_result.json').exists():load_worker(root/a.variant).worker(root/a.variant,job)
