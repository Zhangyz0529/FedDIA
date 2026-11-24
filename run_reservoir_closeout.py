"""Task-end pure ER reservoir; preserves the frozen common training machinery."""
import argparse, hashlib, json, os, pickle, random, shutil, subprocess, sys
from pathlib import Path

def read(p): return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(out,project):
    assert not out.exists(), 'Use a fresh versioned output directory'
    baseline=project/'supplementary_data/feddia_method_suite_20261114_v1'
    extension=project/'supplementary_data/feddia_external_fifo_20261114_v1'
    out.mkdir(parents=True);shutil.copytree(extension/'code',out/'code')
    p=out/'code/run_method_suite.py';s=p.read_text().replace("('FIFO','LWF')","('RESERVOIR',)")
    s=s.replace("buffer_restoration='deterministic task-end class-balanced latest records'","buffer_restoration='task-end Algorithm R, independent client RNG, original records once'")
    s=s.replace("reconstruction='online reservoir' if arm.get('derpp') else 'task-end latest per class'","reconstruction='task-end Algorithm R over original training records once'")
    p.write_text(s);shutil.copy2(__file__,out/'code/run_reservoir_closeout.py')
    shutil.copy2(baseline/'data_manifests.json',out/'data_manifests.json')
    jobs=[dict(id=f'RESERVOIR_5-5-5_s{s}',dataset='CIC_IDS',arm='RESERVOIR',seed=s,schedule=[5,5,5],role='reservoir') for s in (42,43,123)]
    save(out/'plan.json',dict(baseline=str(baseline),extension=str(extension),jobs=jobs,seeds=[42,43,123],
      reservoir='Algorithm R, task-end original train records once, persistent per-client seen count; retained records never fed back as candidates',
      identity='immutable client/task/train/row; distinct original rows remain distinct even if equal-valued; no deduplication of the historical candidate stream',
      order='frozen pickle row order, task chronological index; not claimed physical arrival time',
      rng='independent random.Random(seed*1000000+client*1000+8675309), persistent across tasks; no effect on current or replay RNG',
      control='M0 width48 full updates, ER capacity240 weight0.5, batch64 epochs3, fixed 5/5/5; exact shared Task1 weights; fresh optimizer per call',
      selection='no hyperparameter search; all three runs specified before new test inference',
      limitation='development-used diagnostic split, not independent confirmation; dual input physical pairing uncertified'))
    save(out/'selected_runs.json',jobs)
    save(out/'code_manifest.json',{str(p.relative_to(out/'code')):sha(p) for p in (out/'code').rglob('*.py')})

def worker(out,job,base):
    import torch
    import run_federated_cl_v4 as v4
    original=v4.V4Client;run=out/'runs'/job['id']
    class ReservoirClient(original):
        def store_replay(self,samples,label_map,model=None):
            if not hasattr(self,'reservoir_rng'):
                self.reservoir_rng=random.Random(job['seed']*1000000+self.client_id*1000+8675309)
                self.seen_examples=0;self.reservoir_ids=[];self.reservoir_task=0
            self.reservoir_task+=1;events=[];capacity=self.args.replay_capacity
            for row,s in enumerate(samples):
                identity=f'Client_{self.client_id}/Task_{self.reservoir_task}/train/{row}'
                self.seen_examples+=1
                slot=len(self.replay_buffer) if len(self.replay_buffer)<capacity else self.reservoir_rng.randrange(self.seen_examples)
                kept=slot<capacity
                previous=self.reservoir_ids[slot] if kept and slot<len(self.reservoir_ids) else None
                if kept:
                    item=(torch.tensor(v4.normalize_stat_feature(s['stat_raw']),dtype=torch.float32),torch.tensor(v4.normalize_seq_feature(s['seq_feature']),dtype=torch.float32),label_map[s['label']])
                    if slot==len(self.replay_buffer):self.replay_buffer.append(item);self.reservoir_ids.append(identity)
                    else:self.replay_buffer[slot]=item;self.reservoir_ids[slot]=identity
                events.append(dict(id=identity,seen=self.seen_examples,draw=slot,retained=kept,replaced_id=previous))
            save(run/'reservoir_audit'/f'task{self.reservoir_task}_client{self.client_id}.json',dict(seen_examples=self.seen_examples,retained_ids=self.reservoir_ids,rng_state=self.reservoir_rng.getstate(),events=events))
    v4.V4Client=ReservoirClient
    base.worker(out,job)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','evaluate','run']);p.add_argument('--output',required=True,type=Path);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--job',type=Path);a=p.parse_args();out=a.output.resolve()
    if a.action=='prepare':prepare(out,a.project.resolve())
    else:
        import run_method_suite as base
        base.ARMS['RESERVOIR']=dict(width=48,freeze=False,replay=True)
        if a.action=='worker':worker(out,read(a.job),base)
        elif a.action=='evaluate':base.evaluate(out,'CIC_IDS')
        else:
            jobs=read(out/'plan.json')['jobs'];processes=[]
            for j in jobs:
                run=out/'runs'/j['id'];run.mkdir(parents=True,exist_ok=True);save(run/'job.json',j)
                if (run/'training_result.json').exists():continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(len(processes)%2),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
                log=(run/'console.log').open('a')
                cmd=[sys.executable,'-B',str(out/'code/run_reservoir_closeout.py'),'worker','--output',str(out),'--job',str(run/'job.json')]
                processes.append((subprocess.Popen(cmd,cwd=out/'code',env=env,stdout=log,stderr=subprocess.STDOUT),log,j['id']))
                if len(processes)==2:
                    for proc,log,name in processes:assert proc.wait()==0,name;log.close();print('DONE',name,flush=True)
                    processes=[]
            for proc,log,name in processes:assert proc.wait()==0,name;log.close();print('DONE',name,flush=True)
            assert all((out/'runs'/j['id']/'training_result.json').exists() for j in jobs)
            save(out/'TEST_GATE.json',dict(status='OPEN',all_training_complete=True,plan_sha256=sha(out/'plan.json'),selected_runs_sha256=sha(out/'selected_runs.json')))
            base.evaluate(out,'CIC_IDS')
            save(out/'execution_status.json',dict(status='TRAINING_AND_EVALUATION_COMPLETE_REPORT_PENDING',new_continuations=3,reused_task1_prefixes=3,development_runs=0))
