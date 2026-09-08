#!/usr/bin/env python3
"""Zima tmux controller for sequential, guarded K275 speed experiments."""
import json
import os
from pathlib import Path
import subprocess
import shlex
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import quality_scale_campaign as lab
from long_context_probe import raw_native_completion

PROJECT=lab.PROJECT
ROOT=lab.STATE/'speed/20260908'
STAMP='20260907T224500Z'
MODEL='GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-20260907T210500Z'
PACK='20260907T231500Z-k275-mixed-rank-local-v1'
BENCH=Path('/home/mj-kang/Dev/benchmark/llama-benchy/results/k275-speed-20260908')
rows=[]
current='k275-post-reencode-v1'


def write_status(phase, **kwargs):
    lab.atomic_json(ROOT/'STATUS.json',dict(phase=phase,current=current,rows=rows,updated_at=lab.utc(),**kwargs))


def remote(node,args,**kwargs):return lab.ssh(node,args,**kwargs)


def stop():
    state=lab.STATE/f'real-test/{STAMP}/candidate/attempts/{current}'
    if (state/'READY.json').is_file() and not (state/'STOPPED.json').exists():
        write_status('stopping',label=current)
        with (ROOT/f'{current}-stop.log').open('w') as out:
            subprocess.run(['bash',PROJECT/'scripts/stop_candidate_attempt.sh',STAMP,current],cwd=PROJECT,
                           stdout=out,stderr=subprocess.STDOUT,check=True,timeout=300)


def hardware(since,label):
    path=ROOT/f'{label}-kernel.json'
    subprocess.run(['python3',PROJECT/'scripts/kernel_audit.py','--since',since,'--until',lab.utc(),
                    '--ssh-config',PROJECT/'zima-ssh-config','--output',path],check=True,timeout=90)


def probes(label):
    out=[]
    for prompt,expected in [('What is the capital of France? Reply with only the city name.','Paris'),
                            ('Calculate 17 * 19. Reply with only the integer answer.','323'),
                            ('Return exactly this JSON object and nothing else: {"answer":42}',{'answer':42})]:
        r=raw_native_completion(lab.ENDPOINT,prompt,max_tokens=64,timeout=180)
        content=r['content'].strip()
        try:actual=json.loads(content) if isinstance(expected,dict) else content
        except ValueError:actual=None
        out.append(dict(content=content,expected=expected,passed=actual==expected and r['finish_reason']=='stop'))
    lab.atomic_json(ROOT/f'{label}-probes.json',out)
    if not all(x['passed'] for x in out):raise RuntimeError('native generation gate failed')


def fetch_bench(label):
    remote('mj-spark-1',['test','-s',BENCH/label/'result.json'])
    subprocess.run(['rsync','-rt','-e',f'ssh -F {PROJECT}/zima-ssh-config',
                    f'mj-spark-1:{BENCH/label}/',f'{ROOT/label}/'],check=True,timeout=120)
    value=json.loads((ROOT/label/'result.json').read_text())
    assert value['version']=='0.4.0' and len(value['benchmarks'])==1
    b=value['benchmarks'][0]
    assert (b['prompt_size'],b['response_size'],b['concurrency'])==(2048,256,1)
    assert len(b['tg_throughput']['values'])==3
    return dict(tg=b['tg_throughput'],pp=b['pp_throughput'],ttfr=b['ttfr'])


def benchmark(label):
    """Bound stalled pp2048 benchmark runs, not legitimate long-context tests."""
    cmd=['ssh','-F',str(PROJECT/'zima-ssh-config'),'mj-spark-1',
         shlex.join(['bash',str(PROJECT/'scripts/run_speed_bench.sh'),label])]
    progress=BENCH/label/'progress.jsonl'
    size=-1;changed=time.monotonic();started=changed
    with (ROOT/f'{label}-benchmark-client.log').open('w') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            time.sleep(5)
            try:
                observed=int(remote('mj-spark-1',['stat','-c','%s',progress],capture_output=True,text=True,timeout=10).stdout)
            except (subprocess.SubprocessError,ValueError):observed=size
            if observed!=size:size=observed;changed=time.monotonic()
            quiet=time.monotonic()-changed
            if quiet>120 or time.monotonic()-started>1500:
                lab.atomic_json(ROOT/f'{label}-LIVENESS_STOP.json',dict(reason='benchmark_progress_timeout',quiet_seconds=quiet,attempt=current))
                def halt(rank):
                    return remote(f'mj-spark-{rank+1}',['docker','stop','--timeout','20',f'glm53-k3-cand-{current}-rank{rank}'],timeout=60)
                with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(halt,range(3)))
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=10)
                raise RuntimeError(f'benchmark made no progress for {quiet:.1f}s; candidate stopped')
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,cmd)


def run_case(label,channels=1,buffer=524288,eager=1,mtp=0,prefill=128,graph_mode='',spinwait_ms='stock',kv_bytes=1812613120):
    global current
    config=dict(channels=channels,buffer=buffer,eager=eager,mtp=mtp,prefill=prefill,graph_mode=graph_mode,spinwait_ms=spinwait_ms,kv_bytes=kv_bytes)
    stop()
    current='speed-'+label
    since=lab.utc()
    write_status('starting',label=label,config=config)
    env=dict(os.environ,GLM53_RESIDENT_MIN_AVAILABLE_BYTES='12884901888',GLM53_WATCHDOG_RESERVE_GIB='12',
             GLM53_LAZY_K3_UVA='1',GLM53_LAZY_MAX_NUM_BATCHED_TOKENS=str(prefill),GLM53_ENFORCE_EAGER=str(eager),
             GLM53_CUDA_GRAPH_MODE=graph_mode,
             GLM53_SPINWAIT_MS=str(spinwait_ms),
             GLM53_DCP_SIZE='3',GLM53_CP_KV_INTERLEAVE_SIZE='1',GLM53_LAZY_GPU_MEMORY_UTILIZATION='0.12',
             GLM53_SPEC_METHOD='mtp' if mtp else '',GLM53_SPEC_TOKENS=str(mtp) if mtp else '',
             GLM53_SPEC_DRAFT_TP='1' if mtp else '')
    row=dict(label=label,attempt=current,config=config,started_at=since,accepted=False)
    try:
        with (ROOT/f'{label}-launch.log').open('w') as log:
            subprocess.run(['bash',PROJECT/'scripts/start_candidate_cluster_attempt.sh',STAMP,MODEL,current,
                            str(channels),str(buffer),'lazy','256','arena','3600','resident_uva',str(kv_bytes),
                            PACK,'expandable_segments:True','16'],cwd=PROJECT,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3900)
        probes(label+'-before')
        write_status('benchmarking',label=label,config=config)
        metrics=urllib.request.urlopen(lab.ENDPOINT+'/metrics',timeout=10).read()
        (ROOT/f'{label}-metrics-before.txt').write_bytes(metrics)
        benchmark(label)
        row.update(fetch_bench(label))
        probes(label+'-after')
        lab.audit(STAMP,current,since,ROOT/label)
        (ROOT/f'{label}-metrics-after.txt').write_bytes(urllib.request.urlopen(lab.ENDPOINT+'/metrics',timeout=10).read())
        row['accepted']=True
    except Exception as exc:
        row['error']=repr(exc)
        print('CASE_FAILED',label,repr(exc),flush=True)
    # A real kernel fault is a campaign boundary, not permission for blind retry.
    hardware(since,label)
    row['completed_at']=lab.utc()
    rows.append(row)
    lab.atomic_json(ROOT/'RESULTS.json',dict(phase='running',rows=rows))
    print('CASE_RESULT',json.dumps(row),flush=True)
    return row


def best():
    return max((r for r in rows if r['accepted'] and r.get('eligible_for_selection',True)),key=lambda r:r['tg']['mean'])


def main():
    ROOT.mkdir(parents=True,exist_ok=False)
    os.chdir(PROJECT)
    baseline=fetch_bench('baseline')
    rows.append(dict(label='baseline',attempt=current,accepted=True,config=dict(channels=1,buffer=524288,eager=1,mtp=0,prefill=128),**baseline))
    first=run_case('hotpath-eager')
    if not first['accepted']:raise RuntimeError('hotpath baseline failed; investigate before proceeding')
    graph=run_case('hotpath-graphs',eager=0)
    preferred=best()['config'].copy()
    for c,b in [(2,262144),(4,1048576),(8,2097152)]:
        run_case(f'nccl{c}',**dict(preferred,channels=c,buffer=b))
    preferred=best()['config'].copy()
    m1=run_case('mtp1',**dict(preferred,mtp=1))
    if not m1['accepted'] and preferred['eager']==0:
        m1=run_case('mtp1-eager',**dict(preferred,eager=1,mtp=1))
    if m1['accepted']:
        for tokens in (2,3):
            r=run_case(f'mtp{tokens}',**dict(m1['config'],mtp=tokens))
            if not r['accepted']:break
    preferred=best()['config'].copy()
    run_case('prefill256',**dict(preferred,prefill=256))
    chosen=best()
    if chosen['label']=='baseline':
        raise RuntimeError('original runtime won; restore its saved source before final confirmation')
    confirmation=run_case('winner-confirm',**chosen['config'])
    if not confirmation['accepted']:raise RuntimeError('winner confirmation failed; recover explicitly')
    write_status('complete',selected=chosen['label'],loaded=current)
    lab.atomic_json(ROOT/'RESULTS.json',dict(phase='complete',selected=chosen['label'],loaded=current,rows=rows))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if ROOT.exists():write_status('needs_attention',error=repr(exc))
        raise
