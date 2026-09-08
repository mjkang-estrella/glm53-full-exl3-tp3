#!/usr/bin/env python3
"""Zima: one memory-motivated graph/MTP retry after the follow-up finishes."""
import json
import subprocess
import sys
import speed_campaign as lab


def main():
    state=json.loads((lab.ROOT/'RESULTS.json').read_text())
    status=json.loads((lab.ROOT/'STATUS.json').read_text())
    assert state['phase']=='complete' or status['phase']=='needs_attention'
    lab.rows=state['rows'];lab.current=state.get('loaded') or status['current']
    for row in lab.rows:
        if row['config']['prefill']>128 and row['config']['mtp']:
            row['eligible_for_selection']=False
            row['disqualified_reason']='larger-prefill MTP family excluded after repeated liveness failures; measured values retained'
    lab.lab.atomic_json(lab.ROOT/'RESULTS.json',dict(phase='recovering',rows=lab.rows))
    prior=lab.best()
    candidate=lab.run_case('mtp3-graphs-p128',channels=4,buffer=1048576,eager=0,mtp=3,
                           prefill=128,graph_mode='FULL_DECODE_ONLY',kv_bytes=1073741824)
    chosen=max((r for r in lab.rows if r['accepted'] and r.get('eligible_for_selection',True)
                and r['config'].get('kv_bytes')==1073741824),key=lambda r:r['tg']['mean'])
    if chosen['label']=='baseline':raise RuntimeError('original source restoration required')
    final=lab.run_case('qualified-final',**chosen['config'])
    assert final['accepted'], 'final confirmation failed'
    lab.write_status('long_context_qualification',label='qualified-final')
    subprocess.run([sys.executable,lab.PROJECT/'scripts/long_context_probe.py',
                    '--output',lab.ROOT/'FINAL_LONG_CONTEXT.json','--target-tokens','30000',
                    '--mode','raw-native','--max-tokens','64','--timeout-seconds','900'],
                   cwd=lab.PROJECT,check=True,timeout=950)
    long_result=json.loads((lab.ROOT/'FINAL_LONG_CONTEXT.json').read_text())
    assert long_result['passed'], 'long-context gate failed'
    lab.lab.audit(lab.STAMP,lab.current,final['started_at'],lab.ROOT/'qualified-final')
    lab.write_status('complete',selected=chosen['label'],loaded=lab.current)
    lab.lab.atomic_json(lab.ROOT/'RESULTS.json',dict(phase='complete',selected=chosen['label'],loaded=lab.current,rows=lab.rows))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        lab.write_status('needs_attention',error=repr(exc))
        raise
