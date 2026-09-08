#!/usr/bin/env python3
"""Zima: bounded interaction tests after the primary speed matrix completes."""
import json
import speed_campaign as lab


def main():
    state=json.loads((lab.ROOT/'RESULTS.json').read_text())
    assert state['phase']=='complete', 'primary campaign must finish first'
    lab.rows=state['rows']
    lab.current=state['loaded']
    prior_best=lab.best()
    resized=lab.run_case('kv1g',**dict(prior_best['config'],kv_bytes=1073741824))
    cache_bytes=1073741824 if resized['accepted'] else 1812613120
    if resized['accepted']:
        lab.run_case('prefill256-kv1g',**dict(prior_best['config'],prefill=256,kv_bytes=cache_bytes))
    if resized['accepted'] and any(r['accepted'] and r['config']['mtp']==3 for r in lab.rows):
        seed=max((r for r in lab.rows if r['accepted'] and r['config'].get('kv_bytes')==cache_bytes),key=lambda r:r['tg']['mean'])
        lab.run_case('mtp4-kv1g',**dict(seed['config'],mtp=4,kv_bytes=cache_bytes))
    # The updated launcher must already have been atomically deployed.
    no_mtp=max((r for r in lab.rows if r['accepted'] and not r['config']['mtp']),key=lambda r:r['tg']['mean'])
    lab.run_case('decode-only-graphs',**dict(no_mtp['config'],eager=0,graph_mode='FULL_DECODE_ONLY',kv_bytes=cache_bytes))
    mtp=[r for r in lab.rows if r['accepted'] and r['config']['mtp']]
    if mtp:
        chosen=max(mtp,key=lambda r:r['tg']['mean'])
        if chosen['config']['channels']!=4:
            lab.run_case('mtp-nccl4-interaction',**dict(chosen['config'],channels=4,buffer=1048576,kv_bytes=cache_bytes))
        mtp=[r for r in lab.rows if r['accepted'] and r['config']['mtp']]
        chosen=max(mtp,key=lambda r:r['tg']['mean'])
        lab.run_case('mtp-decode-only-graphs',**dict(chosen['config'],eager=0,graph_mode='FULL_DECODE_ONLY',kv_bytes=cache_bytes))
    winner=lab.best()
    lab.run_case('spinwait16',**dict(winner['config'],spinwait_ms='16',kv_bytes=cache_bytes))
    winner=lab.best()
    assert winner['label']!='baseline', 'restore unmodified source if original baseline wins'
    confirmed=lab.run_case('final-confirm',**winner['config'])
    if not confirmed['accepted']:raise RuntimeError('final confirmation failed')
    lab.write_status('complete',selected=winner['label'],loaded=lab.current)
    lab.lab.atomic_json(lab.ROOT/'RESULTS.json',dict(phase='complete',selected=winner['label'],loaded=lab.current,rows=lab.rows))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        lab.write_status('needs_attention',error=repr(exc))
        raise
