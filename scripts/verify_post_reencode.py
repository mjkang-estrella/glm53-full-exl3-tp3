#!/usr/bin/env python3
"""Zima: real output and unchanged-weight repeat capture after restoration."""
import json
from pathlib import Path
import quality_scale_campaign as lab
from long_context_probe import raw_native_completion

ROOT=Path('/mnt/unas-models/ZAI/GLM-5.3-EXL3-archive-notes-20260908')
ATTEMPT='k275-post-reencode-v1'
STAMP='20260907T224500Z'


def main():
    started=lab.utc()
    for rank,node in enumerate(lab.NODES):
        path=lab.STATE/f'real-test/{STAMP}/candidate/attempts/{ATTEMPT}/rank-{rank}/capture/QUALITY_CONTROL.json'
        lab.ssh(node,['test','!','-e',path])
    probes=[]
    for prompt,expected in [('What is the capital of France? Reply with only the city name.','Paris'),
                            ('Calculate 17 * 19. Reply with only the integer answer.','323'),
                            ('Return exactly this JSON object and nothing else: {"answer": 42}',{'answer':42})]:
        response=raw_native_completion(lab.ENDPOINT,prompt,max_tokens=64,timeout=180)
        text=response['content'].strip()
        try:actual=json.loads(text) if isinstance(expected,dict) else text
        except ValueError:actual=None
        passed=actual==expected and response['finish_reason']=='stop'
        probes.append(dict(prompt=prompt,content=text,passed=passed))
        assert passed, probes[-1]
        print('GENERATION_PASS',json.dumps(probes[-1]),flush=True)
    lab.EXPECTED_ROWS=2063
    scores=[]
    for number in (1,2):
        folder=ROOT/f'unchanged-baseline-repeat-{number}'
        folder.mkdir(exist_ok=False)
        lab.capture_window(STAMP,ATTEMPT,'confirmation-0000',folder,folder/'capture.log')
        score=lab.score_windows(folder,['confirmation-0000'],folder/'SCORE.json')
        scores.append(dict(run=number,kld=score['kld'],top1=score['top1'],positions=score['positions']))
        print('UNCHANGED_REPEAT',json.dumps(scores[-1]),flush=True)
    lab.audit(STAMP,ATTEMPT,started,ROOT)
    receipt=dict(passed=True,attempt=ATTEMPT,probes=probes,unchanged_weight_repeats=scores,
                 kld_spread=abs(scores[1]['kld']-scores[0]['kld']),
                 scope='one 2047-position public window; original K275 weights, no quality-control patch',
                 completed_at=lab.utc())
    lab.atomic_json(ROOT/'POST_RESTORE.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
