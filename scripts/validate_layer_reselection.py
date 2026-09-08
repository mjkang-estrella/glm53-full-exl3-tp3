#!/usr/bin/env python3
"""Spark-only summed layer-33 MoE check for already frozen pilot selections."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from r7_encoder.trellis import CodecConfig, Exl3TrellisCodec
from tp3k3.encoder import StagedSource, atomic_json, file_sha256
from reencode_k2_pilot import LAYER, PROJS, read_pack, predict
from refit_k275_scales import collect_rows


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','inventory','k3','k275','activations','output','pilot'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    selection=json.loads((args.pilot/'RESULTS.json').read_text())
    assert selection['passed'] and selection['phase']=='complete'
    old={int(e) for e in json.loads((args.k275/'expert-bits.json').read_text())['bits'][str(LAYER)]['k2']}
    choices={mode:set(row['full_layer_k2']) for mode,row in selection['results'].items()}
    choices['hessian_fixed']=set(old)
    assert len(old)==64 and all(len(s)==64 for s in choices.values())
    pilot=set(json.loads((args.pilot/'DESIGN.json').read_text())['experts'])
    source=StagedSource(args.source,args.inventory)
    x,ids,weights=collect_rows(args.activations,LAYER)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    assert available>=40*(1<<30)
    codec=Exl3TrellisCodec(CodecConfig(
        numeric_core=Path('/workspace/encoder-r10/lineage/encode_tr3_v31.py'),
        numeric_core_sha256='e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032',
        extension=Path('/usr/local/lib/python3.12/dist-packages/exllamav3_ext.cpython-312-aarch64-linux-gnu.so'),
        extension_sha256='7bba0fe1cb7f018bc188cc1df558b6ed5d329c7178093e0d71da7a8d731907d2'))
    totals={name:torch.zeros((4096,6144),dtype=torch.float32,device='cuda') for name in ('reference','baseline',*choices)}
    started=time.time()
    seen=0
    with torch.no_grad():
        for e in range(256):
            chosen=ids==e
            pos=chosen.any(1).nonzero().flatten()
            if not len(pos):continue
            gpu_pos=pos.cuda()
            xx=x[pos].float().cuda()
            rr=(weights[pos]*chosen[pos]).sum(1).cuda()
            ws={}
            records=[]
            for p in PROJS:
                w,record=source.tensor(LAYER,e,p)
                ws[p]=w
                records.append(record)
            ref=(F.silu(xx@ws['gate_proj'].float().cuda().T)*(xx@ws['up_proj'].float().cuda().T))@ws['down_proj'].float().cuda().T
            assert torch.isfinite(ref).all()
            totals['reference'].index_copy_(0,gpu_pos,totals['reference'].index_select(0,gpu_pos)+ref*rr[:,None])
            packed,path=read_pack(args.k275,e)
            baseline=predict(codec,packed,e,xx,2 if e in old else 3)
            totals['baseline'].index_copy_(0,gpu_pos,totals['baseline'].index_select(0,gpu_pos)+baseline*rr[:,None])
            for mode,k2set in choices.items():
                if e not in pilot or (e not in k2set and e not in old):
                    output=baseline
                elif e not in k2set:
                    alt,_=read_pack(args.k3,e)
                    output=predict(codec,alt,e,xx,3)
                    del alt
                else:
                    artifact_mode='hessian' if mode=='hessian_fixed' else mode
                    alt,path=read_pack(args.pilot/artifact_mode,e)
                    receipt=json.loads((args.pilot/f'expert-{e:03d}.json').read_text())
                    assert file_sha256(path)==receipt[artifact_mode]['file_sha256']
                    output=predict(codec,alt,e,xx,2)
                    del alt
                totals[mode].index_copy_(0,gpu_pos,totals[mode].index_select(0,gpu_pos)+output*rr[:,None])
            seen+=1
            if e%16==0:
                atomic_json(args.output/'PROGRESS.json',dict(experts_visited=e+1,experts_with_rows=seen,total=256,elapsed_seconds=time.time()-started))
                print('LAYER_CHECK',e+1,flush=True)
            del ws,ref,packed,baseline,output,xx,rr
            for record in records:
                with (args.source/record['shard']).open('rb') as f:
                    os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    scores={}
    for mode in ('baseline',*choices):
        scores[mode]={}
        for tag,start,end in [('train',0,2048),('evaluation',2048,4096)]:
            ref=totals['reference'][start:end].double()
            diff=totals[mode][start:end].double()-ref
            scores[mode][tag]=dict(squared_error=float(diff.square().sum()),reference_norm=float(ref.square().sum()),
                                  relative_mse=float(diff.square().sum()/ref.square().sum()))
    for mode in choices:
        scores[mode]['evaluation_improvement']=1-scores[mode]['evaluation']['squared_error']/scores['baseline']['evaluation']['squared_error']
    result=dict(passed=True,phase='complete',layer=LAYER,experts=256,experts_with_rows=seen,
                scores=scores,selection_sha256=file_sha256(args.pilot/'RESULTS.json'),
                cuda_peak_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.time()-started,
                scope='summed routed-expert output only; shared expert and attention unchanged, no full-model KLD',
                inputs_from='original K275 model activation capture',production_promotion=False)
    atomic_json(args.output/'RESULTS.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
