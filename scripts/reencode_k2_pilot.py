#!/usr/bin/env python3
"""Spark 2, offline: bounded layer-33 K2 selection and covariance encoding.

Only new /output files are writable. Training code-window rows and evaluation
reasoning-window rows are separate. No full-model KLD claim is made here.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from r7_encoder.trellis import CodecConfig
from r7_encoder.r10_codec import R10TrellisCodec
from tp3k3.encoder import (IdentityR10Codec, IdentityMetric, LocalTensorId,
                          StagedSource, deterministic_vector, physical_slice,
                          pack_unpack_oracle, file_sha256, atomic_json)
from tp3k3.geometry import layer_geometry, GEOMETRY_ID
from refit_k275_scales import collect_rows

LAYER=33
PROJS=('gate_proj','up_proj','down_proj')


def key(expert, projection, rank):
    return f'model.layers.{LAYER}.mlp.experts.{expert}.{projection}.rank{rank}.'


def read_pack(root, expert):
    path=root/f'k3-layer-{LAYER:03d}-expert-{expert:03d}.safetensors'
    with safe_open(path,framework='pt',device='cpu') as f:
        result={name:f.get_tensor(name).clone() for name in f.keys()}
    return result,path


def predict(codec, packed, expert, x, bits):
    result=torch.zeros((len(x),6144),device='cuda',dtype=torch.float32)
    for rank in range(3):
        mats={}
        for projection in PROJS:
            p=key(expert,projection,rank)
            mats[projection]=codec.decode_to_original(packed[p+'trellis'].cuda(),packed[p+'suh'],packed[p+'svh'],bits)
        gate=x@mats['gate_proj']
        up=x@mats['up_proj']
        result.add_((F.silu(gate.clamp(max=10))*up.clamp(-10,10))@mats['down_proj'])
    return result


def error(pred, target, routing, mask):
    return float((((pred[mask]-target[mask])*routing[mask,None]).double().square().sum()).item())


def encode_expert(ic, hc, expert, source_weights, x, routing, train, old2):
    packs={'identity':{},'hessian':{}}
    weighted=x[train]*routing[train,None]
    covariance=weighted.T@weighted/len(weighted)
    oracles=[]
    for rank,g in enumerate(layer_geometry(LAYER)['ranks']):
        hidden={mode:{} for mode in packs}
        for projection in PROJS:
            weight=physical_slice(source_weights[projection],projection,LAYER,rank)
            n,k=weight.shape
            prefix=key(expert,projection,rank)
            tid=LocalTensorId(prefix,k,n,LAYER,expert,projection)
            if old2 is not None:
                su,sv=old2[prefix+'suh'],old2[prefix+'svh']
            else:
                su,sv,_=ic.normalized_vectors(weight,
                    deterministic_vector(k,LAYER,expert,projection,rank,'suh'),
                    deterministic_vector(n,LAYER,expert,projection,rank,'svh'),2)
            for mode,codec in (('identity',ic),('hessian',hc)):
                if mode=='identity':
                    h=IdentityMetric(k)
                elif projection!='down_proj':
                    h=covariance
                else:
                    activation=F.silu(hidden[mode]['gate_proj'][train].clamp(max=10))*hidden[mode]['up_proj'][train].clamp(-10,10)
                    activation=activation*routing[train,None]
                    h=activation.T@activation/len(activation)
                encoded=codec.encode(tensor_id=tid,weight_hf=weight,covariance=h,bits=2,suh=su,svh=sv,
                                     provenance={'pilot':'20260908-layer33','metric':mode,'fit_window':'selection-0002'})
                oracle=pack_unpack_oracle(ic,encoded.trellis,k,n,2)
                decoded=codec.decode_to_original(encoded.trellis.cuda(),encoded.suh,encoded.svh,2)
                assert torch.equal(decoded,encoded.reconstructed_kn.to(decoded.device))
                assert torch.isfinite(decoded).all()
                assert encoded.trellis.numel()*encoded.trellis.element_size()==k*n*2//8
                for suffix,value in (('trellis',encoded.trellis),('suh',encoded.suh),('svh',encoded.svh),
                                     ('mcg',torch.tensor([-877912083],dtype=torch.int32))):
                    packs[mode][prefix+suffix]=value.detach().cpu().contiguous()
                if mode=='identity' and old2 is not None:
                    for suffix in ('trellis','suh','svh','mcg'):
                        assert torch.equal(packs[mode][prefix+suffix],old2[prefix+suffix]), f'identity replay differs {prefix}{suffix}'
                if projection!='down_proj':
                    hidden[mode][projection]=x@decoded
                oracles.append(dict(rank=rank,projection=projection,mode=mode,packed_sha256=encoded.packed_sha256,
                                    unpacked_sha256=oracle,covariance_sha256=encoded.provenance.get('covariance_sha256')))
                del encoded,decoded
            del weight
        del hidden
        hc.clear_caches()
    return packs,oracles


def summarize(rows, original_k2):
    old=set(original_k2)
    pilot={r['expert'] for r in rows}
    count=len(pilot&old)
    scores={}
    for mode in ('identity','hessian'):
        chosen=set(r['expert'] for r in sorted(rows,key=lambda r:(r[mode]['train_error']-r['k3']['train_error'],r['expert']))[:count])
        full=(old-pilot)|chosen
        assert len(full)==64 and len(set(range(256))-full)==192
        baseline=sum(r['identity' if r['expert'] in old else 'k3']['eval_error'] for r in rows)
        candidate=sum(r[mode if r['expert'] in chosen else 'k3']['eval_error'] for r in rows)
        unchanged_assignment=sum(r[mode if r['expert'] in old else 'k3']['eval_error'] for r in rows)
        scores[mode]=dict(pilot_k2=sorted(chosen),full_layer_k2=sorted(full),
                          promoted_to_k3=sorted((old&pilot)-chosen),demoted_to_k2=sorted(chosen-old),
                          baseline_eval_error=baseline,candidate_eval_error=candidate,
                          relative_eval_improvement=1-candidate/baseline,
                          unchanged_assignment_eval_improvement=1-unchanged_assignment/baseline,
                          local_gate_passed=candidate<baseline,
                          scope='sum of routing-weighted individual expert errors, not aggregate MoE error or full-model KLD')
    return scores


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','inventory','k3','k275','activations','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--per-tier',type=int,default=8)
    args=ap.parse_args()
    assert 1<=args.per_tier<=8
    args.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    assert available>=40*(1<<30), 'serving must be stopped'
    source=StagedSource(args.source,args.inventory)
    x,ids,weights=collect_rows(args.activations,LAYER)
    old=set(json.loads((args.k275/'expert-bits.json').read_text())['bits'][str(LAYER)]['k2'])
    old={int(e) for e in old}
    coverage=[]
    for e in range(256):
        positions=(ids==e).any(1).nonzero().flatten()
        fit=int((positions<2048).sum())
        check=int((positions>=2048).sum())
        coverage.append(dict(expert=e,train_rows=fit,eval_rows=check,old_bits=2 if e in old else 3))
    # Coverage only. No numerical error participates in choosing pilot experts.
    selected=[]
    for bits in (2,3):
        eligible=[r for r in coverage if r['old_bits']==bits and r['train_rows']>=20 and r['eval_rows']>=10]
        eligible.sort(key=lambda r:(-min(r['train_rows'],r['eval_rows']),r['expert']))
        assert len(eligible)>=args.per_tier
        selected.extend(r['expert'] for r in eligible[:args.per_tier])
    atomic_json(args.output/'DESIGN.json',dict(layer=LAYER,experts=selected,coverage=coverage,
        train_window='selection-0002',evaluation_window='selection-0003',sigma_reg=.025,
        geometry_id=GEOMETRY_ID,source_revision=source.inventory.get('source_revision'),
        selected_before_errors=True,bit_budget=2.75))
    config=CodecConfig(sigma_reg=.025,
        numeric_core=Path('/workspace/encoder-r10/lineage/encode_tr3_v31.py'),
        numeric_core_sha256='e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032',
        extension=Path('/usr/local/lib/python3.12/dist-packages/exllamav3_ext.cpython-312-aarch64-linux-gnu.so'),
        extension_sha256='7bba0fe1cb7f018bc188cc1df558b6ed5d329c7178093e0d71da7a8d731907d2')
    ic=IdentityR10Codec(config)
    hc=R10TrellisCodec(config,factor_cache_bytes=256<<20)
    rows=[]
    started=time.time()
    with torch.no_grad():
        for e in selected:
            print('EXPERT_START',e,'old_bits',2 if e in old else 3,flush=True)
            pos=(ids==e).any(1).nonzero().flatten()
            xx=x[pos].float().cuda()
            rr=(weights[pos]*(ids[pos]==e)).sum(1).cuda()
            train=(pos<2048).cuda()
            check=~train
            source_weights={}
            provenance=[]
            for projection in PROJS:
                w,record=source.tensor(LAYER,e,projection)
                source_weights[projection]=w
                provenance.append(record)
            gate=xx@source_weights['gate_proj'].float().cuda().T
            up=xx@source_weights['up_proj'].float().cuda().T
            target=(F.silu(gate)*up)@source_weights['down_proj'].float().cuda().T
            assert torch.isfinite(target).all()
            del gate,up
            old3,path3=read_pack(args.k3,e)
            old2,path2=read_pack(args.k275,e) if e in old else (None,None)
            packs,oracles=encode_expert(ic,hc,e,source_weights,xx,rr,train,old2)
            row=dict(expert=e,old_bits=2 if e in old else 3,train_rows=int(train.sum()),eval_rows=int(check.sum()),
                     source=provenance,oracles=oracles,identity_replay_exact=e in old,k3_sha256=file_sha256(path3))
            for mode,packed,bits in [('k3',old3,3)]+[(m,p,2) for m,p in packs.items()]:
                pred=predict(hc,packed,e,xx,bits)
                repeat=predict(hc,packed,e,xx,bits)
                assert torch.equal(pred,repeat), f'local output not deterministic {e} {mode}'
                row[mode]=dict(train_error=error(pred,target,rr,train),eval_error=error(pred,target,rr,check),
                               repeat_exact=True,packed_bytes=sum(t.numel()*t.element_size() for t in packed.values()))
                if mode!='k3':
                    folder=args.output/mode
                    folder.mkdir(exist_ok=True)
                    path=folder/f'k3-layer-{LAYER:03d}-expert-{e:03d}.safetensors'
                    save_file(packed,str(path),metadata={'geometry_id':GEOMETRY_ID,'pilot_mode':mode,'bits':'2'})
                    row[mode]['file_sha256']=file_sha256(path)
                    path.chmod(0o644)
                del pred,repeat
            assert row['identity']['packed_bytes']==row['hessian']['packed_bytes']
            rows.append(row)
            atomic_json(args.output/f'expert-{e:03d}.json',row)
            atomic_json(args.output/'PROGRESS.json',dict(completed=len(rows),total=len(selected),last_expert=e,elapsed_seconds=time.time()-started))
            print('EXPERT_DONE',e,'identity_eval',row['identity']['eval_error'],'hessian_eval',row['hessian']['eval_error'],flush=True)
            del source_weights,target,old3,old2,packs,xx,rr
            torch.cuda.empty_cache()
            for r in provenance:
                with (args.source/r['shard']).open('rb') as f:
                    os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    summary=summarize(rows,old)
    atomic_json(args.output/'RESULTS.json',dict(passed=True,phase='complete',experts=len(rows),results=summary,
         elapsed_seconds=time.time()-started,cuda_peak_bytes=torch.cuda.max_memory_allocated(),
         full_model_kld_measured=False,native_fused_parity_measured=False,
         production_promotion=False,all_experts_and_routing_retained=True))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
