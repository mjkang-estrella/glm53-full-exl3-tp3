#!/usr/bin/env python3
"""Spark-only bounded BF16 expert comparison and same-size FP16 scale refit.

Run after the TP3 service stops, with one source layer and captured inputs.
Fit every fifth captured row as validation, chosen before fitting. K2 bit
assignments, trellises, and all other expert components stay fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from r7_encoder.trellis import CodecConfig, Exl3TrellisCodec
from tp3k3.encoder import StagedSource
from tp3k3.geometry import layer_geometry


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    os.chmod(temp,0o644)
    os.replace(temp,path)


def digest(path):
    result=hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda:stream.read(8<<20),b""):
            result.update(data)
    return result.hexdigest()


def collect_rows(folder,layer):
    x,ids,weights=[],[],[]
    row=0
    for path in sorted(folder.glob(f"layer-{layer:03d}-rows-*.pt")):
        data=torch.load(path,map_location="cpu",weights_only=True)
        if data["first_row"]!=row:
            raise ValueError("activation row order differs")
        x.append(data["x"]); ids.append(data["ids"]); weights.append(data["weights"])
        row+=data["x"].shape[0]
    if row!=4096:
        raise ValueError(f"expected 4096 calibration rows, got {row}")
    return torch.cat(x),torch.cat(ids),torch.cat(weights)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--layer",type=int,default=33)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--inventory",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--activations",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-experts",type=int,default=64)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    available=int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))*1024
    if available<40*1024**3:
        raise RuntimeError("refit requires stopped serving and 40 GiB available")
    source=StagedSource(args.source,args.inventory)
    x,ids,weights=collect_rows(args.activations,args.layer)
    selected=json.loads((args.checkpoint/"expert-bits.json").read_text())["bits"][str(args.layer)]["k2"]
    selected=sorted(int(e) for e in selected)[:args.max_experts]
    codec=Exl3TrellisCodec(CodecConfig(
        numeric_core=Path("/workspace/encoder-r10/lineage/encode_tr3_v31.py"),
        numeric_core_sha256="e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032",
        extension=Path("/usr/local/lib/python3.12/dist-packages/exllamav3_ext.cpython-312-aarch64-linux-gnu.so"),
        extension_sha256="7bba0fe1cb7f018bc188cc1df558b6ed5d329c7178093e0d71da7a8d731907d2"))
    rows=[]
    gains={}
    rankmap=layer_geometry(args.layer)["ranks"]
    started=time.time()
    with torch.inference_mode():
        for expert in selected:
            chosen=(ids==expert)
            pos=chosen.any(dim=1).nonzero().flatten()
            # Keep sampling bounded without choosing rows by measured loss.
            if len(pos)>512:
                pos=pos[torch.linspace(0,len(pos)-1,512).long()]
            valid=(pos%5==0)
            if len(pos)<30 or valid.sum()<6 or (~valid).sum()<20:
                rows.append({"expert":expert,"rows":len(pos),"skipped":"insufficient routed rows"})
                continue
            inputs=x[pos].float().cuda()
            sample_weights=(weights[pos]*chosen[pos]).sum(dim=1).cuda()
            fit=~valid.cuda(); valid=valid.cuda()
            wg,sg=source.tensor(args.layer,expert,"gate_proj")
            wu,su=source.tensor(args.layer,expert,"up_proj")
            wd,sd=source.tensor(args.layer,expert,"down_proj")
            gate=inputs@wg.float().cuda().T
            up=inputs@wu.float().cuda().T
            target=(F.silu(gate)*up)@wd.float().cuda().T
            fp32_finite=bool(torch.isfinite(target).all())
            if not fp32_finite:
                raise ValueError("non-finite BF16-weight expert reference")
            reference_clipped=(F.silu(gate.clamp(max=10))*up.clamp(-10,10))@wd.float().cuda().T
            native_clipping_rel=float(((reference_clipped-target).square().sum()/target.square().sum().clamp_min(1e-30)).item())
            del wg,wu,wd,gate,up,reference_clipped
            path=args.checkpoint/f"k3-layer-{args.layer:03d}-expert-{expert:03d}.safetensors"
            with safe_open(path,framework="pt",device="cpu") as handle:
                packed={key:handle.get_tensor(key).clone() for key in handle.keys()}
            preds={10.0:torch.zeros_like(target),math.inf:torch.zeros_like(target)}
            for rank in range(3):
                outputs={}
                for projection in ("gate_proj","up_proj"):
                    prefix=f"model.layers.{args.layer}.mlp.experts.{expert}.{projection}.rank{rank}."
                    matrix=codec.decode_to_original(packed[prefix+"trellis"].cuda(),packed[prefix+"suh"],packed[prefix+"svh"],2)
                    outputs[projection]=inputs@matrix
                    del matrix
                prefix=f"model.layers.{args.layer}.mlp.experts.{expert}.down_proj.rank{rank}."
                matrix=codec.decode_to_original(packed[prefix+"trellis"].cuda(),packed[prefix+"suh"],packed[prefix+"svh"],2)
                for limit in preds:
                    gate=outputs["gate_proj"].clamp(max=limit)
                    up=outputs["up_proj"].clamp(-limit,limit)
                    preds[limit].add_((F.silu(gate)*up)@matrix)
                del matrix,outputs,gate,up
            tw=target*sample_weights[:,None]
            summary={"expert":expert,"rows":len(pos),"fit_rows":int(fit.sum()),
                     "validation_rows":int(valid.sum()),"source_sha256s":{s["name"]:s["sha256"] for s in (sg,su,sd)},
                     "baseline_expert_sha256":digest(path),"bf16_clip10_relative_mse":native_clipping_rel}
            for limit,pred in preds.items():
                pw=pred*sample_weights[:,None]
                alpha=float(((pw[fit]*tw[fit]).sum()/pw[fit].square().sum().clamp_min(1e-30)).clamp(.9,1.1).item())
                baseline=float((pw[valid]-tw[valid]).square().sum().item())
                target_norm=float(tw[valid].square().sum().item())
                best=(baseline,0.0,1.0)
                for shrink in (.25,.5,1.0):
                    gain=1.0+shrink*(alpha-1.0)
                    err=float((pw[valid]*gain-tw[valid]).square().sum().item())
                    if err<best[0]:
                        best=(err,shrink,gain)
                tag="clip10" if limit==10.0 else "unclamped"
                summary[tag]={"alpha_fit":alpha,"selected_gain":best[2],"shrinkage":best[1],
                              "baseline_validation_mse_sum":baseline,"refit_validation_mse_sum":best[0],
                              "target_validation_norm":target_norm}
                gains.setdefault(tag,{})[str(expert)]=best[2]
            rows.append(summary)
            atomic_json(args.output/"PROGRESS.json",{"layer":args.layer,"completed_experts":len(rows),
                        "selected_experts":len(selected),"last":summary,"elapsed_seconds":time.time()-started})
            print("EXPERT_REFIT",expert,"rows",len(pos),"clip10gain",summary["clip10"]["selected_gain"],flush=True)
            del packed,inputs,target,preds,tw,pw,pred
            torch.cuda.empty_cache()
            for record in (sg,su,sd):
                with (args.source/record["shard"]).open("rb") as handle:
                    os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    aggregate={}
    for tag in gains:
        chosen=[row[tag] for row in rows if tag in row]
        baseline=sum(row["baseline_validation_mse_sum"] for row in chosen)
        improved=sum(row["refit_validation_mse_sum"] for row in chosen)
        aggregate[tag]={"baseline_mse":baseline,"refit_mse":improved,
                        "relative_improvement":1-improved/max(baseline,1e-30)}
    atomic_json(args.output/"SCALE_FIT.json",{"schema":"glm53.activation-scale-fit.v1","layer":args.layer,
                "bits":2,"gains":gains,"experts":rows,"aggregate":aggregate,
                "elapsed_seconds":time.time()-started,"cuda_peak_bytes":torch.cuda.max_memory_allocated(),
                "source_revision":source.inventory.get("source_revision"),"passed":True,
                "metric_scope":"local expert output error; full-model KL still required"})


if __name__=="__main__":
    main()
