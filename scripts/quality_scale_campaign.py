#!/usr/bin/env python3
"""Zima controller for bounded, same-bitrate expert output-scale experiments.

Uses two distinct public selection windows for tuning; confirmation windows
are evaluated once only after choosing a candidate. No final-logit scaling.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

from score_matched_logits import teacher_memmap, candidate_parts, sha256_file, atomic_json

PROJECT = Path("/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3")
STATE = Path("/home/mj-kang/Dev/state/glm53-full-exl3-tp3")
MANIFEST = STATE / "real-test/20260906T034145Z/teacher-reference-manifests/reference-full-panel/aggregate-manifest.json"
REFERENCE = Path("/mnt/unas-models/ZAI/GLM-5.3-BF16-full-logits-bounded-reference-427368f1")
DATASET_REVISION = "427368f12a4bdc21668bc4171ce0dc54f8990200"
NODES = ["mj-spark-1", "mj-spark-2", "mj-spark-3"]
WINDOWS = ["selection-0000", "selection-0001"]
CONFIRM = [f"confirmation-{i:04d}" for i in range(4)]
MODEL = "GLM-5.3-K3-TP3-CANDIDATE"
ENDPOINT = "http://192.168.0.238:8893"
EXPECTED_ROWS = 2055


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def command(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, timeout=kwargs.pop("timeout", 120), **kwargs)


def ssh(node, argv, **kwargs):
    return command(["ssh", "-F", PROJECT / "zima-ssh-config", node,
                    shlex.join([str(x) for x in argv])], **kwargs)


def download_reference(folder):
    manifest = json.loads(MANIFEST.read_text())
    records = {r["window_id"]: r for r in manifest["logit_files"]}
    tasks = []
    for window in WINDOWS:
        row = records[window]
        tasks += [(row["path"], row["sha256"]),
                  (f"reference-full-panel/calibration/panel-v1/arrays/{window}.tokens.npy",
                   row["token_ids_sha256"])]
    for window in ("selection-0002", "selection-0003"):
        tasks.append((f"reference-full-panel/calibration/panel-v1/arrays/{window}.tokens.npy",
                      records[window]["token_ids_sha256"]))

    def retrieve(task):
        rel, expected = task
        dest = REFERENCE / rel
        if dest.is_file():
            if sha256_file(dest) != expected:
                raise ValueError(f"existing teacher file hash differs: {rel}")
            return {"path": rel, "sha256": expected, "reused": True}
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + ".quality-download")
        url = f"https://huggingface.co/datasets/{manifest['repo_id']}/resolve/{DATASET_REVISION}/{rel}"
        command(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "20",
                 "--max-time", "1800", "--silent", "--show-error", "--output", temp, url], timeout=1900)
        actual = sha256_file(temp)
        if actual != expected:
            raise ValueError(f"downloaded teacher hash differs: {rel}")
        os.replace(temp, dest)
        print("TEACHER_VERIFIED", rel, flush=True)
        return {"path": rel, "sha256": actual, "reused": False}

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(retrieve, tasks))
    atomic_json(folder / "development-reference.json", {
        "dataset_revision": DATASET_REVISION, "files": rows,
        "development_windows": WINDOWS, "reserved_confirmation_windows": CONFIRM,
        "passed": True, "completed_at": utc(),
    })


def score_windows(capture, windows, output):
    """FP64 stable full-vocabulary KL, blocked to avoid slow logaddexp folds."""
    manifest = json.loads(MANIFEST.read_text())
    mapping = {r["window_id"]: r for r in manifest["logit_files"]}
    rows = []
    for window in windows:
        ref = mapping[window]
        teacher = teacher_memmap(REFERENCE / ref["path"])
        receipt, parts = candidate_parts(capture / window)
        if receipt["token_sha256"] != ref["token_ids_sha256"]:
            raise ValueError("candidate/teacher token identity mismatch")
        cursor = 0
        per_token = []
        agreements = []
        for part in parts:
            for begin in range(0, part.shape[0], 8):
                count = min(8, part.shape[0] - begin)
                t = np.array(teacher[cursor:cursor+count], dtype=np.float64)
                c = np.array(part[begin:begin+count], dtype=np.float64)
                if not np.isfinite(t).all() or not np.isfinite(c).all():
                    raise ValueError("non-finite logits")
                ta, ca = t.argmax(axis=1), c.argmax(axis=1)
                t -= t.max(axis=1, keepdims=True)
                c -= c.max(axis=1, keepdims=True)
                t -= np.log(np.exp(t).sum(axis=1, keepdims=True))
                c -= np.log(np.exp(c).sum(axis=1, keepdims=True))
                kl = (np.exp(t) * (t - c)).sum(axis=1)
                if (kl < -1e-9).any():
                    raise ValueError("negative KL exceeds numeric tolerance")
                per_token.extend(kl.tolist())
                agreements.extend((ta == ca).tolist())
                cursor += count
        if cursor != 2047:
            raise ValueError("scoring row count differs")
        rows.append({"window_id": window, "domain": ref["domain"],
                     "kld": float(np.mean(per_token)), "top1": float(np.mean(agreements)),
                     "per_token_kld": per_token, "per_token_agreement": agreements})
    result = {"schema": "glm53.quality-score.v1", "passed": True,
              "kld": float(np.mean([r["kld"] for r in rows])),
              "top1": float(np.mean([r["top1"] for r in rows])),
              "positions": len(rows)*2047, "windows": rows, "completed_at": utc()}
    atomic_json(output, result)
    return result


def set_control(stamp, attempt, value, root):
    path = root / f"control-{value['id']}.json"
    atomic_json(path, value)
    digest = sha256_file(path)
    for rank, node in enumerate(NODES):
        capture = STATE / f"real-test/{stamp}/candidate/attempts/{attempt}/rank-{rank}/capture"
        ssh(node, ["mkdir", "-p", capture])
        command(["scp", "-q", "-F", PROJECT/"zima-ssh-config", path,
                 f"{node}:{capture}/QUALITY_CONTROL.next"])
        ssh(node, ["mv", capture/"QUALITY_CONTROL.next", capture/"QUALITY_CONTROL.json"])
    return digest


def capture_window(stamp, attempt, window, evidence, log):
    if (evidence / window / "CAPTURE_COMPLETE.json").exists():
        return
    with log.open("w") as stream:
        command([sys.executable, PROJECT/"scripts/capture_matched_logits.py",
                 "--stamp", stamp, "--attempt", attempt, "--window-id", window,
                 "--tokens", REFERENCE/f"reference-full-panel/calibration/panel-v1/arrays/{window}.tokens.npy",
                 "--output-root", evidence, "--timeout", "900", "--endpoint", ENDPOINT,
                 "--ssh-config", PROJECT/"zima-ssh-config", "--node", "mj-spark-1",
                 "--scheduler-singletons", "--expected-rows", str(EXPECTED_ROWS)],
                timeout=1100, stdout=stream, stderr=subprocess.STDOUT)


def collect_calibration(args):
    config = {"schema":"glm53.quality-control.v1", "id":"capture-independent-inputs",
              "gains":{"2":1.0,"3":1.0},
              "capture":{"id":"independent-code-reasoning-v1","layers":[4,5,33],"rows":4096}}
    set_control(args.stamp,args.attempt,config,args.root)
    requests=[]
    for window in ("selection-0002","selection-0003"):
        tokens=REFERENCE/f"reference-full-panel/calibration/panel-v1/arrays/{window}.tokens.npy"
        payload={"model":MODEL,"prompt":np.load(tokens).tolist(),"add_special_tokens":False,
                 "temperature":0,"seed":20260908,"max_tokens":1,"ignore_eos":True}
        request=urllib.request.Request(ENDPOINT+"/v1/completions",data=json.dumps(payload).encode(),
                                      headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(request,timeout=600) as response:
            result=json.load(response)
        requests.append({"window":window,"token_sha256":sha256_file(tokens),"response":result})
        print("CALIBRATION_CAPTURE",window,flush=True)
    receipts=[]
    for layer in (4,5,33):
        path=STATE/f"real-test/{args.stamp}/candidate/attempts/{args.attempt}/rank-0/capture/quality-activations/independent-code-reasoning-v1/layer-{layer:03d}-STATUS.json"
        receipt=json.loads(ssh(NODES[0],["cat",path],capture_output=True,text=True).stdout)
        if receipt.get("rows")!=4096 or not receipt.get("passed"):
            raise ValueError("calibration capture did not reach its row count")
        receipts.append(receipt)
    set_control(args.stamp,args.attempt,{"schema":"glm53.quality-control.v1",
                "id":"identity-after-capture","gains":{"2":1.0,"3":1.0}},args.root)
    atomic_json(args.root/"CALIBRATION_CAPTURE.json", {"passed":True,"requests":requests,
                "receipts":receipts,"completed_at":utc()})


def audit(stamp, attempt, started, folder):
    for rank, node in enumerate(NODES):
        container = f"glm53-k3-cand-{attempt}-rank{rank}"
        r = ssh(node, ["docker", "inspect", "-f", "{{.State.Running}} {{.State.OOMKilled}}", container], capture_output=True, text=True)
        if r.stdout.strip() != "true false":
            raise RuntimeError(f"candidate rank {rank} not healthy")
        stop = STATE/f"real-test/{stamp}/candidate/attempts/{attempt}/rank-{rank}/watchdog/STOP.json"
        ssh(node, ["test", "!", "-e", stop])
    command(["python3", PROJECT/"scripts/kernel_audit.py", "--since", started,
             "--until", utc(), "--ssh-config", PROJECT/"zima-ssh-config",
             "--output", folder/"kernel-audit.json"], timeout=90)


def main():
    global EXPECTED_ROWS
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default="20260907T224500Z")
    parser.add_argument("--attempt", default="k275-quality-v1")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--expected-rows", type=int, default=2055)
    parser.add_argument("--expert-fit", type=Path)
    args = parser.parse_args()
    EXPECTED_ROWS = args.expected_rows
    args.root.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    download_reference(args.root)
    if args.download_only:
        return
    if args.capture_only:
        collect_calibration(args)
        return
    start = utc()
    # The search order and bounds are fixed before seeing development scores.
    variants = [("baseline",1.0,1.0,10.0), ("k2-095",.95,1.0,10.0),
                ("k2-0975",.975,1.0,10.0), ("k2-1025",1.025,1.0,10.0),
                ("k2-105",1.05,1.0,10.0), ("k3-099",1.0,.99,10.0),
                ("k3-101",1.0,1.01,10.0), ("clamp20",1.0,1.0,20.0),
                ("clamp40",1.0,1.0,40.0), ("unclamped",1.0,1.0,"unclamped")]
    expert_gains = {}
    if args.expert_fit:
        fit = json.loads(args.expert_fit.read_text())
        if not fit.get("passed"):
            raise ValueError("activation scale fit did not pass")
        expert_gains = {str(fit["layer"]):fit["gains"]["clip10"]}
        variants.append(("refit-layer33",1.0,1.0,10.0))
    results = []
    try:
        for name, k2, k3, limit in variants:
            folder = args.evidence/name
            folder.mkdir(exist_ok=True)
            atomic_json(args.root/"STATUS.json", {"phase":"development", "variant":name,
                        "updated_at":utc(), "results":results})
            control={"schema":"glm53.quality-control.v1", "id":name,
                     "gains":{"2":k2,"3":k3},"swiglu_limit":limit}
            if name == "refit-layer33":
                control["expert_gains"] = expert_gains
            digest=set_control(args.stamp,args.attempt,control,args.root)
            for window in WINDOWS:
                print("CAPTURE",name,window,utc(),flush=True)
                capture_window(args.stamp,args.attempt,window,folder,folder/f"{window}.log")
            for rank,node in enumerate(NODES):
                ack=STATE/f"real-test/{args.stamp}/candidate/attempts/{args.attempt}/rank-{rank}/capture/QUALITY_ACK-rank{rank}.json"
                response=ssh(node,["cat",ack],capture_output=True,text=True)
                if json.loads(response.stdout)["control_sha256"]!=digest:
                    raise ValueError("rank did not apply the intended control")
            score=score_windows(folder,WINDOWS,folder/"SCORE.json")
            audit(args.stamp,args.attempt,start,folder)
            row={"variant":name,"gains":control["gains"],"swiglu_limit":limit,
                 "expert_gains":control.get("expert_gains",{}),"kld":score["kld"],"top1":score["top1"]}
            results.append(row)
            print("RESULT",json.dumps(row),flush=True)
        best=min(results,key=lambda r:r["kld"])
        # Evaluate the selected variant and an identity control under the same
        # fresh process on confirmation data; no further tuning uses this set.
        confirmation=[]
        names=["baseline"] if best["variant"]=="baseline" else ["baseline",best["variant"]]
        for name in names:
            row=next(r for r in results if r["variant"]==name)
            identity="confirm-"+name
            folder=args.evidence/identity
            folder.mkdir(exist_ok=True)
            set_control(args.stamp,args.attempt,{"schema":"glm53.quality-control.v1",
                        "id":identity,"gains":row["gains"],"swiglu_limit":row["swiglu_limit"],
                        "expert_gains":row.get("expert_gains",{})},args.root)
            atomic_json(args.root/"STATUS.json", {"phase":"confirmation","variant":name,
                        "updated_at":utc(),"results":results,"selected":best})
            for window in CONFIRM:
                print("CONFIRM",name,window,utc(),flush=True)
                capture_window(args.stamp,args.attempt,window,folder,folder/f"{window}.log")
            score=score_windows(folder,CONFIRM,folder/"SCORE.json")
            audit(args.stamp,args.attempt,start,folder)
            confirmation.append({"variant":name,"kld":score["kld"],"top1":score["top1"]})
        accepted=(len(confirmation)>1 and confirmation[1]["kld"]<confirmation[0]["kld"]
                  and confirmation[1]["top1"]>=confirmation[0]["top1"]-.001)
        # Identity is the endpoint handoff until a scale is folded into actual
        # stored FP16 vectors and the resulting derivative is tested.
        set_control(args.stamp,args.attempt,{"schema":"glm53.quality-control.v1",
                    "id":"identity-after-sweep","gains":{"2":1.0,"3":1.0}},args.root)
        result={"phase":"complete","started_at":start,"completed_at":utc(),
                "development":results,"selected":best,"confirmation":confirmation,
                "scale_candidate_improved":accepted,"packed_bpw":2.75,
                "endpoint_gains":{"2":1.0,"3":1.0},"logits_scaled":False}
        atomic_json(args.root/"RESULTS.json",result)
        atomic_json(args.root/"STATUS.json",result)
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as exc:
        atomic_json(args.root/"FAILED.json",{"error":repr(exc),"at":utc(),"results":results})
        try:
            set_control(args.stamp,args.attempt,{"schema":"glm53.quality-control.v1",
                        "id":"identity-after-failure","gains":{"2":1.0,"3":1.0}},args.root)
        finally:
            raise


if __name__=="__main__":
    main()
