#!/usr/bin/env python3
"""Zima: repeat a frozen candidate/control comparison, then verify real output."""
from pathlib import Path
import json
import time

import quality_scale_campaign as lab
from long_context_probe import raw_native_completion

ROOT=Path("/home/mj-kang/Dev/state/glm53-full-exl3-tp3/quality/20260908-scale-v2")
EVIDENCE=Path("/mnt/unas-models/ZAI/GLM-5.3-EXL3-K275-quality-experiments-20260908-scale-v2")
STAMP="20260907T224500Z"
ATTEMPT="k275-quality-v2"


def control(row, name):
    return {"schema":"glm53.quality-control.v1","id":name,"gains":row["gains"],
            "swiglu_limit":row.get("swiglu_limit",10.0),"expert_gains":row.get("expert_gains",{})}


def main():
    deadline=time.monotonic()+1800
    while not (ROOT/"RESULTS.json").is_file():
        if (ROOT/"FAILED.json").exists() or time.monotonic()>deadline:
            raise RuntimeError("first campaign did not finish successfully")
        time.sleep(10)
    source=json.loads((ROOT/"RESULTS.json").read_text())
    baseline=source["development"][0]
    best=source["selected"]
    lab.EXPECTED_ROWS=2063
    started=lab.utc()
    repeated=[]
    variants=[baseline] if best["variant"]=="baseline" else [baseline,best]
    for row in variants:
        name="repeat-"+row["variant"]
        folder=EVIDENCE/name
        folder.mkdir(exist_ok=False)
        lab.set_control(STAMP,ATTEMPT,control(row,name),ROOT)
        lab.atomic_json(ROOT/"REPEAT_STATUS.json",{"phase":"capturing","variant":name,
                        "updated_at":lab.utc(),"completed":repeated})
        for window in lab.CONFIRM:
            print("REPEAT_CAPTURE",name,window,lab.utc(),flush=True)
            lab.capture_window(STAMP,ATTEMPT,window,folder,folder/f"{window}.log")
        score=lab.score_windows(folder,lab.CONFIRM,folder/"SCORE.json")
        lab.audit(STAMP,ATTEMPT,started,folder)
        repeated.append({"variant":row["variant"],"kld":score["kld"],"top1":score["top1"]})
        print("REPEAT_RESULT",json.dumps(repeated[-1]),flush=True)
    means=[]
    for first,second in zip(source["confirmation"],repeated):
        assert first["variant"]==second["variant"]
        means.append({"variant":first["variant"],"kld":(first["kld"]+second["kld"])/2,
                      "top1":(first["top1"]+second["top1"])/2,
                      "kld_repeat_spread":abs(first["kld"]-second["kld"])})
    stable=(len(means)>1 and means[1]["kld"]<means[0]["kld"]
            and means[1]["top1"]>=means[0]["top1"]-.001
            and all(b["kld"]<a["kld"] for a,b in [source["confirmation"],repeated]))
    chosen=best if stable else baseline
    selected_control=control(chosen,"selected-after-repeat")
    lab.set_control(STAMP,ATTEMPT,selected_control,ROOT)
    probes=[]
    for prompt,expected in [
        ("What is the capital of France? Reply with only the city name.","Paris"),
        ("Calculate 17 * 19. Reply with only the integer answer.","323"),
        ('Return exactly this JSON object and nothing else: {"answer": 42}',{"answer":42})]:
        result=raw_native_completion(lab.ENDPOINT,prompt,max_tokens=64,timeout=180)
        text=result["content"].strip()
        try:
            ok=(json.loads(text)==expected) if isinstance(expected,dict) else text==expected
        except ValueError:
            ok=False
        probes.append({"prompt":prompt,"expected":expected,"content":text,
                       "finish_reason":result["finish_reason"],"passed":ok and result["finish_reason"]=="stop"})
        print("SEMANTIC_PROBE",json.dumps(probes[-1]),flush=True)
    if not all(row["passed"] for row in probes) and chosen["variant"]!="baseline":
        selected_control=control(baseline,"baseline-after-probe-failure")
        lab.set_control(STAMP,ATTEMPT,selected_control,ROOT)
        chosen=baseline
    lab.audit(STAMP,ATTEMPT,started,ROOT)
    result={"phase":"complete","started_at":started,"completed_at":lab.utc(),
            "first_confirmation":source["confirmation"],"repeat_confirmation":repeated,
            "two_run_means":means,"stable_improvement":stable,
            "loaded_variant":chosen["variant"],"control":selected_control,"probes":probes,
            "runtime_attempt":ATTEMPT,"packed_bpw":2.75}
    lab.atomic_json(ROOT/"REPEATED_CONFIRMATION.json",result)
    lab.atomic_json(ROOT/"REPEAT_STATUS.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
