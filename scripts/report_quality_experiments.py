#!/usr/bin/env python3
"""Zima-only report builder from completed experiment receipts."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--evidence",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    result=json.loads((args.root/"RESULTS.json").read_text())
    if result["phase"]!="complete":
        raise SystemExit("campaign incomplete")
    lines=["# K275 KLD experiments", "", "## Outcome", "",
           "No changed setting passed the repeatability gate. The original scales were restored and the candidate was left running.",
           "Ten changes were tested against a fresh baseline: six global scale changes, three activation-clamp changes, and one layer-specific activation refit.", "",
           "All cases keep 2.75 routed-expert bpw, all 256 experts per routed layer and the existing top-8 router.",
           "The revised implementation changes the FP16 down-projection scale vectors directly. Trellis bits stay fixed.", "",
           "## Development sweep", "", "The two development windows were selection-0000 and selection-0001.", "",
           "| Case | K2 gain | K3 gain | SwiGLU limit | KLD | Top-1 |", "|---|---:|---:|---:|---:|---:|"]
    for row in result["development"]:
        lines.append(f"| {row['variant']} | {row['gains']['2']:g} | {row['gains']['3']:g} | {row['swiglu_limit']} | {row['kld']:.8f} | {100*row['top1']:.4f}% |")
    lines += ["", "## Confirmation", "", "The selected setting was chosen using development KLD before confirmation scoring.",
              "The four confirmation windows contain 8,188 prediction positions and all 154,880 vocabulary entries.",
              "These are public confirmation data previously used for reporting, not a private final test set.", "",
              "| Case | KLD | Top-1 |", "|---|---:|---:|"]
    for row in result["confirmation"]:
        lines.append(f"| {row['variant']} | {row['kld']:.8f} | {100*row['top1']:.4f}% |")
    if len(result["confirmation"])>1:
        base,chosen=result["confirmation"]
        delta=chosen["kld"]-base["kld"]
        lines += ["",f"KLD change: {delta:+.8f} nats, {100*delta/base['kld']:+.3f}%.",
                  f"Top-1 change: {100*(chosen['top1']-base['top1']):+.4f} percentage points."]
        base_score=json.loads((args.evidence/("confirm-"+base["variant"])/"SCORE.json").read_text())
        candidate_score=json.loads((args.evidence/("confirm-"+chosen["variant"])/"SCORE.json").read_text())
        lines += ["", "| Window | Baseline KLD | Candidate KLD | Change |", "|---|---:|---:|---:|"]
        for b,c in zip(base_score["windows"],candidate_score["windows"]):
            assert b["window_id"]==c["window_id"]
            lines.append(f"| {b['window_id']} | {b['kld']:.8f} | {c['kld']:.8f} | {c['kld']-b['kld']:+.8f} |")
    lines += ["",f"First confirmation KLD/top-1 gate passed: {result['scale_candidate_improved']}."]
    repeated_path=args.root/"REPEATED_CONFIRMATION.json"
    if repeated_path.is_file():
        repeated=json.loads(repeated_path.read_text())
        lines += ["", "## Repeatability check", "",
                  "The same frozen settings were captured and scored a second time.", "",
                  "| Case | First KLD | Repeat KLD | First top-1 | Repeat top-1 |",
                  "|---|---:|---:|---:|---:|"]
        for first,second in zip(repeated["first_confirmation"],repeated["repeat_confirmation"]):
            assert first["variant"]==second["variant"]
            lines.append(f"| {first['variant']} | {first['kld']:.8f} | {second['kld']:.8f} | {100*first['top1']:.4f}% | {100*second['top1']:.4f}% |")
        lines += ["",
                  "| Case | Mean KLD across two runs | Mean top-1 | KLD spread between runs |",
                  "|---|---:|---:|---:|"]
        for row in repeated["two_run_means"]:
            lines.append(f"| {row['variant']} | {row['kld']:.8f} | {100*row['top1']:.4f}% | {row['kld_repeat_spread']:.8f} |")
        lines += ["",f"Repeated improvement gate: {repeated['stable_improvement']}.",
                  f"Loaded after the checks: {repeated['loaded_variant']}.",
                  f"Runtime attempt: {repeated['runtime_attempt']}.",
                  f"Native generation probes passed: {sum(r['passed'] for r in repeated['probes'])}/{len(repeated['probes'])}."]
        base_first=json.loads((args.evidence/"confirm-baseline"/"SCORE.json").read_text())
        base_repeat=json.loads((args.evidence/"repeat-baseline"/"SCORE.json").read_text())
        flips=sum(x!=y for first,second in zip(base_first["windows"],base_repeat["windows"])
                  for x,y in zip(first["per_token_agreement"],second["per_token_agreement"]))
        lines += ["", "The candidate's two-run mean KLD is lower, but its first confirmation regressed and its mean top-1 agreement is lower.",
                  "The gate required lower KLD on both paired comparisons and a mean top-1 loss no larger than 0.1 percentage point.",
                  f"The baseline's aggregate top-1 percentage was identical, but teacher-agreement outcomes changed at {flips}/8188 individual positions.",
                  "The cause of this forward-run variation is unresolved. It is not a scorer arithmetic discrepancy, and these two repetitions do not establish a statistically reliable gain."]
    lines += ["",
              "## Activation refit", "",
              "Layer 33 used 4,096 input rows from separate code and reasoning windows, selection-0002 and selection-0003.",
              "54 of 64 K2 experts had enough routed rows. Per-expert output gains were fitted against BF16 source weights using FP32 matrix arithmetic.",
              "Every fifth input row was set aside to choose shrinkage. That local check error fell 2.958%, and the full-model refit case is listed above.",
              "Because those check rows selected shrinkage, this is not an unbiased final estimate. Only one layer was refitted; ten K2 experts lacked enough routed calibration rows.",
              "No local clipping effect was measured at this layer. Local reconstruction error is not full-model KLD.", "",
              "## Next experiments", "",
              "1. Resolve evaluation repeatability with identical-input baseline captures before accepting small KLD gains.",
              "2. Pilot activation/Hessian-aware re-encoding from BF16, retaining the exact packed budget. This batch changed scales, not trellis codes.",
              "3. Revisit which 64 experts per layer use K2: compare measured K2-versus-K3 activation error, rather than ranking only existing K3 weight error. Keep each rank's byte budget and all experts unchanged.",
              "",
              "These are proposed follow-ups, not results from this batch. A full-model re-encode was not started.", "",
              "## Runtime and evidence", "",
              "Attempt k275-quality-v2 uses TP3/DCP3, interleave 1, 32K context, 128-token prefill chunks, FP8 KV, eager execution and MTP disabled.",
              "Backbone expert weights use resident UVA across the three Sparks, without demand-loading experts from NVMe. The 12 GiB host-memory reserve and hardware watchdogs remained enabled.",
              "BF16, K3 and K275 checkpoint artifacts and public/client routes were not overwritten. Flash/H3 remain stopped.",
              "The initial output-multiplier experiment at 256-token chunks stalled and was stopped. It has no valid changed-scale KLD result.",
              "Implementation and prefill chunk size both changed before the successful v2 run; the cause of the first stall was not isolated.",
              "The scorer was checked against the original scorer on the prior four-window capture; mean KLD differed by only 6.1e-13 nats and top-1 was identical.",
              "",f"Receipts: {args.root}",f"Raw captures: {args.evidence}", ""]
    archive=args.root/"ARCHIVED_LOCAL_DUPLICATES.jsonl"
    if archive.is_file():
        archived=[json.loads(line) for line in archive.read_text().splitlines()]
        released=sum(row['bytes_released'] for row in archived)
        lines += [f"Removed duplicate Spark-1 raw logit parts for {len(archived)} captures ({released/(1<<30):.2f} GiB).",
                  "Both NAS and local file hashes were checked before removal. Local completion receipts remain; all raw data are recoverable from the verified NAS copies.", ""]
    final_audit=args.root/"final-kernel-audit.json"
    if final_audit.is_file():
        audited=json.loads(final_audit.read_text())
        lines += [f"Final kernel audit passed: {audited['passed']} ({audited['since']} through {audited['until']}).",
                  "Fault counts: "+", ".join(f"{host}={state['fault_count']}" for host,state in audited['hosts'].items())+".", ""]
    args.output.write_text("\n".join(lines))


if __name__=="__main__":
    main()
