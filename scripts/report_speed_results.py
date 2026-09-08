#!/usr/bin/env python3
"""Build a concise speed report from the Zima campaign's measured receipts."""
import argparse
import json
import math
from pathlib import Path


def counter(path, name):
    if not path.is_file():return None
    values=[float(line.rsplit(' ',1)[1]) for line in path.read_text().splitlines() if line.startswith(name+'{')]
    return sum(values) if values else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    data=json.loads((args.root/'RESULTS.json').read_text())
    baseline=data['rows'][0]['tg']['mean']
    summary=None
    if data['phase']=='complete':
        final=next(r for r in data['rows'] if r['attempt']==data['loaded'])
        def normalized(c):
            return {**dict(graph_mode='',spinwait_ms='stock',kv_bytes=1812613120),**c}
        matches=[r for r in data['rows'] if r['accepted'] and r.get('eligible_for_selection',True)
                 and normalized(r['config'])==normalized(final['config'])]
        values=[v for r in matches for v in r['tg']['values']]
        mean=sum(values)/len(values)
        summary=dict(loaded=data['loaded'],config=normalized(final['config']),runs=len(values),
                     decode_mean=mean,decode_std=math.sqrt(sum((v-mean)**2 for v in values)/len(values)),
                     decode_values=values,baseline_decode_mean=baseline,speedup=mean/baseline,
                     prompt_mean=sum(r['pp']['mean'] for r in matches)/len(matches),
                     ttfr_seconds=sum(r['ttfr']['mean'] for r in matches)/len(matches)/1000)
        for filename,key in [('FINAL_LONG_CONTEXT.json','long_context'),('FINAL_MEMORY_SUMMARY.json','memory'),('FINAL_KERNEL_AUDIT.json','kernel_audit')]:
            path=args.root/filename
            if path.is_file():summary[key]=json.loads(path.read_text())
        (args.root/'FINAL_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# K275 llama-benchy speed results','',
           f"Campaign phase: {data['phase']}.",
           'All cases use original K275 weights, all experts/routing, TP3/DCP3, 32K context, FP8 KV and a 12 GiB host-reserve guard.',
           'Client: Spark 1, llama-benchy 0.4.0 at commit 446dd42fde2ebbaa1d68a0dfe9dc1e5b833f95ad, clean checkout.',
           'Workload: pp2048, requested tg256 with exact-tg, three measured warm runs, concurrency 1, no cache, generation latency mode.',
           'Decode throughput excludes prefill. The standard tool counts observed content/reasoning token IDs and can exclude suppressed formatting tokens; requested length and observed length are not conflated.','',
           '| Case | NCCL channels / KiB | Graphs | MTP tokens | Prefill chunk | KV GiB/rank | Decode tok/s | Change | Prompt tok/s | TTFR seconds | Outcome |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    if summary:
        lines[2:2]=[f"Best qualified profile: **{summary['decode_mean']:.2f} decode tok/s** across {summary['runs']} measured runs, versus {baseline:.2f} before tuning. Speedup: **{summary['speedup']:.2f}x**.",
                    'This is the original K275 model. Higher short-run results that failed repeatability or memory checks are not eligible.','']
    for row in data['rows']:
        c=row['config']
        if row['accepted']:
            tg=f"{row['tg']['mean']:.3f} ± {row['tg']['std']:.3f}"
            change=f"{100*(row['tg']['mean']/baseline-1):+.1f}%"
            pp=f"{row['pp']['mean']:.2f}"
            ttfr=f"{row['ttfr']['mean']/1000:.3f}"
            outcome='passed' if row.get('eligible_for_selection',True) else 'measured; disqualified on repeat'
        else:
            tg=change=pp=ttfr='n/a'
            outcome='rejected; see launch/probe logs'
        graph='off' if c['eager'] else (c.get('graph_mode') or 'default')
        label=row['label']
        if c.get('spinwait_ms','stock')!='stock':label+=f", {c['spinwait_ms']} ms spin"
        kv=c.get('kv_bytes',1812613120)/(1<<30)
        lines.append(f"| {label} | {c['channels']} / {c['buffer']//1024} | {graph} | {c['mtp']} | {c['prefill']} | {kv:.3f} | {tg} | {change} | {pp} | {ttfr} | {outcome} |")
    lines += ['', '## Speculative acceptance','',
              'Counter deltas cover the benchmark, its warmup/latency requests and the post-benchmark generation probes. They are not per-run or code-workload acceptance guarantees.','',
              '| Case | Accepted draft tokens | Proposed draft tokens | Acceptance |','|---|---:|---:|---:|']
    for row in data['rows']:
        if not row['accepted'] or not row['config']['mtp']:continue
        deltas={}
        for label,metric in [('accepted','vllm:spec_decode_num_accepted_tokens_total'),('drafted','vllm:spec_decode_num_draft_tokens_total')]:
            before=counter(args.root/f"{row['label']}-metrics-before.txt",metric)
            after=counter(args.root/f"{row['label']}-metrics-after.txt",metric)
            if before is not None and after is not None:deltas[label]=after-before
        if deltas.get('drafted',0)>0 and 'accepted' in deltas:
            lines.append(f"| {row['label']} | {deltas['accepted']:.0f} | {deltas['drafted']:.0f} | {100*deltas['accepted']/deltas['drafted']:.1f}% |")
    lines += ['', '## Runtime changes','',
              'Resident execution no longer runs LFU eviction bookkeeping. Previously every layer synchronized routing IDs to CPU despite having all experts resident and no evictions. Non-resident caching retains its original accounting.',
              'DCP padding diagnostics now print once rather than once per layer/token. Padding and attention arithmetic are unchanged.',
              'The historical CUDA-graph failure stack points to dynamic routing-index filtering in observe_ids. The resident-only no-op removes that capture blocker. The combined hot-path case changes bookkeeping and logging, so its gain is not attributed to either change in isolation.','',
              'All benchmarks use the same tool/protocol. MTP acceptance and speed depend on text and can differ on code or agent workloads. A throughput winner is not a universal workload guarantee.',
              'No public/client route was changed, and Flash/H3 remain stopped. Sealed K3 and K275 checkpoint files were not modified.','']
    labels={r['label'] for r in data['rows']}
    if 'prefill256' in labels:
        lines += ['## Excluded trials','',
                  'The original approximately 1.69 GiB KV / MTP3 / prefill256 run hit the host-memory guard, about 54 MiB below the 12 GiB floor. Its benchmark was interrupted, and no throughput is accepted. The 1 GiB KV retry is a separately measured configuration.','']
    if 'mtp4-kv1g' in labels:
        lines += ['The four-token MTP trial stalled during its second measured request and was manually stopped. A controller pause had occurred during planning of a separate long-context check; its benchmark child remained active, and resuming the controller did not recover model progress. The long-context helper never ran. Kernel audit was clean. Root cause is unresolved, and partial output is not a valid speed result.','']
    if any(not r.get('eligible_for_selection',True) for r in data['rows']):
        lines += ['A later confirmation of MTP3 with prefill256 also stalled, without a controller pause. Larger-prefill MTP configurations are therefore ineligible for the final selection, including earlier short runs that completed. Their measurements remain visible for audit. The exact stall mechanism is unresolved.',
                  'Recovery benchmarks add a 120-second no-progress watchdog for the fixed pp2048 workload. This watchdog is not applied to legitimate long-context prefill.','']
    if 'mtp3-graphs-p128' in labels:
        lines += ['The tested MTP3/decode-only-graph combinations failed the memory reserve at both 256- and 128-token prefill settings with 1 GiB KV. This is a constraint on those tested configurations, not a claim that all MTP/graph combinations are impossible.','']
    if data['phase']=='complete':
        lines += [f"Selected profile: `{data['selected']}`.",f"Loaded confirmation attempt: `{data['loaded']}`.",'']
        if summary and 'long_context' in summary:
            long=summary['long_context']
            lines += ['## Final qualification','',
                      f"A real request used {long['usage']['prompt_tokens']:,} prompt tokens and returned exactly `{long['content']}` with normal stopping in {long['elapsed_seconds']:.2f} seconds. Passed: {long['passed']}.",
                      'The model limit remains 32,768 tokens, TP3/DCP3, one sequence, FP8 KV at 1 GiB per rank. The winning profile uses MTP3, draft TP1 setting, NCCL4/1 MiB, 128-token prefill, eager execution and stock spin-wait.',
                      'No checkpoint weights, experts or routing were changed. No new KLD/top-1 score is claimed for this speed sweep.','']
            if 'memory' in summary:
                minima=[r['min_host_available_gib'] for r in summary['memory']['ranks'].values()]
                lines += [f"Lowest observed host availability: {min(minima):.3f} GiB. Model-container swap stayed at zero. Kernel audit passed: {summary.get('kernel_audit',{}).get('passed','not attached')}.",'']
    args.output.write_text('\n'.join(lines))


if __name__=='__main__':main()
