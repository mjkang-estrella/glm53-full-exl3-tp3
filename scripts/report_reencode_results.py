#!/usr/bin/env python3
"""Zima: build a report from the NAS-preserved pilot and audit receipts."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/unas-models/ZAI/GLM-5.3-EXL3-archive-notes-20260908')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8<<20):h.update(b)
    return h.hexdigest()


def main():
    pilot=ROOT/'reencode-layer33/pilot-v2'
    result=json.loads((pilot/'RESULTS.json').read_text())
    layer=json.loads((ROOT/'reencode-layer33/pilot-v4/RESULTS.json').read_text())
    prior=json.loads((ROOT/'reencode-layer33/pilot-v3/RESULTS.json').read_text())
    assert result['passed'] and layer['passed']
    for mode in prior['scores']:
        assert prior['scores'][mode]==layer['scores'][mode], 'summed-layer repeat metrics differ'
    verified=[]
    for receipt in sorted(pilot.glob('expert-*.json')):
        row=json.loads(receipt.read_text())
        for mode in ('identity','hessian'):
            path=pilot/mode/f"k3-layer-033-expert-{row['expert']:03d}.safetensors"
            assert sha(path)==row[mode]['file_sha256']
            verified.append(dict(path=str(path),sha256=row[mode]['file_sha256'],bytes=path.stat().st_size))
    assert len(verified)==32
    (ROOT/'PILOT_PAYLOAD_VERIFICATION.json').write_text(json.dumps(dict(passed=True,files=verified),indent=2)+'\n')
    archive=json.loads((ROOT/'PRESERVED.json').read_text())
    cleanup=json.loads((ROOT/'CLEANUP_COMPLETE.json').read_text())
    lines=['# K3/K275 preservation and re-encoding results','',
           'The bounded layer-33 pilot found no accepted improvement. Original checkpoint weights were not replaced.','',
           '## NAS preservation','',f"NAS audit phase: `{archive['phase']}`; verified ledger entries: {archive.get('verified_files','pending')}.",
           'Both assembled checkpoints remain under `/mnt/unas-models/ZAI`; exact directory names and recovery details are in `ARCHIVE_AND_REENCODE.md`.',
           'The audit caught a shared-checksum-ledger bug in the earlier K275 assembler. The original K3 ledger was restored to its sealed SHA-256, breaking that hardlink. A regression-tested atomic writer prevents recurrence. No tensor weights were edited by the repair.','',
           f"Removed {cleanup['invalid_raw_files']} invalid raw parts, {cleanup['bytes_removed']/(1<<30):.3f} GiB, and {len(cleanup['containers_removed'])} stopped failed-test containers.",
           'The invalid raw parts are not recoverable. Their hashes, ordering failures and API receipts remain. Container configuration and logs remain on NAS. No checkpoint was deleted.','',
           '## Experimental design','',
           'Sixteen layer-33 experts were chosen by routing coverage, eight from each existing tier. All original experts/routing and the 2.75-bpw budget were retained.',
           'Fit used 2,048 code-window input rows; evaluation used 2,048 separate reasoning-window rows. Inputs came from the original K275 model. This was not BF16 hidden-state replay or a private final test.',
           'K2 re-encoding used real activation covariance, fixed sigma 0.025 and the original rotations/scales. K2 assignment used only training incremental K2-versus-K3 error. Evaluation rows did not select the assignments or regularization.',
           'All eight existing K2 experts replayed byte-exactly with identity H. All packed-size, pack/unpack, finite reconstruction and local repeat checks passed.',
           f"Encoding took {result['elapsed_seconds']/60:.2f} minutes, with {result['cuda_peak_bytes']/(1<<30):.3f} GiB peak PyTorch CUDA allocation.",
           'The first attempt stopped on a CPU/GPU device mismatch in the reconstruction assertion. Its diagnostics were preserved; the corrected v2 attempt completed.','',
           '## Summed routed-expert output check','',
           'This check evaluated all 256 experts with the captured routes, retaining cross-expert error terms. It is not full-model KLD, task accuracy or native fused-kernel validation.',
           '',
           '| Variant | Training relative MSE | Evaluation relative MSE | Evaluation error change |',
           '|---|---:|---:|---:|']
    labels={'baseline':'Original K275','identity':'Selection only','hessian_fixed':'Re-encode only, original selection','hessian':'Re-encode plus selection'}
    for mode,label in labels.items():
        s=layer['scores'][mode]
        change='baseline' if mode=='baseline' else f"{-100*s['evaluation_improvement']:+.3f}%"
        lines.append(f"| {label} | {s['train']['relative_mse']:.8f} | {s['evaluation']['relative_mse']:.8f} | {change} |")
    lines += ['', 'All changed cases reduced training error but increased evaluation error. This pattern is consistent with overfitting or domain sensitivity in this small calibration set; it does not prove that activation-aware encoding generally fails.',
              'The shared cases produced exactly matching scalar scores in two full-layer passes. Local packed decoding and output checks were deterministic. The separate full-model repeatability issue remains unresolved.','',
              '## Decision','',
              'Reject these candidates. No full-model derivative or bulk re-encode was launched, and no new full-model KLD/top-1 result is claimed.',
              'A follow-up would need broader, balanced calibration, stronger covariance regularization assessed on training-only splits, and a fresh evaluation set. The current results do not justify scaling this recipe to the whole model.','',
              f"NAS evidence: `{ROOT}`",'']
    post=ROOT/'POST_RESTORE.json'
    if post.is_file():
        restore=json.loads(post.read_text())
        lines += ['## Restored serving verification','',
                  f"Attempt `{restore['attempt']}` passed {sum(p['passed'] for p in restore['probes'])}/{len(restore['probes'])} real generation checks.",
                  'Original K275 weights remain loaded at 32K with TP3/DCP3. No quality-control file was present on any rank.',
                  'Two unchanged-weight captures used only public window confirmation-0000, 2,047 positions. These are not the four-window KLD headline.',
                  '',
                  '| Repeat | KLD | Top-1 |','|---|---:|---:|']
        for row in restore['unchanged_weight_repeats']:
            lines.append(f"| {row['run']} | {row['kld']:.8f} | {100*row['top1']:.4f}% |")
        lines += ['',f"Observed KLD spread: {restore['kld_spread']:.8f} nats. The cause remains unresolved; unchanged weights do not make this serving path numerically repeatable.",'']
    (ROOT/'REENCODE_RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
