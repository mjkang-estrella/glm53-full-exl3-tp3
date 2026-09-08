# Wrap-up validation, September 8, 2026

This records source packaging checks, not a new speed measurement.

- `python3 scripts/verify_reproduction.py`: 145 Python/shell syntax checks, three stdlib safety regression tests, deployed encoder hashes and curated result invariants passed.
- Full `tests/` suite: 76 passed in 3.54 s inside the existing pinned image on Spark 3. Separate container, CPU only, runc runtime, no network/GPU/device mounts, 1 GiB memory/swap cap, two CPUs, read-only source snapshot.
- `scripts/test_quality_scale.py`: four CPU tests passed, including exact FP16 restoration and retained data-pointer identity.
- `scripts/inspect_speed_winner.sh`: all three ranks passed running/not-OOMKilled, exact image, read-only K275/rank-pack mounts, winner environment, watchdog-session and no-STOP checks. Endpoint health and 32K model catalog passed.
- Numerical serving files `lazy_k3_patch.py`, `sparse_mla_tp3_patch.py` and `run_candidate_node.sh` match the retrieved deployed source hashes. The atomic K275 assembler also matches the deployed corrected version.

The Mac's default Python lacked pytest/PyTorch; the full CPU suite was therefore run in the isolated image rather than changing the Mac environment or the serving container. No new GPU benchmark, long-context request, KLD evaluation, container restart, public-route update, or checkpoint mutation was performed during this wrap-up.

The original throughput and 30K qualification evidence remains in the curated JSON and NAS archive. Live health is not a new quality or whole-window memory/kernel qualification.
