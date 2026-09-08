# Retrospective source history

This directory had no Git repository when the operator requested per-improvement commits on September 8, 2026. Commit dates describe this version-control wrap-up, not the original experiment execution times. No history was backdated.

The initial commit imports the available pre-speed runtime and launcher baseline. The two small runtime speed changes, the checksum-ledger correction, and optional graph/spin controls are separated into subsequent commits using the exact documented edits. Older loader/DCP work is imported as a snapshot, not presented as newly implemented in those commits.

Intermediate commits are historical code snapshots, not independently qualified deployments. Follow the final README and agent runbook. Model checkpoints, source weights, raw captures, logs, credentials, and generated caches are deliberately excluded.

The Mac snapshot initially lacked several lab-only helper modules. The final dependency inventory states whether they have been retrieved. Never substitute guessed helpers or claim a clean clone can launch until preflight passes.
