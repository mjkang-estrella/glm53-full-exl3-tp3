# Public source scope

The operator authorized public publication on September 8, 2026, after the private reproduction wrap-up. The original ten retrospective commits are retained. Later publication and service-routing changes are separate commits, not backdated experiments.

This repository contains source, tests, operational notes and curated numerical results. It deliberately excludes credentials, private SSH configuration, raw benchmark JSONL/captures, model checkpoints, compiled extensions, Docker images and the external encoder bundle. Hostnames, RFC1918 addresses and canonical lab paths are visible as concrete reproduction examples; configure equivalents for another lab.

Run `python3 scripts/audit_public_history.py` before a push. It scans every reachable commit for common credential patterns, excluded artifacts and binary/large blobs, and reports locations without printing matched values. This supplements human review; it is not a guarantee against every possible secret format.

The external R10 encoder retains its original provenance and licensing. The deployed K2 changes and runtime image are private dependencies described in DEPENDENCIES.md, not files distributed by this repository. No blanket license grant for external components is added by publishing this source. A public clone is not a clean-machine installer and does not include the model weights.

The public web application remains authenticated. Publication of source does not authorize unauthenticated access to the inference API or new network port exposure.
