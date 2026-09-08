#!/usr/bin/env bash
set -euo pipefail
run_stamp=${1:?run stamp required}
exec bash /home/mj-kang/Dev/experiment/glm53-full-exl3-tp3/scripts/start_k275_bulk.sh "$run_stamp"
