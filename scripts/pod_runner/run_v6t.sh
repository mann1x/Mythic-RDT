#!/bin/bash
# v6T TRAMPOLINE: T=8 deferred — exec into v6V chain instead.
#
# Why: 2026-05-10 plan pivoted from "scale T=4→T=8 with same teacher"
# to "upgrade teacher from DSC-self to QC-14B via CTD cross-vocab".
# The sequential launcher (run_v6st_sequential.sh) calls this script
# after v6S completes; we transparently redirect into the v6V chain
# so the launcher's wait/exit semantics still apply.
#
# Original T=8 script preserved as run_v6t_t8_DEFERRED.sh — restore by
# `cp run_v6t_t8_DEFERRED.sh run_v6t.sh` if v6V succeeds and we want
# to scale T.

set -uo pipefail
echo "[v6t-trampoline] $(date -u +%FT%TZ) — deferring T=8; exec'ing v6V chain"
exec bash /workspace/mythic-rdt/scripts/pod_runner/run_v6v_chain.sh
