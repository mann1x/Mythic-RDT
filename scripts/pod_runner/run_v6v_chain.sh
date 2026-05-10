#!/bin/bash
# v6V chain: precompute QC-14B → DSC cross-vocab teacher cache, then train.
# Train is skipped if precompute fails. Eval auto-chains in train script.

set -uo pipefail

LOG=/workspace/mythic-rdt/eval_results/V6V_CHAIN.log
echo "=== v6V chain start $(date -u +%FT%TZ) ===" | tee -a $LOG

bash /workspace/mythic-rdt/scripts/pod_runner/run_v6v_precompute.sh
PRE_RC=$?
echo "=== v6v precompute exit=$PRE_RC at $(date -u +%FT%TZ) ===" | tee -a $LOG

if [ $PRE_RC -ne 0 ]; then
    echo "[v6v-chain] precompute failed (exit $PRE_RC) — SKIPPING train." | tee -a $LOG
    exit $PRE_RC
fi

# Let GPU/disk settle.
sleep 30

bash /workspace/mythic-rdt/scripts/pod_runner/run_v6v_train.sh
TRAIN_RC=$?
echo "=== v6v train+eval exit=$TRAIN_RC at $(date -u +%FT%TZ) ===" | tee -a $LOG
echo "=== v6V chain done $(date -u +%FT%TZ) ===" | tee -a $LOG
exit $TRAIN_RC
