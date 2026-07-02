#!/usr/bin/env bash
# Inspect the dumped SCFA Ascend C (/tmp/scfa_gen.cpp) for the V0 / kvMergeGm
# cross-core sync emission. Run AFTER building with SAS_DUMP_SRC=1.
# Output is small enough to paste back.
set -u
F=/tmp/scfa_gen.cpp
[ -f "$F" ] || { echo "MISSING $F -- build with SAS_DUMP_SRC=1 first"; exit 1; }
echo "===== file size ====="; wc -l "$F"

echo; echo "===== CrossCore call census (count by exact call) ====="
grep -oE 'CrossCore(Set|Wait)Flag(<[^>]*>)?\([0-9]+\)' "$F" | sort | uniq -c | sort -rn

echo; echo "===== every CrossCoreWaitFlag with the 2 lines AROUND it (context: is there a PipeBarrier/SyncAll adjacent?) ====="
grep -n -B2 -A1 'CrossCoreWaitFlag' "$F" | head -80

echo; echo "===== every CrossCoreSetFlag (pipe + id) ====="
grep -n 'CrossCoreSetFlag' "$F" | head -60

echo; echo "===== copy_gm_to_ub_gather (V0 015 gather) emission ====="
grep -n 'copy_gm_to_ub_gather' "$F" | head

echo; echo "===== PipeBarrier census (where are within-core barriers emitted) ====="
grep -oE 'PipeBarrier<PIPE_[A-Z0-9]+>' "$F" | sort | uniq -c

echo; echo "===== SetFlag/WaitFlag census on the MTE2<->MTE3 pair (V0 ping-pong ids 6/7) ====="
grep -oE 'AscendC::(Set|Wait)Flag<AscendC::HardEvent::(MTE2_MTE3|MTE3_MTE2)>\([0-9]+\)' "$F" | sort | uniq -c
