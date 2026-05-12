#!/bin/bash
# Orchestrator: runs each training task in a fresh Python process, so GPU
# memory is fully reclaimed between models (equivalent to a kernel restart).
# Outputs land in /home/ikaros/projects/ay-128/artifacts/.

set -e
cd /home/ikaros/projects/ay-128
source .venv/bin/activate

LOG_DIR="artifacts/_logs"
mkdir -p "$LOG_DIR"

run_task() {
    local task="$1"
    echo ""
    echo "========================================="
    echo " [$(date +%H:%M:%S)] Starting: $task"
    echo "========================================="
    /usr/bin/time -f "  wall: %E   peak RSS: %M KB" \
        python train_all.py "$task" 2>&1 | tee "$LOG_DIR/$task.log"
}

START=$(date +%s)

# Phase 1: independent runs (no inter-deps)
run_task resnet
run_task custom

# Phase 2: ablation variants (only need custom_result.npz to NOT exist — they don't read it)
run_task ablation_0
run_task ablation_1
run_task ablation_2
run_task ablation_3
run_task ablation_4
run_task ablation_summary

# Phase 3: optimisation (needs custom_result.npz)
run_task scheduler
run_task aug_only
run_task combined

# Phase 4: extra credit (needs custom_result.npz; reads best_augmented.pt at eval only)
run_task treeweighted

END=$(date +%s)
echo ""
echo "========================================="
echo " All training complete in $((END - START))s"
echo " Artifacts in: $(pwd)/artifacts/"
echo "========================================="
ls -lh artifacts/ | grep -v _logs
