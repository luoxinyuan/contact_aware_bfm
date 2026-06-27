#!/usr/bin/env bash
set -euo pipefail
export MEMPATH=/home/xl521/ceer/ee-gentle-humanoid/dataset

# ===== Global Configuration =====
PROJECT="luoxinyuan-duke-university/contact_aware_bfm"
export CUDA_VISIBLE_DEVICES=4,5,6,7
MASTER_PORT=29501
NPROC=4
SCRIPT="scripts/train.py"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

run_pipeline() {
  local TASK="$1" TAG="$2" SUFFIX="$3"

  local RUN_TAG="${TAG}_${SUFFIX}_${RUN_TIMESTAMP}"
  local ID_TRAIN="${RUN_TAG}_train"
  local ID_ADAPT="${RUN_TAG}_adapt"
  local ID_FINETUNE="${RUN_TAG}_finetune"

  # ---------- TRAIN ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" "$SCRIPT"
    task="$TASK" +exp=train
    wandb.id="$ID_TRAIN"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- ADAPT ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" "$SCRIPT"
    task="$TASK" +exp=adapt
    checkpoint_path="run:${PROJECT}/${ID_TRAIN}"
    wandb.id="$ID_ADAPT"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- FINETUNE ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" "$SCRIPT"
    task="$TASK" +exp=finetune
    checkpoint_path="run:${PROJECT}/${ID_ADAPT}"
    wandb.id="$ID_FINETUNE"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"
}

# run_pipeline "G1/G1_gentle" "gentle" "kkxx888"
run_pipeline "G1/G1_motion_tracking_future5" "motion_future5" "kkxx8889"
# run_pipeline "G1/G1_no_force" "noforce" "1215"
# run_pipeline "G1/G1_extreme_force" "extremeforce" "1215"
