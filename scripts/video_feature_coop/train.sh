#!/bin/bash

# CoOp Prompt Learning on Pre-Extracted Video Features (MobileCLIP S0)
#
# Directory structure expected:
#   FEATURE_DIR/
#       train/
#           class_name_1/
#               class_name_1_001_x264.npy   # shape: [T, D]
#               ...
#           class_name_2/
#               ...
#       val/
#           (same structure)
#       test/
#           (same structure)

# --- Configuration ---
FEATURE_ROOT="/path/to/features"
MOBILECLIP_PATH="/path/to/mobileclip_s0.pt"
ANNOTATION_DIR=""   # Optional: path to annotation CSV files
OUTPUT_DIR="./output/video_feature_coop"

# CoOp settings
N_CTX=16
CTX_INIT=""         # e.g., "a video of a" for word-based init
CLASS_TOKEN_POSITION="end"
TEMPORAL_AGG="mean"

# Window settings
UNIT_DURATION=1
OVERLAP_RATIO=0.0

# Training settings
BATCH_SIZE=32
LR=0.002
EPOCHS=50
WARMUP_EPOCHS=1
SEED=42

# --- Build command ---
CMD="python train_video_feature_coop.py \
    --feature-dir ${FEATURE_ROOT}/train \
    --val-feature-dir ${FEATURE_ROOT}/val \
    --mobileclip-path ${MOBILECLIP_PATH} \
    --n-ctx ${N_CTX} \
    --class-token-position ${CLASS_TOKEN_POSITION} \
    --temporal-agg ${TEMPORAL_AGG} \
    --unit-duration ${UNIT_DURATION} \
    --overlap-ratio ${OVERLAP_RATIO} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --warmup-epochs ${WARMUP_EPOCHS} \
    --seed ${SEED} \
    --checkpoint-dir ${OUTPUT_DIR}/checkpoints \
    --log-dir ${OUTPUT_DIR}/tensorboard \
    --strict-normal-sampling"

# Add optional annotation dir
if [ -n "${ANNOTATION_DIR}" ]; then
    CMD="${CMD} --annotation-dir ${ANNOTATION_DIR}"
fi

# Add optional context init
if [ -n "${CTX_INIT}" ]; then
    CMD="${CMD} --ctx-init '${CTX_INIT}'"
fi

echo "Running: ${CMD}"
eval ${CMD}
