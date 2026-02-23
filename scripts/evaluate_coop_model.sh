#!/bin/bash
# Evaluation script for CoOp model on test video features

# Configuration
TEST_FEATURE_DIR="/mnt/c/JJS/UCF_Crimes/Features/MCi20-avgpooled/test"
TEST_ANNOTATION_DIR="/mnt/c/JJS/UCF_Crimes/Annotations"

# Model checkpoint (update this path after training completes)
CHECKPOINT_PATH="./output/video_feature_coop/video_feature_coop_best.pth"

# Output directory for results
OUTPUT_DIR="./output/evaluation/video_feature_coop"

# Model configuration (should match training)
MOBILECLIP_MODEL="mobileclip2_s0"
N_CTX=16
CSC_FLAG=""  # Add "--csc" if CSC was used during training

# Run evaluation
python evaluate_video_feature_coop.py \
    --test-feature-dir "$TEST_FEATURE_DIR" \
    --test-annotation-dir "$TEST_ANNOTATION_DIR" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --mobileclip-model "$MOBILECLIP_MODEL" \
    --n-ctx "$N_CTX" \
    --batch-size 32 \
    --num-workers 4 \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    $CSC_FLAG

echo ""
echo "============================================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "Files generated:"
echo "  - metrics.json: Overall metrics"
echo "  - classification_report_frame.json: Frame-level detailed metrics"
echo "  - classification_report_video.json: Video-level detailed metrics (if applicable)"
echo "  - confusion_matrix_frame.png: Frame-level confusion matrix"
echo "  - confusion_matrix_video.png: Video-level confusion matrix (if applicable)"
echo "  - video_predictions.json: Per-video predictions (if applicable)"
