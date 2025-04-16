#!/bin/bash

# --- Configuration ---
# Path to your tokenized data (output from preprocess_data.py)
TOKENIZED_DATA_PATH="/path/to/your/tokenized_data.npy"

# Base model config for size and tokenizer
MODEL_CONFIG="meta-llama/Meta-Llama-3-8B"
TOKENIZER_NAME="meta-llama/Meta-Llama-3-8B"

# Training parameters (adjust!)
OUTPUT_DIR="./bitnet_pretrain_output_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE_PER_DEVICE=2 # START SMALL, increase if memory allows
GRAD_ACCUM_STEPS=16      # Adjust to reach desired global batch size
LEARNING_RATE=3e-4
MAX_STEPS=50000         # Total training steps
SAVE_STEPS=5000
LOGGING_STEPS=50
SEQ_LENGTH=2048

# Precision (uncomment ONE)
PRECISION_ARGS="--bf16" # Use BF16 on Ampere+ GPUs
# PRECISION_ARGS="--fp16" # Use FP16 on older GPUs

# Gradient Checkpointing (highly recommended for large models)
GRAD_CKPT_ARGS="--gradient_checkpointing"
# GRAD_CKPT_ARGS="" # Disable if needed

# --- Accelerator Configuration ---
# Ensure you have configured Accelerate first: `accelerate config`
# This will ask about your hardware setup (multi-GPU, multi-node, etc.)
# Adjust num_processes based on the number of GPUs you want to use.
NUM_GPUS=8 # Example: Use 8 GPUs

echo "Starting BitNet Pre-training..."
echo "Output Directory: ${OUTPUT_DIR}"
echo "Tokenized Data: ${TOKENIZED_DATA_PATH}"
echo "Batch Size Per Device: ${BATCH_SIZE_PER_DEVICE}"
echo "Gradient Accumulation Steps: ${GRAD_ACCUM_STEPS}"
echo "Effective Global Batch Size: $(($BATCH_SIZE_PER_DEVICE * $GRAD_ACCUM_STEPS * $NUM_GPUS))"

# --- Launch Training ---
# Add any HPC-specific commands here (e.g., module load, srun)
# module load cuda/11.8 python/3.10 anaconda3/latest # Example module loads
# source activate your_conda_env                     # Example environment activation

accelerate launch --config_file /path/to/accelerate/config.yaml --num_processes ${NUM_GPUS} train.py \
    --tokenized_data_path ${TOKENIZED_DATA_PATH} \
    --model_config_name ${MODEL_CONFIG} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --seq_length ${SEQ_LENGTH} \
    --batch_size_per_device ${BATCH_SIZE_PER_DEVICE} \
    --grad_accum_steps ${GRAD_ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --max_steps ${MAX_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    ${PRECISION_ARGS} \
    ${GRAD_CKPT_ARGS} \
    # Add other args like --weight_decay, --warmup_steps if needed

echo "Training script finished."