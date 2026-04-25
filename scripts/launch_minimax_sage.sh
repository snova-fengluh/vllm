#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Launch script for MiniMax M2.5 with SAGE attention on 8 A100 GPUs

set -e

# ============================================================================
# Configuration
# ============================================================================

# Model path
MODEL_PATH="${MODEL_PATH:-/import/ml-sc-scratch4/fengluh/minimax/MiniMax-M2.5-bf16}"

# Server configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Tensor parallelism (use all 8 GPUs)
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

# SAGE configuration (can be overridden via environment variables)
SAGE_WINDOW_LENGTH="${SAGE_WINDOW_LENGTH:-8192}"
SAGE_NUM_SINK_TOKENS="${SAGE_NUM_SINK_TOKENS:-4}"
SAGE_TOP_K="${SAGE_TOP_K:-512}"

# Model configuration
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Dtype
DTYPE="${DTYPE:-bfloat16}"

# ============================================================================
# Environment setup
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if it exists
if [ -f "$VLLM_DIR/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$VLLM_DIR/.venv/bin/activate"
fi

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Optimize for A100
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# ============================================================================
# Launch vLLM server with SAGE
# ============================================================================

echo "=============================================="
echo "MiniMax M2.5 with SAGE Attention"
echo "=============================================="
echo "Model path: $MODEL_PATH"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "Max model length: $MAX_MODEL_LEN"
echo "Max num sequences: $MAX_NUM_SEQS"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo ""
echo "SAGE Configuration:"
echo "  Window length: $SAGE_WINDOW_LENGTH"
echo "  Sink tokens: $SAGE_NUM_SINK_TOKENS"
echo "  Top-K: $SAGE_TOP_K"
echo "  Recent window: $((SAGE_WINDOW_LENGTH - SAGE_NUM_SINK_TOKENS - SAGE_TOP_K))"
echo "=============================================="
echo ""
echo "Starting vLLM server with --sage-enabled..."
echo ""

# Launch vLLM with SAGE CLI arguments
# The --sage-enabled flag automatically switches to MiniMaxM2SageForCausalLM
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "minimax-m2.5-sage" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --disable-log-requests \
    --enable-chunked-prefill \
    --sage-enabled \
    --sage-window-length "$SAGE_WINDOW_LENGTH" \
    --sage-num-sink-tokens "$SAGE_NUM_SINK_TOKENS" \
    --sage-top-k "$SAGE_TOP_K" \
    "$@"
