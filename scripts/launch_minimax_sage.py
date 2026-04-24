#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Launch script for MiniMax M2.5 with SAGE attention.

This script provides a more flexible way to launch the vLLM server
with SAGE-enabled MiniMax M2.5 model.

Usage:
    # Launch server with defaults
    python launch_minimax_sage.py

    # Launch with custom SAGE config
    python launch_minimax_sage.py --sage-window-length 16384 --sage-top-k 1024

    # Test mode (offline inference without server)
    python launch_minimax_sage.py --test

    # Custom model path
    python launch_minimax_sage.py --model /path/to/model
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch MiniMax M2.5 with SAGE attention"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="/import/ml-sc-scratch4/fengluh/minimax/MiniMax-M2.5-bf16",
        help="Path to the model",
    )

    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    # Parallelism
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="Number of GPUs for tensor parallelism",
    )

    # SAGE configuration
    parser.add_argument(
        "--sage-window-length",
        type=int,
        default=8192,
        help="SAGE window length (total cache size)",
    )
    parser.add_argument(
        "--sage-num-sink-tokens",
        type=int,
        default=4,
        help="Number of sink tokens to always preserve",
    )
    parser.add_argument(
        "--sage-top-k",
        type=int,
        default=512,
        help="Number of top-k important tokens to select",
    )

    # Model limits
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of concurrent sequences",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization ratio",
    )

    # Other options
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "auto"],
        help="Model dtype",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (offline inference) instead of server",
    )
    parser.add_argument(
        "--test-prompt",
        type=str,
        default="Write a short story about a robot learning to paint.",
        help="Prompt for test mode",
    )

    return parser.parse_args()


def update_model_config(model_path: str, args) -> str:
    """
    Create a temporary config file with SAGE parameters.

    Returns the path to the temporary config file.
    """
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Add SAGE configuration
    config["sage_enabled"] = True
    config["sage_window_length"] = args.sage_window_length
    config["sage_num_sink_tokens"] = args.sage_num_sink_tokens
    config["sage_top_k"] = args.sage_top_k
    config["sage_num_full_kv_layer"] = 0

    # Update architecture to use SAGE model
    if "architectures" in config:
        original_arch = config["architectures"]
        config["architectures"] = ["MiniMaxM2SageForCausalLM"]
        print(f"Updated architecture: {original_arch} -> {config['architectures']}")

    # Write to temporary file
    temp_config = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="minimax_sage_config_"
    )
    json.dump(config, temp_config, indent=2)
    temp_config.close()

    print(f"SAGE config written to: {temp_config.name}")
    return temp_config.name


def run_test_mode(args):
    """Run offline inference test."""
    print("\n" + "=" * 60)
    print("Running in TEST MODE (offline inference)")
    print("=" * 60 + "\n")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Error: vLLM not installed. Please install it first.")
        sys.exit(1)

    # Update config
    temp_config = update_model_config(args.model, args)

    try:
        print(f"Loading model: {args.model}")
        print(f"Tensor parallel size: {args.tensor_parallel_size}")
        print(f"Max model length: {args.max_model_len}")
        print()

        # Initialize model
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=True,
        )

        # Generate
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
        )

        print(f"Prompt: {args.test_prompt}")
        print("-" * 60)

        outputs = llm.generate([args.test_prompt], sampling_params)

        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Generated:\n{generated_text}")
            print("-" * 60)
            print(f"Tokens generated: {len(output.outputs[0].token_ids)}")

    finally:
        # Cleanup temp config
        os.unlink(temp_config)


def run_server_mode(args):
    """Launch the vLLM OpenAI-compatible server."""
    print("\n" + "=" * 60)
    print("Launching vLLM Server with SAGE Attention")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Max model length: {args.max_model_len}")
    print(f"Max num sequences: {args.max_num_seqs}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print()
    print("SAGE Configuration:")
    print(f"  Window length: {args.sage_window_length}")
    print(f"  Sink tokens: {args.sage_num_sink_tokens}")
    print(f"  Top-K: {args.sage_top_k}")
    recent = args.sage_window_length - args.sage_num_sink_tokens - args.sage_top_k
    print(f"  Recent window: {recent}")
    print("=" * 60 + "\n")

    # Update config in model directory
    config_path = Path(args.model) / "config.json"
    backup_path = Path(args.model) / "config.json.backup"

    # Backup original config
    if config_path.exists() and not backup_path.exists():
        import shutil

        shutil.copy(config_path, backup_path)
        print(f"Backed up original config to: {backup_path}")

    # Update config with SAGE parameters
    with open(config_path) as f:
        config = json.load(f)

    config["sage_enabled"] = True
    config["sage_window_length"] = args.sage_window_length
    config["sage_num_sink_tokens"] = args.sage_num_sink_tokens
    config["sage_top_k"] = args.sage_top_k
    config["sage_num_full_kv_layer"] = 0

    # Update architecture
    if "architectures" in config:
        config["architectures"] = ["MiniMaxM2SageForCausalLM"]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated model config with SAGE parameters")

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        "minimax-m2.5-sage",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
        "--trust-remote-code",
        "--disable-log-requests",
        "--enable-chunked-prefill",
    ]

    print(f"Running: {' '.join(cmd)}\n")

    import subprocess

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    finally:
        # Restore original config
        if backup_path.exists():
            import shutil

            shutil.copy(backup_path, config_path)
            print(f"Restored original config from backup")


def main():
    args = parse_args()

    # Validate SAGE configuration
    recent_window = (
        args.sage_window_length - args.sage_num_sink_tokens - args.sage_top_k
    )
    if recent_window <= 0:
        print(
            f"Error: Invalid SAGE config. "
            f"window_length ({args.sage_window_length}) must be > "
            f"sink_tokens ({args.sage_num_sink_tokens}) + top_k ({args.sage_top_k})"
        )
        sys.exit(1)

    # Check model path
    if not Path(args.model).exists():
        print(f"Error: Model path does not exist: {args.model}")
        sys.exit(1)

    if args.test:
        run_test_mode(args)
    else:
        run_server_mode(args)


if __name__ == "__main__":
    main()
