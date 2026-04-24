#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test client for MiniMax M2.5 SAGE inference endpoint.

Usage:
    # Basic test
    python test_minimax_sage_client.py

    # Custom prompt
    python test_minimax_sage_client.py --prompt "Explain quantum computing"

    # Long context test
    python test_minimax_sage_client.py --long-context

    # Streaming
    python test_minimax_sage_client.py --stream
"""

import argparse
import json
import time

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Test MiniMax SAGE endpoint")
    parser.add_argument(
        "--host", type=str, default="localhost", help="Server host"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a detailed explanation of how attention mechanisms work in transformer models.",
        help="Prompt to send",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming output"
    )
    parser.add_argument(
        "--long-context",
        action="store_true",
        help="Test with a long context prompt",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run throughput benchmark",
    )
    return parser.parse_args()


def generate_long_context_prompt(target_tokens: int = 4000) -> str:
    """Generate a long context prompt to test SAGE eviction."""
    base_text = """
    The history of artificial intelligence is a fascinating journey through decades
    of scientific discovery, technological advancement, and philosophical debate.
    From the earliest days of computing, researchers have dreamed of creating
    machines that could think, learn, and reason like humans.

    In 1950, Alan Turing published his seminal paper "Computing Machinery and
    Intelligence," which introduced what we now call the Turing Test. This paper
    posed the fundamental question: "Can machines think?" Turing proposed that
    if a machine could engage in a conversation indistinguishable from a human,
    it could be considered intelligent.

    The field of AI was officially born at the Dartmouth Conference in 1956,
    where John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon
    brought together researchers to discuss "machines that use language, form
    abstractions and concepts, solve kinds of problems now reserved for humans,
    and improve themselves."

    The following decades saw periods of great optimism and funding (known as
    "AI summers") followed by periods of disappointment and reduced funding
    ("AI winters"). Early AI systems were rule-based expert systems that encoded
    human knowledge in explicit if-then rules.

    The resurgence of neural networks in the 2010s, particularly deep learning,
    transformed the field. Breakthroughs in image recognition, natural language
    processing, and game playing demonstrated that AI could achieve superhuman
    performance on specific tasks.

    The transformer architecture, introduced in 2017 with the paper "Attention
    Is All You Need," revolutionized natural language processing. This
    architecture enabled the development of large language models like GPT,
    BERT, and their successors.
    """

    # Repeat to reach target length
    prompt = base_text
    while len(prompt.split()) < target_tokens:
        prompt += "\n\n" + base_text

    prompt += "\n\nBased on this history, summarize the key milestones in AI development:"

    return prompt


def test_completion(args):
    """Test the completion endpoint."""
    url = f"http://{args.host}:{args.port}/v1/completions"

    prompt = args.prompt
    if args.long_context:
        prompt = generate_long_context_prompt()
        print(f"Generated long context prompt with ~{len(prompt.split())} words")

    payload = {
        "model": "minimax-m2.5-sage",
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
    }

    print(f"\n{'='*60}")
    print("Testing MiniMax M2.5 with SAGE Attention")
    print(f"{'='*60}")
    print(f"Endpoint: {url}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Streaming: {args.stream}")
    print(f"{'='*60}\n")

    if not args.long_context:
        print(f"Prompt: {prompt[:200]}...")
    else:
        print(f"Prompt: [Long context - {len(prompt)} chars]")
    print(f"\n{'-'*60}\n")

    start_time = time.time()

    if args.stream:
        # Streaming request
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        print("Generated text:")
        full_text = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        text = chunk["choices"][0].get("text", "")
                        print(text, end="", flush=True)
                        full_text += text
                    except json.JSONDecodeError:
                        pass
        print("\n")
    else:
        # Non-streaming request
        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        full_text = result["choices"][0]["text"]
        print(f"Generated text:\n{full_text}")

    elapsed = time.time() - start_time
    tokens = len(full_text.split())  # Approximate

    print(f"\n{'-'*60}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Approximate tokens: {tokens}")
    print(f"Tokens/second: {tokens/elapsed:.2f}")


def test_chat_completion(args):
    """Test the chat completion endpoint."""
    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": args.prompt},
    ]

    payload = {
        "model": "minimax-m2.5-sage",
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
    }

    print(f"\n{'='*60}")
    print("Testing Chat Completion")
    print(f"{'='*60}\n")

    start_time = time.time()

    if args.stream:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        print("Assistant:")
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        print("\n")
    else:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"Assistant:\n{content}")

    elapsed = time.time() - start_time
    print(f"\nTime elapsed: {elapsed:.2f}s")


def run_benchmark(args):
    """Run a simple throughput benchmark."""
    url = f"http://{args.host}:{args.port}/v1/completions"

    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis.",
        "How does a neural network learn?",
    ]

    print(f"\n{'='*60}")
    print("Running Throughput Benchmark")
    print(f"{'='*60}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Max tokens per prompt: {args.max_tokens}")
    print(f"{'='*60}\n")

    total_tokens = 0
    start_time = time.time()

    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}...")

        payload = {
            "model": "minimax-m2.5-sage",
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens += completion_tokens

    elapsed = time.time() - start_time

    print(f"\n{'-'*60}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Throughput: {total_tokens/elapsed:.2f} tokens/s")
    print(f"Average latency: {elapsed/len(prompts):.2f}s per request")


def check_server_health(args):
    """Check if the server is running."""
    url = f"http://{args.host}:{args.port}/health"
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def main():
    args = parse_args()

    # Check server health
    if not check_server_health(args):
        print(f"Error: Server not reachable at {args.host}:{args.port}")
        print("Make sure the server is running with:")
        print("  python scripts/launch_minimax_sage.py")
        return

    print(f"Server is healthy at {args.host}:{args.port}")

    if args.benchmark:
        run_benchmark(args)
    else:
        test_completion(args)
        print("\n")
        test_chat_completion(args)


if __name__ == "__main__":
    main()
