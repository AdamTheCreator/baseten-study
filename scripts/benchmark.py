#!/usr/bin/env python3
"""
Benchmark an inference endpoint — measures TTFT, TPOT, throughput, and latency percentiles.

Usage:
    # Against Baseten pre-optimized API
    python benchmark.py --endpoint https://bridge.baseten.co/v1/direct \
        --model deepseek-v3 --requests 100 --concurrency 16

    # Against a custom deployment
    python benchmark.py --endpoint https://model-xxxx.api.baseten.co/v1 \
        --model meta-llama/Llama-3.1-70B-Instruct --requests 200

    # Against a self-hosted vLLM server (for baseline comparison)
    python benchmark.py --endpoint http://localhost:8000/v1 \
        --model meta-llama/Llama-3.1-70B-Instruct --requests 100

    # With custom prompts file (JSONL, one {"prompt": "..."} per line)
    python benchmark.py --prompt-file customer_prompts.jsonl --requests 500

    # Save results to file
    python benchmark.py --output results.json
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict

import httpx
import numpy as np


# --- Default test prompts (used if no --prompt-file provided) ---

DEFAULT_PROMPTS = [
    "Explain how a transformer neural network processes a sequence of tokens, step by step.",
    "Write a Python function that implements binary search on a sorted list. Include docstring and type hints.",
    "What are the key differences between TCP and UDP? When would you choose one over the other?",
    "Summarize the main challenges of deploying large language models in production environments.",
    "Describe the CAP theorem and give a real-world example for each tradeoff.",
    "Write a SQL query to find the top 5 customers by total order value in the last 30 days, including their names and order counts.",
    "Explain the concept of prompt caching in LLM inference and why it matters for cost optimization.",
    "What is quantization in the context of neural networks? Compare FP16, FP8, and INT4.",
    "Design a rate limiting system for an API that supports both per-user and global limits.",
    "Explain the difference between tensor parallelism and pipeline parallelism for distributed model inference.",
]


@dataclass
class RequestResult:
    """Result from a single inference request."""
    prompt_tokens_approx: int
    output_tokens: int
    ttft_ms: float           # Time to first token
    total_ms: float          # Total request duration
    tpot_ms: float           # Time per output token (avg)
    status_code: int
    error: str | None


async def send_streaming_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: str,
    api_key: str | None,
    max_tokens: int = 256,
) -> RequestResult:
    """Send a streaming chat completion request and capture timing metrics."""

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    url = f"{endpoint.rstrip('/')}/chat/completions"
    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    error = None

    try:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                body = await response.aread()
                return RequestResult(
                    prompt_tokens_approx=len(prompt.split()),
                    output_tokens=0,
                    ttft_ms=0,
                    total_ms=(time.perf_counter() - start) * 1000,
                    tpot_ms=0,
                    status_code=response.status_code,
                    error=body.decode()[:200],
                )

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1  # Approximate: 1 chunk ≈ 1 token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    except Exception as e:
        error = str(e)
        return RequestResult(
            prompt_tokens_approx=len(prompt.split()),
            output_tokens=0,
            ttft_ms=0,
            total_ms=(time.perf_counter() - start) * 1000,
            tpot_ms=0,
            status_code=0,
            error=error,
        )

    end = time.perf_counter()
    total_ms = (end - start) * 1000

    if first_token_time is not None:
        ttft_ms = (first_token_time - start) * 1000
        decode_time = (end - first_token_time) * 1000
        tpot_ms = decode_time / max(token_count - 1, 1)
    else:
        ttft_ms = total_ms
        tpot_ms = 0

    return RequestResult(
        prompt_tokens_approx=len(prompt.split()),
        output_tokens=token_count,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        tpot_ms=tpot_ms,
        status_code=200,
        error=None,
    )


async def run_benchmark(
    endpoint: str,
    model: str,
    prompts: list[str],
    num_requests: int,
    concurrency: int,
    api_key: str | None,
    max_tokens: int,
) -> list[RequestResult]:
    """Run concurrent benchmark requests with a concurrency limit."""

    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    completed = 0

    async def bounded_request(client: httpx.AsyncClient, prompt: str):
        nonlocal completed
        async with semaphore:
            result = await send_streaming_request(
                client, endpoint, model, prompt, api_key, max_tokens
            )
            results.append(result)
            completed += 1
            if completed % 10 == 0 or completed == num_requests:
                print(f"  Progress: {completed}/{num_requests} requests completed")

    # Cycle through prompts if we need more requests than prompts
    request_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        tasks = [bounded_request(client, p) for p in request_prompts]
        await asyncio.gather(*tasks)

    return results


def compute_summary(results: list[RequestResult]) -> dict:
    """Compute summary statistics from benchmark results."""

    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    if not successful:
        return {"error": "All requests failed", "failures": len(failed)}

    ttfts = [r.ttft_ms for r in successful]
    totals = [r.total_ms for r in successful]
    tpots = [r.tpot_ms for r in successful if r.tpot_ms > 0]
    output_tokens = [r.output_tokens for r in successful]

    # Calculate throughput: total tokens generated / total wall-clock time
    total_wall_time_sec = max(totals) / 1000  # approximate (concurrent)
    total_tokens = sum(output_tokens)
    system_throughput = total_tokens / total_wall_time_sec if total_wall_time_sec > 0 else 0

    return {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "error_rate_pct": (len(failed) / len(results)) * 100,

        "ttft_p50_ms": round(np.percentile(ttfts, 50), 1),
        "ttft_p90_ms": round(np.percentile(ttfts, 90), 1),
        "ttft_p95_ms": round(np.percentile(ttfts, 95), 1),
        "ttft_p99_ms": round(np.percentile(ttfts, 99), 1),
        "ttft_mean_ms": round(np.mean(ttfts), 1),
        "ttft_min_ms": round(min(ttfts), 1),
        "ttft_max_ms": round(max(ttfts), 1),

        "total_latency_p50_ms": round(np.percentile(totals, 50), 1),
        "total_latency_p95_ms": round(np.percentile(totals, 95), 1),
        "total_latency_p99_ms": round(np.percentile(totals, 99), 1),

        "tpot_p50_ms": round(np.percentile(tpots, 50), 1) if tpots else None,
        "tpot_p95_ms": round(np.percentile(tpots, 95), 1) if tpots else None,

        "output_tokens_mean": round(np.mean(output_tokens), 1),
        "system_throughput_tps": round(system_throughput, 1),

        "avg_request_throughput_tps": round(
            np.mean([r.output_tokens / (r.total_ms / 1000) for r in successful if r.total_ms > 0]),
            1
        ),
    }


def print_report(summary: dict, endpoint: str, model: str, concurrency: int):
    """Print a formatted benchmark report."""

    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Endpoint:    {endpoint}")
    print(f"  Model:       {model}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Requests:    {summary['total_requests']} ({summary['successful']} ok, {summary['failed']} failed)")
    print(f"  Error Rate:  {summary['error_rate_pct']:.1f}%")
    print()

    print("  TTFT (Time to First Token):")
    print(f"    p50:  {summary['ttft_p50_ms']:>8.1f} ms")
    print(f"    p90:  {summary['ttft_p90_ms']:>8.1f} ms")
    print(f"    p95:  {summary['ttft_p95_ms']:>8.1f} ms")
    print(f"    p99:  {summary['ttft_p99_ms']:>8.1f} ms")
    print(f"    mean: {summary['ttft_mean_ms']:>8.1f} ms")
    print()

    print("  Total Latency (end-to-end):")
    print(f"    p50:  {summary['total_latency_p50_ms']:>8.1f} ms")
    print(f"    p95:  {summary['total_latency_p95_ms']:>8.1f} ms")
    print(f"    p99:  {summary['total_latency_p99_ms']:>8.1f} ms")
    print()

    if summary.get("tpot_p50_ms"):
        print("  TPOT (Time Per Output Token):")
        print(f"    p50:  {summary['tpot_p50_ms']:>8.1f} ms")
        print(f"    p95:  {summary['tpot_p95_ms']:>8.1f} ms")
        print()

    print("  Throughput:")
    print(f"    System:      {summary['system_throughput_tps']:>8.1f} tokens/sec (total across all requests)")
    print(f"    Per-request: {summary['avg_request_throughput_tps']:>8.1f} tokens/sec (average per request)")
    print(f"    Avg output:  {summary['output_tokens_mean']:>8.1f} tokens/request")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark an inference endpoint")
    parser.add_argument("--endpoint", default="https://bridge.baseten.co/v1/direct",
                        help="Base URL of the OpenAI-compatible API")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct",
                        help="Model name to request")
    parser.add_argument("--api-key", default=None,
                        help="API key (defaults to BASETEN_API_KEY env var)")
    parser.add_argument("--requests", type=int, default=50,
                        help="Number of requests to send")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max output tokens per request")
    parser.add_argument("--prompt-file", default=None,
                        help="JSONL file with prompts (one {\"prompt\": \"...\"} per line)")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("BASETEN_API_KEY")

    # Load prompts
    if args.prompt_file:
        prompts = []
        with open(args.prompt_file) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data.get("prompt", data.get("text", "")))
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"Using {len(prompts)} default prompts")

    print(f"\nBenchmarking: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Requests: {args.requests}, Concurrency: {args.concurrency}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Run benchmark
    start = time.perf_counter()
    results = asyncio.run(run_benchmark(
        endpoint=args.endpoint,
        model=args.model,
        prompts=prompts,
        num_requests=args.requests,
        concurrency=args.concurrency,
        api_key=api_key,
        max_tokens=args.max_tokens,
    ))
    wall_time = time.perf_counter() - start

    # Compute and display results
    summary = compute_summary(results)
    summary["wall_time_sec"] = round(wall_time, 1)
    print_report(summary, args.endpoint, args.model, args.concurrency)
    print(f"\n  Wall time: {wall_time:.1f}s")

    # Save to file
    if args.output:
        output = {
            "config": {
                "endpoint": args.endpoint,
                "model": args.model,
                "requests": args.requests,
                "concurrency": args.concurrency,
                "max_tokens": args.max_tokens,
            },
            "metrics": summary,
            "raw_results": [asdict(r) for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
