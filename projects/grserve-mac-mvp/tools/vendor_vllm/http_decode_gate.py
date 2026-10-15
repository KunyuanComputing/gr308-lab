"""Small reproducible streaming-completions gate for vendor vLLM tuning."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx


async def run_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: list[int],
    output_tokens: int,
    index: int,
) -> dict[str, object]:
    started = time.perf_counter()
    ttft = None
    chunks = 0
    usage = None
    finish_reason = None
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with client.stream("POST", endpoint, json=payload) as response:
        status = response.status_code
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            item = json.loads(data)
            chunks += 1
            if ttft is None:
                ttft = time.perf_counter() - started
            if item.get("usage"):
                usage = item["usage"]
            choices = item.get("choices") or []
            if choices and choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
    elapsed = time.perf_counter() - started
    print("completed", index, round(elapsed, 3), flush=True)
    return {
        "idx": index,
        "status": status,
        "elapsed_s": elapsed,
        "ttft_s": ttft,
        "finish_reason": finish_reason,
        "usage": usage,
        "chunks": chunks,
    }


async def async_main(args: argparse.Namespace) -> None:
    prompt = [args.prompt_token_id] * args.input_tokens
    limits = httpx.Limits(
        max_connections=max(32, args.concurrency * 2),
        max_keepalive_connections=max(32, args.concurrency * 2),
    )
    async with httpx.AsyncClient(
        limits=limits, timeout=httpx.Timeout(args.timeout)
    ) as client:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(
                run_request(
                    client,
                    args.endpoint,
                    args.model,
                    prompt,
                    args.output_tokens,
                    index,
                )
                for index in range(args.concurrency)
            )
        )
        duration = time.perf_counter() - started
    completed = sum(
        (result.get("usage") or {}).get("completion_tokens", 0)
        for result in results
    )
    print(
        json.dumps(
            {
                "config": args.config,
                "input_tokens": args.input_tokens,
                "output_tokens_per_request": args.output_tokens,
                "concurrency": args.concurrency,
                "requests": len(results),
                "duration_s": duration,
                "completed_output_tokens": completed,
                "output_tokens_per_s": completed / duration,
                "results": results,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18000/v1/completions")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--prompt-token-id", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
