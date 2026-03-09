import argparse
import json
import os
import statistics
import time
from typing import Any

import dashscope
from dotenv import load_dotenv
from openai import OpenAI

from deepresearch.config import load_llm_config


def _extract_text(payload: Any) -> str:
	if payload is None:
		return ""
	if isinstance(payload, str):
		return payload
	if isinstance(payload, list):
		texts = [_extract_text(item) for item in payload]
		return "\n".join([text for text in texts if text]).strip()
	if isinstance(payload, dict):
		if "text" in payload and isinstance(payload["text"], str):
			return payload["text"]
		if "content" in payload:
			return _extract_text(payload["content"])
		if "message" in payload:
			return _extract_text(payload["message"])
		if "output" in payload:
			return _extract_text(payload["output"])
		if "choices" in payload:
			return _extract_text(payload["choices"])
	return ""


def call_openai(api_key: str, base_url: str, model: str, prompt: str, max_tokens: int) -> tuple[float, str]:
	client = OpenAI(api_key=api_key, base_url=base_url)
	start = time.perf_counter()
	response = client.chat.completions.create(
		model=model,
		messages=[{"role": "user", "content": prompt}],
		temperature=0,
		max_tokens=max_tokens,
		extra_body={"enable_thinking": True},
		stream=True
	)
	full_text = ""
	for chunk in response:
		if chunk.choices and chunk.choices[0].delta.content:
			full_text += chunk.choices[0].delta.content
	latency = time.perf_counter() - start
	return latency, full_text


def call_dashscope(api_key: str, model: str, prompt: str) -> tuple[float, str]:
	messages = [{"role": "user", "content": [{"text": prompt}]}]
	start = time.perf_counter()
	response = dashscope.MultiModalConversation.call(
		api_key=api_key,
		model=model,
		messages=messages,
		temperature=0,
	)
	latency = time.perf_counter() - start

	if hasattr(response, "status_code") and response.status_code != 200:
		raise RuntimeError(
			f"DashScope request failed: status_code={response.status_code}, "
			f"code={getattr(response, 'code', None)}, message={getattr(response, 'message', None)}"
		)

	if hasattr(response, "output"):
		text = _extract_text(response.output)
	else:
		text = _extract_text(response)

	if not text:
		text = _extract_text(json.loads(json.dumps(response, ensure_ascii=False, default=str)))
	return latency, text


def calc_stats(latencies: list[float]) -> dict[str, float]:
	sorted_values = sorted(latencies)
	p95_index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * 0.95) - 1))
	return {
		"count": float(len(sorted_values)),
		"min": min(sorted_values),
		"avg": statistics.mean(sorted_values),
		"median": statistics.median(sorted_values),
		"p95": sorted_values[p95_index],
		"max": max(sorted_values),
	}


def run_benchmark(rounds: int, prompt: str, openai_model: str, dashscope_model: str, max_tokens: int) -> None:
	cfg = load_llm_config()
	openai_key = os.getenv("OPENAI_API_KEY") or cfg.api_key
	dashscope_key = cfg.api_key

	results: dict[str, list[float]] = {}

	if openai_key:
		results["openai"] = []
		print(f"\n[OpenAI] model={openai_model}")
		for i in range(rounds):
			try:
				latency, text = call_openai(openai_key, cfg.base_url, openai_model, prompt, max_tokens)
				results["openai"].append(latency)
				print(f"  round {i + 1}/{rounds}: {latency:.3f}s | sample={text[:80]!r}")
			except Exception as error:
				print(f"  round {i + 1}/{rounds}: ERROR -> {error}")
	else:
		print("\n[OpenAI] skipped: DASHSCOPE_API_KEY not found.")

	if dashscope_key:
		results["dashscope"] = []
		print(f"\n[DashScope] model={dashscope_model}")
		for i in range(rounds):
			try:
				latency, text = call_dashscope(dashscope_key, dashscope_model, prompt)
				results["dashscope"].append(latency)
				print(f"  round {i + 1}/{rounds}: {latency:.3f}s | sample={text[:80]!r}")
			except Exception as error:
				print(f"  round {i + 1}/{rounds}: ERROR -> {error}")
	else:
		print("\n[DashScope] skipped: DASHSCOPE_API_KEY not found.")

	print("\n=== Latency Summary (seconds) ===")
	print(f"{'provider':<12} {'count':>5} {'min':>8} {'avg':>8} {'median':>8} {'p95':>8} {'max':>8}")
	print("-" * 67)

	for provider in ("openai", "dashscope"):
		latencies = results.get(provider, [])
		if not latencies:
			print(f"{provider:<12} {'0':>5} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
			continue
		stats = calc_stats(latencies)
		print(
			f"{provider:<12} "
			f"{int(stats['count']):>5} "
			f"{stats['min']:>8.3f} "
			f"{stats['avg']:>8.3f} "
			f"{stats['median']:>8.3f} "
			f"{stats['p95']:>8.3f} "
			f"{stats['max']:>8.3f}"
		)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Compare OpenAI and DashScope API latency.")
    parser.add_argument("--prompt", type=str, default="用一句话介绍你自己。", help="测试提示词")
    parser.add_argument("--rounds", type=int, default=3, help="每个提供商调用次数")
    parser.add_argument("--openai-model", type=str, default=os.getenv("OPENAI_MODEL", "qwen3.5-plus"))
    parser.add_argument("--dashscope-model", type=str, default=os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus"))
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.rounds <= 0:
        raise ValueError("--rounds 必须大于 0")

    run_benchmark(
        rounds=args.rounds,
        prompt=args.prompt,
        openai_model=args.openai_model,
        dashscope_model=args.dashscope_model,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
	main()


