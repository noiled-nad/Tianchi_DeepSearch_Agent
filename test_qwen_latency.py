#!/usr/bin/env python3
"""测试 Qwen3.5-Plus 从请求到开始流式输出的延迟时间"""

import os
import time
from dashscope import Generation


def test_streaming_latency():
    """测试流式输出的首字延迟"""
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
        return

    prompt = "请用一句话介绍人工智能"

    print("=" * 50)
    print("测试 Qwen3.5-Plus 流式输出延迟")
    print("=" * 50)

    # 记录请求开始时间
    start_time = time.time()
    first_chunk_time = None
    chunk_count = 0

    responses = Generation.call(
        model='qwen3.5-plus',
        prompt=prompt,
        stream=True,
        result_format='message',
        incremental_output=True,
    )

    for response in responses:
        chunk_count += 1
        if chunk_count == 1:
            first_chunk_time = time.time()
            ttfb = (first_chunk_time - start_time) * 1000
            print(f"\n[首块延迟 TTFB]: {ttfb:.2f} ms")

        # 打印内容
        if response.output and response.output.choices:
            content = response.output.choices[0].message.content
            if content:
                print(content, end='', flush=True)

    end_time = time.time()
    total_time = (end_time - start_time) * 1000

    print(f"\n\n[总耗时]: {total_time:.2f} ms")
    print(f"[总块数]: {chunk_count}")


def test_non_streaming_latency():
    """测试非流式输出的延迟"""
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
        return

    prompt = "请用一句话介绍人工智能"

    print("\n" + "=" * 50)
    print("测试 Qwen3.5-Plus 非流式输出延迟")
    print("=" * 50)

    start_time = time.time()

    response = Generation.call(
        model='qwen3.5-plus',
        prompt=prompt,
        stream=False,
        result_format='message',
    )

    end_time = time.time()
    latency = (end_time - start_time) * 1000

    print(f"\n[响应延迟]: {latency:.2f} ms")
    if response.output and response.output.choices:
        print(f"[响应内容]: {response.output.choices[0].message.content}")


if __name__ == '__main__':
    # 测试多次取平均
    print("\n运行 3 次测试取平均值...\n")

    for i in range(3):
        print(f"\n### 第 {i+1} 次测试 ###")
        test_streaming_latency()
        print()

    # test_non_streaming_latency()
