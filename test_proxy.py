#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代理测试脚本

使用方法:
    python test_proxy.py                    # 使用默认代理 127.0.0.1:1080
    python test_proxy.py --port 7890        # 指定端口
    python test_proxy.py --no-proxy         # 不使用代理
"""

import argparse
import os
import socket
import sys
import time

try:
    import httpx
except ImportError:
    print("请先安装 httpx: pip install httpx")
    sys.exit(1)


def check_port(host: str, port: int) -> bool:
    """检查端口是否开放"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def test_http_request(proxy: str, url: str, timeout: int = 10) -> dict:
    """测试 HTTP 请求"""
    start_time = time.time()
    try:
        if proxy:
            # httpx 新版本使用 proxy 参数（单数）
            with httpx.Client(proxy=proxy, timeout=timeout, verify=False) as client:
                response = client.get(url)
        else:
            with httpx.Client(timeout=timeout, verify=False) as client:
                response = client.get(url)
        elapsed = time.time() - start_time
        return {
            "success": True,
            "status_code": response.status_code,
            "elapsed": elapsed,
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "status_code": None,
            "elapsed": elapsed,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="代理测试脚本")
    parser.add_argument("--host", default="127.0.0.1", help="代理主机地址")
    parser.add_argument("--port", type=int, default=1080, help="代理端口")
    parser.add_argument("--no-proxy", action="store_true", help="不使用代理")
    args = parser.parse_args()

    proxy = None if args.no_proxy else f"http://{args.host}:{args.port}"

    print("=" * 50)
    print("代理测试脚本")
    print("=" * 50)

    # 1. 检查端口
    print(f"\n[1] 检查代理端口 {args.host}:{args.port}")
    if args.no_proxy:
        print("    跳过（使用直连模式）")
    elif check_port(args.host, args.port):
        print(f"    ✅ 端口 {args.port} 开放")
    else:
        print(f"    ❌ 端口 {args.port} 关闭")
        print("    请检查代理服务是否启动")
        return

    # 2. 测试基本连接
    test_urls = [
        ("httpbin.org/ip", "https://httpbin.org/ip"),
        ("Google", "https://www.google.com"),
        ("Cloudflare", "https://1.1.1.1/cdn-cgi/trace"),
    ]

    print(f"\n[2] 测试基本连接")
    for name, url in test_urls:
        result = test_http_request(proxy, url)
        if result["success"]:
            # 2xx, 3xx, 4xx 都说明连接正常
            status = result['status_code']
            if status and status < 500:
                print(f"    ✅ {name}: {status} ({result['elapsed']:.2f}s)")
            else:
                print(f"    ⚠️  {name}: {status} ({result['elapsed']:.2f}s)")
        else:
            print(f"    ❌ {name}: {result['error']} ({result['elapsed']:.2f}s)")

    # 3. 测试搜索服务
    search_apis = [
        ("Serper API", "https://google.serper.dev"),
        ("DuckDuckGo", "https://duckduckgo.com"),
        ("Wikipedia", "https://en.wikipedia.org"),
        ("Jina Reader", "https://r.jina.ai"),
    ]

    print(f"\n[3] 测试搜索服务 (2xx/3xx/4xx = 连接正常)")
    for name, url in search_apis:
        result = test_http_request(proxy, url)
        if result["success"]:
            status = result['status_code']
            if status and status < 500:
                print(f"    ✅ {name}: {status} ({result['elapsed']:.2f}s)")
            else:
                print(f"    ⚠️  {name}: {status} ({result['elapsed']:.2f}s)")
        else:
            error_short = result['error'][:50] + "..." if len(result['error']) > 50 else result['error']
            print(f"    ❌ {name}: {error_short}")

    # 4. 测试国内服务（直连）
    cn_urls = [
        ("阿里云 IQS", "https://cloud-iqs.aliyuncs.com"),
        ("DashScope", "https://dashscope.aliyuncs.com"),
    ]

    print(f"\n[4] 测试国内服务 (2xx/3xx/4xx = 连接正常)")
    for name, url in cn_urls:
        result = test_http_request(proxy, url)
        if result["success"]:
            status = result['status_code']
            if status and status < 500:
                print(f"    ✅ {name}: {status} ({result['elapsed']:.2f}s)")
            else:
                print(f"    ⚠️  {name}: {status} ({result['elapsed']:.2f}s)")
        else:
            error_short = result['error'][:50] + "..." if len(result['error']) > 50 else result['error']
            print(f"    ❌ {name}: {error_short}")

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
