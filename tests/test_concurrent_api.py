"""
TTS Inference API 并发测试脚本

使用 /v1/tts 端点进行测试，支持声音克隆和 VoiceDesign 模式。

测试流程:
1. 发送 2 个顺序请求（基线测试）
2. 发送 4 个并发请求
3. 如果 4 并发正常，再测试 16 并发
"""

import argparse
import base64
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ─── 项目根目录 ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent


# ─── 辅助函数 ────────────────────────────────────────────────────────────────────

def audio_to_bytes(file_path):
    """读取音频文件为 bytes"""
    if not file_path or not Path(file_path).exists():
        return None
    with open(file_path, "rb") as f:
        return f.read()


def read_ref_text(ref_text):
    """读取参考文本（支持文件路径或直接文本）"""
    path = Path(ref_text)
    if path.exists() and path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return f.read()
    return ref_text


# ─── 配置 ───────────────────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://127.0.0.1:8013/v1/tts"
DEFAULT_REF_AUDIO = str(PROJECT_ROOT / "samples" / "0流萤.wav")
DEFAULT_REF_TEXT_FILE = str(PROJECT_ROOT / "samples" / "0流萤.txt")

# 测试文本列表（中英文混合）
TEST_TEXTS = [
    "你好，欢迎使用语音合成系统。今天的天气非常不错，适合出去散步。",
    "Hello, welcome to the speech synthesis system. The weather is wonderful today.",
    "在这个快速发展的时代，人工智能技术正在改变我们的生活方式，让一切变得更加便利。",
    "Technology is advancing at an unprecedented pace, bringing new possibilities every day.",
    "春风得意马蹄疾，一日看尽长安花。这是一首非常经典的唐诗，描写了科举高中后的喜悦。",
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "人工智能的发展让我们对未来充满期待，语音合成技术也在不断进步。",
    "Music has the power to transport us to different worlds and evoke deep emotions within us.",
    "大语言模型和语音合成的结合，将带来全新的人机交互体验。",
    "Every great journey begins with a single step. Let us take that step together today.",
    "这个世界上最美好的事情，就是能够用声音传递温暖和力量。",
    "Innovation distinguishes between a leader and a follower. Stay hungry, stay foolish.",
    "云计算和边缘计算的结合，为实时语音合成提供了强大的算力支持。",
    "The sun sets behind the mountains, painting the sky with shades of orange and purple.",
    "开源社区的力量是无穷的，每一位贡献者都在推动技术的进步。",
    "Dreams are not what you see in your sleep. Dreams are things that do not let you sleep.",
]

OUTPUT_DIR = PROJECT_ROOT / "test_outputs"


# ─── 核心请求函数 ─────────────────────────────────────────────────────────────────

def make_tts_request(
    text: str,
    request_id: int,
    api_url: str,
    ref_audio_bytes: bytes | None,
    ref_text: str,
    output_dir: Path,
    api_key: str | None = None,
    timeout: int = 300,
    instruct: str | None = None,
) -> dict:
    """发送单个 TTS 请求并返回结果统计"""

    # 构建请求体
    payload = {
        "text": text,
    }

    # VoiceDesign 模式（Qwen3 等）
    if instruct:
        payload["instruct"] = instruct
    # 声音克隆模式
    elif ref_audio_bytes:
        payload["references"] = [
            {
                "audio": base64.b64encode(ref_audio_bytes).decode("utf-8"),
                "text": ref_text if ref_text else "",
            }
        ]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    result = {
        "request_id": request_id,
        "text_preview": text[:40] + ("..." if len(text) > 40 else ""),
        "text_length": len(text),
        "success": False,
        "status_code": None,
        "audio_size_bytes": 0,
        "elapsed_seconds": 0.0,
        "error": None,
    }

    start = time.perf_counter()
    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        result["elapsed_seconds"] = round(elapsed, 3)
        result["status_code"] = response.status_code

        if response.status_code == 200:
            audio_content = response.content
            result["audio_size_bytes"] = len(audio_content)
            result["success"] = True

            # 保存到文件
            out_path = output_dir / f"request_{request_id:03d}.wav"
            with open(out_path, "wb") as f:
                f.write(audio_content)
            result["output_file"] = str(out_path)
        else:
            try:
                result["error"] = response.text[:200]
            except Exception:
                result["error"] = f"HTTP {response.status_code}"

    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - start
        result["elapsed_seconds"] = round(elapsed, 3)
        result["error"] = f"TIMEOUT ({timeout}s)"
    except requests.exceptions.ConnectionError as e:
        elapsed = time.perf_counter() - start
        result["elapsed_seconds"] = round(elapsed, 3)
        result["error"] = f"CONNECTION_ERROR: {e}"
    except Exception as e:
        elapsed = time.perf_counter() - start
        result["elapsed_seconds"] = round(elapsed, 3)
        result["error"] = f"EXCEPTION: {type(e).__name__}: {e}"

    return result


# ─── 测试阶段 ─────────────────────────────────────────────────────────────────────

def print_separator(title: str):
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_results(results: list[dict], phase_name: str, total_elapsed: float):
    """打印测试结果汇总"""

    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    elapsed_list = [r["elapsed_seconds"] for r in results if r["success"]]
    audio_sizes = [r["audio_size_bytes"] for r in results if r["success"]]

    print(f"\n{'─' * 60}")
    print(f"  📊 {phase_name} 结果汇总")
    print(f"{'─' * 60}")
    print(f"  总请求数:     {len(results)}")
    print(f"  ✅ 成功:       {success_count}")
    print(f"  ❌ 失败:       {fail_count}")
    print(f"  ⏱️  总耗时:     {total_elapsed:.3f}s")

    if elapsed_list:
        avg_latency = sum(elapsed_list) / len(elapsed_list)
        min_latency = min(elapsed_list)
        max_latency = max(elapsed_list)
        avg_audio = sum(audio_sizes) / len(audio_sizes) / 1024
        print(f"  📈 平均延迟:   {avg_latency:.3f}s")
        print(f"  📉 最小延迟:   {min_latency:.3f}s")
        print(f"  📈 最大延迟:   {max_latency:.3f}s")
        print(f"  🔊 平均音频:   {avg_audio:.1f} KB")

    # 逐请求详情
    print(f"\n  {'ID':>4} | {'状态':^6} | {'耗时':>8} | {'音频大小':>10} | 文本预览")
    print(f"  {'─'*4}-+-{'─'*6}-+-{'─'*8}-+-{'─'*10}-+-{'─'*30}")
    for r in results:
        status = "✅" if r["success"] else "❌"
        elapsed = f"{r['elapsed_seconds']:.3f}s"
        size = f"{r['audio_size_bytes'] / 1024:.1f}KB" if r["success"] else "N/A"
        preview = r["text_preview"]
        print(f"  {r['request_id']:>4} | {status:^6} | {elapsed:>8} | {size:>10} | {preview}")
        if r["error"]:
            print(f"       └─ 错误: {r['error'][:60]}")

    print()
    return success_count == len(results)


def run_sequential_test(
    num_requests: int,
    api_url: str,
    ref_audio_bytes: bytes | None,
    ref_text: str,
    output_dir: Path,
    api_key: str | None = None,
    timeout: int = 300,
    instruct: str | None = None,
) -> bool:
    """运行顺序请求测试"""

    phase_name = f"顺序测试 ({num_requests} 请求)"
    print_separator(phase_name)

    output_sub = output_dir / f"sequential_{num_requests}"
    output_sub.mkdir(parents=True, exist_ok=True)

    results = []
    start_all = time.perf_counter()

    for i in range(num_requests):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        print(f"  ▶ 发送请求 {i + 1}/{num_requests}: {text[:30]}...")
        result = make_tts_request(
            text=text,
            request_id=i + 1,
            api_url=api_url,
            ref_audio_bytes=ref_audio_bytes,
            ref_text=ref_text,
            output_dir=output_sub,
            api_key=api_key,
            timeout=timeout,
            instruct=instruct,
        )
        status_icon = "✅" if result["success"] else "❌"
        print(f"    {status_icon} 完成 - {result['elapsed_seconds']:.3f}s")
        results.append(result)

    total_elapsed = time.perf_counter() - start_all
    return print_results(results, phase_name, total_elapsed)


def run_concurrent_test(
    num_concurrent: int,
    api_url: str,
    ref_audio_bytes: bytes | None,
    ref_text: str,
    output_dir: Path,
    api_key: str | None = None,
    timeout: int = 300,
    instruct: str | None = None,
) -> bool:
    """运行并发请求测试"""

    phase_name = f"并发测试 ({num_concurrent} 并发)"
    print_separator(phase_name)

    output_sub = output_dir / f"concurrent_{num_concurrent}"
    output_sub.mkdir(parents=True, exist_ok=True)

    print(f"  🚀 同时发送 {num_concurrent} 个并发请求...")

    results = []
    start_all = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = {}
        for i in range(num_concurrent):
            text = TEST_TEXTS[i % len(TEST_TEXTS)]
            future = executor.submit(
                make_tts_request,
                text=text,
                request_id=i + 1,
                api_url=api_url,
                ref_audio_bytes=ref_audio_bytes,
                ref_text=ref_text,
                output_dir=output_sub,
                api_key=api_key,
                timeout=timeout,
                instruct=instruct,
            )
            futures[future] = i + 1

        for future in as_completed(futures):
            req_id = futures[future]
            result = future.result()
            status_icon = "✅" if result["success"] else "❌"
            print(f"    {status_icon} 请求 #{req_id} 完成 - {result['elapsed_seconds']:.3f}s")
            results.append(result)

    results.sort(key=lambda r: r["request_id"])
    total_elapsed = time.perf_counter() - start_all
    return print_results(results, phase_name, total_elapsed)


# ─── 主程序 ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TTS Inference API 并发测试脚本",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--url", "-u", type=str, default=DEFAULT_API_URL,
        help=f"API 服务地址 (默认: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--ref-audio", "-ra", type=str, default=DEFAULT_REF_AUDIO,
        help="参考音频文件路径（用于声音克隆）",
    )
    parser.add_argument(
        "--ref-text-file", "-rt", type=str, default=DEFAULT_REF_TEXT_FILE,
        help="参考文本文件路径",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=str(OUTPUT_DIR),
        help="输出目录",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API 密钥（如已配置认证）",
    )
    parser.add_argument(
        "--timeout", "-t", type=int, default=300,
        help="单个请求超时时间，单位秒（默认: 300）",
    )
    parser.add_argument(
        "--skip-sequential", action="store_true",
        help="跳过顺序测试，直接进入并发测试",
    )
    parser.add_argument(
        "--no-ref", action="store_true",
        help="不使用参考音频（跳过声音克隆）",
    )
    parser.add_argument(
        "--instruct", type=str, default=None,
        help="VoiceDesign 声音描述（用于 Qwen3 等模型，例如: 'A cheerful young female voice'）\n"
             "设置后将忽略 --ref-audio",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print_separator("TTS Inference API 并发测试")
    print(f"  API 地址:     {args.url}")
    print(f"  超时时间:     {args.timeout}s")
    print(f"  输出目录:     {args.output_dir}")

    # 加载参考音频和文本
    ref_audio_bytes = None
    ref_text = ""

    if args.instruct:
        print(f"  模式:         VoiceDesign")
        print(f"  声音描述:     {args.instruct}")
    elif not args.no_ref:
        print(f"  参考音频:     {args.ref_audio}")
        print(f"  参考文本文件: {args.ref_text_file}")

        if not os.path.exists(args.ref_audio):
            print(f"\n  ⚠️  参考音频文件不存在: {args.ref_audio}")
            print("  将使用默认语音（无声音克隆）")
        else:
            print("\n  📂 加载参考音频...")
            ref_audio_bytes = audio_to_bytes(args.ref_audio)
            print(f"  ✅ 参考音频大小: {len(ref_audio_bytes) / 1024:.1f} KB")

        if os.path.exists(args.ref_text_file):
            ref_text = read_ref_text(args.ref_text_file)
            print(f"  ✅ 参考文本: {ref_text[:60]}...")
    else:
        print("  模式:         不使用参考音频")

    # 创建输出目录（先清空旧内容）
    import shutil
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"  🗑️  已清空输出目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    instruct = args.instruct

    # ─── 阶段一：2 个顺序请求 ───
    if not args.skip_sequential:
        seq_ok = run_sequential_test(
            num_requests=2,
            api_url=args.url,
            ref_audio_bytes=ref_audio_bytes,
            ref_text=ref_text,
            output_dir=output_dir,
            api_key=args.api_key,
            timeout=args.timeout,
            instruct=instruct,
        )
        if not seq_ok:
            print("\n  ⛔ 顺序测试失败，停止后续测试。")
            sys.exit(1)
        print("  ✅ 顺序测试通过！进入并发测试...\n")

    # ─── 阶段二：4 并发请求 ───
    concurrent_4_ok = run_concurrent_test(
        num_concurrent=4,
        api_url=args.url,
        ref_audio_bytes=ref_audio_bytes,
        ref_text=ref_text,
        output_dir=output_dir,
        api_key=args.api_key,
        timeout=args.timeout,
        instruct=instruct,
    )

    if not concurrent_4_ok:
        print("\n  ⛔ 4 并发测试失败，停止后续测试。")
        sys.exit(1)

    print("  ✅ 4 并发测试通过！进入 16 并发测试...\n")

    # ─── 阶段三：16 并发请求 ───
    concurrent_16_ok = run_concurrent_test(
        num_concurrent=16,
        api_url=args.url,
        ref_audio_bytes=ref_audio_bytes,
        ref_text=ref_text,
        output_dir=output_dir,
        api_key=args.api_key,
        timeout=args.timeout,
        instruct=instruct,
    )

    # ─── 最终汇总 ───
    print_separator("🏁 测试完成 - 最终结果")
    if not args.skip_sequential:
        print("  ✅ 顺序测试 (2 请求):   通过")
    print("  ✅ 并发测试 (4 并发):   通过")
    if concurrent_16_ok:
        print("  ✅ 并发测试 (16 并发):  通过")
    else:
        print("  ❌ 并发测试 (16 并发):  失败")

    print(f"\n  📁 所有生成的音频文件保存在: {output_dir}")
    print()


if __name__ == "__main__":
    main()
