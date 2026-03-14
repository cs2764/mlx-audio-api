"""
Ming-omni-tts-0.5B API 测试脚本

测试流程:
1. 发送 2 个顺序请求（基线测试）
2. 发送 4 个并发请求
3. 发送 16 个并发请求
4. 发送 32 个并发请求
"""

import argparse
import base64
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent

# ─── 辅助函数 ────────────────────────────────────────────────────────────────────

def audio_to_bytes(file_path):
    if not file_path or not Path(file_path).exists():
        return None
    with open(file_path, "rb") as f:
        return f.read()


def read_ref_text(ref_text):
    path = Path(ref_text)
    if path.exists() and path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return f.read().strip()
    return ref_text


# ─── 配置 ───────────────────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://127.0.0.1:8013/v1/tts"
DEFAULT_REF_AUDIO = str(PROJECT_ROOT / "samples" / "0流萤.wav")
DEFAULT_REF_TEXT_FILE = str(PROJECT_ROOT / "samples" / "0流萤.txt")

TEST_TEXTS = [
    "你好，我是Ming语音合成模型，很高兴为你服务，请多关照。",           # 22 chars
    "Hello, I am the Ming TTS model, nice to meet you today.",          # 55 chars
    "人工智能技术正在改变我们的生活，语音合成让机器说话更自然流畅。",    # 26 chars
    "The weather is wonderful today, perfect for a walk in the park.",   # 63 chars
    "春眠不觉晓，处处闻啼鸟，夜来风雨声，花落知多少，这是唐诗名句。",    # 27 chars
    "To be or not to be, that is the question we all must answer.",      # 61 chars
    "Ming模型支持中英文混合输入，可以流畅切换语言，非常实用好用。",      # 25 chars
    "Machine learning has transformed how we interact with computers.",   # 63 chars
    "今天天气晴朗，阳光明媚，非常适合出门散步，享受美好的自然风光。",    # 27 chars
    "Every great journey begins with a single step forward in life.",     # 62 chars
    "语音合成技术的进步让人机交互变得更加自然，未来充满无限可能。",       # 25 chars
    "Innovation distinguishes between a leader and a follower always.",   # 64 chars
    "云计算为实时语音合成提供了强大算力，让高质量TTS成为可能实现。",     # 26 chars
    "The sun sets behind the mountains, painting the sky with colors.",   # 64 chars
    "开源社区的力量是无穷的，每一位贡献者都在推动技术不断向前进步。",    # 27 chars
    "Dreams are not what you see in sleep, they keep you from sleeping.", # 66 chars
    "深度学习模型在语音领域取得了突破性进展，合成质量媲美真人声音。",     # 26 chars
    "Artificial intelligence is reshaping industries across the globe.",  # 64 chars
    "我们相信技术的力量可以让世界变得更美好，让生活更加便利舒适。",       # 25 chars
    "The quick brown fox jumps over the lazy dog near the riverbank.",    # 63 chars
    "自然语言处理和语音合成的结合，将带来全新的人机交互体验感受。",       # 25 chars
    "Science and technology are the driving forces of human progress.",   # 63 chars
    "音频质量和推理速度是衡量TTS系统性能的两个最重要的核心指标。",       # 25 chars
    "A good voice synthesis system should sound natural and expressive.", # 66 chars
    "并发处理能力是生产环境中TTS服务最关键的性能指标之一，非常重要。",   # 27 chars
    "Real-time voice generation requires both speed and high quality.",   # 63 chars
    "模型量化技术可以在保持音质的同时大幅降低内存占用和推理延迟。",       # 25 chars
    "The best way to predict the future is to create it yourself now.",   # 64 chars
    "多语言支持让TTS系统能够服务全球用户，打破语言障碍促进交流。",       # 25 chars
    "Voice cloning technology enables personalized speech synthesis.",    # 62 chars
    "高并发场景下的稳定性测试是确保服务质量的重要环节，不可忽视。",       # 25 chars
    "Streaming audio output reduces latency for real-time applications.", # 65 chars
]

OUTPUT_DIR = PROJECT_ROOT / "test_outputs_ming"


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
) -> dict:
    payload = {"text": text}

    if ref_audio_bytes:
        payload["references"] = [
            {
                "audio": base64.b64encode(ref_audio_bytes).decode("utf-8"),
                "text": ref_text or "",
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
        response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        elapsed = time.perf_counter() - start
        result["elapsed_seconds"] = round(elapsed, 3)
        result["status_code"] = response.status_code

        if response.status_code == 200:
            audio_content = response.content
            result["audio_size_bytes"] = len(audio_content)
            result["success"] = True
            out_path = output_dir / f"request_{request_id:03d}.wav"
            out_path.write_bytes(audio_content)
            result["output_file"] = str(out_path)
        else:
            result["error"] = response.text[:200]

    except requests.exceptions.Timeout:
        result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        result["error"] = f"TIMEOUT ({timeout}s)"
    except requests.exceptions.ConnectionError as e:
        result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        result["error"] = f"CONNECTION_ERROR: {e}"
    except Exception as e:
        result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        result["error"] = f"EXCEPTION: {type(e).__name__}: {e}"

    return result


# ─── 输出工具 ─────────────────────────────────────────────────────────────────────

def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_results(results: list[dict], phase_name: str, total_elapsed: float) -> tuple[bool, dict]:
    """返回 (all_success, stats) 供后续对比分析使用"""
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    elapsed_list = [r["elapsed_seconds"] for r in results if r["success"]]
    audio_sizes = [r["audio_size_bytes"] for r in results if r["success"]]

    avg_latency = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0
    min_latency = min(elapsed_list) if elapsed_list else 0
    max_latency = max(elapsed_list) if elapsed_list else 0

    print(f"\n{'─' * 60}")
    print(f"  {phase_name} 结果汇总")
    print(f"{'─' * 60}")
    print(f"  总请求数:   {len(results)}")
    print(f"  成功:       {success_count}  失败: {fail_count}")
    print(f"  总耗时:     {total_elapsed:.3f}s")

    if elapsed_list:
        print(f"  平均延迟:   {avg_latency:.3f}s")
        print(f"  最小/最大:  {min_latency:.3f}s / {max_latency:.3f}s")
        print(f"  平均音频:   {sum(audio_sizes)/len(audio_sizes)/1024:.1f} KB")

    print(f"\n  {'ID':>4} | {'状态':^4} | {'耗时':>8} | {'大小':>10} | 文本预览")
    print(f"  {'─'*4}-+-{'─'*4}-+-{'─'*8}-+-{'─'*10}-+-{'─'*30}")
    for r in sorted(results, key=lambda x: x["request_id"]):
        status = "OK" if r["success"] else "FAIL"
        elapsed = f"{r['elapsed_seconds']:.3f}s"
        size = f"{r['audio_size_bytes']/1024:.1f}KB" if r["success"] else "N/A"
        print(f"  {r['request_id']:>4} | {status:^4} | {elapsed:>8} | {size:>10} | {r['text_preview']}")
        if r["error"]:
            print(f"       └─ 错误: {r['error'][:70]}")

    print()
    stats = {
        "phase": phase_name,
        "n": len(results),
        "success": success_count,
        "total_elapsed": total_elapsed,
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
    }
    return success_count == len(results), stats


# ─── 测试阶段 ─────────────────────────────────────────────────────────────────────

def run_sequential_test(num_requests, api_url, ref_audio_bytes, ref_text, output_dir, api_key, timeout):
    phase_name = f"顺序测试 ({num_requests} 请求)"
    print_separator(phase_name)
    sub = output_dir / f"sequential_{num_requests}"
    sub.mkdir(parents=True, exist_ok=True)

    results = []
    start_all = time.perf_counter()
    for i in range(num_requests):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        print(f"  > 请求 {i+1}/{num_requests}: {text[:35]}...")
        r = make_tts_request(text, i+1, api_url, ref_audio_bytes, ref_text, sub, api_key, timeout)
        print(f"    {'OK' if r['success'] else 'FAIL'} {r['elapsed_seconds']:.3f}s")
        results.append(r)

    return print_results(results, phase_name, time.perf_counter() - start_all)


def run_concurrent_test(num_concurrent, api_url, ref_audio_bytes, ref_text, output_dir, api_key, timeout):
    phase_name = f"并发测试 ({num_concurrent} 并发)"
    print_separator(phase_name)
    sub = output_dir / f"concurrent_{num_concurrent}"
    sub.mkdir(parents=True, exist_ok=True)

    print(f"  同时发送 {num_concurrent} 个请求...")
    results = []
    start_all = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = {
            executor.submit(
                make_tts_request,
                TEST_TEXTS[i % len(TEST_TEXTS)], i+1,
                api_url, ref_audio_bytes, ref_text, sub, api_key, timeout,
            ): i+1
            for i in range(num_concurrent)
        }
        for future in as_completed(futures):
            r = future.result()
            print(f"    {'OK' if r['success'] else 'FAIL'} 请求 #{r['request_id']} - {r['elapsed_seconds']:.3f}s")
            results.append(r)

    return print_results(results, phase_name, time.perf_counter() - start_all)


# ─── 主程序 ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Ming-omni-tts-0.5B API 测试脚本")
    parser.add_argument("--url", "-u", default=DEFAULT_API_URL, help="API 地址")
    parser.add_argument("--ref-audio", "-ra", default=DEFAULT_REF_AUDIO, help="参考音频路径")
    parser.add_argument("--ref-text-file", "-rt", default=DEFAULT_REF_TEXT_FILE, help="参考文本路径")
    parser.add_argument("--output-dir", "-o", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--api-key", default=None, help="API 密钥")
    parser.add_argument("--timeout", "-t", type=int, default=300, help="请求超时秒数")
    parser.add_argument("--skip-sequential", action="store_true", help="跳过顺序测试")
    parser.add_argument("--no-ref", action="store_true", help="不使用参考音频")
    return parser.parse_args()


def print_comparison(all_stats: list[dict]):
    """对比各阶段延迟，分析并发是否缩短了平均请求时间"""
    print_separator("并发性能对比分析")

    # 以顺序测试的平均延迟为基准
    baseline = next((s for s in all_stats if s["n"] <= 4), None)
    if not baseline:
        return

    baseline_avg = baseline["avg_latency"]

    print(f"\n  {'阶段':<22} | {'并发数':>6} | {'平均延迟':>10} | {'vs 基准':>10} | {'总耗时':>10} | {'成功率':>6}")
    print(f"  {'─'*22}-+-{'─'*6}-+-{'─'*10}-+-{'─'*10}-+-{'─'*10}-+-{'─'*6}")

    for s in all_stats:
        n = s["n"]
        avg = s["avg_latency"]
        total = s["total_elapsed"]
        rate = f"{s['success']}/{n}"

        if baseline_avg > 0:
            delta = avg - baseline_avg
            pct = (delta / baseline_avg) * 100
            vs = f"{'+' if delta >= 0 else ''}{pct:.1f}%"
        else:
            vs = "N/A"

        label = s["phase"]
        print(f"  {label:<22} | {n:>6} | {avg:>9.3f}s | {vs:>10} | {total:>9.3f}s | {rate:>6}")

    print()
    print("  结论:")

    concurrent_stats = [s for s in all_stats if s["n"] > 2]
    for s in concurrent_stats:
        avg = s["avg_latency"]
        if baseline_avg > 0:
            delta = avg - baseline_avg
            pct = abs(delta / baseline_avg) * 100
            if delta < 0:
                print(f"    {s['phase']}: 平均延迟缩短 {pct:.1f}%（{baseline_avg:.3f}s → {avg:.3f}s）✅ 并发有加速效果")
            elif delta < baseline_avg * 0.1:
                print(f"    {s['phase']}: 平均延迟基本持平（+{pct:.1f}%），队列处理高效 ✅")
            else:
                print(f"    {s['phase']}: 平均延迟增加 {pct:.1f}%（{baseline_avg:.3f}s → {avg:.3f}s），请求在队列中等待 ⚠️")
    print()


def main():
    args = parse_args()

    print_separator("Ming-omni-tts-0.5B API 测试")
    print(f"  API 地址:   {args.url}")
    print(f"  超时时间:   {args.timeout}s")
    print(f"  输出目录:   {args.output_dir}")

    ref_audio_bytes = None
    ref_text = ""

    if not args.no_ref:
        print(f"  参考音频:   {args.ref_audio}")
        if not os.path.exists(args.ref_audio):
            print(f"\n  警告: 参考音频不存在: {args.ref_audio}，将使用默认语音")
        else:
            ref_audio_bytes = audio_to_bytes(args.ref_audio)
            print(f"  参考音频大小: {len(ref_audio_bytes)/1024:.1f} KB")

        if os.path.exists(args.ref_text_file):
            ref_text = read_ref_text(args.ref_text_file)
            print(f"  参考文本: {ref_text[:60]}...")
    else:
        print("  模式: 不使用参考音频")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    kw = dict(api_url=args.url, ref_audio_bytes=ref_audio_bytes, ref_text=ref_text,
              output_dir=output_dir, api_key=args.api_key, timeout=args.timeout)

    # 阶段一：2 顺序请求（基准）
    if not args.skip_sequential:
        ok, stats = run_sequential_test(num_requests=2, **kw)
        all_stats.append(stats)
        if not ok:
            print("\n  顺序测试失败，停止。")
            sys.exit(1)
        print("  顺序测试通过，进入并发测试...\n")

    # 阶段二：4 并发
    ok, stats = run_concurrent_test(num_concurrent=4, **kw)
    all_stats.append(stats)
    if not ok:
        print("\n  4 并发测试失败，停止。")
        sys.exit(1)
    print("  4 并发测试通过，进入 16 并发测试...\n")

    # 阶段三：16 并发
    ok, stats = run_concurrent_test(num_concurrent=16, **kw)
    all_stats.append(stats)
    if not ok:
        print("\n  16 并发测试失败，停止。")
        sys.exit(1)
    print("  16 并发测试通过，进入 32 并发测试...\n")

    # 阶段四：32 并发
    _, stats = run_concurrent_test(num_concurrent=32, **kw)
    all_stats.append(stats)

    # 对比分析
    print_comparison(all_stats)


if __name__ == "__main__":
    main()
