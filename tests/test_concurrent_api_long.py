"""
TTS Inference API 并发测试脚本（长文本版）

与 test_concurrent_api.py 相同逻辑，但每条测试文本不低于 200 字，
用于测试模型在长文本输入下的稳定性和并发表现。

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

# 测试文本列表 —— 每条均不低于 200 字（中英文混合）
TEST_TEXTS = [
    # 1. 中文 - 人工智能与未来
    "人工智能技术正在以前所未有的速度改变着我们的世界。从智能手机上的语音助手，到自动驾驶汽车，再到医疗影像诊断，人工智能的应用已经渗透到生活的方方面面。语音合成作为人工智能的重要分支，让机器能够以自然流畅的方式与人类交流。未来，随着大语言模型和神经网络技术的不断进步，语音合成的质量将越来越接近真人水平，为教育、娱乐、无障碍服务等领域带来深远影响。我们正站在一个技术变革的十字路口，每一次创新都在重新定义人与机器之间的边界。",

    # 2. English - Technology and Human Connection
    "Technology has always been a double-edged sword, offering both tremendous opportunities and significant challenges. In the realm of artificial intelligence, we are witnessing a transformation that touches every aspect of human life. Speech synthesis, once a novelty confined to science fiction, has become a practical tool used by millions of people every day. From helping individuals with visual impairments to enabling seamless multilingual communication, text-to-speech systems are breaking down barriers and creating new possibilities. As we continue to push the boundaries of what machines can do, it is essential that we remain mindful of the human values that should guide technological progress.",

    # 3. 中文 - 自然与季节
    "春天来了，大地从漫长的冬眠中苏醒过来。田野里的麦苗开始返青，山坡上的桃花竞相绽放，空气中弥漫着泥土和花朵混合的清新气息。燕子从南方飞回，在屋檐下筑起新巢，孩子们脱下厚重的棉衣，在阳光下奔跑嬉戏。农民们开始忙碌地耕种，播下希望的种子。夏天的热烈、秋天的丰收、冬天的宁静，每一个季节都有它独特的美丽和意义。大自然用它亘古不变的节律，提醒着我们生命的循环与更新，让我们在忙碌的现代生活中，依然能感受到那份原始而纯粹的美好。",

    # 4. English - The Art of Storytelling
    "Storytelling is one of the oldest and most powerful forms of human communication. Long before the invention of writing, our ancestors gathered around fires to share tales of adventure, wisdom, and wonder. These stories served not only as entertainment but as a means of preserving cultural knowledge and transmitting values from one generation to the next. Today, in the age of digital media and artificial intelligence, the art of storytelling continues to evolve. Podcasts, audiobooks, and voice assistants have given new life to the spoken word, reminding us that the human voice carries a unique emotional resonance that no other medium can fully replicate. A well-told story can inspire, comfort, and transform the listener.",

    # 5. 中文 - 城市与乡村
    "城市是现代文明的结晶，高楼大厦、霓虹灯光、川流不息的人群，构成了一幅充满活力的画卷。然而，在这繁华背后，许多人心中依然保留着对乡村的深深眷恋。乡村的清晨，鸡鸣声唤醒沉睡的村庄，炊烟袅袅升起，老人们在院子里打太极，孩子们背着书包走向学校。那里没有城市的喧嚣，却有着城市无法给予的宁静与踏实。城乡之间的差距正在逐渐缩小，越来越多的年轻人选择回到家乡创业，用新技术和新理念为古老的土地注入新的生机与活力。城市与乡村，各有其美。",

    # 6. English - Music and Emotion
    "Music is a universal language that transcends cultural boundaries and speaks directly to the human soul. Whether it is the soaring melody of a symphony orchestra, the rhythmic pulse of a jazz ensemble, or the intimate strumming of an acoustic guitar, music has the power to evoke emotions that words alone cannot express. Scientists have discovered that listening to music activates multiple regions of the brain simultaneously, triggering memories, influencing mood, and even affecting physical responses such as heart rate and breathing. For centuries, musicians and composers have harnessed this power to create works that endure across generations. In our modern world, music remains one of the most profound ways in which human beings connect with one another and with their own inner lives.",

    # 7. 中文 - 科技与教育
    "教育是民族振兴的基石，而科技的发展正在深刻地改变教育的面貌。在线学习平台让优质教育资源突破地域限制，偏远山区的孩子也能聆听名师讲课。人工智能辅助教学系统能够根据每个学生的学习进度和薄弱环节，提供个性化的练习和反馈，大大提高了学习效率。虚拟现实技术让学生能够身临其境地探索历史遗迹、宇宙星空或人体内部结构，使抽象的知识变得生动具体。语音合成技术在语言学习领域也发挥着重要作用，帮助学习者纠正发音、提升口语表达能力。未来的教育将是人与技术深度融合的全新体验。",

    # 8. English - Ocean and Exploration
    "The ocean covers more than seventy percent of our planet's surface, yet it remains one of the least explored frontiers on Earth. Beneath the waves lies a world of extraordinary diversity and mystery, from the sunlit coral reefs teeming with colorful fish to the crushing darkness of the deep sea trenches where bizarre creatures have evolved to survive in conditions that would be lethal to most life forms. Ocean exploration has yielded countless scientific discoveries, including new species, geological formations, and insights into the history of our planet. As climate change threatens marine ecosystems, understanding and protecting the ocean has never been more urgent. The sea has inspired poets, navigators, and dreamers throughout human history, and it continues to call to those who seek the unknown.",

    # 9. 中文 - 传统文化与现代生活
    "中华文明拥有五千年的悠久历史，积淀了丰富而深厚的文化遗产。从诗词歌赋到书法绘画，从京剧昆曲到民间工艺，每一种传统艺术形式都承载着先人的智慧与情感。在现代化浪潮的冲击下，如何保护和传承这些宝贵的文化遗产，成为了一个重要的时代课题。令人欣慰的是，越来越多的年轻人开始对传统文化产生浓厚兴趣，汉服、茶道、古琴等传统元素在当代生活中焕发出新的光彩。将传统与现代有机融合，让古老的文化在新时代继续生长，是我们这一代人共同的责任与使命。",

    # 10. English - Space and the Cosmos
    "Since the dawn of civilization, human beings have looked up at the night sky with a mixture of awe and curiosity. The stars, planets, and galaxies that populate the cosmos have inspired myths, guided navigators, and fueled the imagination of scientists and dreamers alike. In the twentieth century, humanity took its first tentative steps beyond the confines of our home planet, sending astronauts to the Moon and robotic probes to the far reaches of the solar system. Today, space exploration is entering a new era, driven by both government agencies and private companies with ambitious plans for Mars colonization and beyond. The universe is vast beyond comprehension, containing billions of galaxies each with billions of stars, and the question of whether we are alone in this immensity remains one of the most profound mysteries of our existence.",

    # 11. 中文 - 饮食文化
    "中国饮食文化博大精深，源远流长，是中华文明的重要组成部分。八大菜系各具特色，川菜的麻辣鲜香、粤菜的清淡精致、鲁菜的醇厚大气、淮扬菜的细腻雅致，无不体现着不同地域的风土人情和历史积淀。一道好菜不仅仅是味觉的享受，更是文化的传承和情感的寄托。逢年过节，一家人围坐在餐桌旁，共享丰盛的佳肴，那种温馨与幸福是任何语言都难以完全表达的。随着全球化的深入，中国美食走向世界，成为连接不同文化、增进相互理解的重要纽带。食物，是最温柔的外交官。",

    # 12. English - The Power of Reading
    "Reading is one of the most transformative activities available to human beings. Through books, we can travel to distant lands, inhabit the minds of characters utterly unlike ourselves, and gain access to the accumulated wisdom of countless generations. A single book has the power to change a life, to open a mind that was previously closed, or to provide comfort in times of darkness and uncertainty. In an age of social media and instant gratification, the slow, immersive experience of reading a book offers a rare opportunity for deep reflection and sustained attention. Libraries and bookstores remain sacred spaces where the full range of human experience is preserved and made accessible to all. Whether fiction or nonfiction, poetry or prose, every book is an invitation to see the world through new eyes.",

    # 13. 中文 - 环境保护
    "地球是我们共同的家园，保护环境是每一个地球公民的责任。近年来，气候变化、空气污染、水资源短缺等环境问题日益严峻，引起了全球社会的广泛关注。可再生能源的开发利用、绿色出行方式的推广、垃圾分类回收制度的建立，都是应对环境挑战的重要举措。每个人的日常行为都与环境息息相关，节约用水用电、减少一次性塑料制品的使用、选择低碳的生活方式，这些看似微小的改变，汇聚在一起就能产生巨大的力量。我们有责任为子孙后代留下一个山清水秀、天蓝地绿的美好世界。",

    # 14. English - Friendship and Human Bonds
    "Friendship is one of the most precious gifts that life has to offer. True friends are those who stand by us in times of difficulty, celebrate our successes without envy, and offer honest counsel when we need it most. Unlike family relationships, which are determined by birth, friendships are chosen, and this element of choice makes them particularly meaningful. Research in psychology and sociology has consistently shown that strong social connections are among the most important predictors of happiness, health, and longevity. In our increasingly digital world, maintaining deep and authentic friendships requires intentional effort. The conversations that matter most often happen face to face, over a shared meal or a long walk, when we allow ourselves to be fully present with another person.",

    # 15. 中文 - 创新与创业
    "创新是推动社会进步的核心动力，而创业则是将创新理念转化为现实价值的重要途径。在这个充满机遇与挑战的时代，越来越多的年轻人选择走上创业之路，用自己的智慧和勇气去开创一番事业。从硅谷的科技独角兽到中关村的创业孵化器，从深圳的硬件创新到杭州的电商生态，中国的创业浪潮正在席卷全球。成功的创业者不仅需要敏锐的市场洞察力和扎实的专业技能，更需要坚韧不拔的意志和不断学习的精神。失败是创业路上不可避免的一部分，每一次跌倒都是宝贵的经验积累，为下一次的腾飞奠定基础。",

    # 16. English - The Importance of Sleep
    "Sleep is one of the most fundamental biological needs of the human body, yet it is also one of the most commonly neglected aspects of modern health. During sleep, the brain consolidates memories, processes emotions, and clears out metabolic waste products that accumulate during waking hours. The body repairs tissues, synthesizes proteins, and releases hormones essential for growth and development. Chronic sleep deprivation has been linked to a wide range of health problems, including cardiovascular disease, obesity, diabetes, and impaired immune function. Despite this, millions of people around the world regularly sacrifice sleep in favor of work, entertainment, or social obligations. Prioritizing adequate, high-quality sleep is not a luxury but a necessity for maintaining physical health, mental clarity, and emotional well-being.",
]

OUTPUT_DIR = PROJECT_ROOT / "test_outputs_long"


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

    payload = {"text": text}

    if instruct:
        payload["instruct"] = instruct
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
        response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        elapsed = time.perf_counter() - start
        result["elapsed_seconds"] = round(elapsed, 3)
        result["status_code"] = response.status_code

        if response.status_code == 200:
            audio_content = response.content
            result["audio_size_bytes"] = len(audio_content)
            result["success"] = True
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
    """打印测试结果汇总，返回 (all_ok, stats_dict)"""

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

    avg_latency = 0.0
    if elapsed_list:
        avg_latency = sum(elapsed_list) / len(elapsed_list)
        min_latency = min(elapsed_list)
        max_latency = max(elapsed_list)
        avg_audio = sum(audio_sizes) / len(audio_sizes) / 1024
        print(f"  📈 平均延迟:   {avg_latency:.3f}s")
        print(f"  📉 最小延迟:   {min_latency:.3f}s")
        print(f"  📈 最大延迟:   {max_latency:.3f}s")
        print(f"  🔊 平均音频:   {avg_audio:.1f} KB")

    print(f"\n  {'ID':>4} | {'状态':^6} | {'耗时':>8} | {'音频大小':>10} | {'字数':>5} | 文本预览")
    print(f"  {'─'*4}-+-{'─'*6}-+-{'─'*8}-+-{'─'*10}-+-{'─'*5}-+-{'─'*30}")
    for r in results:
        status = "✅" if r["success"] else "❌"
        elapsed = f"{r['elapsed_seconds']:.3f}s"
        size = f"{r['audio_size_bytes'] / 1024:.1f}KB" if r["success"] else "N/A"
        chars = str(r["text_length"])
        preview = r["text_preview"]
        print(f"  {r['request_id']:>4} | {status:^6} | {elapsed:>8} | {size:>10} | {chars:>5} | {preview}")
        if r["error"]:
            print(f"       └─ 错误: {r['error'][:60]}")

    print()
    all_ok = success_count == len(results)
    avg_per_req = total_elapsed / len(results) if results else 0.0
    stats = {
        "phase_name": phase_name,
        "num_requests": len(results),
        "total_elapsed": total_elapsed,
        "avg_per_request": avg_per_req,
        "avg_latency": avg_latency,
        "all_ok": all_ok,
    }
    return all_ok, stats


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
    phase_name = f"顺序测试 ({num_requests} 请求)"
    print_separator(phase_name)

    output_sub = output_dir / f"sequential_{num_requests}"
    output_sub.mkdir(parents=True, exist_ok=True)

    results = []
    start_all = time.perf_counter()

    for i in range(num_requests):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        print(f"  ▶ 发送请求 {i + 1}/{num_requests} ({len(text)} 字): {text[:30]}...")
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
    ok, stats = print_results(results, phase_name, total_elapsed)
    return ok, stats


def run_concurrent_test(
    num_concurrent: int,
    api_url: str,
    ref_audio_bytes: bytes | None,
    ref_text: str,
    output_dir: Path,
    api_key: str | None = None,
    timeout: int = 300,
    instruct: str | None = None,
    stagger_ms: int = 10,
) -> bool:
    phase_name = f"并发测试 ({num_concurrent} 并发)"
    print_separator(phase_name)

    output_sub = output_dir / f"concurrent_{num_concurrent}"
    output_sub.mkdir(parents=True, exist_ok=True)

    print(f"  🚀 发送 {num_concurrent} 个并发请求（间隔 {stagger_ms}ms，每条文本 ≥200 字）...")

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
            if i < num_concurrent - 1:
                time.sleep(stagger_ms / 1000)

        for future in as_completed(futures):
            req_id = futures[future]
            result = future.result()
            status_icon = "✅" if result["success"] else "❌"
            print(f"    {status_icon} 请求 #{req_id} 完成 - {result['elapsed_seconds']:.3f}s ({result['text_length']} 字)")
            results.append(result)

    results.sort(key=lambda r: r["request_id"])
    total_elapsed = time.perf_counter() - start_all
    ok, stats = print_results(results, phase_name, total_elapsed)
    return ok, stats


# ─── 性能对比总结 ──────────────────────────────────────────────────────────────────

def print_performance_summary(all_stats: list[dict]):
    W = 80
    print("\n" + "=" * W)
    print("📈 性能对比总结")
    print("=" * W)

    col_phase = 26
    col_n     = 9
    col_total = 13
    col_avg   = 13
    col_lat   = 13
    col_ok    = 6

    header = (
        f"  {'阶段':<{col_phase}}| {'请求数':>{col_n}} | {'总耗时':>{col_total}} |"
        f" {'平均/请求':>{col_avg}} | {'平均延迟':>{col_lat}} | {'状态':^{col_ok}}"
    )
    sep = (
        f"  {'─'*col_phase}-+-{'─'*col_n}-+-{'─'*col_total}-+-"
        f"{'─'*col_avg}-+-{'─'*col_lat}-+-{'─'*col_ok}"
    )
    print(header)
    print(sep)

    for s in all_stats:
        status = "✅" if s["all_ok"] else "❌"
        print(
            f"  {s['phase_name']:<{col_phase}}| {s['num_requests']:>{col_n}} |"
            f" {s['total_elapsed']:>{col_total - 1}.3f}s |"
            f" {s['avg_per_request']:>{col_avg - 1}.3f}s |"
            f" {s['avg_latency']:>{col_lat - 1}.3f}s | {status:^{col_ok}}"
        )

    print("─" * W)

    baseline = next((s for s in all_stats if "顺序" in s["phase_name"]), None)
    concurrent_stats = [s for s in all_stats if "并发" in s["phase_name"]]

    if baseline and concurrent_stats:
        print(f"\n📊 并发效率分析（以「{baseline['phase_name']}」为基线）：")
        baseline_avg = baseline["avg_per_request"]
        for s in concurrent_stats:
            if s["avg_per_request"] > 0:
                speedup = baseline_avg / s["avg_per_request"]
                print(f"  {s['phase_name']:<28}: 平均 {s['avg_per_request']:.3f}s/请求, 吞吐量提升 {speedup:.1f}x")
    print()


# ─── 主程序 ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TTS Inference API 并发测试脚本（长文本版，每条文本 ≥200 字）",
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

    print_separator("TTS Inference API 并发测试（长文本版）")
    print(f"  API 地址:     {args.url}")
    print(f"  超时时间:     {args.timeout}s")
    print(f"  输出目录:     {args.output_dir}")
    print(f"  文本长度:     每条 ≥200 字")

    # 验证所有测试文本长度
    short_texts = [(i + 1, len(t)) for i, t in enumerate(TEST_TEXTS) if len(t) < 200]
    if short_texts:
        print(f"\n  ⚠️  以下文本不足 200 字: {short_texts}")
        sys.exit(1)
    print(f"  ✅ 共 {len(TEST_TEXTS)} 条测试文本，最短 {min(len(t) for t in TEST_TEXTS)} 字，最长 {max(len(t) for t in TEST_TEXTS)} 字")

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

    import shutil
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"  🗑️  已清空输出目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    instruct = args.instruct

    all_stats = []

    # ─── 阶段一：2 个顺序请求 ───
    if not args.skip_sequential:
        seq_ok, seq_stats = run_sequential_test(
            num_requests=2,
            api_url=args.url,
            ref_audio_bytes=ref_audio_bytes,
            ref_text=ref_text,
            output_dir=output_dir,
            api_key=args.api_key,
            timeout=args.timeout,
            instruct=instruct,
        )
        all_stats.append(seq_stats)
        if not seq_ok:
            print("\n  ⛔ 顺序测试失败，停止后续测试。")
            sys.exit(1)
        print("  ✅ 顺序测试通过！进入并发测试...\n")

    # ─── 阶段二：4 并发请求 ───
    concurrent_4_ok, stats_4 = run_concurrent_test(
        num_concurrent=4,
        api_url=args.url,
        ref_audio_bytes=ref_audio_bytes,
        ref_text=ref_text,
        output_dir=output_dir,
        api_key=args.api_key,
        timeout=args.timeout,
        instruct=instruct,
    )
    all_stats.append(stats_4)

    if not concurrent_4_ok:
        print("\n  ⛔ 4 并发测试失败，停止后续测试。")
        print_performance_summary(all_stats)
        sys.exit(1)

    print("  ✅ 4 并发测试通过！进入 16 并发测试...\n")

    # ─── 阶段三：16 并发请求 ───
    concurrent_16_ok, stats_16 = run_concurrent_test(
        num_concurrent=16,
        api_url=args.url,
        ref_audio_bytes=ref_audio_bytes,
        ref_text=ref_text,
        output_dir=output_dir,
        api_key=args.api_key,
        timeout=args.timeout,
        instruct=instruct,
    )
    all_stats.append(stats_16)

    # ─── 阶段四：32 并发请求 ───
    concurrent_32_ok, stats_32 = run_concurrent_test(
        num_concurrent=32,
        api_url=args.url,
        ref_audio_bytes=ref_audio_bytes,
        ref_text=ref_text,
        output_dir=output_dir,
        api_key=args.api_key,
        timeout=args.timeout,
        instruct=instruct,
    )
    all_stats.append(stats_32)

    # ─── 阶段五：64 并发请求 ───
    concurrent_64_ok, stats_64 = run_concurrent_test(
        num_concurrent=64,
        api_url=args.url,
        ref_audio_bytes=ref_audio_bytes,
        ref_text=ref_text,
        output_dir=output_dir,
        api_key=args.api_key,
        timeout=args.timeout,
        instruct=instruct,
    )
    all_stats.append(stats_64)

    # ─── 性能对比总结 ───
    print_performance_summary(all_stats)

    # ─── 最终状态 ───
    print_separator("🏁 测试完成 - 最终结果")
    if not args.skip_sequential:
        print("  ✅ 顺序测试 (2 请求):   通过")
    print("  ✅ 并发测试 (4 并发):   通过")
    status_16 = "✅" if concurrent_16_ok else "❌"
    status_32 = "✅" if concurrent_32_ok else "❌"
    status_64 = "✅" if concurrent_64_ok else "❌"
    print(f"  {status_16} 并发测试 (16 并发):  {'通过' if concurrent_16_ok else '失败'}")
    print(f"  {status_32} 并发测试 (32 并发):  {'通过' if concurrent_32_ok else '失败'}")
    print(f"  {status_64} 并发测试 (64 并发):  {'通过' if concurrent_64_ok else '失败'}")

    print(f"\n  📁 所有生成的音频文件保存在: {output_dir}")
    print()


if __name__ == "__main__":
    main()
