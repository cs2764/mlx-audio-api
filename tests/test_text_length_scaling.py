"""
TTS Inference API 文本长度扩展测试

每次只发送一个 API 请求，文本长度从 50 字开始，每次增加 50 字，直到 1000 字。
文本全部为汉字+标点，从预置长文本中截取到精确字数。
每次调用记录耗时，最后汇总输出对比表格。

用法:
    uv run python tests/test_text_length_scaling.py
    uv run python tests/test_text_length_scaling.py --no-ref
    uv run python tests/test_text_length_scaling.py --instruct "A calm female voice"
    uv run python tests/test_text_length_scaling.py --url http://127.0.0.1:8013/v1/tts --api-key <KEY>
"""

import argparse
import base64
import os
import time
from pathlib import Path

import requests

# ─── 项目根目录 ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── 配置 ───────────────────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://127.0.0.1:8013/v1/tts"
DEFAULT_REF_AUDIO = str(PROJECT_ROOT / "samples" / "0流萤.wav")
DEFAULT_REF_TEXT_FILE = str(PROJECT_ROOT / "samples" / "0流萤.txt")
OUTPUT_DIR = PROJECT_ROOT / "test_outputs_scaling"

# 步长和范围
STEP = 50
MIN_LEN = 50
MAX_LEN = 1000

# ─── 语料：《三国演义》开篇（约 1500 字，用于截取） ──────────────────────────────

CHINESE_CORPUS = (
    "词曰：滚滚长江东逝水，浪花淘尽英雄。是非成败转头空：青山依旧在，几度夕阳红。"
    "白发渔樵江渚上，惯看秋月春风。一壶浊酒喜相逢：古今多少事，都付笑谈中。"
    "话说天下大势，分久必合，合久必分。周末七国分争，并入于秦。及秦灭之后，楚、汉分争，又并入于汉。"
    "汉朝自高祖斩白蛇而起义，一统天下。后来光武中兴，传至献帝，遂分为三国。"
    "推其致乱之由，殆始于桓、灵二帝。桓帝禁锢善类，崇信宦官。及桓帝崩，灵帝即位，"
    "大将军窦武、太傅陈蕃，共相辅佐。时有宦官曹节等弄权，窦武、陈蕃谋诛之，"
    "作事不密，反为所害，中涓自此愈横。建宁二年四月望日，帝御温德殿。方陞座，"
    "殿角狂风骤起，只见一条大青蛇，从梁上飞将下来，蟠于椅上。帝惊倒，左右急救入宫，"
    "百官俱奔避。须臾，蛇不见了。忽然大雷大雨，加以冰雹，落到半夜方止，坏却房屋无数。"
    "建宁四年二月，洛阳地震；又海水泛滥，沿海居民，尽被大浪卷入海中。"
    "光和元年，雌鸡化雄。六月朔，黑气十馀丈，飞入温德殿中。秋七月，有虹现于玉堂，"
    "五原山岸，尽皆崩裂。种种不祥，非止一端。帝下诏问群臣以灾异之由，"
    "议郎蔡邕上疏，以为霓堕鸡化，乃妇寺干政之所致，言颇切直。帝览奏叹息，因起更衣。"
    "曹节在后窃视，悉宣告左右，遂以他事陷邕于罪，放归田里。后张让、赵忠、封谞、段圭、"
    "曹节、侯览、蹇硕、程旷、夏恽、郭胜十人朋比为奸，号为十常侍。"
    "帝尊信张让，呼为阿父。朝政日非，以致天下人心思乱，盗贼蜂起。"
    "时钜鹿郡有兄弟三人：一名张角，一名张宝，一名张梁。那张角本是个不第秀才，"
    "因入山采药，遇一老人，碧眼童颜，手执藜杖，唤角至一洞中，以天书三卷授之，曰："
    "此名太平要术。汝得之，当代天宣化，普救世人。若萌异心，必获恶报。"
    "角拜问姓名。老人曰：吾乃南华老仙也。言讫，化阵清风而去。角得此书，晓夜功习，能呼风唤雨，"
    "号为太平道人。中平元年正月内，疫气流行，张角散施符水，为人治病，自称大贤良师。"
    "角有徒弟五百馀人，云游四方，皆能书符念咒。次后徒众日多，角乃立三十六方，"
    "大方万馀人，小方六七千，各立渠帅，称为将军；讹言：苍天已死，黄天当立；"
    "岁在甲子，天下大吉。令人各以白土，书甲子二字于家中大门上。"
    "青、幽、徐、冀、荆、扬、兖、豫八州之人，家家侍奉大贤良师张角名字。"
    "角遣其党马元义，暗赍金帛，结交中涓封胥，以为内应。角与二弟商议曰："
    "至难得者，民心也。今民心已顺，若不乘势取天下，诚为可惜。遂一面私造黄旗，"
    "约期举事；一面使弟子唐周，持书报封谞。唐周乃径赴省中告变。帝召大将军何进调兵擒马元义，"
    "斩之；次收封谞等一干人下狱。张角闻知事露，星夜举兵，自称天公将军，"
    "张宝称地公将军，张梁称人公将军；申言于众曰：今汉运将终，大圣人出。"
    "汝等皆宜顺天从正，以乐太平。四方百姓，裹黄巾从张角反者四五十万。"
    "贼势浩大，官军望风而靡。何进奏帝火速降诏，令各处备御，讨贼立功；"
    "一面遣中郎将卢植、皇甫嵩、朱隽，各引精兵，分三路讨之。"
)


def get_text_at_length(target_len: int) -> str:
    """从语料中截取精确字数的文本（含标点计入字数）。"""
    if target_len > len(CHINESE_CORPUS):
        raise ValueError(
            f"语料长度不足：需要 {target_len} 字，语料仅 {len(CHINESE_CORPUS)} 字"
        )
    return CHINESE_CORPUS[:target_len]


def audio_to_bytes(file_path: str) -> bytes | None:
    if not file_path or not Path(file_path).exists():
        return None
    with open(file_path, "rb") as f:
        return f.read()


def read_ref_text(ref_text: str) -> str:
    path = Path(ref_text)
    if path.exists() and path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return f.read()
    return ref_text


# ─── 单次请求 ─────────────────────────────────────────────────────────────────────

def make_tts_request(
    text: str,
    api_url: str,
    ref_audio_bytes: bytes | None,
    ref_text: str,
    output_path: Path,
    api_key: str | None = None,
    timeout: int = 600,
    instruct: str | None = None,
) -> dict:
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
        "text_length": len(text),
        "success": False,
        "status_code": None,
        "audio_size_bytes": 0,
        "audio_duration_seconds": 0.0,
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
            # Parse WAV duration from header (no extra deps needed)
            # WAV: bytes 24-27 = sample rate, 34-35 = bits per sample, 22-23 = channels
            # data chunk size at bytes 40-43 (standard PCM WAV)
            try:
                import struct
                sample_rate = struct.unpack_from("<I", audio_content, 24)[0]
                num_channels = struct.unpack_from("<H", audio_content, 22)[0]
                bits_per_sample = struct.unpack_from("<H", audio_content, 34)[0]
                data_size = struct.unpack_from("<I", audio_content, 40)[0]
                bytes_per_sample = bits_per_sample // 8
                total_samples = data_size // (num_channels * bytes_per_sample)
                result["audio_duration_seconds"] = round(total_samples / sample_rate, 3)
            except Exception:
                result["audio_duration_seconds"] = 0.0
            with open(output_path, "wb") as f:
                f.write(audio_content)
            result["output_file"] = str(output_path)
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


# ─── 主流程 ───────────────────────────────────────────────────────────────────────

def print_separator(title: str):
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_summary(results: list[dict], total_elapsed: float):
    print_separator("📊 汇总结果")

    success_results = [r for r in results if r["success"]]
    fail_results = [r for r in results if not r["success"]]

    # 表头
    header = f"  {'字数':>6} | {'状态':^4} | {'耗时(s)':>8} | {'ms/字':>7} | {'字/秒':>7} | {'时长(s)':>8} | {'ms音频/字':>9} | {'音频大小':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in results:
        status = "✅" if r["success"] else "❌"
        elapsed = f"{r['elapsed_seconds']:.3f}"
        size = f"{r['audio_size_bytes'] / 1024:.1f}KB" if r["success"] else "N/A"
        if r["success"] and r["elapsed_seconds"] > 0 and r["text_length"] > 0:
            ms_per_char = f"{r['elapsed_seconds'] * 1000 / r['text_length']:.1f}"
            chars_per_sec = f"{r['text_length'] / r['elapsed_seconds']:.1f}"
            dur = r["audio_duration_seconds"]
            dur_str = f"{dur:.2f}"
            ms_audio_per_char = f"{dur * 1000 / r['text_length']:.0f}"
        else:
            ms_per_char = "N/A"
            chars_per_sec = "N/A"
            dur_str = "N/A"
            ms_audio_per_char = "N/A"
        print(f"  {r['text_length']:>6} | {status:^4} | {elapsed:>8} | {ms_per_char:>7} | {chars_per_sec:>7} | {dur_str:>8} | {ms_audio_per_char:>9} | {size:>10}")
        if r["error"]:
            print(f"         └─ {r['error'][:70]}")

    print()
    print(f"  总计:         {len(results)} 次请求")
    print(f"  成功:         {len(success_results)}")
    print(f"  失败:         {len(fail_results)}")
    print(f"  总耗时:       {total_elapsed:.3f}s")

    if success_results:
        elapsed_list = [r["elapsed_seconds"] for r in success_results]
        ms_per_char_list = [r["elapsed_seconds"] * 1000 / r["text_length"] for r in success_results if r["text_length"] > 0]
        dur_list = [r["audio_duration_seconds"] for r in success_results if r["audio_duration_seconds"] > 0]
        ms_audio_per_char_list = [r["audio_duration_seconds"] * 1000 / r["text_length"] for r in success_results if r["text_length"] > 0 and r["audio_duration_seconds"] > 0]
        avg_elapsed = sum(elapsed_list) / len(elapsed_list)
        avg_ms_per_char = sum(ms_per_char_list) / len(ms_per_char_list) if ms_per_char_list else 0
        avg_dur = sum(dur_list) / len(dur_list) if dur_list else 0
        avg_ms_audio_per_char = sum(ms_audio_per_char_list) / len(ms_audio_per_char_list) if ms_audio_per_char_list else 0
        print(f"  最快:         {min(elapsed_list):.3f}s（{min(success_results, key=lambda r: r['elapsed_seconds'])['text_length']} 字）")
        print(f"  最慢:         {max(elapsed_list):.3f}s（{max(success_results, key=lambda r: r['elapsed_seconds'])['text_length']} 字）")
        print(f"  平均耗时:     {avg_elapsed:.3f}s")
        print(f"  平均ms/字:    {avg_ms_per_char:.1f}ms")
        print(f"  平均音频时长: {avg_dur:.2f}s")
        print(f"  平均ms音频/字:{avg_ms_audio_per_char:.0f}ms")

    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="TTS 文本长度扩展测试：从 50 字到 1000 字，每次递增 50 字，逐一顺序请求",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--url", "-u", type=str, default=DEFAULT_API_URL,
                        help=f"API 地址（默认: {DEFAULT_API_URL}）")
    parser.add_argument("--ref-audio", "-ra", type=str, default=DEFAULT_REF_AUDIO,
                        help="参考音频文件路径")
    parser.add_argument("--ref-text-file", "-rt", type=str, default=DEFAULT_REF_TEXT_FILE,
                        help="参考文本文件路径")
    parser.add_argument("--output-dir", "-o", type=str, default=str(OUTPUT_DIR),
                        help="输出目录")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API 密钥")
    parser.add_argument("--timeout", "-t", type=int, default=600,
                        help="单请求超时秒数（默认: 600）")
    parser.add_argument("--no-ref", action="store_true",
                        help="不使用参考音频")
    parser.add_argument("--instruct", type=str, default=None,
                        help="VoiceDesign 声音描述（Qwen3 等模型）")
    parser.add_argument("--step", type=int, default=STEP,
                        help=f"字数步长（默认: {STEP}）")
    parser.add_argument("--min-len", type=int, default=MIN_LEN,
                        help=f"起始字数（默认: {MIN_LEN}）")
    parser.add_argument("--max-len", type=int, default=MAX_LEN,
                        help=f"最大字数（默认: {MAX_LEN}）")
    return parser.parse_args()


def main():
    args = parse_args()

    lengths = list(range(args.min_len, args.max_len + 1, args.step))
    total_requests = len(lengths)

    print_separator(f"TTS 文本长度扩展测试（{args.min_len}~{args.max_len} 字，步长 {args.step}，共 {total_requests} 次）")
    print(f"  API 地址:   {args.url}")
    print(f"  超时时间:   {args.timeout}s")
    print(f"  输出目录:   {args.output_dir}")

    # 验证语料足够长
    if args.max_len > len(CHINESE_CORPUS):
        print(f"\n  ❌ 语料长度不足：需要 {args.max_len} 字，语料仅 {len(CHINESE_CORPUS)} 字")
        return

    # 加载参考音频
    ref_audio_bytes = None
    ref_text = ""

    if args.instruct:
        print(f"  模式:       VoiceDesign")
        print(f"  声音描述:   {args.instruct}")
    elif not args.no_ref:
        if not os.path.exists(args.ref_audio):
            print(f"  ⚠️  参考音频不存在: {args.ref_audio}，将使用默认语音")
        else:
            ref_audio_bytes = audio_to_bytes(args.ref_audio)
            print(f"  参考音频:   {args.ref_audio} ({len(ref_audio_bytes) / 1024:.1f} KB)")
        if os.path.exists(args.ref_text_file):
            ref_text = read_ref_text(args.ref_text_file)
            print(f"  参考文本:   {ref_text[:50]}...")
    else:
        print("  模式:       无参考音频")

    # 准备输出目录
    import shutil
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── 逐步测试 ───
    results = []
    total_start = time.perf_counter()

    for idx, length in enumerate(lengths, 1):
        text = get_text_at_length(length)
        out_path = output_dir / f"len_{length:04d}.wav"

        print(f"\n  [{idx:02d}/{total_requests}] 字数: {length}  文本: {text[:20]}…{text[-10:]}")

        result = make_tts_request(
            text=text,
            api_url=args.url,
            ref_audio_bytes=ref_audio_bytes,
            ref_text=ref_text,
            output_path=out_path,
            api_key=args.api_key,
            timeout=args.timeout,
            instruct=args.instruct,
        )

        status = "✅" if result["success"] else "❌"
        elapsed = result["elapsed_seconds"]
        if result["success"] and elapsed > 0 and length > 0:
            ms_per_char = elapsed * 1000 / length
            chars_per_sec = length / elapsed
            size_str = f"{result['audio_size_bytes'] / 1024:.1f} KB"
            dur = result["audio_duration_seconds"]
            ms_audio_per_char = dur * 1000 / length if length > 0 else 0
            print(f"         {status} 耗时: {elapsed:.3f}s  ms/字: {ms_per_char:.1f}  字/秒: {chars_per_sec:.1f}  音频: {size_str}  时长: {dur:.1f}s  ms音频/字: {ms_audio_per_char:.0f}")
        else:
            print(f"         {status} 耗时: {elapsed:.3f}s  音频: N/A")
        if result["error"]:
            print(f"         └─ 错误: {result['error'][:70]}")

        results.append(result)

    total_elapsed = time.perf_counter() - total_start

    # ─── 汇总 ───
    print_summary(results, total_elapsed)
    print(f"  📁 音频文件保存在: {output_dir}")
    print()


if __name__ == "__main__":
    main()
