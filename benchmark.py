import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from translator import make_openai_client

BENCH_SENTENCES = {
    "ja": [
        "こんにちは、今日はいい天気ですね。",
        "この映画はとても面白かったです。",
        "明日の会議は何時からですか？",
        "日本の桜は本当に美しいですね。",
        "新しいレストランに行ってみましょう。",
    ],
    "en": [
        "Hello, the weather is nice today.",
        "That movie was really interesting.",
        "What time does tomorrow's meeting start?",
        "The cherry blossoms in Japan are truly beautiful.",
        "Let's try going to the new restaurant.",
    ],
    "zh": [
        "你好，今天天气真不错。",
        "那部电影真的很有意思。",
        "明天的会议几点开始？",
        "日本的樱花真的很美丽。",
        "我们去试试那家新餐厅吧。",
    ],
    "ko": [
        "안녕하세요, 오늘 날씨가 좋네요.",
        "그 영화 정말 재미있었어요.",
        "내일 회의는 몇 시부터인가요?",
        "일본의 벚꽃은 정말 아름답네요.",
        "새로운 레스토랑에 가볼까요?",
    ],
    "fr": [
        "Bonjour, il fait beau aujourd'hui.",
        "Ce film était vraiment intéressant.",
        "À quelle heure commence la réunion demain?",
        "Les cerisiers en fleurs au Japon sont magnifiques.",
        "Allons essayer le nouveau restaurant.",
    ],
    "de": [
        "Hallo, heute ist schönes Wetter.",
        "Der Film war wirklich interessant.",
        "Um wie viel Uhr beginnt das Meeting morgen?",
        "Die Kirschblüten in Japan sind wunderschön.",
        "Lass uns das neue Restaurant ausprobieren.",
    ],
}


def run_benchmark(models, source_lang, target_lang, timeout_s, prompt, result_callback):
    """Run benchmark in a background thread. Calls result_callback(str) for each output line."""
    sentences = BENCH_SENTENCES.get(source_lang, BENCH_SENTENCES["en"])
    rounds = len(sentences)

    result_callback(
        f"Testing {len(models)} model(s) x {rounds} rounds  |  "
        f"timeout={timeout_s}s  |  {source_lang} -> {target_lang}\n"
        f"{'=' * 60}\n"
    )

    def _test_model(m):
        name = m["name"]
        lines = [f"Model: {name}", f"  {'─' * 50}"]
        try:
            client = make_openai_client(
                m["api_base"],
                m["api_key"],
                proxy=m.get("proxy", "none"),
                timeout=timeout_s,
            )
            ttfts = []
            totals = []

            for i, text in enumerate(sentences):
                if m.get("no_system_role"):
                    messages = [{"role": "user", "content": f"{prompt}\n{text}"}]
                else:
                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ]
                try:
                    t0 = time.perf_counter()
                    stream = client.chat.completions.create(
                        model=m["model"],
                        messages=messages,
                        max_tokens=256,
                        temperature=0.3,
                        stream=True,
                    )
                    ttft = None
                    chunks = []
                    for chunk in stream:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        delta = chunk.choices[0].delta
                        if delta.content:
                            chunks.append(delta.content)
                    total_ms = (time.perf_counter() - t0) * 1000
                    result_text = "".join(chunks).strip()
                    ttft = ttft or total_ms
                except Exception:
                    t0 = time.perf_counter()
                    resp = client.chat.completions.create(
                        model=m["model"],
                        messages=messages,
                        max_tokens=256,
                        temperature=0.3,
                        stream=False,
                    )
                    total_ms = (time.perf_counter() - t0) * 1000
                    ttft = total_ms
                    result_text = resp.choices[0].message.content.strip()

                ttfts.append(ttft)
                totals.append(total_ms)
                lines.append(
                    f"  Round {i + 1}: {total_ms:7.0f}ms "
                    f"(TTFT {ttft:6.0f}ms) | {result_text[:60]}"
                )

            avg_total = statistics.mean(totals)
            std_total = statistics.stdev(totals) if len(totals) > 1 else 0
            avg_ttft = statistics.mean(ttfts)
            std_ttft = statistics.stdev(ttfts) if len(ttfts) > 1 else 0
            lines.append(
                f"  Avg: {avg_total:.0f}ms \u00b1 {std_total:.0f}ms  "
                f"(TTFT: {avg_ttft:.0f}ms \u00b1 {std_ttft:.0f}ms)"
            )

            result_callback("\n".join(lines))
            return {
                "name": name,
                "avg_ttft": avg_ttft,
                "std_ttft": std_ttft,
                "avg_total": avg_total,
                "std_total": std_total,
                "error": None,
            }

        except Exception as e:
            err_msg = str(e).split("\n")[0][:120]
            lines.append(f"  FAILED: {err_msg}")
            result_callback("\n".join(lines))
            return {
                "name": name,
                "avg_ttft": 0,
                "std_ttft": 0,
                "avg_total": 0,
                "std_total": 0,
                "error": err_msg,
            }

    def _run_all():
        results = []
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {pool.submit(_test_model, m): m for m in models}
            for fut in as_completed(futures):
                results.append(fut.result())

        ok = [r for r in results if not r["error"]]
        ok.sort(key=lambda r: r["avg_ttft"])
        result_callback(f"\n{'=' * 60}")
        result_callback("Ranking by Avg TTFT:")
        for i, r in enumerate(ok):
            result_callback(
                f"  #{i + 1}  TTFT {r['avg_ttft']:6.0f}ms \u00b1 {r['std_ttft']:4.0f}ms  "
                f"Total {r['avg_total']:6.0f}ms \u00b1 {r['std_total']:4.0f}ms  "
                f"{r['name']}"
            )
        failed = [r for r in results if r["error"]]
        for r in failed:
            result_callback(f"  FAIL  {r['name']}: {r['error']}")
        result_callback("__DONE__")

    threading.Thread(target=_run_all, daemon=True).start()
