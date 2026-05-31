"""多模态线手动实测（2026-05-30）：用真实截图测 verify_image 能否稳定提文字 + 判断。

这是命题人校准后多模态线（🔽降级）的「先实测再下结论」脚本。
不是 pytest 用例——直接 `python tests/manual_image_smoke.py` 跑，打真实 VL API。

两步：
- A 步（全图）：只测 extract_text_from_image，看 OCR/提取稳不稳。
- B 步（少量）：测完整 verify_image（提取→判断）端到端出 verdict。

用法：
  python tests/manual_image_smoke.py extract     # 只跑 A 步
  python tests/manual_image_smoke.py full N       # 跑 A 步 + 对前 N 张跑完整核查
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 关流式，让输出干净（提取阶段不需要逐字打印）
os.environ.setdefault("STREAM_LLM", "0")

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthnote import image  # noqa: E402

# 6 张真实谣言场景截图，覆盖：纯图 / 富文字 / 灰色地带 / 不值得核查
IMAGES = [
    ("洪崖洞洪水照（纯图·复用图嫌疑）", "assets/demo_cases/flood_chongqing_hongyadong.png"),
    ("玉粟源洪水照（纯图·复用图嫌疑）", "assets/demo_cases/flood_reused_01.png"),
    ("北大学硕取消（社交帖·富文字·灰区）", "assets/screenshots/grey_zone_01_beida_xueshuo.png"),
    ("梅姨落网（公告型·张维平拐卖案）", "assets/demo_cases/meiyi_claim_pending.png"),
    ("教室占用吐槽（小红书·不值得核查）", "assets/screenshots/not_checkworthy_classroom_rant.png"),
    ("pony 关注（社交帖·不值得核查）", "assets/screenshots/not_checkworthy_pony_follow.png"),
]


def run_extract() -> list[dict]:
    print("=" * 70)
    print("A 步：extract_text_from_image 提取稳定性")
    print("=" * 70)
    rows = []
    for name, rel in IMAGES:
        path = str(ROOT / rel)
        ok_path = os.path.isfile(path)
        t0 = time.time()
        text = image.extract_text_from_image(path) if ok_path else ""
        dt = time.time() - t0
        row = {
            "name": name,
            "rel": rel,
            "exists": ok_path,
            "chars": len(text),
            "secs": round(dt, 1),
            "preview": text[:160].replace("\n", " ⏎ "),
        }
        rows.append(row)
        status = "✅" if text else ("⚠️空" if ok_path else "❌缺图")
        print(f"\n[{status}] {name}  ({row['chars']}字 / {row['secs']}s)")
        print(f"     {rel}")
        if text:
            print(f"     提取: {row['preview']}")
        elif ok_path:
            print("     ⚠️ 提取为空")
    return rows


def run_full(n: int) -> None:
    print("\n" + "=" * 70)
    print(f"B 步：verify_image 完整核查（提取→判断），前 {n} 张")
    print("=" * 70)
    for name, rel in IMAGES[:n]:
        path = str(ROOT / rel)
        print(f"\n──── {name} ────")
        t0 = time.time()
        result = image.verify_image(path)
        dt = time.time() - t0
        ext = result.get("extracted_text", "")
        vr = result.get("verify_result")
        print(f"  提取 {len(ext)} 字 / 总耗时 {round(dt, 1)}s")
        if result.get("error"):
            print(f"  ❌ error: {result['error']}")
            continue
        if vr is None:
            print("  ⚠️ verify_result 为空")
            continue
        verdict = getattr(vr, "verdict", None) or (
            vr.get("verdict") if isinstance(vr, dict) else None
        )
        print(f"  ✅ verdict = {verdict}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "extract"
    rows = run_extract()

    ok = sum(1 for r in rows if r["chars"] > 0)
    print("\n" + "=" * 70)
    print(f"A 步汇总：{ok}/{len(rows)} 张成功提取到文字")
    avg = round(sum(r["secs"] for r in rows) / max(len(rows), 1), 1)
    print(f"平均耗时 {avg}s/张")
    print("=" * 70)

    if mode == "full":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        run_full(n)


if __name__ == "__main__":
    main()
