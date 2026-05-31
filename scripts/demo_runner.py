"""路演 Demo 预热脚本。

跑 8 个 demo case 预热搜索缓存和记忆库，
确保路演时即使网络不好也能秒回。

用法：
    python scripts/demo_runner.py              # 预热全部
    python scripts/demo_runner.py --check      # 只检查环境
    python scripts/demo_runner.py --index 0    # 跑单个 case
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def check_env():
    """检查 demo 环境。"""
    print("=" * 50)
    print("TruthNote Demo 环境检查")
    print("=" * 50)

    from truthnote.config import settings

    checks = [
        ("LLM 提供商", settings.llm_provider, settings.llm_provider != "claude_cli"),
        ("搜索提供商", settings.search_provider, True),
        (
            "百炼 API Key",
            "已配置" if settings.dashscope_api_key else "未配置",
            bool(settings.dashscope_api_key),
        ),
        ("Tavily API Key", "已配置" if settings.tavily_api_key else "未配置（用 mock）", True),
    ]

    all_ok = True
    for name, value, ok in checks:
        status = "OK" if ok else "WARN"
        print(f"  [{status}] {name}: {value}")
        if not ok:
            all_ok = False

    print()

    from truthnote.memory import MemoryStore

    try:
        store = MemoryStore()
        stats = store.get_stats()
        print(f"  记忆库：{stats['total_cases']} 条案例，{stats['total_memories']} 条记忆")
    except Exception as e:
        print(f"  [WARN] 记忆库：{e}")

    print()
    if all_ok:
        print("环境检查通过，可以开始 demo")
    else:
        print("有警告项，建议修复后再 demo")
    return all_ok


def run_demo_cases(index=None):
    """预热 demo case。"""
    cases_path = ROOT / "scenarios" / "demo_cases.json"
    if not cases_path.exists():
        print(f"demo case 文件不存在：{cases_path}")
        return

    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)

    if index is not None:
        cases = [cases[index]]

    from truthnote.pipeline import verify_message

    print(f"\n预热 {len(cases)} 个 demo case...\n")

    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {case['id']} ({case['category']})")
        print(f"  消息：{case['message'][:50]}...")

        t0 = time.perf_counter()
        try:
            result = verify_message(case["message"])
            elapsed = time.perf_counter() - t0
            print(f"  判定：{result.overall_verdict.value}")
            print(f"  回复：{result.friendly_reply[:50]}...")
            print(f"  耗时：{elapsed:.1f}s")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  错误：{e} ({elapsed:.1f}s)")
        print()

    print("预热完成。第二次 demo 将从记忆/缓存秒回。")


def main():
    parser = argparse.ArgumentParser(description="TruthNote Demo Runner")
    parser.add_argument("--check", action="store_true", help="只检查环境")
    parser.add_argument("--index", type=int, help="只跑指定索引")
    args = parser.parse_args()

    if args.check:
        check_env()
        return

    check_env()
    run_demo_cases(args.index)


if __name__ == "__main__":
    main()
