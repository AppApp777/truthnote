"""TruthNote CLI — 命令行直接核查一条消息。

用法：
  python cli.py "紧急通知！存款超5万要交税！"
  python cli.py --file scenarios/rumor_testset.json --index 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.truthnote.orchestrator import Orchestrator
from src.truthnote.search import get_search_provider


def main():
    parser = argparse.ArgumentParser(description="TruthNote 事实核查 CLI")
    parser.add_argument("message", nargs="?", help="要核查的消息")
    parser.add_argument("--file", help="从测试集 JSON 文件读取")
    parser.add_argument("--index", type=int, default=0, help="测试集中的索引")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            testset = json.load(f)
        item = testset[args.index]
        message = item["message"]
        print(f"[测试用例] {item['id']} ({item['category']})")
        print(f"[期望判定] {item['expected_verdict']}")
        print()
    elif args.message:
        message = args.message
    else:
        print("请输入要核查的消息（Ctrl+Z 结束）：")
        message = sys.stdin.read().strip()
        if not message:
            print("未输入消息")
            return

    print(f"{'=' * 60}")
    print(f"核查消息：{message}")
    print(f"{'=' * 60}\n")

    orchestrator = Orchestrator(search_provider=get_search_provider())
    result = orchestrator.run(message)

    print(f"\n{'=' * 60}")
    print(f"总判定：{result.overall_verdict.value}")
    print(f"{'=' * 60}\n")

    for i, cv in enumerate(result.claims, 1):
        print(f"声明 {i}：{cv.claim.text}")
        print(f"  类别：{cv.claim.category.value}")
        print(f"  判定：{cv.verdict.value} (置信度 {cv.confidence:.0%})")
        print(f"  推理：{cv.reasoning}")
        print(f"  证据：{len(cv.evidence_chain)} 条")
        for e in cv.evidence_chain:
            print(f"    - [{e.source}] {e.title}")
        print()

    print(f"{'—' * 60}")
    print(f"核查总结：\n{result.summary}\n")
    print(f"{'—' * 60}")
    print(f"发给爸妈版：\n{result.friendly_reply}\n")

    print(
        f"\n[执行统计] {orchestrator.trace.total_llm_calls} 次 LLM 调用，"
        f"总耗时 {orchestrator.trace.total_duration_ms}ms"
    )
    for step in orchestrator.trace.steps:
        print(f"  - [{step.agent}] {step.action}: {step.duration_ms}ms")


if __name__ == "__main__":
    main()
