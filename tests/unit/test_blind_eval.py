"""阶段 0.1 失败测试先行 — 盲测评测的指标计算（纯函数，不调 LLM）。

验收预注册 docs/preregistration.md §D/§E/§F 的映射与指标定义。
先写本测试（红）→ 再实现 scripts/run_blind_eval.py（绿）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_blind_eval as rbe  # noqa: E402

# verdict 中文值（与 schemas.Verdict 一致）
TRUE = "属实"
PARTLY = "部分属实"
MISLEADING = "误导性信息"
MOSTLY_FALSE = "大部分不实"
FALSE = "谣言"
UNVERIFIABLE = "无法核实"


def test_verdict_to_correct_mapping():
    # 真消息(0)：只有"属实"算对
    assert rbe.verdict_to_correct(0, TRUE) is True
    assert rbe.verdict_to_correct(0, PARTLY) is False  # 警告=对真消息泼脏水
    assert rbe.verdict_to_correct(0, FALSE) is False
    assert rbe.verdict_to_correct(0, UNVERIFIABLE) is False  # 弃权算错
    # 谣言(1)：任何警告类算对
    assert rbe.verdict_to_correct(1, FALSE) is True
    assert rbe.verdict_to_correct(1, MOSTLY_FALSE) is True
    assert rbe.verdict_to_correct(1, PARTLY) is True
    assert rbe.verdict_to_correct(1, TRUE) is False  # 漏判
    assert rbe.verdict_to_correct(1, UNVERIFIABLE) is False  # 弃权算错


def _fixture_rows():
    return [
        {"gold_label": 0, "verdict": TRUE},  # 真，正确接受
        {"gold_label": 0, "verdict": FALSE},  # 真，误报（警告）
        {"gold_label": 0, "verdict": UNVERIFIABLE},  # 真，弃权（错）
        {"gold_label": 1, "verdict": MOSTLY_FALSE},  # 谣，抓到
        {"gold_label": 1, "verdict": TRUE},  # 谣，漏判
        {"gold_label": 1, "verdict": UNVERIFIABLE},  # 谣，弃权（错）
    ]


def test_compute_metrics_values():
    m = rbe.compute_metrics(_fixture_rows())
    assert m["n"] == 6
    assert m["n_true"] == 3
    assert m["n_rumor"] == 3
    # 正确：row0 + row3 = 2
    assert abs(m["accuracy"] - 2 / 6) < 1e-9
    # 覆盖：非弃权 = 4 条
    assert abs(m["coverage"] - 4 / 6) < 1e-9
    # 覆盖内准确率：2/4
    assert abs(m["covered_accuracy"] - 0.5) < 1e-9
    # 真消息误报率：1/3（row1 被警告）
    assert abs(m["true_false_positive_rate"] - 1 / 3) < 1e-9
    # 真消息弃权率：1/3（row2）
    assert abs(m["true_abstain_rate"] - 1 / 3) < 1e-9
    # 谣言召回：1/3（row3）
    assert abs(m["rumor_recall"] - 1 / 3) < 1e-9
    # 警告精确率：警告 2 条(row1真, row3谣)，其中谣 1 → 1/2
    assert abs(m["warning_precision"] - 0.5) < 1e-9
    # 平衡准确率：(真recall 1/3 + 谣recall 1/3)/2 = 1/3
    assert abs(m["balanced_accuracy"] - 1 / 3) < 1e-9


def test_compute_metrics_empty_safe():
    m = rbe.compute_metrics([])
    assert m["n"] == 0
    assert m["accuracy"] == 0.0
    assert m["coverage"] == 0.0
    assert m["covered_accuracy"] == 0.0  # 不能除零崩


def test_all_abstain_scores_zero_accuracy():
    """全弃权不许刷出高准确率——堵住沉默刷分。"""
    rows = [{"gold_label": i % 2, "verdict": UNVERIFIABLE} for i in range(10)]
    m = rbe.compute_metrics(rows)
    assert m["accuracy"] == 0.0
    assert m["coverage"] == 0.0
