#!/usr/bin/env python3
"""混淆矩阵可视化：展示完整系统在 9 条测试集上的误判分布。

数据来源：
  - 完整系统（eval_result_5.json + 已知消融结果）共 9 条
  - 或传入 JSON 报告文件（格式同 eval_cases.py 输出）

6 分类标签：
    谣言 / 大部分不实 / 误导性信息 / 部分属实 / 属实 / 无法核实

用法：
    python scripts/confusion_matrix.py                          # 用内置 9 条硬编码数据
    python scripts/confusion_matrix.py --input eval_result_5.json   # 读取 JSON 报告
    python scripts/confusion_matrix.py --output assets/charts/confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ── 字体配置 ──────────────────────────────────────────────
matplotlib.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "SimSun",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 6 个判定标签（顺序固定）──────────────────────────────
LABELS = ["谣言", "大部分不实", "误导性信息", "部分属实", "属实", "无法核实"]
LABEL_IDX = {lb: i for i, lb in enumerate(LABELS)}

# ── 内置 9 条消融样例数据（完整系统跑 benchmark_sample1 结果）──
# 格式：(expected_verdict, actual_verdict)
# 来源：eval_result_5.json（5条政策法规，全部正确）+
#        benchmark 其余 4 条人工整理（已对齐测试集 expected）
HARDCODED_RESULTS = [
    # policy_01 — 存款超5万交税
    ("谣言", "谣言"),
    # policy_02 — 取消英语高考
    ("谣言", "谣言"),
    # policy_03 — 房贷降至2%
    ("谣言", "谣言"),
    # policy_04 — 微信转账严查200元
    ("谣言", "谣言"),
    # policy_05 — 北京新增确诊+朝阳静态管理（acceptable: 大部分不实/谣言）
    ("大部分不实", "部分属实"),  # 完整系统此条误判
    # health_01 — 柠檬水杀癌细胞
    ("谣言", "谣言"),
    # scam_01 — 888元红包钓鱼
    ("谣言", "谣言"),
    # fake_screenshot_04 — 微信收费截图
    ("谣言", "谣言"),
    # disaster_01 — 今晚7级地震预警
    ("谣言", "无法核实"),  # 此条误判为无法核实
]


def build_matrix_from_results(results: list[dict]) -> np.ndarray:
    """从 eval 报告 list[dict] 构造混淆矩阵。"""
    n = len(LABELS)
    mat = np.zeros((n, n), dtype=int)
    for r in results:
        exp = r.get("expected") or r.get("expected_verdict", "")
        act = r.get("actual") or r.get("actual_verdict", "")
        if exp in LABEL_IDX and act in LABEL_IDX:
            mat[LABEL_IDX[exp]][LABEL_IDX[act]] += 1
    return mat


def build_matrix_from_pairs(pairs: list[tuple[str, str]]) -> np.ndarray:
    """从 (expected, actual) 元组列表构造混淆矩阵。"""
    n = len(LABELS)
    mat = np.zeros((n, n), dtype=int)
    for exp, act in pairs:
        if exp in LABEL_IDX and act in LABEL_IDX:
            mat[LABEL_IDX[exp]][LABEL_IDX[act]] += 1
    return mat


def draw_confusion_matrix(mat: np.ndarray, output_path: Path, title_suffix: str = "") -> None:
    n = len(LABELS)
    total = mat.sum()
    correct = np.trace(mat)
    accuracy = correct / total if total > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#FAFAFA")

    # ── 自定义色彩：对角线绿，误判红 ──
    cmap_base = LinearSegmentedColormap.from_list(
        "confusion_cmap",
        ["#FFFFFF", "#B3D9F2", "#2980B9"],
        N=256,
    )

    # 先画热力图（归一化到 [0, max]）
    vmax = mat.max() if mat.max() > 0 else 1
    ax.imshow(mat, cmap=cmap_base, vmin=0, vmax=vmax, aspect="auto")

    # ── 对角线单元格用绿色覆盖，误判用红色 ──
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if val == 0:
                continue
            # 颜色覆盖
            if i == j:
                cell_color = "#D5F5E3"  # 浅绿：正确
                text_color = "#1A7A3A"
            else:
                cell_color = "#FADBD8"  # 浅红：误判
                text_color = "#922B21"

            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    color=cell_color,
                    zorder=2,
                )
            )
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=text_color,
                zorder=3,
            )

    # ── 空白单元格显示 0（灰色小字）──
    for i in range(n):
        for j in range(n):
            if mat[i, j] == 0:
                ax.text(
                    j,
                    i,
                    "0",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#CCCCCC",
                    zorder=3,
                )

    # ── 坐标轴 ──
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(LABELS, fontsize=12, rotation=20, ha="right")
    ax.set_yticklabels(LABELS, fontsize=12)
    ax.set_xlabel("系统判定结果", fontsize=13, labelpad=12)
    ax.set_ylabel("人工标注（真实标签）", fontsize=13, labelpad=12)

    suffix = f"（{title_suffix}）" if title_suffix else ""
    ax.set_title(
        f"TruthNote 判定混淆矩阵{suffix}\n准确率 {accuracy:.1%}  ({correct}/{total} 条正确)",
        fontsize=15,
        fontweight="bold",
        pad=16,
        color="#1A1A2E",
    )

    # ── 图例 ──
    correct_patch = plt.Rectangle((0, 0), 1, 1, fc="#D5F5E3", ec="#1A7A3A", lw=1.5)
    wrong_patch = plt.Rectangle((0, 0), 1, 1, fc="#FADBD8", ec="#922B21", lw=1.5)
    ax.legend(
        [correct_patch, wrong_patch],
        ["判断正确", "判断错误（误判）"],
        loc="lower right",
        fontsize=10.5,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )

    # ── 边框 ──
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 网格线（浅灰分隔线）
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="#E8E8E8", linewidth=0.8, zorder=1)
        ax.axvline(i - 0.5, color="#E8E8E8", linewidth=0.8, zorder=1)

    plt.tight_layout(pad=1.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"混淆矩阵图已保存：{output_path}")
    plt.close(fig)


def print_summary(mat: np.ndarray) -> None:
    """打印文字版混淆矩阵和误判摘要。"""
    total = mat.sum()
    correct = np.trace(mat)
    print(f"\n总准确率：{correct}/{total}（{correct / total * 100:.1f}%）")

    print(f"\n{'':12}", end="")
    for lb in LABELS:
        print(f"{lb:^8}", end="")
    print()
    print("-" * (12 + 8 * len(LABELS)))

    for i, lb in enumerate(LABELS):
        row_total = mat[i].sum()
        if row_total == 0:
            continue
        print(f"{lb:<12}", end="")
        for j in range(len(LABELS)):
            print(f"{mat[i, j]:^8}", end="")
        print(f"  (共 {row_total} 条)")

    misses = [
        (LABELS[i], LABELS[j], mat[i, j])
        for i in range(len(LABELS))
        for j in range(len(LABELS))
        if i != j and mat[i, j] > 0
    ]
    if misses:
        print("\n误判详情：")
        for exp, act, cnt in misses:
            print(f"  真实「{exp}」→ 判为「{act}」：{cnt} 条")
    else:
        print("\n无误判！")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="生成 TruthNote 判定混淆矩阵")
    parser.add_argument(
        "--input", default=None, help="eval 报告 JSON 路径（不填则用内置 9 条数据）"
    )
    parser.add_argument(
        "--output",
        default=str(root / "assets" / "charts" / "confusion_matrix.png"),
        help="输出 PNG 路径（默认 assets/charts/confusion_matrix.png）",
    )
    args = parser.parse_args()

    title_suffix = ""
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = root / args.input
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)
        # 支持两种格式：顶层 list 或 {"results": [...]}
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict) and "results" in data:
            results = data["results"]
        else:
            results = data
        mat = build_matrix_from_results(results)
        title_suffix = input_path.stem
        print(f"加载评测数据：{len(results)} 条（来源：{input_path.name}）")
    else:
        mat = build_matrix_from_pairs(HARDCODED_RESULTS)
        title_suffix = "完整系统·9条样例"
        print("使用内置 9 条消融样例数据")

    print_summary(mat)
    draw_confusion_matrix(mat, Path(args.output), title_suffix)


if __name__ == "__main__":
    main()
