#!/usr/bin/env python3
"""消融实验可视化：输出横向柱状图 PNG。

消融数据（9 条样例，基于 benchmark_sample1_v4 + 完整系统结果）：
  纯 LLM              6/9  ≈ 66.7%
  +搜索（GLM-v3）      2/5  ≈ 40.0%  ← 加搜索反降（注：5条子集）
  +规则（structured）  3/5  ≈ 60.0%  ← 加规则部分恢复
  完整系统             7/9  ≈ 77.8%  ← 全部组件联动

用法：
    python scripts/visualize_ablation.py
    python scripts/visualize_ablation.py --output assets/charts/ablation.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── 字体配置（Windows 优先 SimHei / Microsoft YaHei）──
matplotlib.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "SimSun",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号乱码


# ── 消融数据（硬编码，来自实测结果）──────────────────────
ABLATION_DATA = [
    {
        "label": "纯 LLM\n（无搜索无规则）",
        "accuracy": 66.7,
        "correct": 6,
        "total": 9,
        "color": "#5B8DB8",  # 冷蓝
        "note": None,
    },
    {
        "label": "+网络搜索\n（加 Tavily 实时检索）",
        "accuracy": 40.0,
        "correct": 2,
        "total": 5,
        "color": "#D95F5F",  # 警示红——搜索降准
        "note": "搜索引入噪声\n准确率反降",
    },
    {
        "label": "+输出规则约束\n（结构化 Prompt）",
        "accuracy": 60.0,
        "correct": 3,
        "total": 5,
        "color": "#F0A500",  # 琥珀橙——部分恢复
        "note": None,
    },
    {
        "label": "完整系统\n（多 Agent + 证据链）",
        "accuracy": 77.8,
        "correct": 7,
        "total": 9,
        "color": "#2DB27D",  # 翠绿——最优
        "note": None,
    },
]


def build_chart(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    labels = [d["label"] for d in ABLATION_DATA]
    accuracies = [d["accuracy"] for d in ABLATION_DATA]
    colors = [d["color"] for d in ABLATION_DATA]
    n = len(labels)

    y_pos = np.arange(n)
    bar_height = 0.52

    # ── 背景斑马纹 ──
    for i in range(n):
        ax.axhspan(i - 0.5, i + 0.5, alpha=0.04 if i % 2 == 0 else 0.0, color="gray")

    # ── 基准线（纯 LLM 基线） ──
    baseline = accuracies[0]
    ax.axvline(baseline, color="#999999", linewidth=1.2, linestyle="--", alpha=0.6, zorder=1)
    ax.text(
        baseline + 0.5,
        n - 0.05,
        f"基线 {baseline:.0f}%",
        color="#888888",
        fontsize=10,
        va="top",
        ha="left",
    )

    # ── 柱状图 ──
    bars = ax.barh(
        y_pos,
        accuracies,
        height=bar_height,
        color=colors,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    # ── 数值标签（柱右侧）──
    for bar, d in zip(bars, ABLATION_DATA, strict=False):
        w = bar.get_width()
        ax.text(
            w + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%  ({d['correct']}/{d['total']})",
            va="center",
            ha="left",
            fontsize=13,
            fontweight="bold",
            color="#333333",
            zorder=4,
        )

    # ── 红色下箭头 + 注释（+搜索那根柱） ──
    search_idx = 1
    search_bar = bars[search_idx]
    ax_x = search_bar.get_width() + 22  # 箭头 x 位置（柱外右侧留白区）
    ax_y = search_bar.get_y() + search_bar.get_height() / 2

    # 用 annotate 画红色带框注释
    ax.annotate(
        "! 搜索引入噪声\n  准确率降 26.7 pp",
        xy=(search_bar.get_width(), ax_y),
        xytext=(ax_x, ax_y + 0.55),
        fontsize=10.5,
        color="#C0392B",
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            color="#C0392B",
            lw=2.0,
            connectionstyle="arc3,rad=0.25",
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#FDECEA",
            edgecolor="#C0392B",
            linewidth=1.5,
            alpha=0.9,
        ),
        zorder=5,
    )

    # ── 完整系统高亮边框 ──
    best_bar = bars[-1]
    ax.patches  # noqa: B018  确保渲染
    rect = mpatches.FancyBboxPatch(
        (-1, best_bar.get_y() - 0.06),
        best_bar.get_width() + 2,
        best_bar.get_height() + 0.12,
        boxstyle="round,pad=0.02",
        linewidth=2.5,
        edgecolor="#2DB27D",
        facecolor="none",
        zorder=6,
    )
    ax.add_patch(rect)
    ax.text(
        best_bar.get_width() + 22,
        best_bar.get_y() + best_bar.get_height() / 2 - 0.55,
        "[最优]",
        color="#2DB27D",
        fontsize=11,
        fontweight="bold",
        va="center",
        ha="left",
    )

    # ── 坐标轴设置 ──
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("判定准确率（%）", fontsize=13, labelpad=10)
    ax.set_title(
        "TruthNote 消融实验：各组件对准确率的贡献\n（9 条谣言测试集，6 类别覆盖）",
        fontsize=16,
        fontweight="bold",
        pad=18,
        color="#1A1A2E",
    )

    # x 轴刻度
    ax.set_xticks([0, 20, 40, 60, 77.8, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "77.8%", "80%", "100%"], fontsize=10)
    ax.tick_params(axis="x", which="both", bottom=True, top=False)
    ax.tick_params(axis="y", which="both", left=False)

    # 去掉多余边框
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")

    # ── 图例说明 ──
    legend_items = [
        mpatches.Patch(color="#5B8DB8", label="基础 LLM"),
        mpatches.Patch(color="#D95F5F", label="+搜索（准确率下降）"),
        mpatches.Patch(color="#F0A500", label="+规则约束（恢复）"),
        mpatches.Patch(color="#2DB27D", label="完整多 Agent 系统（最优）"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=10,
        framealpha=0.85,
        edgecolor="#CCCCCC",
    )

    plt.tight_layout(pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"消融实验图已保存：{output_path}")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="生成消融实验可视化图")
    parser.add_argument(
        "--output",
        default=str(root / "assets" / "charts" / "ablation.png"),
        help="输出 PNG 路径（默认 assets/charts/ablation.png）",
    )
    args = parser.parse_args()
    build_chart(Path(args.output))


if __name__ == "__main__":
    main()
