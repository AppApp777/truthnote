"""阶段 1.2 — 从 design 标注计数算"类型对数赔率提升 Δ"（oracle Q1）。

不是算原始假率（等量 50:50 抽样下假率无效），而是算 case-control 下仍有效的
对数似然比 Δ = log[P(type|谣)/P(type|真)]，Jeffreys 平滑 + 向全局 0 收缩。

运行时先验 = sigmoid(logit(部署基础率) + λ·Δ)，封顶 0.60。
同一套 Δ，只换基础率：CANDY 跑分用 ≈0.48，真实产品用 ≈0.10。

输入：data/eval/design_message_types.json（阶段 1.1 产出）
产出：data/eval/type_log_odds_lift.json（Δ 表 + 两档基础率下的先验 + 新旧对比）
纯确定性，不调 LLM。
"""

import json
import math

ROOT = r"d:/VIBE CODING/A-hackthon/02-wenke-song"
IN = ROOT + r"/data/eval/design_message_types.json"
OUT = ROOT + r"/data/eval/type_log_odds_lift.json"

# 7 个运行时 MessageType（.value）
MESSAGE_TYPES = [
    "health_product_promo",
    "financial_scam",
    "political_rumor",
    "health_advice",
    "fact_assertion",
    "personal_experience",
    "other",
]

# 现状手写先验（对照）
OLD_PRIOR = {
    "health_product_promo": 0.85,
    "financial_scam": 0.90,
    "political_rumor": 0.60,
    "health_advice": 0.35,
    "fact_assertion": 0.30,
    "personal_experience": 0.10,
    "other": 0.30,
}

JEFFREYS_A = 0.5
KAPPA_M = 100  # MessageType 级收缩强度：n=100 时半信半疑，向全局 0 收
PRIOR_CAP = 0.60  # 先验封顶："先验只能升怀疑，定罪靠证据"
CANDY_BASE_RATE = 0.482  # holdout 自然谣率 7438/(7438+7997)，跑分档
PRODUCT_BASE_RATE = 0.10  # 真实产品档（路演讲故事用）
LAMBDA = 0.5  # 类型提升强度


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def main() -> None:
    data = json.load(open(IN, encoding="utf-8"))
    counts = data["counts"]
    # 把 fallback "其他" 折进 other
    if "其他" in counts:
        for k in ("true", "false"):
            counts.setdefault("other", {"true": 0, "false": 0})[k] += counts["其他"][k]
        del counts["其他"]

    total_false = sum(v["false"] for v in counts.values())
    total_true = sum(v["true"] for v in counts.values())
    K = len(MESSAGE_TYPES)

    lift = {}
    rows = []
    for mt in MESSAGE_TYPES:
        c = counts.get(mt, {"true": 0, "false": 0})
        f, t = c["false"], c["true"]
        n = f + t
        # Jeffreys 平滑的对数似然比
        p_f = (f + JEFFREYS_A) / (total_false + JEFFREYS_A * K)
        p_t = (t + JEFFREYS_A) / (total_true + JEFFREYS_A * K)
        delta_obs = math.log(p_f / p_t)
        # 向全局 0 收缩（小样本类型不窜）
        shrink = n / (n + KAPPA_M) if n else 0.0
        delta = shrink * delta_obs
        lift[mt] = round(delta, 4)
        rows.append(
            {
                "type": mt,
                "n": n,
                "true": t,
                "false": f,
                "delta_obs": round(delta_obs, 3),
                "shrink_B": round(shrink, 2),
                "delta": round(delta, 3),
            }
        )

    def prior_at(base_rate: float) -> dict:
        out = {}
        for mt in MESSAGE_TYPES:
            p = sigmoid(logit(base_rate) + LAMBDA * lift[mt])
            out[mt] = round(min(p, PRIOR_CAP), 4)
        return out

    prior_candy = prior_at(CANDY_BASE_RATE)
    prior_product = prior_at(PRODUCT_BASE_RATE)

    result = {
        "params": {
            "jeffreys_a": JEFFREYS_A,
            "kappa_m": KAPPA_M,
            "lambda": LAMBDA,
            "prior_cap": PRIOR_CAP,
            "candy_base_rate": CANDY_BASE_RATE,
            "product_base_rate": PRODUCT_BASE_RATE,
        },
        "totals": {"true": total_true, "false": total_false},
        "log_odds_lift": lift,
        "prior_candy_mode": prior_candy,
        "prior_product_mode": prior_product,
        "rows": rows,
    }
    json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 控制台对照表
    print(f"design 计数：真 {total_true} / 谣 {total_false}\n")
    print(
        f"{'MessageType':<22}{'n':>5}{'Δobs':>7}{'收缩':>6}{'Δ':>7} | "
        f"{'旧先验':>7}{'新(CANDY)':>10}{'新(产品)':>9}"
    )
    for r in rows:
        mt = r["type"]
        print(
            f"{mt:<22}{r['n']:>5}{r['delta_obs']:>7.2f}{r['shrink_B']:>6.2f}{r['delta']:>7.2f} | "
            f"{OLD_PRIOR[mt]:>7.2f}{prior_candy[mt]:>10.3f}{prior_product[mt]:>9.3f}"
        )
    print(f"\n✅ 落盘 {OUT}")
    print("说明：political_rumor / personal_experience 无样本→Δ=0→先验=基础率（运行时本就不触发）")


if __name__ == "__main__":
    main()
