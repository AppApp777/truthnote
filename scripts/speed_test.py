"""GLM vs DeepSeek 速度对比测试"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from truthnote.llm import chat  # noqa: E402

PROMPT = "判断这条消息是否为谣言：紧急通知！存款超5万要交税！请用JSON回复{verdict, reasoning}"
MSG = [{"role": "user", "content": PROMPT}]

MODELS = [
    ("glm-4-flash", "openai"),
    ("deepseek-v4-flash", "openai"),
]

# 跑 3 轮取平均
ROUNDS = 2

for model, provider in MODELS:
    times = []
    for r in range(ROUNDS):
        t0 = time.time()
        try:
            result = chat(MSG, model=model, provider=provider)
            elapsed = time.time() - t0
            times.append(elapsed)
            if r == 0:
                content_len = len(result["content"])
                in_tok = result["usage"]["input_tokens"]
                out_tok = result["usage"]["output_tokens"]
        except Exception as e:
            print(f"{model}: ERROR - {e}")
            break

    if times:
        avg = sum(times) / len(times)
        print(f"{model:30s} | 平均 {avg:.2f}s | {ROUNDS}轮: {[f'{t:.2f}s' for t in times]}")
        print(f"{'':30s} | {content_len} chars, in={in_tok} out={out_tok}")
