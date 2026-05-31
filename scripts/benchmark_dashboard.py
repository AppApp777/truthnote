#!/usr/bin/env python3
"""DEPRECATED — benchmark 看板已并入主 app（http://localhost:8000/benchmark）。

保留本脚本仅为向后兼容：直接启动一个最小 FastAPI app，把同样的 router 挂在 5050。
新功能请用主 app（python -m uvicorn src.truthnote.main:app --port 8000）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402

from truthnote.benchmark import router as benchmark_router  # noqa: E402

app = FastAPI(title="TruthNote Benchmark Dashboard (legacy)")
app.include_router(benchmark_router)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = ROOT / "templates" / "benchmark.html"
    return FileResponse(html_path, media_type="text/html")


if __name__ == "__main__":
    import uvicorn

    print("⚠  DEPRECATED — 请改用主 app: http://localhost:8000/benchmark")
    print("  本脚本保留向后兼容，端口 5050")
    print(f"  项目根: {ROOT}")
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="info")
