$ErrorActionPreference = "Stop"

$COVERAGE_MIN = if ($env:COVERAGE_MIN) { $env:COVERAGE_MIN } else { 50 }

Write-Host "== 编译检查 =="
try { python -m compileall -q . 2>$null } catch {}

Write-Host "== Ruff 格式检查 =="
python -m ruff format --check . 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "格式不合规，运行 python -m ruff format . 修复"
    exit 1
}

Write-Host "== Ruff lint =="
python -m ruff check . 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Lint 不通过"
    exit 1
}

Write-Host "== 契约测试 =="
if ((Test-Path "tests/contracts") -and (Get-ChildItem "tests/contracts/*.py" -ErrorAction SilentlyContinue)) {
    pytest -q tests/contracts --maxfail=1
    if ($LASTEXITCODE -ne 0) { exit 1 }
} else {
    Write-Host "（跳过：tests/contracts/ 为空）"
}

Write-Host "== 单元测试 =="
if ((Test-Path "tests/unit") -and (Get-ChildItem "tests/unit/*.py" -ErrorAction SilentlyContinue)) {
    pytest -q tests/unit --maxfail=1
    if ($LASTEXITCODE -ne 0) { exit 1 }
} else {
    Write-Host "（跳过：tests/unit/ 为空）"
}

Write-Host "== 门禁通过 ✓ =="
