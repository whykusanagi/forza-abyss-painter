# VRAM strategy harness — sync packages, run A/B/C on logo sample.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File `
#     "\\QUASAR\ContentCreation\ForzaAbyssPainter_build\diagnostics\run_vram_strategy.ps1"
#
#   run_vram_strategy.ps1 -Quick
#
# Outputs: diagnostics\smoke_results_upstream\vram_strategy_report.{json,md}

param(
    [switch]$Quick,
    [double]$Budget = 8.0,
    [int]$K = 0,
    [int]$Res = 0
)

$ErrorActionPreference = 'Stop'

$buildRoot = Split-Path $PSScriptRoot -Parent
$desktopBuild = Join-Path $env:USERPROFILE 'Desktop\ForzaAbyssPainter_build'
if (-not (Test-Path (Join-Path $buildRoot 'source\forza_abyss_painter')) -and (Test-Path (Join-Path $desktopBuild 'source\forza_abyss_painter'))) {
    $buildRoot = $desktopBuild
}

$py = Join-Path $env:LOCALAPPDATA 'ForzaAbyssPainter\runtime\python311\python.exe'
if (-not (Test-Path $py)) {
    Write-Error "Embedded python not found: $py — install GPU runtime from ForzaAbyssPainter.exe first."
}

$srcPkg = Join-Path $buildRoot 'source\forza_abyss_painter'
$dstPkg = Join-Path $env:LOCALAPPDATA 'ForzaAbyssPainter\runtime\python311\Lib\site-packages\forza_abyss_painter'
foreach ($sub in @('shapegen', 'io', 'runtime', 'cli')) {
    $from = Join-Path $srcPkg $sub
    $to = Join-Path $dstPkg $sub
    if (-not (Test-Path $from)) {
        Write-Warning "Skip sync (missing): $from"
        continue
    }
    Write-Host "Sync $sub -> site-packages"
    & robocopy $from $to /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $sub (exit $LASTEXITCODE)" }
}

$harness = Join-Path $PSScriptRoot 'vram_strategy_harness.py'
$args = @($harness)
if ($Quick) { $args += '--quick' }
if ($Budget -gt 0) { $args += @('--budget', [string]$Budget) }
if ($K -gt 0) { $args += @('--K', [string]$K) }
if ($Res -gt 0) { $args += @('--res', [string]$Res) }

Write-Host ""
Write-Host "Running vram_strategy_harness.py ..."
& $py @args
$code = $LASTEXITCODE
if ($code -ne 0) { exit $code }

$report = Join-Path $PSScriptRoot 'smoke_results_upstream\vram_strategy_report.md'
if (Test-Path $report) {
    Write-Host ""
    Get-Content $report
}
exit 0
