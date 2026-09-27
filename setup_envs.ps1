<#
.SYNOPSIS
  Create a .venv next to every requirements.txt in this repo and install it.

.EXAMPLE
  .\setup_envs.ps1                      # all environments
  .\setup_envs.ps1 churn                # only folders whose path matches "churn"
  .\setup_envs.ps1 -Recreate foyer      # delete and rebuild matching .venv(s)
#>
param(
    [string]$Filter = "",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$reqs = Get-ChildItem -Path $root -Recurse -Filter requirements.txt -File |
    Where-Object { $_.FullName -notmatch '\\\.venv\\' -and $_.DirectoryName -ne $root } |
    Where-Object { $_.DirectoryName -match [regex]::Escape($Filter) } |
    Sort-Object DirectoryName

$failed = @()
foreach ($req in $reqs) {
    $dir  = $req.DirectoryName
    $venv = Join-Path $dir ".venv"
    $name = $dir.Substring($root.Length + 1)
    Write-Host "`n=== $name" -ForegroundColor Cyan

    if ($Recreate -and (Test-Path $venv)) { Remove-Item -Recurse -Force $venv }
    if (-not (Test-Path $venv)) { python -m venv $venv }

    $py = Join-Path $venv "Scripts\python.exe"
    & $py -m pip install --quiet --upgrade pip
    & $py -m pip install --quiet -r $req.FullName
    if ($LASTEXITCODE -ne 0) { $failed += $name; continue }

    # Register notebooks' kernel so Jupyter/VS Code can pick this env by name
    if (Select-String -Path $req.FullName -Pattern '^ipykernel' -Quiet) {
        $kernel = ($name -replace '[\\/]', '_')
        & $py -m ipykernel install --user --name $kernel --display-name "ML: $kernel" | Out-Null
    }
    if (Select-String -Path $req.FullName -Pattern '^playwright' -Quiet) {
        & $py -m playwright install chromium
    }
    Write-Host "ok -> $venv" -ForegroundColor Green
}

if ($failed) {
    Write-Host "`nFailed: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "`nAll environments ready." -ForegroundColor Green
