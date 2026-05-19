# Libère l'espace disque utilisé par MLflow (runs + registry)
$base = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $base

Write-Host "Taille mlruns avant nettoyage..."
if (Test-Path "mlruns") {
    $size = (Get-ChildItem -Recurse mlruns -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host ("  {0:N1} Mo" -f $size)
}

$uri = if ($env:MLFLOW_API) { $env:MLFLOW_API } else { "http://localhost:8000" }
try {
    $r = Invoke-RestMethod -Method POST -Uri "$uri/maintenance/cleanup"
    Write-Host "Cleanup API:" ($r | ConvertTo-Json -Compress)
} catch {
    Write-Host "API indisponible — suppression locale de mlruns et mlflow.db..."
    Remove-Item -Recurse -Force mlruns -ErrorAction SilentlyContinue
    Remove-Item -Force mlflow.db, mlflow.db-wal, mlflow.db-shm -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path mlruns -Force | Out-Null
    Write-Host "Réinitialisation locale terminée."
}

Write-Host "Espace libre sur le disque:"
Get-PSDrive -Name C | Select-Object @{N="FreeGB";E={[math]::Round($_.Free/1GB,2)}}
