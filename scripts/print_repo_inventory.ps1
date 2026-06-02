$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path ".").Path
$excludedNames = @(
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules"
)

function Get-RelativePath {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )

    $baseUri = [System.Uri]($BasePath.TrimEnd('\') + '\')
    $targetUri = [System.Uri]$TargetPath
    $relative = $baseUri.MakeRelativeUri($targetUri).ToString()
    return [System.Uri]::UnescapeDataString($relative).Replace('/', '\')
}

Write-Output "# Repository inventory"
Write-Output "# Root: $repoRoot"
Write-Output "# Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "# Excluded: $($excludedNames -join ', ')"
Write-Output ""

Write-Output "[DIR] ."

Get-ChildItem -LiteralPath $repoRoot -Force -Recurse |
    Where-Object {
        $parts = $_.FullName.Substring($repoRoot.Length).TrimStart('\').Split('\')
        -not ($parts | Where-Object { $excludedNames -contains $_ })
    } |
    Sort-Object FullName |
    ForEach-Object {
        $relativePath = Get-RelativePath -BasePath $repoRoot -TargetPath $_.FullName
        if ($_.PSIsContainer) {
            Write-Output "[DIR] $relativePath"
        }
        else {
            Write-Output "[FILE] $relativePath"
        }
    }
