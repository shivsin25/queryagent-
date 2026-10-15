<#
.SYNOPSIS
Deploys the last commit to a Hugging Face Space (Docker SDK).

.DESCRIPTION
Exports the committed code (never .env or storage/), swaps in the Space's README, which
carries the configuration Hugging Face reads, and pushes it to the Space, which then builds
the Docker image. The Space repository is a deployment target: each deploy replaces its
contents and history. When git asks you to sign in, use your Hugging Face username and an
access token with write permission as the password.

.EXAMPLE
.\scripts\deploy_hf_space.ps1 -Space shivsin25/query-tool
.\scripts\deploy_hf_space.ps1 -Space shivsin25/query-tool -DryRun
#>
param(
    [Parameter(Mandatory = $true)][string]$Space,
    [switch]$DryRun  # build the deploy folder, but don't push
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (git -C $root status --porcelain) {
    Write-Warning "You have uncommitted changes. Only the last commit is deployed."
}
$sha = git -C $root rev-parse --short HEAD
$build = Join-Path ([IO.Path]::GetTempPath()) "hf-space-$sha"
if (Test-Path $build) { Remove-Item $build -Recurse -Force }
New-Item -ItemType Directory $build | Out-Null

git -C $root archive --format=tar --output "$build.tar" HEAD
tar -xf "$build.tar" -C $build
Remove-Item "$build.tar"
Copy-Item "$root\deploy\huggingface\README.md" "$build\README.md" -Force
Remove-Item "$build\docs" -Recurse -Force -ErrorAction SilentlyContinue  # screenshots

Push-Location $build
try {
    git init -q -b main
    git lfs install --local | Out-Null
    git lfs track "*.pdf" "*.png" | Out-Null  # the Hub stores binary files with LFS
    git add -A
    git -c "user.name=$(git -C $root config user.name)" `
        -c "user.email=$(git -C $root config user.email)" `
        commit -q -m "Deploy $sha"
    if ($DryRun) {
        Write-Host "Dry run: deploy folder ready at $build" -ForegroundColor Yellow
        return
    }
    git push --force "https://huggingface.co/spaces/$Space" main
    if ($LASTEXITCODE -ne 0) { throw "Push to the Space failed." }
    Write-Host "Deployed $sha. Watch the build: https://huggingface.co/spaces/$Space" -ForegroundColor Green
}
finally {
    Pop-Location
}
