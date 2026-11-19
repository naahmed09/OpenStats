<#
.SYNOPSIS
  Builds the OpenStats Windows installers: an .msi (WiX) and a setup .exe (NSIS).

.DESCRIPTION
  Needs Windows 10/11, Rust (rustup, MSVC toolchain), the Visual Studio Build Tools, the Tauri
  CLI (cargo install tauri-cli --version "^2" --locked) and Git for Windows (its bash runs the
  shared web-bundle script). Output goes to dist\: OpenStats_VERSION_ARCH.msi and
  OpenStats_VERSION_ARCH-setup.exe, plus SHA256SUMS.

  Code signing is optional: set OPENSTATS_WIN_CERT_THUMBPRINT to the thumbprint of a certificate in
  the Windows certificate store (and optionally OPENSTATS_WIN_TIMESTAMP_URL). Unsigned installers
  trigger SmartScreen warnings.

.PARAMETER Arch
  x64 (default) or arm64.
#>
param(
  [ValidateSet('x64', 'arm64')] [string] $Arch = 'x64',
  [switch] $DryRun
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Dist = if ($env:OPENSTATS_DIST) { $env:OPENSTATS_DIST } else { Join-Path $Root 'dist' }
$TauriDir = Join-Path $Root 'openstats-desktop\tauri\src-tauri'

function Invoke-Step([string] $Description, [scriptblock] $Block) {
  Write-Host "==> $Description"
  if ($DryRun) { Write-Host "[dry-run] $Block"; return }
  & $Block
  if ($LASTEXITCODE -ne 0) { throw "$Description failed with exit code $LASTEXITCODE" }
}

foreach ($tool in 'cargo', 'rustup', 'bash') {
  if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "'$tool' is required but was not found on PATH" }
}
# On GitHub's Windows runners (and any PC with WSL) a bare `bash` is the WSL launcher, which fails
# without a Linux distribution. Use the bash that ships with Git for Windows instead.
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
  $gitBash = Join-Path (Split-Path -Parent (Split-Path -Parent $git.Source)) 'bin\bash.exe'
  if (Test-Path $gitBash) { function bash { & $gitBash @args } }
}
$RootPosix = $Root -replace '\\', '/'
& cargo tauri --version *> $null
if ($LASTEXITCODE -ne 0) { throw 'The Tauri CLI is required: cargo install tauri-cli --version "^2" --locked' }

$triple = if ($Arch -eq 'arm64') { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
$version = (Select-String -Path (Join-Path $Root 'Cargo.toml') -Pattern '^version = "(.*)"$' | Select-Object -First 1).Matches.Groups[1].Value

Invoke-Step 'checking versions' { bash "$RootPosix/scripts/check-versions.sh" }
Invoke-Step "adding the $triple target" { rustup target add $triple }
Invoke-Step 'building the web bundle' { bash "$RootPosix/scripts/build-web.sh" }

$extra = @()
if ($env:OPENSTATS_WIN_CERT_THUMBPRINT) {
  $stamp = if ($env:OPENSTATS_WIN_TIMESTAMP_URL) { $env:OPENSTATS_WIN_TIMESTAMP_URL } else { 'http://timestamp.digicert.com' }
  $cfg = @{ bundle = @{ windows = @{ certificateThumbprint = $env:OPENSTATS_WIN_CERT_THUMBPRINT; digestAlgorithm = 'sha256'; timestampUrl = $stamp } } } | ConvertTo-Json -Compress -Depth 5
  $extra = @('--config', $cfg)
} else {
  Write-Warning 'OPENSTATS_WIN_CERT_THUMBPRINT is not set: the installers will be unsigned'
}

Push-Location $TauriDir
try {
  Invoke-Step "building the installers for $triple" { cargo tauri build --target $triple --bundles msi,nsis @extra }
} finally {
  Pop-Location
}

if ($DryRun) { Write-Host "dry run: would copy installers to $Dist"; return }

New-Item -ItemType Directory -Force -Path $Dist | Out-Null
$bundle = Join-Path $TauriDir "target\$triple\release\bundle"
$msi = Get-ChildItem -Path (Join-Path $bundle 'msi') -Filter '*.msi' | Select-Object -First 1
$exe = Get-ChildItem -Path (Join-Path $bundle 'nsis') -Filter '*-setup.exe' | Select-Object -First 1
if (-not $msi) { throw "Tauri produced no .msi under $bundle\msi" }
if (-not $exe) { throw "Tauri produced no setup .exe under $bundle\nsis" }
Copy-Item $msi.FullName (Join-Path $Dist "OpenStats_${version}_$Arch.msi") -Force
Copy-Item $exe.FullName (Join-Path $Dist "OpenStats_${version}_$Arch-setup.exe") -Force

if ($env:OPENSTATS_WIN_CERT_THUMBPRINT) {
  foreach ($f in "OpenStats_${version}_$Arch.msi", "OpenStats_${version}_$Arch-setup.exe") {
    $sig = Get-AuthenticodeSignature (Join-Path $Dist $f)
    if ($sig.Status -ne 'Valid') { throw "$f is not validly signed ($($sig.Status))" }
  }
}

$sums = Get-ChildItem -Path $Dist -File | Where-Object { $_.Name -ne 'SHA256SUMS' } | ForEach-Object {
  '{0}  {1}' -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name
}
Set-Content -Path (Join-Path $Dist 'SHA256SUMS') -Value $sums -Encoding ascii
Write-Host "wrote installers to $Dist"
