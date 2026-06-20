<#
.SYNOPSIS
  Downloads the OpenCV 4.10.0 Windows prebuilt installer, verifies its SHA-256,
  and extracts it to C:\nf-opencv so cibuildwheel can link against it.

  Called via [tool.cibuildwheel.windows].before-all in pyproject.toml.
  The resulting layout:
    C:\nf-opencv\opencv\build\OpenCVConfig.cmake   <- OpenCV_DIR
    C:\nf-opencv\opencv\build\x64\vc16\bin\opencv_world4100.dll  <- delvewheel target

  ponytail: the self-extractor SHA-256 and the extraction path are version-specific.
  Ceiling: on OPENCV_VERSION bump —
    1. Re-download the new .exe and re-compute the hash with
         (Get-FileHash <exe> -Algorithm SHA256).Hash.ToLower()
    2. Update $EXPECTED_SHA256 below.
    3. Verify the new build still uses x64\vc16 (MSVC 2019 ABI); if vc17 appears,
       update the PATH/OpenCV_DIR entries in pyproject.toml [tool.cibuildwheel.windows].
#>

$ErrorActionPreference = "Stop"

$OPENCV_VERSION  = "4.10.0"
$DEST            = "C:\nf-opencv"
$EXE             = Join-Path $env:RUNNER_TEMP "opencv-$OPENCV_VERSION-windows.exe"
$URL             = "https://github.com/opencv/opencv/releases/download/$OPENCV_VERSION/opencv-$OPENCV_VERSION-windows.exe"

# SHA-256 of the official opencv-4.10.0-windows.exe release asset (175 MiB).
# Obtained by direct download + sha256sum on 2026-06-20.
# ponytail: ceiling = re-pin when OPENCV_VERSION changes (hash is asset-specific).
$EXPECTED_SHA256 = "bff38466091c313dac21a0b73eea8278316a89c1d434c6f0b10697e087670168"

# Short-circuit if already extracted (e.g. on a self-hosted runner with a warm cache).
$BUILD_DIR = "$DEST\opencv\build"
if (Test-Path "$BUILD_DIR\OpenCVConfig.cmake") {
    Write-Host "OpenCV already present at $BUILD_DIR — skipping download."
    exit 0
}

# Download
Write-Host "Downloading OpenCV $OPENCV_VERSION ..."
Write-Host "  URL: $URL"
Invoke-WebRequest -Uri $URL -OutFile $EXE -UseBasicParsing
Write-Host "  Saved to $EXE"

# Integrity check — fail hard on mismatch (supply-chain protection).
Write-Host "Verifying SHA-256 ..."
$actual = (Get-FileHash -Path $EXE -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $EXPECTED_SHA256) {
    Write-Error (
        "SHA-256 mismatch — aborting.`n" +
        "  Expected : $EXPECTED_SHA256`n" +
        "  Actual   : $actual`n" +
        "Re-pin the hash in ci/build-opencv-windows.ps1 after verifying the new release."
    )
    exit 1
}
Write-Host "  Hash OK: $actual"

# Extract.  opencv-4.10.0-windows.exe is a 7-zip SFX.
# Flags: -o<dest> sets the output directory (no space between -o and path),
#        -y assumes yes to all prompts.
# ponytail: -o/-y are 7-zip SFX flags; ceiling = if a future release switches to
# an NSIS or WiX installer, update the extraction command here.
Write-Host "Extracting to $DEST ..."
$proc = Start-Process `
    -FilePath  $EXE `
    -ArgumentList "-o$DEST -y" `
    -Wait -PassThru -NoNewWindow
if ($proc.ExitCode -ne 0) {
    Write-Error "Extractor exited with code $($proc.ExitCode)"
    exit 1
}

# Sanity check — confirm the key file landed where we expect it.
if (-not (Test-Path "$BUILD_DIR\OpenCVConfig.cmake")) {
    Write-Error "Extraction succeeded but OpenCVConfig.cmake not found at $BUILD_DIR"
    exit 1
}

Write-Host "OpenCV $OPENCV_VERSION ready at $BUILD_DIR"
