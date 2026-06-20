#!/usr/bin/env bash
# Build a minimal STATIC OpenCV inside the cibuildwheel manylinux container.
# Only the modules nanofractal needs are built (core, imgproc, calib3d,
# features2d), with -fPIC so they can be linked into the extension .so. The wheel
# is then self-contained (no system OpenCV needed at runtime).
set -euo pipefail

OPENCV_VERSION="${OPENCV_VERSION:-4.10.0}"
PREFIX="${OPENCV_CACHE_DIR:-/project/.opencv-cache}"

# Short-circuit on cache hit — checked before any package installs so a
# restored actions/cache is respected without touching the network.
if [ -f "$PREFIX/lib/libopencv_core.a" ] || [ -f "$PREFIX/lib64/libopencv_core.a" ]; then
  echo "OpenCV cache hit at $PREFIX — skipping build."
  exit 0
fi

mkdir -p "$PREFIX"

# zlib is the only external dependency of the selected modules.
yum install -y zlib-devel || dnf install -y zlib-devel || true

cd /tmp
curl -L -o opencv.tar.gz \
  "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz"
# Integrity check against a supply-chain tarball swap.
# ponytail: GitHub auto-generated archive hashes are stable in practice but NOT
# contractually guaranteed (a recompression would change them). Ceiling: if this
# check fails after bumping OPENCV_VERSION (or a rare GitHub recompress), update
# the pinned hash. Upgrade path: switch to a release asset with a published digest.
OPENCV_SHA256_4_10_0="b2171af5be6b26f7a06b1229948bbb2bdaa74fcf5cd097e0af6378fce50a6eb9"
if [ "$OPENCV_VERSION" = "4.10.0" ]; then
  echo "${OPENCV_SHA256_4_10_0}  opencv.tar.gz" | sha256sum --check --strict
fi
tar xzf opencv.tar.gz

# SIMD flags differ by architecture.
# x86_64: pin SSE4_2 baseline (matches the x86-64-v3 wheel floor) with AVX/AVX2/
#         AVX512 runtime dispatch so vectorised kernels are CPU-selected at runtime.
# aarch64: NEON is mandatory in ARMv8-A; use it as the baseline and omit the x86
#          dispatch list entirely (passing SSE4_2 on ARM would break the build).
arch="$(uname -m)"
if [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then
  CPU_FLAGS="-DCPU_BASELINE=NEON -DCPU_DISPATCH="
else
  CPU_FLAGS="-DCPU_BASELINE=SSE4_2 -DCPU_DISPATCH=AVX,AVX2,FP16,AVX512_SKX"
fi

cmake -S "opencv-${OPENCV_VERSION}" -B ocv-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  ${CPU_FLAGS} \
  -DBUILD_LIST=core,imgproc,calib3d,features2d \
  -DBUILD_opencv_apps=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
  -DBUILD_opencv_python3=OFF -DBUILD_opencv_python2=OFF -DBUILD_JAVA=OFF \
  -DBUILD_opencv_gapi=OFF -DWITH_ADE=OFF -DWITH_ITT=OFF \
  -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF \
  -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF \
  -DWITH_OPENEXR=OFF -DWITH_JASPER=OFF -DWITH_OPENJPEG=OFF \
  -DWITH_V4L=OFF -DWITH_1394=OFF -DWITH_PROTOBUF=OFF \
  -DWITH_IPP=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_EIGEN=OFF \
  -DOPENCV_GENERATE_PKGCONFIG=OFF

cmake --build ocv-build --target install -j"$(nproc)"
echo "OpenCV ${OPENCV_VERSION} (static, minimal) installed at $PREFIX"
