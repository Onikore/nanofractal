#!/usr/bin/env bash
# Build a minimal STATIC OpenCV (cibuildwheel manylinux container, macOS
# runners, or the OpenCV 5 CI job). Only the modules nanofractal needs are built
# (4.x: core, imgproc, calib3d, features2d; 5.x: core, imgproc, geometry,
# features), with -fPIC so they can be linked into the extension. The wheel is
# then self-contained (no system OpenCV needed at runtime).
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

# zlib is the only external dependency of the selected modules (manylinux;
# elsewhere it is present or OpenCV builds its bundled copy).
if command -v yum >/dev/null || command -v dnf >/dev/null; then
  yum install -y zlib-devel || dnf install -y zlib-devel || true
fi

cd /tmp
curl -L -o opencv.tar.gz \
  "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz"
# Integrity check against a supply-chain tarball swap.
# ponytail: GitHub auto-generated archive hashes are stable in practice but NOT
# contractually guaranteed (a recompression would change them). Ceiling: if this
# check fails after bumping OPENCV_VERSION (or a rare GitHub recompress), update
# the pinned hash. Upgrade path: switch to a release asset with a published digest.
case "$OPENCV_VERSION" in
  4.10.0) sha="b2171af5be6b26f7a06b1229948bbb2bdaa74fcf5cd097e0af6378fce50a6eb9" ;;
  5.0.0) sha="b0528f5a1d379d59d4701cb28c36e22214cc51cf64594e5b56f2d3e6c0233095" ;;
  *) echo "no pinned SHA-256 for OpenCV $OPENCV_VERSION" >&2; exit 1 ;;
esac
if [ "$(uname -s)" = "Darwin" ]; then # macOS 15 ships a BSD sha256sum
  echo "${sha}  opencv.tar.gz" | shasum -a 256 --check
else
  echo "${sha}  opencv.tar.gz" | sha256sum --check --strict
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

case "$OPENCV_VERSION" in
  5.*) MODULES="core,imgproc,geometry,features" ;;
  *) MODULES="core,imgproc,calib3d,features2d" ;;
esac

# macOS: keep the static libs free of framework dependencies (OpenCL,
# Accelerate LAPACK) that nanofractal never uses, and use the pthreads
# parallel backend like Linux: OpenCV picks Grand Central Dispatch on Apple
# unconditionally, and GCD ignores setNumThreads() (set_num_threads(2) would
# still use every core, get_num_threads() always reports the core count).
EXTRA=""
if [ "$(uname -s)" = "Darwin" ]; then
  EXTRA="-DWITH_OPENCL=OFF -DWITH_LAPACK=OFF -DWITH_PTHREADS_PF=ON"
  perl -pi -e 's/if\(APPLE AND NOT HAVE_TBB\)/if(FALSE) # nanofractal: no GCD/' "opencv-${OPENCV_VERSION}/cmake/OpenCVFindFrameworks.cmake"
  grep -q "nanofractal: no GCD" "opencv-${OPENCV_VERSION}/cmake/OpenCVFindFrameworks.cmake"
  # parallel.cpp also defines HAVE_GCD itself on __APPLE__, ahead of pthreads
  perl -0pi -e 's/#ifdef __APPLE__\n(\s*)#define HAVE_GCD/#if 0 \/\/ nanofractal: no GCD\n$1#define HAVE_GCD/' \
    "opencv-${OPENCV_VERSION}/modules/core/src/parallel.cpp"
  grep -q "nanofractal: no GCD" "opencv-${OPENCV_VERSION}/modules/core/src/parallel.cpp"
fi
NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

cmake -S "opencv-${OPENCV_VERSION}" -B ocv-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  ${CPU_FLAGS} ${EXTRA} \
  -DBUILD_LIST=${MODULES} \
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

cmake --build ocv-build --target install -j"${NPROC}"
echo "OpenCV ${OPENCV_VERSION} (static, minimal) installed at $PREFIX"
