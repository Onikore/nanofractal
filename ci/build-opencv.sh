#!/usr/bin/env bash
# Build a minimal STATIC OpenCV inside the cibuildwheel manylinux container.
# Only the modules nanofractal needs are built (core, imgproc, calib3d,
# features2d), with -fPIC so they can be linked into the extension .so. The wheel
# is then self-contained (no system OpenCV needed at runtime).
set -euo pipefail

OPENCV_VERSION="${OPENCV_VERSION:-4.10.0}"
PREFIX="${OPENCV_PREFIX:-/opt/opencv-static}"

if [ -f "$PREFIX/lib64/cmake/opencv4/OpenCVConfig.cmake" ] || \
   [ -f "$PREFIX/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
  echo "OpenCV already installed at $PREFIX — skipping build."
  exit 0
fi

# zlib is the only external dependency of the selected modules.
yum install -y zlib-devel || dnf install -y zlib-devel || true

cd /tmp
curl -L -o opencv.tar.gz \
  "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz"
tar xzf opencv.tar.gz

# SIMD: pin the baseline + runtime dispatch explicitly so the vectorised kernels
# (adaptiveThreshold, resize, cornerSubPix) are always compiled and CPU-selected
# at runtime, independent of OpenCV's per-version defaults. SSE4_2 is a safe
# baseline for the x86-64-v3 wheel floor; AVX2/AVX512 are dispatched on top.
cmake -S "opencv-${OPENCV_VERSION}" -B ocv-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCPU_BASELINE=SSE4_2 \
  -DCPU_DISPATCH=AVX,AVX2,FP16,AVX512_SKX \
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
