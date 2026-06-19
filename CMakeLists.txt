cmake_minimum_required(VERSION 3.18)
project(nanofractal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind shipped with the build environment
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

# OpenCV: system by default. CI (Plan B) overrides via -DOpenCV_DIR=<minimal build>
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d features2d)

nanobind_add_module(_nanofractal NB_STATIC src/_bindings.cpp)

target_include_directories(_nanofractal PRIVATE
  "${CMAKE_SOURCE_DIR}/third_party"
  "${CMAKE_SOURCE_DIR}/src"
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(_nanofractal PRIVATE ${OpenCV_LIBS})

# Release already implies -O3 on GCC/Clang; keep it explicit but MSVC-safe.
target_compile_options(_nanofractal PRIVATE
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>)

# ---------------------------------------------------------------------------
# Architecture & FP tuning (GCC / Clang only; MSVC is skipped silently).
#
#   -DNF_NATIVE=ON          local dev: -march=native -ffast-math
#   -DNF_MARCH=x86-64-v3   portable:  -march=<isa>  -ffast-math (CI wheels)
#
# x86-64-v3 = AVX2 + FMA + BMI1/2 (Haswell 2013+).  Enables auto-vectorisation
# of inner pixel loops without breaking pre-Skylake CPUs in the wild.
# ffast-math: allows fp-reassociation and removes errno overhead; safe for
# this library (no signalling NaNs produced, no errno-dependent callers).
#
# aarch64 / arm64: NEON is mandatory in the ARMv8-A base ISA, so GCC/Clang
# already enable it by default.  We add -ftree-vectorize to encourage the
# auto-vectorizer and skip -march=native / NF_MARCH entirely (the x86 ISA
# string is invalid on ARM and would cause a hard build failure).
# ---------------------------------------------------------------------------
option(NF_NATIVE "Enable -march=native -ffast-math (local dev only; not for distributed wheels)" OFF)
set(NF_MARCH "" CACHE STRING "ISA baseline passed to -march= (e.g. x86-64-v3). Used in CI wheels.")

if(NOT MSVC)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8")
    # ARM / aarch64 path: rely on the compiler's default ARMv8-A target
    # (NEON always available).  Do NOT apply NF_MARCH — an x86 ISA string
    # would break the build.  -ffast-math is also safe here.
    target_compile_options(_nanofractal PRIVATE -O3 -ftree-vectorize -ffast-math)
  else()
    # x86 / x86_64 path: honour NF_NATIVE or NF_MARCH as before.
    if(NF_NATIVE)
      target_compile_options(_nanofractal PRIVATE -march=native -ffast-math)
    elseif(NF_MARCH)
      target_compile_options(_nanofractal PRIVATE -march=${NF_MARCH} -ffast-math)
    endif()
  endif()
endif()

# Single source of truth for the version: injected by scikit-build-core from
# pyproject.toml. Falls back for non-skbuild direct cmake builds.
if(DEFINED SKBUILD_PROJECT_VERSION)
  set(_NF_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
  set(_NF_VERSION "0.0.0")
endif()
target_compile_definitions(_nanofractal PRIVATE NF_VERSION="${_NF_VERSION}")

include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg)
if(_ipo_ok)
  set_property(TARGET _nanofractal PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

install(TARGETS _nanofractal LIBRARY DESTINATION nanofractal)
