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
