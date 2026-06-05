#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <opencv2/core.hpp>
#include <cassert>
#include <vector>

namespace nb = nanobind;

// NOTE: nb::ndarray with constrained template args (e.g. const uint8_t, c_contig)
// SILENTLY COERCES mismatched inputs (casting dtype / making a contiguous copy)
// rather than raising. That would defeat both zero-copy and input validation. So we
// take an unconstrained RawArray as the parameter type and validate dtype +
// C-contiguity ourselves (raising TypeError), then zero-copy wrap in as_mat().
using RawArray = nb::ndarray<nb::numpy>;

// Typed contiguous CPU float arrays for numeric I/O (e.g. pose: corners,
// camera matrix, distortion). Unlike image inputs these are small and the
// Python facade pre-converts them, so nanobind coercing to the exact type is
// acceptable here.
using F32Arr = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using F64Arr = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using I32Arr = nb::ndarray<const int32_t, nb::c_contig, nb::device::cpu>;

// Validate dtype==uint8 and C-contiguous on the raw (uncoerced) ndarray.
inline void validate_image_array(const RawArray &arr) {
    // Check dtype: must be uint8 (DLPack UInt, 8 bits, 1 lane).
    auto dt = arr.dtype();
    if (dt.code != static_cast<uint8_t>(nb::dlpack::dtype_code::UInt) ||
        dt.bits != 8 || dt.lanes != 1) {
        throw nb::type_error("image array must have dtype uint8");
    }
    // Check C-contiguity: stride[i] (in elements) must equal the product of the
    // trailing dims. A zero-size dimension makes numpy strides 0 yet the array is
    // vacuously contiguous, so skip the check in that case.
    if (arr.ndim() >= 1) {
        for (size_t i = 0; i < arr.ndim(); ++i)
            if (arr.shape(i) == 0) return;  // empty array is contiguous
        size_t expected = 1;
        for (int i = (int)arr.ndim() - 1; i >= 0; --i) {
            if (arr.stride(i) != (int64_t)expected) {
                throw nb::type_error("image array must be C-contiguous");
            }
            expected *= arr.shape(i);
        }
    }
}

// Wrap a numpy buffer as cv::Mat WITHOUT copying.
// Accepts a RawArray so nanobind cannot coerce it before we validate.
// Raises TypeError for wrong dtype or non-C-contiguous input.
// The source array must outlive the returned Mat.
inline cv::Mat as_mat(const RawArray &arr) {
    validate_image_array(arr);
    auto *data = static_cast<uint8_t *>(arr.data());
    if (arr.ndim() == 2) {
        return cv::Mat((int)arr.shape(0), (int)arr.shape(1), CV_8UC1, data);
    }
    if (arr.ndim() == 3 && arr.shape(2) == 3) {
        return cv::Mat((int)arr.shape(0), (int)arr.shape(1), CV_8UC3, data);
    }
    throw nb::value_error(
        "image must be uint8 (H,W) grayscale or (H,W,3) BGR, C-contiguous");
}

// Like as_mat but also rejects degenerate (0-size) images. Detection entry points
// use this so an empty frame yields a clear ValueError instead of an opaque
// OpenCV assertion failure deep in the detector.
inline cv::Mat as_image(const RawArray &arr) {
    cv::Mat m = as_mat(arr);
    if (m.rows == 0 || m.cols == 0)
        throw nb::value_error("image is empty");
    return m;
}

// Build an owned numpy array by copying a flat vector. The heap vector is freed
// when numpy releases the capsule.
template <typename T>
nb::ndarray<nb::numpy, T> make_owned(std::vector<T> &&data,
                                     std::initializer_list<size_t> shape) {
    auto *heap = new std::vector<T>(std::move(data));
    nb::capsule owner(heap, [](void *p) noexcept {
        delete static_cast<std::vector<T> *>(p);
    });
    return nb::ndarray<nb::numpy, T>(heap->data(), shape, owner);
}

// Convenience: int32 ids (N,) and float32 corners (N,4,2).
inline nb::ndarray<nb::numpy, int32_t> ids_to_numpy(std::vector<int32_t> &&ids) {
    size_t n = ids.size();
    return make_owned<int32_t>(std::move(ids), {n});
}

inline nb::ndarray<nb::numpy, float> corners_to_numpy(std::vector<float> &&c,
                                                      size_t n) {
    assert(c.size() == n * 8 && "corners buffer must hold n*4*2 floats");
    return make_owned<float>(std::move(c), {n, (size_t)4, (size_t)2});
}
