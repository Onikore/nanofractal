#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <opencv2/core.hpp>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

// Input image: uint8, C-contiguous, CPU. nanobind enforces dtype/contiguity/device
// at the call boundary (raising TypeError on mismatch).
using ImageArray = nb::ndarray<const uint8_t, nb::c_contig, nb::device::cpu>;

// Raw unconstrained ndarray used as the actual function parameter type so that
// nanobind does NOT silently coerce dtype or strides before we can inspect them.
// We then validate dtype and contiguity ourselves and raise TypeError on mismatch.
using RawArray = nb::ndarray<nb::numpy>;

// Validate dtype==uint8 and C-contiguous on the raw (uncoerced) ndarray.
inline void validate_image_array(const RawArray &arr) {
    // Check dtype: must be uint8 (DLPack UInt, 8 bits, 1 lane).
    auto dt = arr.dtype();
    if (dt.code != static_cast<uint8_t>(nb::dlpack::dtype_code::UInt) || dt.bits != 8) {
        throw nb::type_error("image array must have dtype uint8");
    }
    // Check C-contiguity: stride[i] (bytes) must equal product of remaining dims.
    // For uint8, stride in bytes == stride in elements.
    if (arr.ndim() >= 1) {
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
    throw std::invalid_argument(
        "image must be uint8 (H,W) grayscale or (H,W,3) BGR, C-contiguous");
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
    return make_owned<float>(std::move(c), {n, (size_t)4, (size_t)2});
}
