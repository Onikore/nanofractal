#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <opencv2/core.hpp>
#include "ndarray_cv.hpp"

namespace nb = nanobind;

NB_MODULE(_nanofractal, m) {
    m.attr("__version__") = NF_VERSION;
    m.def("_opencv_version", []() { return std::string(cv::getVersionString()); });

    m.def("_img_info", [](RawArray arr) {
        cv::Mat im = as_mat(arr);
        return nb::make_tuple(im.rows, im.cols, im.channels());
    });

    m.def("_img_data_ptr", [](RawArray arr) {
        return (uintptr_t)as_mat(arr).data;
    });

    m.def("_img_mean", [](RawArray arr) {
        cv::Mat im = as_mat(arr);
        double mean;
        {
            nb::gil_scoped_release rel;
            mean = cv::mean(im)[0];
        }
        return mean;
    });

    m.def("_echo_corners", []() {
        std::vector<float> c = {1.5f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,
                                1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        return corners_to_numpy(std::move(c), 2);
    });
}
