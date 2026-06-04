#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <opencv2/core.hpp>

namespace nb = nanobind;

NB_MODULE(_nanofractal, m) {
    m.attr("__version__") = "0.1.0";
    m.def("_opencv_version", []() { return std::string(cv::getVersionString()); });
}
