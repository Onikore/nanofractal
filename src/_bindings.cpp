#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <opencv2/core.hpp>

namespace nb = nanobind;

NB_MODULE(_nanofractal, m) {
    m.attr("__version__") = NF_VERSION;
    m.def("_opencv_version", []() { return std::string(cv::getVersionString()); });
}
