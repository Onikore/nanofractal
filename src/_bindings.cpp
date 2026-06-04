#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <opencv2/core.hpp>
#include "ndarray_cv.hpp"
#include "aruco_nano_v6.h"
#include "aruco_dicts.hpp"

namespace nb = nanobind;

// ---- ArUco Nano v6 ----
// Defined at namespace scope (not inside NB_MODULE) so it is a non-local type,
// which is the conventional, portable way to use it as a nb::class_ template arg.
struct ArucoDetectorImpl {
    int dict;
    unsigned max_attempts;
    ArucoDetectorImpl(int dictionary, unsigned attempts) {
        if (dictionary != 0 && dictionary != 1)
            throw nb::value_error(
                "dictionary must be 0 (ARUCO_MIP_36h12) or 1 (APRILTAG_36h11)");
        dict = dictionary;
        max_attempts = attempts ? attempts : 1u;
    }

    nb::tuple detect(RawArray arr) {
        cv::Mat im = as_mat(arr);
        std::vector<aruconano::Marker> markers;
        {
            nb::gil_scoped_release rel;
            markers = aruconano::MarkerDetector::detect(
                im, max_attempts, (aruconano::MarkerDetector::Dict)dict);
        }
        size_t n = markers.size();
        std::vector<int32_t> ids(n);
        std::vector<float> corners(n * 8);
        for (size_t i = 0; i < n; i++) {
            ids[i] = markers[i].id;
            for (int c = 0; c < 4; c++) {
                corners[i * 8 + c * 2 + 0] = markers[i][c].x;
                corners[i * 8 + c * 2 + 1] = markers[i][c].y;
            }
        }
        return nb::make_tuple(ids_to_numpy(std::move(ids)),
                              corners_to_numpy(std::move(corners), n));
    }
};

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

    // Render an 8x8 cell grid (1-cell black border + 6x6 inner code) for a marker
    // id, matching the bit ordering of touulong() in aruco_nano_v6.h. Test/tool use.
    m.def("_aruco_marker_image8", [](int dict, int id) {
        const std::vector<uint64_t> &codes =
            dict == 0 ? aruco_dicts::mip_36h12() : aruco_dicts::apriltag_36h11();
        if (id < 0 || (size_t)id >= codes.size())
            throw nb::value_error("marker id out of range");
        uint64_t code = codes[id];

        std::vector<uint8_t> grid(8 * 8, 0);  // border stays 0 (black)
        int b = 0;
        for (int y = 5; y >= 0; y--)
            for (int x = 5; x >= 0; x--) {
                int bit = (int)((code >> b) & 1ULL);
                grid[(y + 1) * 8 + (x + 1)] = bit ? 255 : 0;  // +1 for border
                b++;
            }
        return make_owned<uint8_t>(std::move(grid), {(size_t)8, (size_t)8});
    });

    // ---- ArUco Nano v6 ----
    nb::class_<ArucoDetectorImpl>(m, "ArucoDetector")
        .def(nb::init<int, unsigned>(), nb::arg("dictionary"),
             nb::arg("max_attempts"))
        .def_ro("max_attempts", &ArucoDetectorImpl::max_attempts)
        .def_ro("dictionary", &ArucoDetectorImpl::dict)
        .def("detect", &ArucoDetectorImpl::detect, nb::arg("image"));
}
