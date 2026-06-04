#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>  // FastFeatureDetector, used by nanofractal.h
#include <memory>
#include <thread>
#include <atomic>
#include "ndarray_cv.hpp"
#include "aruco_nano_v6.h"
#include "aruco_dicts.hpp"
#include "nanofractal.h"

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

    // Batched pose: for each marker's 4 corners, solvePnP (IPPE) against a
    // square of side marker_size. Returns (rvecs (N,3), tvecs (N,3)) float64.
    nb::tuple estimate_pose(F32Arr corners, F64Arr cam, F64Arr dist,
                            double marker_size) {
        if (corners.ndim() != 3 || corners.shape(1) != 4 || corners.shape(2) != 2)
            throw nb::value_error("corners must be float32 (N,4,2)");
        if (cam.ndim() != 2 || cam.shape(0) != 3 || cam.shape(1) != 3)
            throw nb::value_error("camera_matrix must be float64 (3,3)");
        if (dist.ndim() != 1)
            throw nb::value_error("dist_coeffs must be 1-D float64");
        size_t n = corners.shape(0);
        // const_cast is safe: solvePnP only reads camera/dist, and the facade
        // passes freshly-made contiguous arrays (no aliasing with outputs).
        cv::Mat camMat(3, 3, CV_64F, const_cast<double *>(cam.data()));
        cv::Mat distMat((int)dist.shape(0), 1, CV_64F,
                        const_cast<double *>(dist.data()));
        std::vector<double> rvecs(n * 3), tvecs(n * 3);
        {
            nb::gil_scoped_release rel;
            for (size_t i = 0; i < n; i++) {
                aruconano::Marker mk;
                for (int c = 0; c < 4; c++)
                    mk.push_back(cv::Point2f(corners.data()[i * 8 + c * 2 + 0],
                                             corners.data()[i * 8 + c * 2 + 1]));
                auto rt = mk.estimatePose(camMat, distMat, marker_size);
                cv::Mat rv = rt.first, tv = rt.second;
                for (int k = 0; k < 3; k++) {
                    rvecs[i * 3 + k] = rv.at<double>(k);
                    tvecs[i * 3 + k] = tv.at<double>(k);
                }
            }
        }
        return nb::make_tuple(make_owned<double>(std::move(rvecs), {n, (size_t)3}),
                              make_owned<double>(std::move(tvecs), {n, (size_t)3}));
    }

    // Parallel batch detection. MarkerDetector::detect is stateless, so all worker
    // threads can share this detector. Inputs are validated/wrapped (GIL held),
    // detection runs GIL-released across a thread pool, then results are marshaled
    // to numpy (GIL held).
    std::vector<nb::object> detect_batch(std::vector<RawArray> imgs,
                                         int num_threads) {
        size_t N = imgs.size();
        std::vector<cv::Mat> mats(N);
        for (size_t i = 0; i < N; i++) mats[i] = as_mat(imgs[i]);

        std::vector<std::vector<int32_t>> all_ids(N);
        std::vector<std::vector<float>> all_corners(N);
        int T = num_threads > 0 ? num_threads
                                : (int)std::thread::hardware_concurrency();
        if (T < 1) T = 1;
        int dict_ = dict;
        unsigned attempts_ = max_attempts;
        if (N > 0) {
            nb::gil_scoped_release rel;
            std::atomic<size_t> next{0};
            auto worker = [&]() {
                size_t i;
                while ((i = next.fetch_add(1)) < N) {
                    auto markers = aruconano::MarkerDetector::detect(
                        mats[i], attempts_, (aruconano::MarkerDetector::Dict)dict_);
                    size_t n = markers.size();
                    all_ids[i].resize(n);
                    all_corners[i].resize(n * 8);
                    for (size_t k = 0; k < n; k++) {
                        all_ids[i][k] = markers[k].id;
                        for (int c = 0; c < 4; c++) {
                            all_corners[i][k * 8 + c * 2 + 0] = markers[k][c].x;
                            all_corners[i][k * 8 + c * 2 + 1] = markers[k][c].y;
                        }
                    }
                }
            };
            std::vector<std::thread> ths;
            for (int t = 0; t < T; t++) ths.emplace_back(worker);
            for (auto &x : ths) x.join();
        }
        std::vector<nb::object> out;
        out.reserve(N);
        for (size_t i = 0; i < N; i++) {
            size_t n = all_ids[i].size();
            out.push_back(nb::make_tuple(
                ids_to_numpy(std::move(all_ids[i])),
                corners_to_numpy(std::move(all_corners[i]), n)));
        }
        return out;
    }
};

// ---- Fractal ----
// FractalMarkerDetector::detect is non-const (map::operator[] is non-const; the
// with-inner-points path also mutates a lazy getKeypts() cache), so it is NOT safe
// to call concurrently on one instance. We keep a pool of independent detectors
// (one per worker thread); detect() uses pool[0]. The fractal config is expensive
// to build, so it is constructed once per pooled detector.
struct FractalDetectorImpl {
    std::string config;
    float marker_size;
    std::vector<std::unique_ptr<nanofractal::FractalMarkerDetector>> pool;

    FractalDetectorImpl(std::string cfg, float msize)
        : config(std::move(cfg)), marker_size(msize) {
        pool.push_back(make_detector());
    }

    // Move-only: the detector pool holds unique_ptrs. Explicitly deleting copy
    // keeps std::is_copy_constructible false so nanobind does not try to emit a
    // (ill-formed) copy of the unique_ptr vector.
    FractalDetectorImpl(const FractalDetectorImpl &) = delete;
    FractalDetectorImpl &operator=(const FractalDetectorImpl &) = delete;
    FractalDetectorImpl(FractalDetectorImpl &&) = default;
    FractalDetectorImpl &operator=(FractalDetectorImpl &&) = default;

    std::unique_ptr<nanofractal::FractalMarkerDetector> make_detector() const {
        auto d = std::make_unique<nanofractal::FractalMarkerDetector>();
        d->setParams(config, marker_size > 0 ? marker_size : -1.f);
        return d;
    }

    static void fill(const std::vector<nanofractal::FractalMarker> &markers,
                     std::vector<int32_t> &ids, std::vector<float> &corners) {
        size_t n = markers.size();
        ids.resize(n);
        corners.resize(n * 8);
        for (size_t i = 0; i < n; i++) {
            ids[i] = markers[i].id;
            for (int c = 0; c < 4; c++) {
                corners[i * 8 + c * 2 + 0] = markers[i][c].x;
                corners[i * 8 + c * 2 + 1] = markers[i][c].y;
            }
        }
    }

    nb::tuple detect(RawArray arr) {
        cv::Mat im = as_mat(arr);
        std::vector<nanofractal::FractalMarker> markers;
        {
            nb::gil_scoped_release rel;
            markers = pool[0]->detect(im);
        }
        std::vector<int32_t> ids;
        std::vector<float> corners;
        fill(markers, ids, corners);
        size_t n = ids.size();
        return nb::make_tuple(ids_to_numpy(std::move(ids)),
                              corners_to_numpy(std::move(corners), n));
    }

    // detect + all visible (inner) corner correspondences for occlusion-robust
    // pose: returns (ids, corners, points_2d (M,2), points_3d (M,3)).
    nb::tuple detect_full(RawArray arr) {
        cv::Mat im = as_mat(arr);
        std::vector<nanofractal::FractalMarker> markers;
        std::vector<cv::Point3f> p3d;
        std::vector<cv::Point2f> p2d;
        {
            nb::gil_scoped_release rel;
            markers = pool[0]->detect(im, p3d, p2d);
        }
        std::vector<int32_t> ids;
        std::vector<float> corners;
        fill(markers, ids, corners);
        size_t n = ids.size();

        size_t m2 = p2d.size();
        std::vector<float> pts2(m2 * 2), pts3(m2 * 3);
        for (size_t i = 0; i < m2; i++) {
            pts2[i * 2 + 0] = p2d[i].x;
            pts2[i * 2 + 1] = p2d[i].y;
            pts3[i * 3 + 0] = p3d[i].x;
            pts3[i * 3 + 1] = p3d[i].y;
            pts3[i * 3 + 2] = p3d[i].z;
        }
        return nb::make_tuple(
            ids_to_numpy(std::move(ids)),
            corners_to_numpy(std::move(corners), n),
            make_owned<float>(std::move(pts2), {m2, (size_t)2}),
            make_owned<float>(std::move(pts3), {m2, (size_t)3}));
    }

    // Parallel batch detection. The fractal detector is not thread-safe, so each
    // worker thread t uses its own pool[t] (built once, lazily grown to T).
    std::vector<nb::object> detect_batch(std::vector<RawArray> imgs,
                                         int num_threads) {
        size_t N = imgs.size();
        std::vector<cv::Mat> mats(N);
        for (size_t i = 0; i < N; i++) mats[i] = as_mat(imgs[i]);

        int T = num_threads > 0 ? num_threads
                                : (int)std::thread::hardware_concurrency();
        if (T < 1) T = 1;
        while ((int)pool.size() < T) pool.push_back(make_detector());

        std::vector<std::vector<int32_t>> all_ids(N);
        std::vector<std::vector<float>> all_corners(N);
        if (N > 0) {
            nb::gil_scoped_release rel;
            std::atomic<size_t> next{0};
            auto worker = [&](int t) {
                size_t i;
                while ((i = next.fetch_add(1)) < N) {
                    auto markers = pool[t]->detect(mats[i]);
                    fill(markers, all_ids[i], all_corners[i]);
                }
            };
            std::vector<std::thread> ths;
            for (int t = 0; t < T; t++) ths.emplace_back(worker, t);
            for (auto &x : ths) x.join();
        }
        std::vector<nb::object> out;
        out.reserve(N);
        for (size_t i = 0; i < N; i++) {
            size_t n = all_ids[i].size();
            out.push_back(nb::make_tuple(
                ids_to_numpy(std::move(all_ids[i])),
                corners_to_numpy(std::move(all_corners[i]), n)));
        }
        return out;
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
        if (dict != 0 && dict != 1)
            throw nb::value_error("dict must be 0 (mip_36h12) or 1 (apriltag_36h11)");
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
        .def("detect", &ArucoDetectorImpl::detect, nb::arg("image"))
        .def("estimate_pose", &ArucoDetectorImpl::estimate_pose,
             nb::arg("corners"), nb::arg("camera_matrix"), nb::arg("dist_coeffs"),
             nb::arg("marker_size"))
        .def("detect_batch", &ArucoDetectorImpl::detect_batch,
             nb::arg("images"), nb::arg("num_threads") = 0);

    // ---- Fractal ----
    nb::class_<FractalDetectorImpl>(m, "FractalDetector")
        .def(nb::init<std::string, float>(), nb::arg("config"),
             nb::arg("marker_size"))
        .def("detect", &FractalDetectorImpl::detect, nb::arg("image"))
        .def("detect_full", &FractalDetectorImpl::detect_full, nb::arg("image"))
        .def("detect_batch", &FractalDetectorImpl::detect_batch,
             nb::arg("images"), nb::arg("num_threads") = 0);

    m.def("_fractal_external_id", [](std::string config) {
        nanofractal::FractalMarkerSet s(config);
        return s.idExternal;
    });

    m.def("_fractal_external_image8", [](std::string config) {
        nanofractal::FractalMarkerSet s(config);
        cv::Mat M = s.fractalMarkerCollection[s.idExternal].mat();  // KxK, 0/1
        int K = M.rows;
        std::vector<uint8_t> grid((K + 2) * (K + 2), 0);  // black border
        for (int y = 0; y < K; y++)
            for (int x = 0; x < K; x++)
                grid[(y + 1) * (K + 2) + (x + 1)] = M.at<uint8_t>(y, x) ? 255 : 0;
        return make_owned<uint8_t>(std::move(grid),
                                   {(size_t)(K + 2), (size_t)(K + 2)});
    });
}
