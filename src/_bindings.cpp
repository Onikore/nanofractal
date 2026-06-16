#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>  // FastFeatureDetector, used by nanofractal.h
#include <memory>
#include <thread>
#include <atomic>
#include <cmath>
#include <string>
#include "ndarray_cv.hpp"
#include "detector_params.h"
#include "aruco_nano_v6.h"
#include "aruco_dicts.hpp"
#include "aruco_std_dicts.hpp"
#include "nanofractal.h"

namespace nb = nanobind;

// ---- ArUco Nano v6 ----
// Defined at namespace scope (not inside NB_MODULE) so it is a non-local type,
// which is the conventional, portable way to use it as a nb::class_ template arg.
struct ArucoDetectorImpl {
    int dict;
    unsigned max_attempts;
    DetectorParams params;

    ArucoDetectorImpl(int dictionary, unsigned attempts, DetectorParams p = {})
        : params(p) {
        if (dictionary < 0 || dictionary > 17)
            throw nb::value_error("dictionary must be a valid Dict enum value (0–17)");
        dict = dictionary;
        max_attempts = attempts ? attempts : 1u;
    }

    nb::tuple detect(RawArray arr) {
        cv::Mat im = as_image(arr);
        std::vector<aruconano::Marker> markers;
        {
            nb::gil_scoped_release rel;
            markers = aruconano::MarkerDetector::detect(
                im, max_attempts, (aruconano::MarkerDetector::Dict)dict, params);
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
        for (size_t i = 0; i < N; i++) mats[i] = as_image(imgs[i]);

        std::vector<std::vector<int32_t>> all_ids(N);
        std::vector<std::vector<float>> all_corners(N);
        int T = num_threads > 0 ? num_threads
                                : (int)std::thread::hardware_concurrency();
        if (T < 1) T = 1;
        if (T > (int)N) T = (int)N;  // no point spawning more workers than images
        int dict_ = dict;
        unsigned attempts_ = max_attempts;
        DetectorParams params_ = params;
        if (N > 0) {
            nb::gil_scoped_release rel;
            std::atomic<size_t> next{0};
            auto worker = [&]() {
                size_t i;
                while ((i = next.fetch_add(1)) < N) {
                    auto markers = aruconano::MarkerDetector::detect(
                        mats[i], attempts_, (aruconano::MarkerDetector::Dict)dict_, params_);
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
// (one per worker thread); detect()/detect_full() and detect_batch's thread 0 all
// use pool[0] -- concurrent calls on the same Python object are serialized by the
// GIL. The fractal config is expensive to build, so it is constructed once per
// pooled detector.
struct FractalDetectorImpl {
    std::string config;
    float marker_size;
    DetectorParams params;
    std::vector<std::unique_ptr<nanofractal::FractalMarkerDetector>> pool;

    FractalDetectorImpl(std::string cfg, float msize, DetectorParams p = {})
        : config(std::move(cfg)), marker_size(msize), params(p) {
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
        d->params = params;
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
        cv::Mat im = as_image(arr);
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
        cv::Mat im = as_image(arr);
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
        for (size_t i = 0; i < N; i++) mats[i] = as_image(imgs[i]);

        int T = num_threads > 0 ? num_threads
                                : (int)std::thread::hardware_concurrency();
        if (T < 1) T = 1;
        if (T > (int)N) T = (int)N;  // avoid building detectors we won't use
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

    // Single fractal-marker pose via solvePnP (IPPE, planar). Uses the rich
    // inner+outer correspondences when available (>=4 points); otherwise falls
    // back to the 4 outer corners with a square of side marker_size. Returns
    // (rvec(3,), tvec(3,), rms_reproj_px) float64, or None when no marker / <4 pts.
    nb::object estimate_pose(F32Arr p2d, F32Arr p3d, F32Arr corners,
                             F64Arr cam, F64Arr dist) {
        if (cam.ndim() != 2 || cam.shape(0) != 3 || cam.shape(1) != 3)
            throw nb::value_error("camera_matrix must be float64 (3,3)");
        if (dist.ndim() != 1)
            throw nb::value_error("dist_coeffs must be 1-D float64");
        cv::Mat camMat(3, 3, CV_64F, const_cast<double *>(cam.data()));
        cv::Mat distMat((int)dist.shape(0), 1, CV_64F,
                        const_cast<double *>(dist.data()));

        std::vector<cv::Point3f> obj;
        std::vector<cv::Point2f> img;
        size_t M = (p2d.ndim() == 2) ? p2d.shape(0) : 0;
        bool have_inner = (M >= 4 && p3d.ndim() == 2 && p3d.shape(0) == M &&
                           p3d.shape(1) == 3 && p2d.shape(1) == 2);
        if (have_inner) {
            obj.reserve(M);
            img.reserve(M);
            for (size_t i = 0; i < M; i++) {
                obj.emplace_back(p3d.data()[i * 3 + 0], p3d.data()[i * 3 + 1],
                                 p3d.data()[i * 3 + 2]);
                img.emplace_back(p2d.data()[i * 2 + 0], p2d.data()[i * 2 + 1]);
            }
        } else {
            size_t N = (corners.ndim() == 3 && corners.shape(1) == 4 &&
                        corners.shape(2) == 2) ? corners.shape(0) : 0;
            if (N < 1) return nb::none();  // no marker visible
            float h = (marker_size > 0 ? marker_size : 1.0f) / 2.f;
            obj = {{-h, h, 0.f}, {h, h, 0.f}, {h, -h, 0.f}, {-h, -h, 0.f}};
            for (int c = 0; c < 4; c++)  // outer corners of the first marker
                img.emplace_back(corners.data()[c * 2 + 0], corners.data()[c * 2 + 1]);
        }

        cv::Mat rvec, tvec;
        double rms = 0;
        bool ok;
        {
            nb::gil_scoped_release rel;
            ok = cv::solvePnP(obj, img, camMat, distMat, rvec, tvec, false,
                              cv::SOLVEPNP_IPPE);
            if (ok) {
                std::vector<cv::Point2f> proj;
                cv::projectPoints(obj, rvec, tvec, camMat, distMat, proj);
                double se = 0;
                for (size_t i = 0; i < proj.size(); i++) {
                    double dx = proj[i].x - img[i].x, dy = proj[i].y - img[i].y;
                    se += dx * dx + dy * dy;
                }
                rms = std::sqrt(se / (double)proj.size());
            }
        }
        if (!ok) return nb::none();
        std::vector<double> r(3), t(3);
        for (int k = 0; k < 3; k++) {
            r[k] = rvec.at<double>(k);
            t[k] = tvec.at<double>(k);
        }
        return nb::make_tuple(make_owned<double>(std::move(r), {(size_t)3}),
                              make_owned<double>(std::move(t), {(size_t)3}), rms);
    }

    // Draw detected marker outlines (+ id) and, when axes=true, the pose frame
    // axes onto `image` in place. `image` must be a writable, contiguous uint8
    // array (BGR for colour). Uses the linked OpenCV (no cv2 on the Python side).
    void draw(RawArray image, F32Arr corners, I32Arr ids, bool axes,
              F64Arr cam, F64Arr dist, F64Arr rvec, F64Arr tvec,
              double axis_length) {
        cv::Mat im = as_image(image);
        int lw = std::max(1, im.cols / 500);
        cv::Scalar green(0, 255, 0), red(0, 0, 255), blue(255, 0, 0);
        size_t N = (corners.ndim() == 3 && corners.shape(1) == 4 &&
                    corners.shape(2) == 2) ? corners.shape(0) : 0;
        size_t nids = (ids.ndim() == 1) ? ids.shape(0) : 0;
        for (size_t i = 0; i < N; i++) {
            cv::Point2f c[4];
            for (int k = 0; k < 4; k++)
                c[k] = cv::Point2f(corners.data()[i * 8 + k * 2 + 0],
                                   corners.data()[i * 8 + k * 2 + 1]);
            for (int k = 0; k < 4; k++)
                cv::line(im, c[k], c[(k + 1) % 4], green, lw);
            cv::circle(im, c[0], 3 * lw, red, -1);  // first corner
            if (i < nids) {
                cv::Point2f cen = (c[0] + c[1] + c[2] + c[3]) * 0.25f;
                cv::putText(im, std::to_string(ids.data()[i]), cen,
                            cv::FONT_HERSHEY_SIMPLEX,
                            std::max(0.4, im.cols / 1500.0), blue, lw, cv::LINE_AA);
            }
        }
        if (axes) {
            cv::Mat camMat(3, 3, CV_64F, const_cast<double *>(cam.data()));
            cv::Mat distMat((int)dist.shape(0), 1, CV_64F,
                            const_cast<double *>(dist.data()));
            cv::Mat rv(3, 1, CV_64F, const_cast<double *>(rvec.data()));
            cv::Mat tv(3, 1, CV_64F, const_cast<double *>(tvec.data()));
            cv::drawFrameAxes(im, camMat, distMat, rv, tv, (float)axis_length, lw);
        }
    }
};

NB_MODULE(_nanofractal, m) {
    // Silence nanobind's shutdown "leaked instances" warning. It is a benign
    // cosmetic diagnostic: it fires whenever user code still holds a detector
    // reference at interpreter teardown (e.g. a VideoCapture worker thread left
    // running after Ctrl+C, or an interactive shell pinning the frame via
    // sys.last_traceback). Nothing actually leaks during operation and the OS
    // reclaims the memory at exit; the bindings hold no Python references. The
    // library cannot release a reference owned by user code, so the only
    // library-side lever is to disable the warning itself.
    nb::set_leak_warnings(false);

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

    // Render a marker as a (gsize x gsize) uint8 grid: black border + inner bits.
    // Bit ordering matches touulong() in aruco_nano_v6.h. Test/tool use only.
    auto get_dict_codes = [](int dict) -> const std::vector<uint64_t>* {
        using D = aruconano::MarkerDetector::Dict;
        switch ((D)dict) {
        case D::ARUCO_MIP_36h12:  return &aruco_dicts::mip_36h12();
        case D::APRILTAG_36h11:   return &aruco_dicts::apriltag_36h11();
        case D::DICT_4X4_50:      return &aruco_std_dicts::DICT_4X4_50();
        case D::DICT_4X4_100:     return &aruco_std_dicts::DICT_4X4_100();
        case D::DICT_4X4_250:     return &aruco_std_dicts::DICT_4X4_250();
        case D::DICT_4X4_1000:    return &aruco_std_dicts::DICT_4X4_1000();
        case D::DICT_5X5_50:      return &aruco_std_dicts::DICT_5X5_50();
        case D::DICT_5X5_100:     return &aruco_std_dicts::DICT_5X5_100();
        case D::DICT_5X5_250:     return &aruco_std_dicts::DICT_5X5_250();
        case D::DICT_5X5_1000:    return &aruco_std_dicts::DICT_5X5_1000();
        case D::DICT_6X6_50:      return &aruco_std_dicts::DICT_6X6_50();
        case D::DICT_6X6_100:     return &aruco_std_dicts::DICT_6X6_100();
        case D::DICT_6X6_250:     return &aruco_std_dicts::DICT_6X6_250();
        case D::DICT_6X6_1000:    return &aruco_std_dicts::DICT_6X6_1000();
        case D::DICT_7X7_50:      return &aruco_std_dicts::DICT_7X7_50();
        case D::DICT_7X7_100:     return &aruco_std_dicts::DICT_7X7_100();
        case D::DICT_7X7_250:     return &aruco_std_dicts::DICT_7X7_250();
        case D::DICT_7X7_1000:    return &aruco_std_dicts::DICT_7X7_1000();
        }
        return nullptr;
    };
    m.def("_aruco_marker_image8", [get_dict_codes](int dict, int id) {
        const std::vector<uint64_t>* codes = get_dict_codes(dict);
        if (!codes)
            throw nb::value_error("unknown dict id");
        if (id < 0 || (size_t)id >= codes->size())
            throw nb::value_error("marker id out of range");
        uint64_t code = (*codes)[id];
        int gsize = aruconano::MarkerDetector::dictGridSize(
            (aruconano::MarkerDetector::Dict)dict);
        int n_bits = gsize - 2;
        std::vector<uint8_t> grid((size_t)(gsize * gsize), 0);
        int b = 0;
        for (int y = n_bits - 1; y >= 0; y--)
            for (int x = n_bits - 1; x >= 0; x--) {
                grid[(y + 1) * gsize + (x + 1)] = ((code >> b++) & 1ULL) ? 255 : 0;
            }
        return make_owned<uint8_t>(std::move(grid),
                                   {(size_t)gsize, (size_t)gsize});
    });

    // ---- DetectorParams ----
    nb::class_<DetectorParams>(m, "DetectorParams")
        .def(nb::init<>())
        .def_rw("min_contour_size",    &DetectorParams::min_contour_size,
                "-1 = detector default (ArUco: 50, Fractal: 120)")
        .def_rw("adaptive_block_size", &DetectorParams::adaptive_block_size,
                "-1 = auto-scale with image width")
        .def_rw("adaptive_c",          &DetectorParams::adaptive_c,
                "constant subtracted from adaptive threshold mean (default 7)")
        .def_rw("approx_poly_rate",    &DetectorParams::approx_poly_rate,
                "contour approx epsilon = perimeter * rate (default 0.05)")
        .def_rw("subpix_win_size",     &DetectorParams::subpix_win_size,
                "corner sub-pixel half-window, Fractal only (-1 = default 4, 0 = off)")
        .def_rw("kfilter_min_dist",    &DetectorParams::kfilter_min_dist,
                "min distance (px) between FAST keypoints, Fractal only (default 10)")
        .def("__repr__", [](const DetectorParams &p) {
            return "DetectorParams(min_contour_size=" + std::to_string(p.min_contour_size) +
                   ", adaptive_block_size=" + std::to_string(p.adaptive_block_size) +
                   ", adaptive_c=" + std::to_string(p.adaptive_c) +
                   ", approx_poly_rate=" + std::to_string(p.approx_poly_rate) +
                   ", subpix_win_size=" + std::to_string(p.subpix_win_size) +
                   ", kfilter_min_dist=" + std::to_string(p.kfilter_min_dist) + ")";
        });

    // ---- ArUco Nano v6 ----
    nb::class_<ArucoDetectorImpl>(m, "ArucoDetector")
        .def(nb::init<int, unsigned, DetectorParams>(), nb::arg("dictionary"),
             nb::arg("max_attempts"), nb::arg("params") = DetectorParams{})
        .def_ro("max_attempts", &ArucoDetectorImpl::max_attempts)
        .def_ro("dictionary", &ArucoDetectorImpl::dict)
        .def_rw("params", &ArucoDetectorImpl::params)
        .def("detect", &ArucoDetectorImpl::detect, nb::arg("image"))
        .def("estimate_pose", &ArucoDetectorImpl::estimate_pose,
             nb::arg("corners"), nb::arg("camera_matrix"), nb::arg("dist_coeffs"),
             nb::arg("marker_size"))
        .def("detect_batch", &ArucoDetectorImpl::detect_batch,
             nb::arg("images"), nb::arg("num_threads") = 0);

    // ---- Fractal ----
    nb::class_<FractalDetectorImpl>(m, "FractalDetector")
        .def(nb::init<std::string, float, DetectorParams>(), nb::arg("config"),
             nb::arg("marker_size"), nb::arg("params") = DetectorParams{})
        .def_rw("params", &FractalDetectorImpl::params)
        .def("detect", &FractalDetectorImpl::detect, nb::arg("image"))
        .def("detect_full", &FractalDetectorImpl::detect_full, nb::arg("image"))
        .def("detect_batch", &FractalDetectorImpl::detect_batch,
             nb::arg("images"), nb::arg("num_threads") = 0)
        .def("estimate_pose", &FractalDetectorImpl::estimate_pose,
             nb::arg("points_2d"), nb::arg("points_3d"), nb::arg("corners"),
             nb::arg("camera_matrix"), nb::arg("dist_coeffs"))
        .def("draw", &FractalDetectorImpl::draw, nb::arg("image"),
             nb::arg("corners"), nb::arg("ids"), nb::arg("axes"), nb::arg("cam"),
             nb::arg("dist"), nb::arg("rvec"), nb::arg("tvec"),
             nb::arg("axis_length"));

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
