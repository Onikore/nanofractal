#include "detector_params.h"
#include "fractal_fast.hpp"
#include "nanofractal.h"
#include "ndarray_cv.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <opencv2/core.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp> // solvePnP, projectPoints, cv::fisheye
#else
#include <opencv2/calib3d.hpp> // solvePnP, projectPoints, cv::fisheye
#endif
#include <opencv2/imgproc.hpp> // cornerSubPix, cvtColor
#include <string>
#include <thread>

namespace nb = nanobind;

// Drawing primitive: green marker outline, a red dot on the first corner, and
// the blue id text at the marker centre. Operates on the cv::Mat in place.
static void draw_outlines_ids(cv::Mat &im, const F32Arr &corners,
                              const I32Arr &ids) {
  int lw = std::max(1, im.cols / 500);
  cv::Scalar green(0, 255, 0), red(0, 0, 255), blue(255, 0, 0);
  size_t N =
      (corners.ndim() == 3 && corners.shape(1) == 4 && corners.shape(2) == 2)
          ? corners.shape(0)
          : 0;
  size_t nids = (ids.ndim() == 1) ? ids.shape(0) : 0;
  for (size_t i = 0; i < N; i++) {
    cv::Point2f c[4];
    for (int k = 0; k < 4; k++)
      c[k] = cv::Point2f(corners.data()[i * 8 + k * 2 + 0],
                         corners.data()[i * 8 + k * 2 + 1]);
    for (int k = 0; k < 4; k++)
      cv::line(im, c[k], c[(k + 1) % 4], green, lw);
    cv::circle(im, c[0], 3 * lw, red, -1); // first corner
    if (i < nids) {
      cv::Point2f cen = (c[0] + c[1] + c[2] + c[3]) * 0.25f;
      cv::putText(im, std::to_string(ids.data()[i]), cen,
                  cv::FONT_HERSHEY_SIMPLEX, std::max(0.4, im.cols / 1500.0),
                  blue, lw, cv::LINE_AA);
    }
  }
}

// ---- Fractal ----
// FastFractalDetector keeps per-frame scratch buffers, so it is NOT safe to
// call concurrently on one instance. We keep a pool of independent detectors
// (one per worker thread); detect()/detect_full() and detect_batch's thread 0
// all use pool[0] -- concurrent calls on the same Python object are serialized
// by the GIL. The fractal config is expensive to build, so it is constructed
// once per pooled detector. `params` is copied into a pooled detector before
// every call, so changes made after construction take effect on the next
// detection.
using nanofractal_fast::Detection;
using nanofractal_fast::FastFractalDetector;

static void fill(const std::vector<Detection> &markers,
                 std::vector<int32_t> &ids, std::vector<float> &corners,
                 float ox = 0.f, float oy = 0.f) {
  size_t n = markers.size();
  ids.resize(n);
  corners.resize(n * 8);
  for (size_t i = 0; i < n; i++) {
    ids[i] = markers[i].id;
    for (int c = 0; c < 4; c++) {
      corners[i * 8 + c * 2 + 0] = markers[i].corners[c].x + ox;
      corners[i * 8 + c * 2 + 1] = markers[i].corners[c].y + oy;
    }
  }
}

static nb::object dropped_ids(const FastFractalDetector &d) {
  std::vector<int32_t> v(d.inconsistent_ids().begin(),
                         d.inconsistent_ids().end());
  return nb::cast(ids_to_numpy(std::move(v)));
}

static nb::tuple full_result(std::vector<int32_t> &&ids,
                             std::vector<float> &&corners,
                             const std::vector<cv::Point2f> &p2d,
                             const std::vector<cv::Point3f> &p3d, float ox,
                             float oy, bool multiple, nb::object dropped,
                             const std::vector<int> &levels) {
  size_t n = ids.size();
  size_t m2 = p2d.size();
  std::vector<int32_t> lv(levels.begin(), levels.end());
  std::vector<float> pts2(m2 * 2), pts3(m2 * 3);
  for (size_t i = 0; i < m2; i++) {
    pts2[i * 2 + 0] = p2d[i].x + ox;
    pts2[i * 2 + 1] = p2d[i].y + oy;
    pts3[i * 3 + 0] = p3d[i].x;
    pts3[i * 3 + 1] = p3d[i].y;
    pts3[i * 3 + 2] = p3d[i].z;
  }
  return nb::make_tuple(ids_to_numpy(std::move(ids)),
                        corners_to_numpy(std::move(corners), n),
                        make_owned<float>(std::move(pts2), {m2, (size_t)2}),
                        make_owned<float>(std::move(pts3), {m2, (size_t)3}),
                        multiple, dropped, ids_to_numpy(std::move(lv)));
}

struct FractalDetectorImpl {
  std::string config;
  float marker_size;
  DetectorParams params;
  std::vector<std::unique_ptr<FastFractalDetector>> pool;

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

  std::unique_ptr<FastFractalDetector> make_detector() const {
    return std::make_unique<FastFractalDetector>(
        config, marker_size > 0 ? marker_size : -1.f);
  }

  FastFractalDetector &detector(size_t t) {
    pool[t]->params = params;
    pool[t]->serial = false;
    return *pool[t];
  }

  nb::tuple detect(RawArray arr, int rx = -1, int ry = -1, int rw = -1,
                   int rh = -1) {
    cv::Mat im = as_image(arr);
    bool has_roi = (rw > 0 && rh > 0);
    cv::Mat work = has_roi ? im(cv::Rect(rx, ry, rw, rh)) : im;
    std::vector<Detection> markers;
    FastFractalDetector &d = detector(0);
    {
      nb::gil_scoped_release rel;
      d.detect(work, markers);
    }
    std::vector<int32_t> ids;
    std::vector<float> corners;
    // Offset corners to full-image coordinates when an ROI was used.
    fill(markers, ids, corners, has_roi ? (float)rx : 0.f,
         has_roi ? (float)ry : 0.f);
    size_t n = ids.size();
    return nb::make_tuple(ids_to_numpy(std::move(ids)),
                          corners_to_numpy(std::move(corners), n),
                          d.multiple_instances(), dropped_ids(d));
  }

  // detect + why every convex quad was kept or dropped (debug aid):
  // (ids, corners, multiple, n_contours, n_not_quad, quad_corners (K,4,2),
  //  cells, reason, border_errors, bit_errors, nearest_id, contrast (K,)).
  nb::tuple diagnose(RawArray arr, int rx = -1, int ry = -1, int rw = -1,
                     int rh = -1) {
    cv::Mat im = as_image(arr);
    bool has_roi = (rw > 0 && rh > 0);
    cv::Mat work = has_roi ? im(cv::Rect(rx, ry, rw, rh)) : im;
    std::vector<Detection> markers;
    std::vector<FastFractalDetector::QuadInfo> quads;
    int n_contours = 0, n_not_quad = 0;
    FastFractalDetector &d = detector(0);
    {
      nb::gil_scoped_release rel;
      d.detect(work, markers);
      d.diagnose(quads, n_contours, n_not_quad);
    }
    const float ox = has_roi ? (float)rx : 0.f, oy = has_roi ? (float)ry : 0.f;
    std::vector<int32_t> ids;
    std::vector<float> corners;
    fill(markers, ids, corners, ox, oy);
    size_t n = ids.size(), K = quads.size();
    std::vector<float> qc(K * 8), contrast(K);
    std::vector<int32_t> cells(K), reason(K), border(K), bits(K), nearest(K);
    for (size_t i = 0; i < K; i++) {
      const auto &q = quads[i];
      for (int c = 0; c < 4; c++) {
        qc[i * 8 + c * 2 + 0] = q.c[c].x + ox;
        qc[i * 8 + c * 2 + 1] = q.c[c].y + oy;
      }
      cells[i] = q.cells;
      reason[i] = q.reason;
      border[i] = q.border_errors;
      bits[i] = q.bit_errors;
      nearest[i] = q.nearest_id;
      contrast[i] = q.contrast;
    }
    return nb::make_tuple(
        ids_to_numpy(std::move(ids)), corners_to_numpy(std::move(corners), n),
        d.multiple_instances(), dropped_ids(d), n_contours, n_not_quad,
        corners_to_numpy(std::move(qc), K), ids_to_numpy(std::move(cells)),
        ids_to_numpy(std::move(reason)), ids_to_numpy(std::move(border)),
        ids_to_numpy(std::move(bits)), ids_to_numpy(std::move(nearest)),
        make_owned<float>(std::move(contrast), {K}));
  }

  // detect + all visible (inner) corner correspondences for occlusion-robust
  // pose: returns (ids, corners, points_2d (M,2), points_3d (M,3)).
  nb::tuple detect_full(RawArray arr, int rx = -1, int ry = -1, int rw = -1,
                        int rh = -1) {
    cv::Mat im = as_image(arr);
    bool has_roi = (rw > 0 && rh > 0);
    cv::Mat work = has_roi ? im(cv::Rect(rx, ry, rw, rh)) : im;
    std::vector<Detection> markers;
    std::vector<cv::Point3f> p3d;
    std::vector<cv::Point2f> p2d;
    FastFractalDetector &d = detector(0);
    {
      nb::gil_scoped_release rel;
      d.detect(work, markers, p3d, p2d);
    }
    const float ox = has_roi ? (float)rx : 0.f, oy = has_roi ? (float)ry : 0.f;
    std::vector<int32_t> ids;
    std::vector<float> corners;
    fill(markers, ids, corners, ox, oy);
    return full_result(std::move(ids), std::move(corners), p2d, p3d, ox, oy,
                       d.multiple_instances(), dropped_ids(d),
                       d.points_level());
  }

  // Parallel batch detection. The detector is not thread-safe, so each worker
  // thread t uses its own pool[t] (built once, lazily grown to T).
  std::vector<nb::object> detect_batch(std::vector<RawArray> imgs,
                                       int num_threads, int rx = -1,
                                       int ry = -1, int rw = -1, int rh = -1) {
    bool has_roi = (rw > 0 && rh > 0);
    size_t N = imgs.size();
    std::vector<cv::Mat> mats(N);
    for (size_t i = 0; i < N; i++) {
      cv::Mat full_im = as_image(imgs[i]);
      mats[i] = has_roi ? full_im(cv::Rect(rx, ry, rw, rh)) : full_im;
    }

    int T = num_threads > 0 ? num_threads
                            : (int)std::thread::hardware_concurrency();
    if (T < 1)
      T = 1;
    if (T > (int)N)
      T = (int)N; // avoid building detectors we won't use
    while ((int)pool.size() < T)
      pool.push_back(make_detector());
    for (int t = 0; t < T; t++)
      detector(t).serial = true; // sync params; one frame per core

    float ox_ = has_roi ? (float)rx : 0.f;
    float oy_ = has_roi ? (float)ry : 0.f;
    std::vector<std::vector<int32_t>> all_ids(N);
    std::vector<std::vector<float>> all_corners(N);
    std::vector<char> all_multiple(N, 0);
    std::vector<std::vector<int32_t>> all_dropped(N);
    if (N > 0) {
      nb::gil_scoped_release rel;
      std::atomic<size_t> next{0};
      auto worker = [&](int t) {
        std::vector<Detection> markers;
        size_t i;
        while ((i = next.fetch_add(1)) < N) {
          pool[t]->detect(mats[i], markers);
          fill(markers, all_ids[i], all_corners[i], ox_, oy_);
          all_multiple[i] = pool[t]->multiple_instances();
          all_dropped[i].assign(pool[t]->inconsistent_ids().begin(),
                                pool[t]->inconsistent_ids().end());
        }
      };
      std::vector<std::thread> ths;
      for (int t = 0; t < T; t++)
        ths.emplace_back(worker, t);
      for (auto &x : ths)
        x.join();
    }
    std::vector<nb::object> out;
    out.reserve(N);
    for (size_t i = 0; i < N; i++) {
      size_t n = all_ids[i].size();
      out.push_back(nb::make_tuple(
          ids_to_numpy(std::move(all_ids[i])),
          corners_to_numpy(std::move(all_corners[i]), n), (bool)all_multiple[i],
          ids_to_numpy(std::move(all_dropped[i]))));
    }
    return out;
  }

  // Single fractal-marker pose via solvePnP (IPPE, planar). Uses the rich
  // inner+outer correspondences when available (>=4 points, method "inner");
  // otherwise the corners of every detected level with their real 3D
  // coordinates from the config (method "corners"). One outlier pass, then
  // returns (rvec(3,), tvec(3,), inlier rms px, method, n_points,
  // n_inliers), or None when no marker.
  nb::object estimate_pose(F32Arr p2d, F32Arr p3d, I32Arr ids, F32Arr corners,
                           F64Arr cam, F64Arr dist, bool fisheye = false) {
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
                  corners.shape(2) == 2)
                     ? corners.shape(0)
                     : 0;
      if (N < 1 || ids.ndim() != 1 || ids.shape(0) != N)
        return nb::none(); // no marker visible
      // Level keypoints are in metres when marker_size > 0, otherwise
      // normalized (external side 2): scale those to an external side of 1.
      const auto &coll = pool[0]->marker_set().fractalMarkerCollection;
      const float k = marker_size > 0 ? 1.f : 0.5f;
      std::vector<int> used;
      for (size_t i = 0; i < N; i++) {
        int id = ids.data()[i];
        auto it = coll.find(id);
        if (it == coll.end() ||
            std::find(used.begin(), used.end(), id) != used.end())
          continue; // unknown id or a second copy of a level
        used.push_back(id);
        for (int c = 0; c < 4; c++) {
          const cv::Point2f &q = it->second.keypts[c].pt;
          obj.emplace_back(q.x * k, q.y * k, 0.f);
          img.emplace_back(corners.data()[(i * 4 + c) * 2 + 0],
                           corners.data()[(i * 4 + c) * 2 + 1]);
        }
      }
      if (obj.empty())
        return nb::none();
    }
    // One level: IPPE_SQUARE wants a square centred on the origin, so solve
    // for the level's own frame and shift the translation back afterwards.
    cv::Point3f centre(0.f, 0.f, 0.f);
    if (!have_inner && obj.size() == 4) {
      for (const auto &q : obj)
        centre += q * 0.25f;
      for (auto &q : obj)
        q -= centre;
    }

    // IPPE_SQUARE for a single level (a square: more robust); IPPE for
    // several levels or the inner+outer points (general coplanar).
    int solver = obj.size() == 4 && !have_inner ? cv::SOLVEPNP_IPPE_SQUARE
                                                : cv::SOLVEPNP_IPPE;
    cv::Mat rvec, tvec;
    double rms = 0;
    bool ok = false;
    size_t n_inliers = obj.size();
    bool cv_threw = false;
    std::string cv_err;
    {
      nb::gil_scoped_release rel;
      try {
        // Fisheye: undistort observed points to pinhole pixels (P = K) and
        // solve with zero distortion (solvePnP has no fisheye model).
        std::vector<cv::Point2f> img_solve;
        cv::Mat dist_solve;
        if (fisheye) {
          cv::fisheye::undistortPoints(img, img_solve, camMat, distMat,
                                       cv::noArray(), camMat);
          dist_solve = cv::Mat::zeros(4, 1, CV_64F);
        } else {
          img_solve = img;
          dist_solve = distMat;
        }
        // Per-point image-space errors vs the ORIGINAL observations; returns
        // their RMS.
        auto residuals = [&](const std::vector<cv::Point3f> &o,
                             const std::vector<cv::Point2f> &im,
                             const cv::Mat &rv, const cv::Mat &tv,
                             std::vector<double> *err) {
          std::vector<cv::Point2f> proj;
          if (fisheye)
            cv::fisheye::projectPoints(cv::Mat(o).reshape(3, (int)o.size()),
                                       proj, rv, tv, camMat, distMat);
          else
            cv::projectPoints(o, rv, tv, camMat, distMat, proj);
          double se = 0;
          for (size_t i = 0; i < proj.size(); i++) {
            double dx = proj[i].x - im[i].x, dy = proj[i].y - im[i].y;
            se += dx * dx + dy * dy;
            if (err)
              err->push_back(std::sqrt(dx * dx + dy * dy));
          }
          return std::sqrt(se / (double)proj.size());
        };
        // IPPE degenerates when the marker faces the camera exactly (it
        // returns R = I for a rotation of pi): a solution worse than 1 px is
        // re-solved with SQPNP and the better pose kept.
        auto solve = [&](const std::vector<cv::Point3f> &o,
                         const std::vector<cv::Point2f> &im,
                         const std::vector<cv::Point2f> &ims, int method,
                         cv::Mat &rv, cv::Mat &tv, double &e) {
          bool good =
              cv::solvePnP(o, ims, camMat, dist_solve, rv, tv, false, method);
          if (good)
            e = residuals(o, im, rv, tv, nullptr);
          if (!good || e > 1.0) {
            cv::Mat rv2, tv2;
            if (cv::solvePnP(o, ims, camMat, dist_solve, rv2, tv2, false,
                             cv::SOLVEPNP_SQPNP)) {
              double e2 = residuals(o, im, rv2, tv2, nullptr);
              if (!good || e2 < e) {
                rv = rv2;
                tv = tv2;
                e = e2;
                good = true;
              }
            }
          }
          return good;
        };
        ok = solve(obj, img, img_solve, solver, rvec, tvec, rms);
        // One outlier pass: drop points off by more than max(1.5 px,
        // 3 x median) and solve again when enough of them remain.
        if (ok && obj.size() >= 8) {
          std::vector<double> err;
          residuals(obj, img, rvec, tvec, &err);
          std::vector<double> sorted = err;
          std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2,
                           sorted.end());
          double thr = std::max(1.5, 3.0 * sorted[sorted.size() / 2]);
          std::vector<cv::Point3f> o2;
          std::vector<cv::Point2f> im2, ims2;
          for (size_t i = 0; i < err.size(); i++)
            if (err[i] <= thr) {
              o2.push_back(obj[i]);
              im2.push_back(img[i]);
              ims2.push_back(img_solve[i]);
            }
          if (o2.size() < obj.size() && o2.size() >= 8 &&
              o2.size() * 10 >= obj.size() * 6) {
            cv::Mat rv2, tv2;
            double e2 = 0;
            if (solve(o2, im2, ims2, cv::SOLVEPNP_IPPE, rv2, tv2, e2)) {
              rvec = rv2;
              tvec = tv2;
              rms = e2;
              n_inliers = o2.size();
            }
          }
        }
        if (ok) {
          if (centre != cv::Point3f(0.f, 0.f, 0.f)) {
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            tvec -= R * cv::Mat(cv::Vec3d(centre.x, centre.y, centre.z));
          }
        }
      } catch (const cv::Exception &e) {
        cv_threw = true;
        cv_err = e.what();
      }
    }
    // Re-throw OpenCV failures as a readable Python ValueError (GIL held).
    if (cv_threw)
      throw nb::value_error(("pose estimation failed: " + cv_err).c_str());
    if (!ok)
      return nb::none();
    std::vector<double> r(3), t(3);
    for (int k = 0; k < 3; k++) {
      r[k] = rvec.at<double>(k);
      t[k] = tvec.at<double>(k);
    }
    return nb::make_tuple(make_owned<double>(std::move(r), {(size_t)3}),
                          make_owned<double>(std::move(t), {(size_t)3}), rms,
                          have_inner ? "inner" : "corners", (int)obj.size(),
                          (int)n_inliers);
  }

  // Draw detected marker outlines (+ id) and, when axes=true, the pose frame
  // axes onto `image` in place. `image` must be a writable, contiguous uint8
  // array (BGR for colour). Uses the linked OpenCV (no cv2 on the Python side).
  void draw(RawArray image, F32Arr corners, I32Arr ids, bool axes, F64Arr cam,
            F64Arr dist, F64Arr rvec, F64Arr tvec, double axis_length) {
    cv::Mat im = as_image(image);
    draw_outlines_ids(im, corners, ids);
    if (axes) {
      int lw = std::max(1, im.cols / 500);
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
  m.def("_opencv_version",
        []() { return std::string(cv::getVersionString()); });

  // Threads used inside one detect() call (OpenCV's parallel_for_ pool):
  // n >= 1 caps it, n <= 0 restores the default (all cores).
  m.def(
      "set_num_threads", [](int n) { cv::setNumThreads(n >= 1 ? n : -1); },
      nb::arg("n"));
  m.def("get_num_threads", []() { return cv::getNumThreads(); });

  m.def("_img_info", [](RawArray arr) {
    cv::Mat im = as_mat(arr);
    return nb::make_tuple(im.rows, im.cols, im.channels());
  });

  m.def("_img_data_ptr",
        [](RawArray arr) { return (uintptr_t)as_mat(arr).data; });

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
                            1.f,  2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    return corners_to_numpy(std::move(c), 2);
  });

  // ---- DetectorParams ----
  nb::class_<DetectorParams>(m, "DetectorParams")
      .def(nb::init<>())
      .def_rw("min_contour_size", &DetectorParams::min_contour_size,
              "-1 = detector default (120)")
      .def_rw("adaptive_block_size", &DetectorParams::adaptive_block_size,
              "-1 = auto-scale with image width")
      .def_rw("adaptive_c", &DetectorParams::adaptive_c,
              "constant subtracted from adaptive threshold mean (default 7)")
      .def_rw("approx_poly_rate", &DetectorParams::approx_poly_rate,
              "contour approx epsilon = perimeter * rate (default 0.05)")
      .def_rw("subpix_win_size", &DetectorParams::subpix_win_size,
              "corner sub-pixel half-window (-1 = default 4, 0 = off)")
      .def_rw("kfilter_min_dist", &DetectorParams::kfilter_min_dist,
              "min distance (px) between FAST keypoints (default 10)")
      .def_rw("detection_scale", &DetectorParams::detection_scale,
              "downscale factor for the detection stage "
              "(1.0 = full res; 0.5 ~= 4x faster; corners refined at full res)")
      .def("__repr__", [](const DetectorParams &p) {
        return "DetectorParams(min_contour_size=" +
               std::to_string(p.min_contour_size) + ", adaptive_block_size=" +
               std::to_string(p.adaptive_block_size) +
               ", adaptive_c=" + std::to_string(p.adaptive_c) +
               ", approx_poly_rate=" + std::to_string(p.approx_poly_rate) +
               ", subpix_win_size=" + std::to_string(p.subpix_win_size) +
               ", kfilter_min_dist=" + std::to_string(p.kfilter_min_dist) +
               ", detection_scale=" + std::to_string(p.detection_scale) + ")";
      });

  // ---- Fractal ----
  nb::class_<FractalDetectorImpl>(m, "FractalDetector")
      .def(nb::init<std::string, float, DetectorParams>(), nb::arg("config"),
           nb::arg("marker_size"), nb::arg("params") = DetectorParams{})
      .def_rw("params", &FractalDetectorImpl::params)
      .def("detect", &FractalDetectorImpl::detect, nb::arg("image"),
           nb::arg("rx") = -1, nb::arg("ry") = -1, nb::arg("rw") = -1,
           nb::arg("rh") = -1)
      .def("detect_full", &FractalDetectorImpl::detect_full, nb::arg("image"),
           nb::arg("rx") = -1, nb::arg("ry") = -1, nb::arg("rw") = -1,
           nb::arg("rh") = -1)
      .def("diagnose", &FractalDetectorImpl::diagnose, nb::arg("image"),
           nb::arg("rx") = -1, nb::arg("ry") = -1, nb::arg("rw") = -1,
           nb::arg("rh") = -1)
      .def("detect_batch", &FractalDetectorImpl::detect_batch,
           nb::arg("images"), nb::arg("num_threads") = 0, nb::arg("rx") = -1,
           nb::arg("ry") = -1, nb::arg("rw") = -1, nb::arg("rh") = -1)
      .def("estimate_pose", &FractalDetectorImpl::estimate_pose,
           nb::arg("points_2d"), nb::arg("points_3d"), nb::arg("ids"),
           nb::arg("corners"), nb::arg("camera_matrix"), nb::arg("dist_coeffs"),
           nb::arg("fisheye") = false)
      .def("draw", &FractalDetectorImpl::draw, nb::arg("image"),
           nb::arg("corners"), nb::arg("ids"), nb::arg("axes"), nb::arg("cam"),
           nb::arg("dist"), nb::arg("rvec"), nb::arg("tvec"),
           nb::arg("axis_length"));

  // ---- B2: sub-pixel corner refinement ----
  // Refines detected corners to sub-pixel accuracy via cv::cornerSubPix.
  // Accepts grayscale (H,W) or BGR (H,W,3) uint8 images; converts to gray
  // internally. corners must be float32 (N,4,2); returns float32 (N,4,2).
  m.def("_refine_corners", [](RawArray arr, F32Arr corners, int win_size) {
    cv::Mat im = as_mat(arr);
    // Convert to single-channel; cornerSubPix requires grayscale.
    cv::Mat gray;
    if (im.channels() == 3)
      cv::cvtColor(im, gray, cv::COLOR_BGR2GRAY);
    else
      gray = im;
    if (corners.ndim() != 3 || corners.shape(1) != 4 || corners.shape(2) != 2)
      throw nb::value_error("corners must be float32 (N, 4, 2)");
    size_t N = corners.shape(0);
    std::vector<cv::Point2f> pts;
    pts.reserve(N * 4);
    for (size_t i = 0; i < N * 4; i++)
      pts.emplace_back(corners.data()[i * 2 + 0], corners.data()[i * 2 + 1]);
    if (!pts.empty()) {
      cv::TermCriteria crit(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                            30, 0.01);
      cv::cornerSubPix(gray, pts, cv::Size(win_size, win_size),
                       cv::Size(-1, -1), crit);
    }
    std::vector<float> out(N * 8);
    for (size_t i = 0; i < N * 4; i++) {
      out[i * 2 + 0] = pts[i].x;
      out[i * 2 + 1] = pts[i].y;
    }
    return make_owned<float>(std::move(out), {N, (size_t)4, (size_t)2});
  });

  // The upstream reference detector (third_party/nanofractal.h), used by the
  // test-suite to check that the optimised pipeline returns identical output.
  m.def(
      "_reference_detect",
      [](std::string config, float marker_size, DetectorParams p, RawArray arr,
         bool inner) {
        cv::Mat im = as_image(arr);
        nanofractal::FractalMarkerDetector d;
        d.setParams(config, marker_size > 0 ? marker_size : -1.f);
        d.params = p;
        std::vector<nanofractal::FractalMarker> markers;
        std::vector<cv::Point3f> p3d;
        std::vector<cv::Point2f> p2d;
        {
          nb::gil_scoped_release rel;
          markers = inner ? d.detect(im, p3d, p2d) : d.detect(im);
        }
        std::vector<Detection> det(markers.size());
        for (size_t i = 0; i < markers.size(); i++) {
          det[i].id = markers[i].id;
          for (int c = 0; c < 4; c++)
            det[i].corners[c] = markers[i][c];
        }
        std::vector<int32_t> ids;
        std::vector<float> corners;
        fill(det, ids, corners);
        return full_result(std::move(ids), std::move(corners), p2d, p3d, 0.f,
                           0.f, false, nb::cast(ids_to_numpy({})),
                           std::vector<int>(p2d.size(), -1));
      },
      nb::arg("config"), nb::arg("marker_size"), nb::arg("params"),
      nb::arg("image"), nb::arg("inner"));

  // Test hook: AreaDownscale vs cv::resize(INTER_AREA) on a grayscale image;
  // returns (max |difference|, number of differing pixels, same size).
  m.def("_area_downscale_diff", [](RawArray arr, double f, int threads) {
    cv::Mat im = as_mat(arr);
    if (im.channels() != 1)
      throw nb::value_error("expected a single-channel image");
    cv::Mat mine, ref;
    nanofractal_fast::AreaDownscale a;
    a.max_threads = threads;
    a.run(im, f, mine);
    cv::resize(im, ref, cv::Size(), f, f, cv::INTER_AREA);
    if (mine.size() != ref.size())
      return nb::make_tuple(255.0, -1, false);
    cv::Mat d;
    cv::absdiff(mine, ref, d);
    double mx = 0;
    cv::minMaxLoc(d, nullptr, &mx);
    return nb::make_tuple(mx, cv::countNonZero(d), true);
  });

  // Test hook: AdaptiveThreshold (and its neighbour masks) against
  // cv::adaptiveThreshold(MEAN_C, BINARY_INV) and neighbour_masks(); returns
  // (pixels that differ, mask entries that differ).
  m.def("_threshold_diff", [](RawArray arr, int k, double delta, int threads) {
    cv::Mat im = as_mat(arr);
    if (im.channels() != 1)
      throw nb::value_error("expected a single-channel image");
    nanofractal_fast::AdaptiveThreshold t;
    t.max_threads = threads;
    cv::Mat mine, nbm, ref, nbm_ref;
    t.run(im, k, delta, mine, &nbm);
    cv::adaptiveThreshold(im, ref, 1, cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY_INV, k, delta);
    cv::Mat inner = mine(cv::Rect(1, 1, im.cols, im.rows)), mine8;
    inner.convertTo(mine8, CV_8U);
    int bad = cv::countNonZero(mine8 != ref);
    nanofractal_fast::neighbour_masks(mine, nbm_ref);
    return nb::make_tuple(bad, cv::countNonZero(nbm != nbm_ref));
  });

  // Test hook: trace a 0/1 uint8 image with the parallel band tracer
  // (band_rows rows per band, all threads) and report whether the contours of
  // length >= min_size equal sequential cv::findContours(RETR_LIST, NONE).
  m.def("_tracer_matches_findcontours", [](RawArray arr, int min_size,
                                           int band_rows) {
    cv::Mat im = as_mat(arr);
    if (im.channels() != 1)
      throw nb::value_error("expected a single-channel 0/1 image");
    std::vector<std::vector<cv::Point>> ref, refk, mine;
    bool same;
    {
      nb::gil_scoped_release rel;
      const int threads = cv::getNumThreads();
      cv::setNumThreads(1); // OpenCV 5.0's parallel findContours is the suspect
      cv::Mat r = im != 0;
      cv::findContours(r, ref, cv::noArray(), cv::RETR_LIST,
                       cv::CHAIN_APPROX_NONE);
      cv::setNumThreads(threads);
      for (auto &c : ref)
        if ((int)c.size() >= min_size)
          refk.push_back(c);
      cv::Mat bin(im.rows + 2, im.cols + 2, CV_8SC1, cv::Scalar(0));
      cv::Mat(im != 0).convertTo(bin(cv::Rect(1, 1, im.cols, im.rows)), CV_8S,
                                 1.0 / 255);
      nanofractal_fast::ContourTracer tr;
      tr.min_band_rows = std::max(1, band_rows);
      tr.run(bin, min_size,
             [&](const cv::Point *p, int n) { mine.emplace_back(p, p + n); });
      same = mine == refk;
    }
    return nb::make_tuple(same, (int)refk.size());
  });

  m.def("_fractal_external_id", [](std::string config) {
    nanofractal::FractalMarkerSet s(config);
    return s.idExternal;
  });

  // Every marker level of a config as a list of (id, corners float32 (4,2) in
  // normalized marker coords [-1,1] with y up, bits uint8 (K,K) 0/1). Used to
  // render the full nested composite (generate_fractal) and by benchmarks.
  m.def("_fractal_markers", [](std::string config) {
    nanofractal::FractalMarkerSet s(config);
    nb::list out;
    for (auto &kv : s.fractalMarkerCollection) {
      const nanofractal::FractalMarker &fm = kv.second;
      std::vector<float> c(8);
      for (int k = 0; k < 4; k++) {
        c[k * 2 + 0] = fm.keypts[k].pt.x;
        c[k * 2 + 1] = fm.keypts[k].pt.y;
      }
      cv::Mat M = fm.mat();
      size_t K = (size_t)M.rows;
      std::vector<uint8_t> bits(M.data, M.data + K * K);
      out.append(nb::make_tuple(
          fm.id, make_owned<float>(std::move(c), {(size_t)4, (size_t)2}),
          make_owned<uint8_t>(std::move(bits), {K, K})));
    }
    return out;
  });

  m.def("_fractal_external_image8", [](std::string config) {
    nanofractal::FractalMarkerSet s(config);
    cv::Mat M = s.fractalMarkerCollection[s.idExternal].mat(); // KxK, 0/1
    int K = M.rows;
    std::vector<uint8_t> grid((K + 2) * (K + 2), 0); // black border
    for (int y = 0; y < K; y++)
      for (int x = 0; x < K; x++)
        grid[(y + 1) * (K + 2) + (x + 1)] = M.at<uint8_t>(y, x) ? 255 : 0;
    return make_owned<uint8_t>(std::move(grid),
                               {(size_t)(K + 2), (size_t)(K + 2)});
  });
}
