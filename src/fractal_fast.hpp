// Optimised fractal-marker detection pipeline.
//
// Produces the same detections as the reference nanofractal::FractalMarkerDetector
// (third_party/nanofractal.h) while replacing its hot path:
//
//   * adaptive threshold: a fused box-sum + compare that reproduces
//     cv::adaptiveThreshold(MEAN_C, THRESH_BINARY_INV) bit for bit (including
//     OpenCV's fixed-point / float rounding of the box mean) and writes straight
//     into a zero-bordered buffer, so contour tracing needs no extra copy;
//   * contour tracing: Suzuki-Abe border following identical to cv::findContours
//     (RETR_LIST, CHAIN_APPROX_NONE), but contours shorter than the minimum size
//     are traced in place and never allocated;
//   * no per-frame heap traffic: all scratch buffers live in the detector and are
//     reused across frames, and detected markers are returned as (id, 4 corners)
//     instead of copies of the full FractalMarker.
//
// The inner-point path additionally restricts FAST to the detected markers'
// bounding box and replaces the O(n^2) / kd-tree neighbour searches with grids.
#pragma once

#include "detector_params.h"
#include "nanofractal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#if defined(__SSE2__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace nanofractal_fast {

struct Detection {
  int id;
  cv::Point2f corners[4];
};

// ---------------------------------------------------------------------------
// Adaptive threshold (MEAN_C, BINARY_INV) into a zero-bordered 0/1 buffer.
// ---------------------------------------------------------------------------
class AdaptiveThreshold {
public:
  // dst: (H+2) x (W+2) int8 buffer; interior pixel (x,y) lands at (x+1,y+1).
  // Foreground (dark relative to the local mean) = 1, background and border = 0.
  void run(const cv::Mat &src, int k, double delta, cv::Mat &dst) {
    CV_Assert(src.type() == CV_8UC1 && k % 2 == 1 && k > 1);
    dst.create(src.rows + 2, src.cols + 2, CV_8SC1);
    std::memset(dst.ptr(0), 0, dst.cols);
    std::memset(dst.ptr(src.rows + 1), 0, dst.cols);
    if (k > 257) { // column sums would overflow uint16: defer to OpenCV
      cv::Mat t;
      cv::adaptiveThreshold(src, t, 1, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, k, delta);
      for (int y = 0; y < src.rows; y++) {
        int8_t *o = dst.ptr<int8_t>(y + 1);
        o[0] = o[src.cols + 1] = 0;
        std::memcpy(o + 1, t.ptr(y), src.cols);
      }
      return;
    }
    // window sums fit uint16 up to 16x16 windows (16*16*255 = 65280)
    if (k * k * 255 <= 65535) run_t<uint16_t>(src, k, delta, dst);
    else run_t<uint32_t>(src, k, delta, dst);
  }

private:
  template <typename S> void run_t(const cv::Mat &src, int k, double delta, cv::Mat &dst) {
    const int W = src.cols, H = src.rows, r = k / 2;
    const int idelta = cvFloor(delta);
    // OpenCV's box filter: windows of <= 256 px sum in uint16 and divide in fixed
    // point; larger windows sum in int32 and round s*scale in float (8-lane SIMD
    // groups) or double (the last W % 8 pixels of a row).
    const bool fixed = k * k <= 256;
    uint32_t dd = 0, ds = 1;
    if (fixed) {
      const int d = k * k;
      double scalef = double(1 << 23) / d;
      int dsi = cvFloor(scalef);
      scalef -= dsi;
      int ddi = d / 2;
      if (scalef < 0.5) ddi++;
      else dsi++;
      dd = (uint32_t)ddi;
      ds = (uint32_t)dsi;
    }
    const double scale = 1.0 / (double(k) * k);
    const float scalef32 = (float)scale;
    const int simd_end = W - W % 8;

    const int L = W + 2 * r;
    col_.assign(L, 0);
    for (auto &v : pw_) v.resize(L);  // k <= 257 -> at most 9 levels
    acc_.resize(L);
    mask_.resize(W);
    uint16_t *col = col_.data() + r; // col[-r .. W+r-1], replicate-extended

    for (int dy = -r; dy <= r; dy++) {
      const uint8_t *s = src.ptr<uint8_t>(std::min(std::max(dy, 0), H - 1));
      for (int x = 0; x < W; x++) col[x] = (uint16_t)(col[x] + s[x]);
    }
    for (int y = 0; y < H; y++) {
      if (y > 0) {
        const uint8_t *add = src.ptr<uint8_t>(std::min(y + r, H - 1));
        const uint8_t *sub = src.ptr<uint8_t>(std::max(y - r - 1, 0));
        for (int x = 0; x < W; x++) col[x] = (uint16_t)(col[x] + add[x] - sub[x]);
      }
      for (int i = 1; i <= r; i++) {
        col[-i] = col[0];
        col[W - 1 + i] = col[W - 1];
      }
      const S *s = box_row<S>(col - r, W, k);
      const uint8_t *px = src.ptr<uint8_t>(y);
      int8_t *o = dst.ptr<int8_t>(y + 1) + 1;
      o[-1] = 0;
      o[W] = 0;
      if (fixed) {
        for (int x = 0; x < W; x++) {
          const int mean = (int)(((uint32_t)s[x] + dd) * ds >> 23);
          o[x] = (int8_t)(mean >= px[x] + idelta);
        }
      } else {
        // mean = round_half_even(s * scale) >= T  <=>  p > T - 0.5, or a tie
        // (p == T - 0.5) that rounds up to an even T. Branch-free, vectorisable.
        // (int32 mask first, then narrow: GCC will not vectorise a float
        // compare stored straight to int8, nor an unsigned->float convert)
        int32_t *m = mask_.data();
        for (int x = 0; x < simd_end; x++) {
          const float p = (float)(int32_t)s[x] * scalef32;
          const int32_t T = (int32_t)px[x] + idelta;
          const float t = (float)T - 0.5f;
          const int32_t gt = p > t ? 1 : 0;
          const int32_t eq = p == t ? 1 : 0;
          m[x] = gt | (eq & ~T & 1);
        }
        for (int x = 0; x < simd_end; x++) o[x] = (int8_t)m[x];
        for (int x = simd_end; x < W; x++) {
          const int mean = cv::saturate_cast<uchar>(s[x] * scale);
          o[x] = (int8_t)(mean >= px[x] + idelta);
        }
      }
    }
  }

  // Horizontal window sums out[x] = sum_{i<k} c[x+i], x in [0, W). Builds the
  // power-of-two window sums c, c2, c4, ... (one vectorisable pass each) and adds
  // the ones selected by the bits of k in a single fused pass.
  template <typename S> const S *box_row(const uint16_t *c, int W, int k) {
    S *pw[9];
    int nlev = 0;
    for (int w = 1; w <= k; w *= 2) nlev++;
    for (int i = 0; i < nlev; i++) pw[i] = reinterpret_cast<S *>(pw_[i].data());
    const int L = W + k - 1;
    for (int i = 0; i < L; i++) pw[0][i] = (S)c[i];
    int len = L;
    for (int lev = 1, w = 1; lev < nlev; lev++, w *= 2) {
      len -= w;
      const S *a = pw[lev - 1];
      S *b = pw[lev];
      for (int i = 0; i < len; i++) b[i] = (S)(a[i] + a[i + w]);
    }
    // select terms: k = sum of 2^b; term for bit b starts at offset of the
    // lower selected bits.
    const S *t[9];
    int nt = 0, off = 0;
    for (int b = 0; b < nlev; b++)
      if (k >> b & 1) {
        t[nt++] = pw[b] + off;
        off += 1 << b;
      }
    S *out = reinterpret_cast<S *>(acc_.data());
    switch (nt) {
    case 1:
      return t[0];
    case 2:
      for (int x = 0; x < W; x++) out[x] = (S)(t[0][x] + t[1][x]);
      break;
    case 3:
      for (int x = 0; x < W; x++) out[x] = (S)(t[0][x] + t[1][x] + t[2][x]);
      break;
    case 4:
      for (int x = 0; x < W; x++) out[x] = (S)(t[0][x] + t[1][x] + t[2][x] + t[3][x]);
      break;
    default:
      for (int x = 0; x < W; x++) out[x] = (S)(t[0][x] + t[1][x] + t[2][x] + t[3][x]);
      for (int j = 4; j < nt; j++) {
        const S *tj = t[j];
        for (int x = 0; x < W; x++) out[x] = (S)(out[x] + tj[x]);
      }
    }
    return out;
  }

  std::vector<uint16_t> col_;
  std::vector<int32_t> mask_;
  std::vector<uint32_t> pw_[9], acc_; // sized in uint32 units, reused as S arrays
};

// ---------------------------------------------------------------------------
// Suzuki-Abe border following, identical to cv::findContours(RETR_LIST,
// CHAIN_APPROX_NONE) on a zero-bordered 0/1 int8 image (modified in place).
// Calls on_contour(points, n) for every contour with n >= min_size, in the
// order cv::findContours would return them.
// ---------------------------------------------------------------------------
class ContourTracer {
public:
  template <typename F> void run(cv::Mat &img, int min_size, F &&on_contour) {
    CV_Assert(img.isContinuous());
    const int step = (int)img.step;
    const int W = img.cols, H = img.rows; // padded size
    int deltas[16] = {1, -step + 1, -step, -step - 1, -1, step - 1, step, step + 1};
    std::memcpy(deltas + 8, deltas, 8 * sizeof(int));

    kept_.clear();
    pts_.clear();
    const int end = W - 1; // scan x in [1, W-1)
    for (int y = 1; y < H - 1; y++) {
      int8_t *row = img.ptr<int8_t>(y);
      // A contour starts at x when (left, pixel) = (0, 1) -- outer border -- or
      // (>= 1, 0) -- hole border, left pixel foreground and not yet marked as a
      // right boundary. Tracing only relabels pixels at or right of x on this
      // row, so evaluating the condition on the live row is exactly the
      // sequential raster scan of cv::findContours, just many pixels at a time.
      int x = 1;
      while ((x = next_start(row, x, end)) < end) {
        const int is_hole = row[x] == 0;
        const size_t start = pts_.size();
        trace(row + x - is_hole, deltas, cv::Point(x - is_hole - 1, y - 1), is_hole != 0);
        const size_t n = pts_.size() - start;
        if ((int)n >= min_size) kept_.push_back({start, n});
        else pts_.resize(start);
        x++;
      }
    }
    // cv::findContours(RETR_LIST) lists contours in reverse discovery order
    for (size_t i = kept_.size(); i-- > 0;)
      on_contour(pts_.data() + kept_[i].first, (int)kept_[i].second);
  }

private:
  static inline bool is_start(int8_t l, int8_t p) { return (l == 0 && p == 1) || (p == 0 && l > 0); }

  // first x in [x, end) where a contour starts (end if none)
  static inline int next_start(const int8_t *row, int x, int end) {
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256(), one = _mm256_set1_epi8(1);
    for (; x + 32 <= end; x += 32) {
      const __m256i l = _mm256_loadu_si256((const __m256i *)(row + x - 1));
      const __m256i p = _mm256_loadu_si256((const __m256i *)(row + x));
      const __m256i outer = _mm256_and_si256(_mm256_cmpeq_epi8(l, zero), _mm256_cmpeq_epi8(p, one));
      const __m256i hole = _mm256_and_si256(_mm256_cmpeq_epi8(p, zero), _mm256_cmpgt_epi8(l, zero));
      const unsigned m = (unsigned)_mm256_movemask_epi8(_mm256_or_si256(outer, hole));
      if (m) return x + ctz32(m);
    }
#endif
#if defined(__SSE2__) || defined(_M_X64)
    const __m128i zero16 = _mm_setzero_si128(), one16 = _mm_set1_epi8(1);
    for (; x + 16 <= end; x += 16) {
      const __m128i l = _mm_loadu_si128((const __m128i *)(row + x - 1));
      const __m128i p = _mm_loadu_si128((const __m128i *)(row + x));
      const __m128i outer = _mm_and_si128(_mm_cmpeq_epi8(l, zero16), _mm_cmpeq_epi8(p, one16));
      const __m128i hole = _mm_and_si128(_mm_cmpeq_epi8(p, zero16), _mm_cmpgt_epi8(l, zero16));
      const unsigned m = (unsigned)_mm_movemask_epi8(_mm_or_si128(outer, hole));
      if (m) return x + ctz32(m);
    }
#endif
    for (; x < end; x++)
      if (is_start(row[x - 1], row[x])) return x;
    return end;
  }

  static inline int ctz32(unsigned v) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long idx;
    _BitScanForward(&idx, v);
    return (int)idx;
#else
    return __builtin_ctz(v);
#endif
  }

  // icvFetchContour (8-bit, nbd = 2, CHAIN_APPROX_NONE)
  inline void trace(int8_t *i0, const int *deltas, cv::Point pt, bool hole) {
    static const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    static const int dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    const int8_t nbd = 2;
    const int8_t right = (int8_t)(nbd | -128);
    int s, s_end;
    s_end = s = hole ? 0 : 4;
    int8_t *i1;
    do {
      s = (s - 1) & 7;
      i1 = i0 + deltas[s];
    } while (*i1 == 0 && s != s_end);

    if (s == s_end) { // single pixel domain
      *i0 = right;
      pts_.push_back(pt);
      return;
    }
    int8_t *i3 = i0, *i4;
    for (;;) {
      s_end = s;
      // next non-zero neighbour counter-clockwise from s (s+1 .. s+8): gather
      // the 8-neighbour occupancy into a bit mask and take its lowest set bit
      // after rotation, instead of a data-dependent probing loop.
      unsigned m = 0;
      for (int d = 0; d < 8; d++) m |= (unsigned)(i3[deltas[d]] != 0) << d;
      m |= m << 8;
      s = (s + 1 + ctz32(m >> (s + 1))) & 7;
      i4 = i3 + deltas[s];
      if ((unsigned)(s - 1) < (unsigned)s_end) *i3 = right;
      else if (*i3 == 1) *i3 = nbd;
      pts_.push_back(pt);
      pt.x += dx[s];
      pt.y += dy[s];
      if (i4 == i0 && i3 == i1) break;
      i3 = i4;
      s = (s + 4) & 7;
    }
  }

  std::vector<cv::Point> pts_;
  std::vector<std::pair<size_t, size_t>> kept_;
};

// ---------------------------------------------------------------------------
// The detector
// ---------------------------------------------------------------------------
class FastFractalDetector {
public:
  DetectorParams params;

  explicit FastFractalDetector(const std::string &config, float marker_size = -1.f) : set_(config) {
    if (marker_size != -1.f) set_.convertToMeters(marker_size);
    for (auto &kv : set_.bits_ids) {
      Group g;
      g.n = (int)std::lround(std::sqrt((double)kv.first)) + 2;
      for (int id : kv.second) {
        const nanofractal::FractalMarker &fm = set_.fractalMarkerCollection.at(id);
        Code c;
        c.id = id;
        const cv::Mat M = fm.mat(), K = fm.mask();
        const int sz = M.rows;
        c.bits.resize(sz * sz);
        c.mask.resize(sz * sz);
        for (int r = 0; r < sz; r++)
          for (int q = 0; q < sz; q++) {
            c.bits[r * sz + q] = (uint8_t)(M.at<uchar>(r, q) * 255);
            c.mask[r * sz + q] = K.at<uchar>(r, q);
          }
        g.codes.push_back(std::move(c));
      }
      groups_.push_back(std::move(g));
    }
  }

  // Outer-corner detection. `out` is cleared and filled with one entry per
  // detected marker level, in the reference detector's order.
  void detect(const cv::Mat &img, std::vector<Detection> &out) {
    to_gray(img);
    detect_gray(out);
  }

  // Detection plus inner-corner correspondences for pose estimation.
  void detect(const cv::Mat &img, std::vector<Detection> &out, std::vector<cv::Point3f> &p3d,
              std::vector<cv::Point2f> &p2d) {
    to_gray(img);
    detect_gray(out);
    if (!out.empty()) inner_points(out, p3d, p2d);
  }

  const nanofractal::FractalMarkerSet &marker_set() const { return set_; }

private:
  struct Code {
    int id;
    std::vector<uint8_t> bits; // 0 / 255, row-major sz x sz
    std::vector<uint8_t> mask; // 1 = compare this cell
  };
  struct Group {
    int n; // cells per side including the black border
    std::vector<Code> codes;
  };
  struct Candidate {
    int id;
    cv::Point2f c[4];
  };

  void to_gray(const cv::Mat &img) {
    if (img.channels() == 3) cv::cvtColor(img, gray_, cv::COLOR_BGR2GRAY);
    else gray_ = img;
  }

  int subpix() const { return params.subpix_win_size == -1 ? 4 : params.subpix_win_size; }

  void detect_gray(std::vector<Detection> &out) {
    out.clear();
    cands_.clear();
    float scale = params.detection_scale;
    if (!(scale > 0.f && scale < 1.f)) scale = 1.f;
    const float inv_scale = 1.f / scale;
    if (scale < 1.f) cv::resize(gray_, proc_, cv::Size(), scale, scale, cv::INTER_AREA);
    else proc_ = gray_;

    int win;
    if (params.adaptive_block_size == -1) {
      win = std::max(int(3), int(15 * float(proc_.cols) / 1920.));
    } else {
      win = params.adaptive_block_size;
      if (win < 3) win = 3;
    }
    if (win % 2 == 0) win++;
    int min_contour = (params.min_contour_size == -1) ? 120 : params.min_contour_size;
    if (scale < 1.f) min_contour = std::max(4, int(min_contour * scale + 0.5f));

    thresh_.run(proc_, win, params.adaptive_c, bin_);
    const double rate = params.approx_poly_rate;
    tracer_.run(bin_, min_contour, [&](const cv::Point *pts, int n) {
      const cv::Mat contour(n, 1, CV_32SC2, const_cast<cv::Point *>(pts));
      cv::approxPolyDP(contour, approx_, double(n) * rate, true);
      if (approx_.size() != 4 || !cv::isContourConvex(approx_)) return;
      decode(approx_);
    });

    // duplicates: sort by id and, within an id, largest perimeter first
    std::sort(cands_.begin(), cands_.end(), [](const Candidate &a, const Candidate &b) {
      if (a.id < b.id) return true;
      else if (a.id == b.id) return perimeter(a.c) > perimeter(b.c);
      else return false;
    });
    auto ip = std::unique(cands_.begin(), cands_.end(),
                          [](const Candidate &a, const Candidate &b) { return a.id == b.id; });
    cands_.resize(std::distance(cands_.begin(), ip));
    if (cands_.empty()) return;

    if (scale < 1.f)
      for (auto &m : cands_)
        for (auto &pt : m.c) {
          pt.x *= inv_scale;
          pt.y *= inv_scale;
        }
    corners_.clear();
    for (const auto &m : cands_) corners_.insert(corners_.end(), m.c, m.c + 4);
    const int sp = subpix();
    if (sp > 0)
      cv::cornerSubPix(gray_, corners_, cv::Size(sp, sp), cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
    out.resize(cands_.size());
    for (size_t i = 0; i < cands_.size(); i++) {
      out[i].id = cands_[i].id;
      for (int c = 0; c < 4; c++) out[i].corners[c] = corners_[i * 4 + c];
    }
  }

  // Sample every bit-grid size through the candidate's homography and match it
  // against the codes of that size (FractalMarkerDetector::detect, inner loop).
  void decode(const std::vector<cv::Point> &quad) {
    cv::Point2f mc[4];
    for (int j = 0; j < 4; j++) mc[j] = cv::Point2f((float)quad[j].x, (float)quad[j].y);
    { // sort corners clockwise
      const double dx1 = mc[1].x - mc[0].x, dy1 = mc[1].y - mc[0].y;
      const double dx2 = mc[2].x - mc[0].x, dy2 = mc[2].y - mc[0].y;
      if ((dx1 * dy2) - (dy1 * dx2) < 0.0) std::swap(mc[1], mc[3]);
    }
    const cv::Point2f unit[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    const cv::Mat Hm = cv::getPerspectiveTransform(unit, mc);
    const double *h = Hm.ptr<double>(0);
    for (const Group &g : groups_) {
      const int n = g.n;
      uint8_t bits[18 * 18];
      int sum = 0;
      for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
          const float px = float(c + 0.5) / float(n), py = float(r + 0.5) / float(n);
          const double a = h[0] * px + h[1] * py + h[2];
          const double b = h[3] * px + h[4] * py + h[5];
          const double w = h[6] * px + h[7] * py + h[8];
          const uint8_t v = uchar(0.5 + subpixel_value(proc_, cv::Point2f(a / w, b / w)));
          bits[r * n + c] = v;
          sum += v;
        }
      const double mean = double(sum) / double(n * n);
      for (int i = 0; i < n * n; i++) bits[i] = bits[i] > mean ? 255 : 0;
      int nrot = 0;
      const int id = marker_id(bits, n, g, nrot);
      if (id == -1) continue;
      // (the reference rotates its working copy in place, so a second matching
      // grid size sees the already-rotated corners)
      std::rotate(mc, mc + 4 - nrot, mc + 4);
      Candidate cand;
      cand.id = id;
      std::copy(mc, mc + 4, cand.c);
      cands_.push_back(cand);
    }
  }

  // FractalMarkerDetector::getMarkerId on an n x n 0/255 grid.
  static int marker_id(const uint8_t *bits, int n, const Group &g, int &nrot) {
    for (int x = 0; x < n; x++)
      if (bits[x] || bits[(n - 1) * n + x] || bits[x * n] || bits[x * n + n - 1]) return -1;
    const int sz = n - 2;
    uint8_t inner[256], tmp[256];
    for (int r = 0; r < sz; r++)
      for (int c = 0; c < sz; c++) inner[r * sz + c] = bits[(r + 1) * n + c + 1];
    for (nrot = 0; nrot < 4; nrot++) {
      for (const Code &code : g.codes) {
        bool match = true;
        for (int i = 0; i < sz * sz && match; i++)
          if (code.mask[i] && inner[i] != code.bits[i]) match = false;
        if (match) return code.id;
      }
      for (int i = 0; i < sz; i++)
        for (int j = 0; j < sz; j++) tmp[i * sz + j] = inner[(sz - 1 - j) * sz + i];
      std::memcpy(inner, tmp, (size_t)sz * sz);
    }
    return -1;
  }

  // FractalMarkerDetector::getSubpixelValue, bit-exact. The reference reads
  // (x+1, y+1) without a bounds check; those reads are clamped to the buffer.
  static float subpixel_value(const cv::Mat &im, const cv::Point2f &p) {
    float ix, iy;
    const float fx = std::modf(p.x, &ix);
    const float fy = std::modf(p.y, &iy);
    cv::Point tl;
    if (fx > 0.5) {
      if (fy > 0.5) tl = cv::Point(ix, iy);
      else tl = cv::Point(ix, iy - 1);
    } else {
      if (fy > 0.5) tl = cv::Point(ix - 1, iy);
      else tl = cv::Point(ix - 1, iy - 1);
    }
    if (tl.x < 0) tl.x = 0;
    if (tl.y < 0) tl.y = 0;
    if (tl.x >= im.cols) tl.x = im.cols - 1;
    if (tl.y >= im.cols) tl.y = im.rows - 1; // (sic) reference compares with cols
    const uchar *d = im.data;
    const size_t st = im.step;
    const size_t last = (size_t)(im.rows - 1) * st + (size_t)(im.cols - 1);
    auto px = [&](int y, int x) -> float { return float(d[std::min((size_t)y * st + (size_t)x, last)]); };
    return (1.f - fy) * (1. - fx) * px(tl.y, tl.x) + fx * (1 - fy) * px(tl.y, tl.x + 1) +
           (1 - fx) * fy * px(tl.y + 1, tl.x) + fx * fy * px(tl.y + 1, tl.x + 1);
  }

  static int perimeter(const cv::Point2f *a) {
    int sum = 0;
    for (int i = 0; i < 4; i++) sum += cv::norm(a[i] - a[(i + 1) % 4]);
    return sum;
  }

  // ---- inner corners (FractalMarkerDetector::detect(img, p3d, p2d)) ----
  void inner_points(const std::vector<Detection> &det, std::vector<cv::Point3f> &p3d,
                    std::vector<cv::Point2f> &p2d) {
    std::vector<cv::Point2f> imgpoints;
    std::vector<cv::Point3f> objpoints;
    for (const Detection &m : det) {
      const auto &kp = set_.fractalMarkerCollection.at(m.id).keypts;
      for (int c = 0; c < 4; c++) {
        imgpoints.push_back(m.corners[c]);
        objpoints.push_back(cv::Point3f(kp[c].pt.x, kp[c].pt.y, 0));
      }
    }
    kpts_.clear();
    cv::FAST(gray_, kpts_, 10, true, cv::FastFeatureDetector::TYPE_9_16);
    nanofractal::_private::kfilter(kpts_, params.kfilter_min_dist * params.kfilter_min_dist);
    nanofractal::_private::assignClass(gray_, kpts_);
    if (kpts_.empty()) return;
    const cv::Mat H = cv::findHomography(objpoints, imgpoints);

    std::vector<cv::Point2f> objPts, imgPts;
    for (const auto &fm : set_.fractalMarkerCollection) {
      const std::vector<cv::KeyPoint> &okp = fm.second.keypts;
      objPts.resize(okp.size());
      for (size_t i = 0; i < okp.size(); i++) objPts[i] = okp[i].pt;
      cv::perspectiveTransform(objPts, imgPts, H);
      bool consider = true;
      for (size_t i = 0; i < imgPts.size() - 1 && consider; i++)
        for (size_t j = i + 1; j < imgPts.size() && consider; j++) {
          const double dx = imgPts[i].x - imgPts[j].x, dy = imgPts[i].y - imgPts[j].y;
          if (dx * dx + dy * dy < 150) consider = false;
        }
      if (consider) {
        for (size_t idx = 0; idx < imgPts.size(); idx++) {
          const cv::Point2f q = imgPts[idx];
          if (!(q.x > 0 && q.x < gray_.cols && q.y > 0 && q.y < gray_.rows)) continue;
          int hit = -1, nhits = 0;
          for (size_t k = 0; k < kpts_.size() && nhits < 2; k++) {
            const double dx = kpts_[k].pt.x - q.x;
            const double dy = kpts_[k].pt.y - q.y;
            if (dx * dx + dy * dy < 100) {
              hit = (int)k;
              nhits++;
            }
          }
          if (nhits == 1 && kpts_[hit].class_id == okp[idx].class_id) {
            p2d.push_back(kpts_[hit].pt);
            p3d.push_back(cv::Point3f(objPts[idx].x, objPts[idx].y, 0));
          }
        }
      } else {
        for (const Detection &m : det)
          if (m.id == fm.first) {
            const auto &kp = set_.fractalMarkerCollection.at(m.id).keypts;
            for (int c = 0; c < 4; c++) {
              p3d.push_back(cv::Point3f(kp[c].pt.x, kp[c].pt.y, 0));
              p2d.push_back(m.corners[c]);
            }
            break;
          }
      }
    }
    const int sp = subpix();
    if (!p2d.empty() && sp > 0)
      cv::cornerSubPix(gray_, p2d, cv::Size(sp, sp), cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
  }

  nanofractal::FractalMarkerSet set_;
  std::vector<Group> groups_;
  AdaptiveThreshold thresh_;
  ContourTracer tracer_;
  cv::Mat gray_, proc_, bin_;
  std::vector<cv::Point> approx_;
  std::vector<Candidate> cands_;
  std::vector<cv::Point2f> corners_;
  std::vector<cv::KeyPoint> kpts_;
};

} // namespace nanofractal_fast
