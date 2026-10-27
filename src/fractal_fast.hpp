// Optimised fractal-marker detection pipeline.
//
// Produces the same detections as the reference
// nanofractal::FractalMarkerDetector (third_party/nanofractal.h) while
// replacing its hot path:
//
//   * adaptive threshold: a fused box-sum + compare that reproduces
//     cv::adaptiveThreshold(MEAN_C, THRESH_BINARY_INV) bit for bit (including
//     OpenCV's fixed-point / float rounding of the box mean) and writes
//     straight into a zero-bordered buffer, so contour tracing needs no extra
//     copy;
//   * contour tracing: Suzuki-Abe border following identical to
//   cv::findContours
//     (RETR_LIST, CHAIN_APPROX_NONE), but contours shorter than the minimum
//     size are traced in place and never allocated;
//   * no per-frame heap traffic: all scratch buffers live in the detector and
//   are
//     reused across frames, and detected markers are returned as (id, 4
//     corners) instead of copies of the full FractalMarker.
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
#include <type_traits>
#include <vector>
#if defined(__SSE2__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h> // _BitScanForward / _BitScanReverse
#endif

namespace nanofractal_fast {

struct Detection {
  int id;
  cv::Point2f corners[4];
};

// ---------------------------------------------------------------------------
// 8-neighbour occupancy masks of a zero-bordered 0/1 int8 image: bit d of
// nbm(y, x) is set when the neighbour in direction d is non-zero (0 E, 1 NE,
// 2 N, 3 NW, 4 W, 5 SW, 6 S, 7 SE). Tracing only relabels non-zero pixels with
// other non-zero values, so the masks stay valid for the whole contour pass.
// ---------------------------------------------------------------------------
inline void neighbour_mask_row(const cv::Mat &bin, cv::Mat &nbm, int y) {
  const int8_t *u = bin.ptr<int8_t>(y - 1), *r = bin.ptr<int8_t>(y),
               *d = bin.ptr<int8_t>(y + 1);
  uint8_t *m = nbm.ptr<uint8_t>(y);
  const int W = bin.cols;
  m[0] = m[W - 1] = 0;
  for (int x = 1; x < W - 1; x++)
    m[x] = (uint8_t)((r[x + 1] != 0) | (u[x + 1] != 0) << 1 | (u[x] != 0) << 2 |
                     (u[x - 1] != 0) << 3 | (r[x - 1] != 0) << 4 |
                     (d[x - 1] != 0) << 5 | (d[x] != 0) << 6 |
                     (d[x + 1] != 0) << 7);
}

inline void neighbour_masks(const cv::Mat &bin, cv::Mat &nbm) {
  nbm.create(bin.size(), CV_8UC1);
  std::memset(nbm.ptr(0), 0, nbm.cols);
  std::memset(nbm.ptr(nbm.rows - 1), 0, nbm.cols);
  for (int y = 1; y < bin.rows - 1; y++)
    neighbour_mask_row(bin, nbm, y);
}

// ---------------------------------------------------------------------------
// Adaptive threshold (MEAN_C, BINARY_INV) into a zero-bordered 0/1 buffer.
// ---------------------------------------------------------------------------
class AdaptiveThreshold {
public:
  int max_threads = 0; // row stripes; 0 = cv::getNumThreads()

  // dst: (H+2) x (W+2) int8 buffer; interior pixel (x,y) lands at (x+1,y+1).
  // Foreground (dark relative to the local mean) = 1, background and border =
  // 0. nbm (optional): neighbour masks of dst, computed on the fly while the
  // rows are still in cache.
  void run(const cv::Mat &src, int k, double delta, cv::Mat &dst,
           cv::Mat *nbm = nullptr) {
    CV_Assert(src.type() == CV_8UC1 && k % 2 == 1 && k > 1);
    dst.create(src.rows + 2, src.cols + 2, CV_8SC1);
    std::memset(dst.ptr(0), 0, dst.cols);
    std::memset(dst.ptr(src.rows + 1), 0, dst.cols);
    if (k > 257) { // column sums would overflow uint16: defer to OpenCV
      cv::Mat t;
      cv::adaptiveThreshold(src, t, 1, cv::ADAPTIVE_THRESH_MEAN_C,
                            cv::THRESH_BINARY_INV, k, delta);
      for (int y = 0; y < src.rows; y++) {
        int8_t *o = dst.ptr<int8_t>(y + 1);
        o[0] = o[src.cols + 1] = 0;
        std::memcpy(o + 1, t.ptr(y), src.cols);
      }
      if (nbm)
        neighbour_masks(dst, *nbm);
      return;
    }
    // Row stripes run in parallel (cv::parallel_for_; serial inside
    // detect_batch workers). Each stripe primes its own vertical window, so the
    // result does not depend on the split.
    const int H = src.rows;
    const int r = k / 2;
    int nstripes =
        max_threads > 0 ? max_threads : std::max(1, cv::getNumThreads());
    nstripes = std::min(
        nstripes, std::max(1, H / std::max(48, 2 * k))); // keep priming cheap
    if ((int)scratch_.size() < nstripes)
      scratch_.resize(nstripes);
    const bool small = k * k * 255 <= 65535; // full window sums fit uint16
    cv::parallel_for_(
        cv::Range(0, nstripes),
        [&](const cv::Range &rg) {
          for (int i = rg.start; i < rg.end; i++) {
            const int y0 = (int)((int64_t)H * i / nstripes),
                      y1 = (int)((int64_t)H * (i + 1) / nstripes);
            if (small)
              stripe<uint16_t>(src, k, delta, dst, y0, y1, scratch_[i]);
            else
              stripe<uint32_t>(src, k, delta, dst, y0, y1, scratch_[i]);
          }
        },
        nstripes);
    (void)r;
    if (nbm) {
      nbm->create(dst.size(), CV_8UC1);
      std::memset(nbm->ptr(0), 0, nbm->cols);
      std::memset(nbm->ptr(nbm->rows - 1), 0, nbm->cols);
      cv::Mat &m = *nbm;
      cv::parallel_for_(
          cv::Range(1, dst.rows - 1),
          [&](const cv::Range &rg) {
            for (int y = rg.start; y < rg.end; y++)
              neighbour_mask_row(dst, m, y);
          },
          nstripes);
    }
  }

private:
  static constexpr int kStrip = 512;

  struct Scratch {
    std::vector<uint16_t> ring, ext, col;
    std::vector<int32_t> mask;
    std::vector<uint32_t> pw[9],
        acc; // sized in uint32 units, reused as S arrays
  };
  std::vector<Scratch> scratch_;

  // Rows [y0, y1) of the threshold. Separable box sum, horizontal first: each
  // source row's horizontal window sums (<= k*255, always uint16) are computed
  // once into a ring of k+2 rows, and the vertical window is a running sum over
  // that ring (V = uint16 when the full k*k window fits, else uint32). Rows and
  // columns replicate at the border, like OpenCV's BORDER_REPLICATE box filter.
  template <typename V>
  static void stripe(const cv::Mat &src, int k, double delta, cv::Mat &dst,
                     int y0, int y1, Scratch &sc) {
    if (y0 >= y1)
      return;
    const int W = src.cols, H = src.rows, r = k / 2;
    const int idelta = cvFloor(delta);
    // OpenCV's box filter: windows of <= 256 px sum in uint16 and divide in
    // fixed point; larger windows sum in int32 and round s*scale in float
    // (8-lane SIMD groups) or double (the last W % 8 pixels of a row).
    const bool fixed = k * k <= 256;
    uint32_t dd = 0, ds = 1;
    if (fixed) {
      const int d = k * k;
      double scalef = double(1 << 23) / d;
      int dsi = cvFloor(scalef);
      scalef -= dsi;
      int ddi = d / 2;
      if (scalef < 0.5)
        ddi++;
      else
        dsi++;
      dd = (uint32_t)ddi;
      ds = (uint32_t)dsi;
    }
    const double scale = 1.0 / (double(k) * k);
    const float scalef32 = (float)scale;
    const int simd_end = W - W % 8;

    const int R = k + 2; // ring rows: y+r is written while y-r-1 is still read
    sc.ring.resize((size_t)R * W);
    sc.ext.resize(W + 2 * r);
    for (auto &v : sc.pw)
      v.resize(std::min(W, kStrip) + 2 * r);
    sc.acc.resize(std::min(W, kStrip) + 2 * r);
    sc.mask.resize(W);
    sc.col.assign((size_t)W * sizeof(V) / sizeof(uint16_t) + 1, 0);
    V *col = reinterpret_cast<V *>(sc.col.data());
    // (lambdas capture by value: a by-reference capture lets W escape, after
    // which GCC cannot prove stores through col don't alias it and stops
    // vectorising every loop bounded by W)
    uint16_t *const ring = sc.ring.data();
    Scratch *scp = &sc;
    auto hrow = [ring, R, W](int j) { return ring + (size_t)(j % R) * W; };
    auto hsum = [scp, &src, hrow, r, W,
                 k](int j) { // horizontal window sums of source row j
      const uint8_t *sp = src.ptr<uint8_t>(j);
      uint16_t *e = scp->ext.data();
      for (int i = 0; i < r; i++)
        e[i] = sp[0];
      for (int x = 0; x < W; x++)
        e[r + x] = sp[x];
      for (int i = 0; i < r; i++)
        e[r + W + i] = sp[W - 1];
      uint16_t *out = hrow(j);
      for (int x0 = 0; x0 < W; x0 += kStrip) {
        const int w = std::min(kStrip, W - x0);
        const uint16_t *hs = box_row<uint16_t>(*scp, e + x0, w, k);
        std::memcpy(out + x0, hs, w * sizeof(uint16_t));
      }
    };

    // vertical window of row y0: rows clamp(y0-r .. y0+r)
    for (int j = std::max(y0 - r, 0); j <= std::min(y0 + r, H - 1); j++)
      hsum(j);
    {
      const uint16_t *h0 = hrow(std::min(std::max(y0 - r, 0), H - 1));
      for (int x = 0; x < W; x++)
        col[x] = (V)h0[x];
      for (int dy = -r + 1; dy <= r; dy++) {
        const uint16_t *h = hrow(std::min(std::max(y0 + dy, 0), H - 1));
        for (int x = 0; x < W; x++)
          col[x] = (V)(col[x] + h[x]);
      }
    }
    for (int y = y0; y < y1; y++) {
      if (y > y0) {
        if (y + r <= H - 1)
          hsum(y + r);
        const uint16_t *add = hrow(std::min(y + r, H - 1));
        const uint16_t *sub = hrow(std::max(y - r - 1, 0));
        for (int x = 0; x < W; x++)
          col[x] = (V)(col[x] + add[x] - sub[x]);
      }
      const uint8_t *px = src.ptr<uint8_t>(y);
      int8_t *orow = dst.ptr<int8_t>(y + 1);
      orow[0] = 0;
      orow[W + 1] = 0;
      int8_t *o = orow + 1;
      if (fixed) {
        for (int x = 0; x < W; x++) {
          const int mean = (int)(((uint32_t)col[x] + dd) * ds >> 23);
          o[x] = (int8_t)(mean >= px[x] + idelta);
        }
      } else {
        int x = 0;
        int32_t *m = sc.mask.data();
#if defined(__AVX2__)
        // mean = round_half_even(float(s) * float(scale)) exactly like OpenCV's
        // SIMD path (cvtps2dq rounds to nearest even)
        const __m256 vs = _mm256_set1_ps(scalef32);
        const __m256i vd = _mm256_set1_epi32(idelta);
        for (; x + 8 <= simd_end; x += 8) {
          const __m256i sv = _mm256_loadu_si256((const __m256i *)(col + x));
          const __m256i mean =
              _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(sv), vs));
          const __m256i t = _mm256_add_epi32(
              _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(px + x))),
              vd);
          // mean >= t  <=>  !(t > mean)
          const __m256i ge = _mm256_andnot_si256(_mm256_cmpgt_epi32(t, mean),
                                                 _mm256_set1_epi32(1));
          _mm256_storeu_si256((__m256i *)(m + x), ge);
        }
#endif
        // mean = round_half_even(s * scale) >= T  <=>  p > T - 0.5, or a tie
        // (p == T - 0.5) that rounds up to an even T. Branch-free; int32 mask
        // first, then narrow (GCC will not vectorise a float compare stored
        // straight to int8, nor an unsigned->float convert).
        for (; x < simd_end; x++) {
          const float p = (float)(int32_t)col[x] * scalef32;
          const int32_t T = (int32_t)px[x] + idelta;
          const float t = (float)T - 0.5f;
          const int32_t gt = p > t ? 1 : 0;
          const int32_t eq = p == t ? 1 : 0;
          m[x] = gt | (eq & ~T & 1);
        }
        for (x = 0; x < simd_end; x++)
          o[x] = (int8_t)m[x];
        for (x = simd_end; x < W; x++) {
          const int mean = cv::saturate_cast<uchar>(col[x] * scale);
          o[x] = (int8_t)(mean >= px[x] + idelta);
        }
      }
    }
  }

  // Horizontal window sums out[x] = sum_{i<k} c[x+i], x in [0, W). Builds the
  // power-of-two window sums c, c2, c4, ... (one vectorisable pass each) and
  // adds the ones selected by the bits of k in a single fused pass.
  template <typename S>
  static const S *box_row(Scratch &sc, const uint16_t *c, int W, int k) {
    S *pw[9];
    int nlev = 0;
    for (int w = 1; w <= k; w *= 2)
      nlev++;
    for (int i = 0; i < nlev; i++)
      pw[i] = reinterpret_cast<S *>(sc.pw[i].data());
    const int L = W + k - 1;
    if (std::is_same<S, uint16_t>::value)
      pw[0] = reinterpret_cast<S *>(const_cast<uint16_t *>(c));
    else
      for (int i = 0; i < L; i++)
        pw[0][i] = (S)c[i];
    int len = L;
    for (int lev = 1, w = 1; lev < nlev; lev++, w *= 2) {
      len -= w;
      const S *a = pw[lev - 1];
      S *b = pw[lev];
      for (int i = 0; i < len; i++)
        b[i] = (S)(a[i] + a[i + w]);
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
    S *out = reinterpret_cast<S *>(sc.acc.data());
    switch (nt) {
    case 1:
      return t[0];
    case 2:
      for (int x = 0; x < W; x++)
        out[x] = (S)(t[0][x] + t[1][x]);
      break;
    case 3:
      for (int x = 0; x < W; x++)
        out[x] = (S)(t[0][x] + t[1][x] + t[2][x]);
      break;
    case 4:
      for (int x = 0; x < W; x++)
        out[x] = (S)(t[0][x] + t[1][x] + t[2][x] + t[3][x]);
      break;
    default:
      for (int x = 0; x < W; x++)
        out[x] = (S)(t[0][x] + t[1][x] + t[2][x] + t[3][x]);
      for (int j = 4; j < nt; j++) {
        const S *tj = t[j];
        for (int x = 0; x < W; x++)
          out[x] = (S)(out[x] + tj[x]);
      }
    }
    return out;
  }
};

// ---------------------------------------------------------------------------
// Suzuki-Abe border following, identical to cv::findContours(RETR_LIST,
// CHAIN_APPROX_NONE) on a zero-bordered 0/1 int8 image (modified in place).
// Calls on_contour(points, n) for every contour with n >= min_size, in the
// order cv::findContours would return them.
// ---------------------------------------------------------------------------
class ContourTracer {
public:
  // Bands narrower than this are not worth a thread (tests lower it to force
  // many band boundaries).
  int min_band_rows = 96;
  int max_threads = 0; // bands; 0 = cv::getNumThreads()

  template <typename F> void run(cv::Mat &img, int min_size, F &&on_contour) {
    neighbour_masks(img, own_nbm_);
    run(img, own_nbm_, min_size, std::forward<F>(on_contour));
  }

  // nbm: neighbour_masks(img), e.g. produced by AdaptiveThreshold::run.
  //
  // Parallel, exact. The image rows are split into bands traced concurrently.
  // A trace's path depends only on which pixels are non-zero (nbm, shared and
  // read-only); the pixel labels findContours writes only decide which later
  // pixels still start a contour. Each band therefore keeps the labels of its
  // own rows in a private copy and drops label writes outside them (as in
  // OpenCV's data-race-free TRUCO tracer). A band also meets borders that
  // started in an earlier band and reach into it; it traces them once (their
  // labels then suppress their other starts in the band) and discards them:
  // a border is kept only by the band holding its canonical start -- the top
  // row of an outer border, the row below the top row of a hole border (the
  // top row of the hole itself). Everything else is identical to the
  // sequential raster scan, so are the contours and, after merging the bands
  // in order, their order.
  template <typename F>
  void run(cv::Mat &img, const cv::Mat &nbm, int min_size, F &&on_contour) {
    CV_Assert(img.isContinuous() && nbm.isContinuous() &&
              nbm.size() == img.size() && img.rows >= 3);
    const int H = img.rows;
    int nb = max_threads > 0 ? max_threads : std::max(1, cv::getNumThreads());
    nb = std::min(nb, std::max(1, (H - 2) / min_band_rows));
    if ((int)bands_.size() < nb)
      bands_.resize(nb);
    for (int b = 0; b < nb; b++) {
      Band &bd = bands_[b];
      bd.y0 = 1 + (int)((int64_t)(H - 2) * b / nb);
      bd.y1 = 1 + (int)((int64_t)(H - 2) * (b + 1) / nb);
    }
    if (nb == 1) {
      run_band(img, nbm, img.ptr<int8_t>(1), min_size, bands_[0]);
    } else {
      cv::parallel_for_(
          cv::Range(0, nb),
          [&](const cv::Range &rg) {
            for (int b = rg.start; b < rg.end; b++) {
              Band &bd = bands_[b];
              // private labels of the band's rows (initially the 0/1 image)
              // (kPad zero bytes in front: the scan reads row[x - 1] at x = 0)
              const size_t bytes = (size_t)(bd.y1 - bd.y0) * img.step;
              bd.labels.resize(bytes + kPad);
              std::memset(bd.labels.data(), 0, kPad);
              std::memcpy(bd.labels.data() + kPad, img.ptr(bd.y0), bytes);
              run_band(img, nbm, bd.labels.data() + kPad, min_size, bd);
            }
          },
          nb);
    }
    // cv::findContours(RETR_LIST) lists contours in reverse discovery order
    for (int b = nb; b-- > 0;) {
      const Band &bd = bands_[b];
      for (size_t i = bd.kept.size(); i-- > 0;)
        on_contour(bd.pts.data() + bd.kept[i].first, (int)bd.kept[i].second);
    }
  }

private:
  static constexpr size_t kPad = 64;
  struct Band {
    int y0 = 1, y1 = 1;         // padded rows [y0, y1)
    std::vector<int8_t> labels; // private labels of rows [y0, y1)
    std::vector<cv::Point> pts; // sized as capacity; npts in use
    size_t npts = 0;
    std::vector<std::pair<size_t, size_t>> kept;
    std::vector<uint64_t> ev;
  };
  std::vector<Band> bands_;
  cv::Mat own_nbm_;

  // lab: labels of padded row bd.y0 (rows y0..y1-1 follow with img's step)
  void run_band(const cv::Mat &img, const cv::Mat &nbm, int8_t *lab,
                int min_size, Band &bd) {
    const int step = (int)img.step;
    const int W = img.cols;
    int deltas[16] = {1,  -step + 1, -step, -step - 1,
                      -1, step - 1,  step,  step + 1};
    std::memcpy(deltas + 8, deltas, 8 * sizeof(int));
    Trace tr;
    tr.deltas = deltas;
    tr.lo = nbm.ptr<uint8_t>(bd.y0);
    tr.hi = nbm.ptr<uint8_t>(bd.y1);
    tr.loff = lab - (const int8_t *)tr.lo;
    tr.bd = &bd;

    bd.kept.clear();
    bd.npts = 0;
    if (bd.pts.size() < 4096)
      bd.pts.resize(4096);
    const int end = W - 1; // scan x in [1, W-1)
    const bool mark_isolated = min_size > 1;
    bd.ev.assign((size_t)(end + 63) / 64, 0);
    // a contour is ours iff its top row (outer) / top row + 1 (hole) is in
    // the band; pt.y is the padded row - 1
    const int keep_outer = bd.y0 - 1, keep_hole = bd.y0 - 2;
    for (int y = bd.y0; y < bd.y1; y++) {
      int8_t *row = lab + (size_t)(y - bd.y0) * step;
      const uint8_t *nrow = nbm.ptr<uint8_t>(y);
      // A contour starts at x when (left, pixel) = (0, 1) -- outer border -- or
      // (>= 1, 0) -- hole border, left pixel foreground and not yet marked as a
      // right boundary. Pixels left of the scan position are never relabelled
      // and pixels that are 0 never change, so tracing a contour can only
      // remove later starts on this row, never create one: the starts of a row
      // can be collected up front in bulk and re-checked when reached, which is
      // exactly the sequential raster scan of cv::findContours.
      collect_starts(row, nrow, end, mark_isolated, bd.ev.data());
      for (size_t wi = 0; wi < bd.ev.size(); wi++) {
        uint64_t bits = bd.ev[wi];
        while (bits) {
          const int x = (int)(wi * 64) + ctz64(bits);
          bits &= bits - 1;
          if (!is_start(row[x - 1], row[x]))
            continue; // removed by an earlier trace
          const int is_hole = row[x] == 0;
          const size_t start = bd.npts;
          const int miny = tr.run(nrow + x - is_hole,
                                  cv::Point(x - is_hole - 1, y - 1), is_hole);
          const size_t n = bd.npts - start;
          if ((int)n >= min_size && miny >= (is_hole ? keep_hole : keep_outer))
            bd.kept.push_back({start, n});
          else
            bd.npts = start;
        }
      }
    }
  }

  static inline bool is_start(int8_t l, int8_t p) {
    return (l == 0 && p == 1) || (p == 0 && l > 0);
  }

  // ev bit x = a contour starts at x given the row as it is now.
  //
  // With skip_isolated, foreground pixels whose 8 neighbours are all
  // background are dropped: findContours traces each as a one-point "single
  // pixel domain" contour (always below min_size when this is used) whose only
  // effect is to mark the pixel as a right boundary. Such a pixel is its own
  // component, so no trace ever reads it; the one thing its mark changes is the
  // hole test of its right neighbour, which is suppressed here too. On noisy
  // frames these speckles are the large majority of all contours.
  static void collect_starts(const int8_t *row, const uint8_t *nb, int end,
                             bool skip_isolated, uint64_t *ev) {
    uint32_t iso_carry = 0; // isolated flag of pixel x-1
    int x = 0;              // x = 0 is the zero border: never a start
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256(), one = _mm256_set1_epi8(1);
    // 32-pixel blocks aligned to 0 so two of them fill one 64-bit word
    auto block = [&](int x0) -> uint32_t {
      const __m256i l = _mm256_loadu_si256((const __m256i *)(row + x0 - 1));
      const __m256i p = _mm256_loadu_si256((const __m256i *)(row + x0));
      const __m256i p1 = _mm256_cmpeq_epi8(p, one);
      const __m256i outer = _mm256_and_si256(_mm256_cmpeq_epi8(l, zero), p1);
      const __m256i hole = _mm256_and_si256(_mm256_cmpeq_epi8(p, zero),
                                            _mm256_cmpgt_epi8(l, zero));
      uint32_t mo = (uint32_t)_mm256_movemask_epi8(outer);
      uint32_t mh = (uint32_t)_mm256_movemask_epi8(hole);
      uint32_t iso = 0;
      if (skip_isolated) {
        const __m256i m = _mm256_loadu_si256((const __m256i *)(nb + x0));
        iso = (uint32_t)_mm256_movemask_epi8(
            _mm256_and_si256(p1, _mm256_cmpeq_epi8(m, zero)));
      }
      mo &= ~iso;
      mh &= ~((iso << 1) | iso_carry);
      iso_carry = iso >> 31;
      return mo | mh;
    };
    for (; x + 64 <= end; x += 64) {
      const uint64_t lo = block(x);
      const uint64_t hi = block(x + 32);
      ev[x >> 6] = lo | (hi << 32);
    }
#endif
    // scalar tail (and the whole row without AVX2), word by word
    for (; x < end; x += 64) {
      uint64_t word = 0;
      const int stop = std::min(x + 64, end);
      for (int i = std::max(x, 1); i < stop; i++) {
        const int8_t l = row[i - 1], pv = row[i];
        const uint32_t iso = skip_isolated && pv == 1 && nb[i] == 0;
        bool start = is_start(l, pv) && !iso;
        if (pv == 0 && iso_carry)
          start = false;
        iso_carry = iso;
        if (start)
          word |= 1ULL << (i & 63);
      }
      ev[x >> 6] = word;
    }
  }

  static inline int ctz64(uint64_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long idx;
    _BitScanForward64(&idx, v);
    return (int)idx;
#else
    return __builtin_ctzll(v);
#endif
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

  static inline int msb32(unsigned v) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long idx;
    _BitScanReverse(&idx, v);
    return (int)idx;
#else
    return 31 - __builtin_clz(v);
#endif
  }

  // icvFetchContour (8-bit, nbd = 2, CHAIN_APPROX_NONE) walking the neighbour
  // masks; labels are written to the band's private rows only.
  struct Trace {
    const int *deltas;
    const uint8_t *lo, *hi; // mask rows of the band
    ptrdiff_t loff;         // mask pointer -> label pointer, inside the band
    Band *bd;

    inline void label(const uint8_t *p, bool is_right) const {
      if (p < lo || p >= hi)
        return; // another band's row: its owner keeps its own labels
      int8_t *q = (int8_t *)(p + loff);
      const int8_t cur = *q;
      // right boundary, else first visit 1 -> nbd (branch-free: the outcome
      // is data dependent and mispredicts badly as a branch)
      const int8_t visited = cur == 1 ? (int8_t)2 : cur;
      *q = is_right ? (int8_t)(2 | -128) : visited;
    }

    // returns the smallest pt.y visited; appends the points to bd->pts
    inline int run(const uint8_t *i0, cv::Point pt, int hole) const {
      static const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
      static const int dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};
      Band &b = *bd;
      cv::Point *w = b.pts.data() + b.npts, *wend = b.pts.data() + b.pts.size();
      int miny = pt.y;

      // initial direction: first non-zero neighbour clockwise from s_end - 1
      // (s_end itself is always background here). None -> single pixel.
      int s_end = hole ? 0 : 4;
      unsigned m0 = *i0;
      m0 |= m0 << 8;
      const unsigned cand =
          m0 & (((1u << (s_end + 8)) - 1) & ~((1u << (s_end + 1)) - 1));
      if (!cand) {
        label(i0, true);
        if (w == wend)
          w = grow(b, w, wend);
        *w++ = pt;
        b.npts = w - b.pts.data();
        return miny;
      }
      int s = msb32(cand) & 7;
      const uint8_t *const i1 = i0 + deltas[s];
      const uint8_t *i3 = i0, *i4;
      for (;;) {
        s_end = s;
        // next non-zero neighbour counter-clockwise from s (s+1 .. s+8)
        unsigned m = *i3;
        m |= m << 8;
        s = (s + 1 + ctz32(m >> (s + 1))) & 7;
        i4 = i3 + deltas[s];
        label(i3, (unsigned)(s - 1) < (unsigned)s_end);
        if (w == wend)
          w = grow(b, w, wend);
        *w++ = pt;
        miny = std::min(miny, pt.y);
        pt.x += dx[s];
        pt.y += dy[s];
        if (i4 == i0 && i3 == i1)
          break;
        i3 = i4;
        s = (s + 4) & 7;
      }
      b.npts = w - b.pts.data();
      return miny;
    }

    static cv::Point *grow(Band &b, cv::Point *w, cv::Point *&wend) {
      const size_t used = w - b.pts.data();
      b.pts.resize(b.pts.size() * 2);
      wend = b.pts.data() + b.pts.size();
      return b.pts.data() + used;
    }
  };
};

// ---------------------------------------------------------------------------
// cv::approxPolyDP(closed contour of int points), exact. OpenCV's
// implementation is replicated without the per-call heap buffers, InputArray
// dispatch and modulo-wrapped point reads; every quantity it computes from
// integer points is an integer below 2^53, so int64 arithmetic gives identical
// values, and the eps comparisons keep its double expressions. The split step
// differs between versions: OpenCV 5 (geometry module) measures the distance
// to the segment, OpenCV 4 (imgproc) to the line through it.
// ---------------------------------------------------------------------------
class ApproxPolyDP {
public:
  // returns the vertices in out (cleared first)
  void run(const cv::Point *src, int count, double eps,
           std::vector<cv::Point> &out) {
    out.resize(count);
    const int n = approx(src, count, eps, out.data());
    out.resize(n);
  }

private:
  int approx(const cv::Point *src, int count, double eps, cv::Point *dst) {
    if (count == 0)
      return 0;
    if (stack_.size() < (size_t)count)
      stack_.resize(count);
    cv::Range *stack = stack_.data();
    size_t top = 0;
    auto push = [&](const cv::Range &r) {
      if (top >= stack_.size()) {
        stack_.resize(stack_.size() * 3 / 2 + 1);
        stack = stack_.data();
      }
      stack[top++] = r;
    };
    eps *= eps;
    int new_count = 0;
    cv::Range slice(0, 0), right_slice(0, 0);
    cv::Point start_pt;
    bool le_eps = false;

    // 1. approximately the two farthest points (three passes, closed contour)
    int pos = 0;
    right_slice.start = 0;
    for (int it = 0; it < 3; it++) {
      int64_t max_dist = 0;
      pos = (pos + right_slice.start) % count;
      start_pt = src[pos];
      // points pos+1 .. pos+count-1 (mod count), j = 1 .. count-1
      int j = 1;
      for (int q = pos + 1; q < count; q++, j++) {
        const int64_t dx = src[q].x - start_pt.x, dy = src[q].y - start_pt.y;
        const int64_t d = dx * dx + dy * dy;
        if (d > max_dist) {
          max_dist = d;
          right_slice.start = j;
        }
      }
      for (int q = 0; q < pos; q++, j++) {
        const int64_t dx = src[q].x - start_pt.x, dy = src[q].y - start_pt.y;
        const int64_t d = dx * dx + dy * dy;
        if (d > max_dist) {
          max_dist = d;
          right_slice.start = j;
        }
      }
      le_eps = (double)max_dist <= eps;
    }

    // 2. initialise the stack
    if (!le_eps) {
      right_slice.end = slice.start = pos % count;
      slice.end = right_slice.start = (right_slice.start + slice.start) % count;
      push(right_slice);
      push(slice);
    } else {
      dst[new_count++] = start_pt;
    }

    // 3. recursive split
    while (top > 0) {
      slice = stack[--top];
      const cv::Point end_pt = src[slice.end];
      int p = slice.start;
      start_pt = src[p];
      if (++p >= count)
        p = 0;
      if (p != slice.end) {
        const int64_t dx = end_pt.x - start_pt.x, dy = end_pt.y - start_pt.y;
        const int64_t seg2 = dx * dx + dy * dy;
        CV_Assert(dx != 0 || dy != 0);
        int64_t max_d = 0;
        while (p != slice.end) {
          const cv::Point pt = src[p];
          if (++p >= count)
            p = 0;
          const int64_t ax = pt.x - start_pt.x, ay = pt.y - start_pt.y;
#if CV_VERSION_MAJOR >= 5
          // squared distance to the segment, times its squared length
          const int64_t proj = ax * dx + ay * dy;
          int64_t d;
          if (proj < 0) {
            d = (ax * ax + ay * ay) * seg2;
          } else if (proj > seg2) {
            const int64_t bx = pt.x - end_pt.x, by = pt.y - end_pt.y;
            d = (bx * bx + by * by) * seg2;
          } else {
            const int64_t dist = ay * dx - ax * dy;
            d = dist * dist;
          }
#else
          // distance to the line times the segment length, squared (same
          // order as OpenCV 4's |dist|, and its max_dist^2 exactly)
          const int64_t dist = ay * dx - ax * dy;
          const int64_t d = dist * dist;
#endif
          if (d > max_d) {
            max_d = d;
            right_slice.start = (p + count - 1) % count;
          }
        }
        le_eps = (double)max_d <= eps * (double)seg2;
      } else {
        le_eps = true;
        start_pt = src[slice.start];
      }
      if (le_eps) {
        dst[new_count++] = start_pt;
      } else {
        right_slice.end = slice.end;
        slice.end = right_slice.start;
        push(right_slice);
        push(slice);
      }
    }

    // 4. remove extra points on [almost] straight lines
    const int cnt = new_count;
    int rpos = cnt - 1;
    auto read = [&](cv::Point &q) {
      q = dst[rpos];
      if (++rpos >= cnt)
        rpos = 0;
    };
    cv::Point pt, end_pt;
    read(start_pt);
    int wpos = rpos;
    read(pt);
    for (int i = 0; i < cnt && new_count > 2; i++) {
      read(end_pt);
      const double dx = end_pt.x - start_pt.x, dy = end_pt.y - start_pt.y;
      const double dist =
          std::fabs((pt.x - start_pt.x) * dy - (pt.y - start_pt.y) * dx);
      const double sip = (pt.x - start_pt.x) * (end_pt.x - pt.x) +
                         (pt.y - start_pt.y) * (end_pt.y - pt.y);
      if (dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx != 0 &&
          dy != 0 && sip >= 0) {
        new_count--;
        dst[wpos] = start_pt = end_pt;
        if (++wpos >= cnt)
          wpos = 0;
        read(pt);
        i++;
        continue;
      }
      dst[wpos] = start_pt = pt;
      if (++wpos >= cnt)
        wpos = 0;
      pt = end_pt;
    }
    return new_count;
  }

  std::vector<cv::Range> stack_;
};

// ---------------------------------------------------------------------------
// The detector
// ---------------------------------------------------------------------------
class FastFractalDetector {
public:
  DetectorParams params;
  // true: never split a frame across threads (detect_batch workers, which
  // already run one frame per core; nested parallel_for_ would run serially
  // anyway, so splitting would only add band overhead)
  bool serial = false;

  explicit FastFractalDetector(const std::string &config,
                               float marker_size = -1.f)
      : set_(config) {
    if (marker_size != -1.f)
      set_.convertToMeters(marker_size);
    for (auto &kv : set_.bits_ids) {
      Group g;
      g.n = (int)std::lround(std::sqrt((double)kv.first)) + 2;
      for (int id : kv.second) {
        const nanofractal::FractalMarker &fm =
            set_.fractalMarkerCollection.at(id);
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
  void detect(const cv::Mat &img, std::vector<Detection> &out,
              std::vector<cv::Point3f> &p3d, std::vector<cv::Point2f> &p2d) {
    to_gray(img);
    detect_gray(out);
    if (!out.empty())
      inner_points(out, p3d, p2d);
  }

  const nanofractal::FractalMarkerSet &marker_set() const { return set_; }

  // True when the last detection saw one marker level in two separate places
  // (two copies of the marker); only the largest of each id is returned.
  bool multiple_instances() const { return multiple_; }

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
    if (img.channels() == 3)
      cv::cvtColor(img, gray_, cv::COLOR_BGR2GRAY);
    else
      gray_ = img;
  }

  int subpix() const {
    return params.subpix_win_size == -1 ? 4 : params.subpix_win_size;
  }

  void detect_gray(std::vector<Detection> &out) {
    out.clear();
    cands_.clear();
    multiple_ = false;
    float scale = params.detection_scale;
    if (!(scale > 0.f && scale < 1.f))
      scale = 1.f;
    const float inv_scale = 1.f / scale;
    if (scale < 1.f)
      cv::resize(gray_, proc_, cv::Size(), scale, scale, cv::INTER_AREA);
    else
      proc_ = gray_;

    int win;
    if (params.adaptive_block_size == -1) {
      win = std::max(int(3), int(15 * float(proc_.cols) / 1920.));
    } else {
      win = params.adaptive_block_size;
      if (win < 3)
        win = 3;
    }
    if (win % 2 == 0)
      win++;
    int min_contour =
        (params.min_contour_size == -1) ? 120 : params.min_contour_size;
    if (scale < 1.f)
      min_contour = std::max(4, int(min_contour * scale + 0.5f));

    thresh_.max_threads = tracer_.max_threads = serial ? 1 : 0;
    thresh_.run(proc_, win, params.adaptive_c, bin_, &nbm_);
    const double rate = params.approx_poly_rate;
    conts_.clear();
    tracer_.run(bin_, nbm_, min_contour, [&](const cv::Point *pts, int n) {
      conts_.push_back({pts, n});
    });

    // polygon fit + decoding per contour in parallel; each contour writes its
    // (at most one per grid size) candidates to its own slots, merged below in
    // contour order, so the result is independent of the split
    const int nc = (int)conts_.size();
    const int G = (int)groups_.size();
    slots_.resize((size_t)nc * G);
    nslots_.assign(nc, 0);
    const int nchunks =
        std::max(1, std::min(nc, serial ? 1 : 2 * cv::getNumThreads()));
    if ((int)workers_.size() < nchunks)
      workers_.resize(nchunks);
    if (nc > 0)
      cv::parallel_for_(
          cv::Range(0, nchunks),
          [&](const cv::Range &rg) {
            for (int c = rg.start; c < rg.end; c++) {
              Worker &w = workers_[c];
              const int i0 = (int)((int64_t)nc * c / nchunks),
                        i1 = (int)((int64_t)nc * (c + 1) / nchunks);
              for (int i = i0; i < i1; i++) {
                const int n = conts_[i].second;
                w.dp.run(conts_[i].first, n, double(n) * rate, w.approx);
                if (w.approx.size() != 4 || !cv::isContourConvex(w.approx))
                  continue;
                nslots_[i] = decode(w.approx, &slots_[(size_t)i * G]);
              }
            }
          },
          nchunks);
    for (int i = 0; i < nc; i++)
      for (int j = 0; j < nslots_[i]; j++)
        cands_.push_back(slots_[(size_t)i * G + j]);

    // duplicates: sort by id and, within an id, largest perimeter first
    std::sort(cands_.begin(), cands_.end(),
              [](const Candidate &a, const Candidate &b) {
                if (a.id < b.id)
                  return true;
                else if (a.id == b.id)
                  return perimeter(a.c) > perimeter(b.c);
                else
                  return false;
              });
    // A smaller candidate of the same id whose centre lies outside the
    // largest one is another copy of the marker (the same square seen twice,
    // e.g. both borders of its black frame, is nested instead).
    for (size_t i = 0, j; i < cands_.size() && !multiple_; i = j) {
      for (j = i + 1; j < cands_.size() && cands_[j].id == cands_[i].id; j++) {
        const cv::Point2f *c = cands_[j].c;
        if (!inside(cands_[i].c, (c[0] + c[1] + c[2] + c[3]) * 0.25f)) {
          multiple_ = true;
          break;
        }
      }
      while (j < cands_.size() && cands_[j].id == cands_[i].id)
        j++;
    }
    auto ip = std::unique(
        cands_.begin(), cands_.end(),
        [](const Candidate &a, const Candidate &b) { return a.id == b.id; });
    cands_.resize(std::distance(cands_.begin(), ip));
    if (cands_.empty())
      return;

    if (scale < 1.f)
      for (auto &m : cands_)
        for (auto &pt : m.c) {
          pt.x *= inv_scale;
          pt.y *= inv_scale;
        }
    corners_.clear();
    for (const auto &m : cands_)
      corners_.insert(corners_.end(), m.c, m.c + 4);
    const int sp = subpix();
    if (sp > 0)
      cv::cornerSubPix(
          gray_, corners_, cv::Size(sp, sp), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                           12, 0.005));
    out.resize(cands_.size());
    for (size_t i = 0; i < cands_.size(); i++) {
      out[i].id = cands_[i].id;
      for (int c = 0; c < 4; c++)
        out[i].corners[c] = corners_[i * 4 + c];
    }
  }

  // Sample every bit-grid size through the candidate's homography and match it
  // against the codes of that size (FractalMarkerDetector::detect, inner loop).
  int decode(const std::vector<cv::Point> &quad, Candidate *out) const {
    int nout = 0;
    cv::Point2f mc[4];
    for (int j = 0; j < 4; j++)
      mc[j] = cv::Point2f((float)quad[j].x, (float)quad[j].y);
    { // sort corners clockwise
      const double dx1 = mc[1].x - mc[0].x, dy1 = mc[1].y - mc[0].y;
      const double dx2 = mc[2].x - mc[0].x, dy2 = mc[2].y - mc[0].y;
      if ((dx1 * dy2) - (dy1 * dx2) < 0.0)
        std::swap(mc[1], mc[3]);
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
          const float px = float(c + 0.5) / float(n),
                      py = float(r + 0.5) / float(n);
          const double a = h[0] * px + h[1] * py + h[2];
          const double b = h[3] * px + h[4] * py + h[5];
          const double w = h[6] * px + h[7] * py + h[8];
          const uint8_t v =
              uchar(0.5 + subpixel_value(proc_, cv::Point2f(a / w, b / w)));
          bits[r * n + c] = v;
          sum += v;
        }
      const double mean = double(sum) / double(n * n);
      for (int i = 0; i < n * n; i++)
        bits[i] = bits[i] > mean ? 255 : 0;
      int nrot = 0;
      const int id = marker_id(bits, n, g, nrot);
      if (id == -1)
        continue;
      // (the reference rotates its working copy in place, so a second matching
      // grid size sees the already-rotated corners)
      std::rotate(mc, mc + 4 - nrot, mc + 4);
      Candidate &cand = out[nout++];
      cand.id = id;
      std::copy(mc, mc + 4, cand.c);
    }
    return nout;
  }

  // FractalMarkerDetector::getMarkerId on an n x n 0/255 grid.
  static int marker_id(const uint8_t *bits, int n, const Group &g, int &nrot) {
    for (int x = 0; x < n; x++)
      if (bits[x] || bits[(n - 1) * n + x] || bits[x * n] ||
          bits[x * n + n - 1])
        return -1;
    const int sz = n - 2;
    uint8_t inner[256], tmp[256];
    for (int r = 0; r < sz; r++)
      for (int c = 0; c < sz; c++)
        inner[r * sz + c] = bits[(r + 1) * n + c + 1];
    for (nrot = 0; nrot < 4; nrot++) {
      for (const Code &code : g.codes) {
        bool match = true;
        for (int i = 0; i < sz * sz && match; i++)
          if (code.mask[i] && inner[i] != code.bits[i])
            match = false;
        if (match)
          return code.id;
      }
      for (int i = 0; i < sz; i++)
        for (int j = 0; j < sz; j++)
          tmp[i * sz + j] = inner[(sz - 1 - j) * sz + i];
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
      if (fy > 0.5)
        tl = cv::Point(ix, iy);
      else
        tl = cv::Point(ix, iy - 1);
    } else {
      if (fy > 0.5)
        tl = cv::Point(ix - 1, iy);
      else
        tl = cv::Point(ix - 1, iy - 1);
    }
    if (tl.x < 0)
      tl.x = 0;
    if (tl.y < 0)
      tl.y = 0;
    if (tl.x >= im.cols)
      tl.x = im.cols - 1;
    if (tl.y >= im.cols)
      tl.y = im.rows - 1; // (sic) reference compares with cols
    const uchar *d = im.data;
    const size_t st = im.step;
    const size_t last = (size_t)(im.rows - 1) * st + (size_t)(im.cols - 1);
    auto px = [&](int y, int x) -> float {
      return float(d[std::min((size_t)y * st + (size_t)x, last)]);
    };
    return (1.f - fy) * (1. - fx) * px(tl.y, tl.x) +
           fx * (1 - fy) * px(tl.y, tl.x + 1) +
           (1 - fx) * fy * px(tl.y + 1, tl.x) +
           fx * fy * px(tl.y + 1, tl.x + 1);
  }

  // Point inside a convex quad (either winding).
  static bool inside(const cv::Point2f *q, cv::Point2f p) {
    int pos = 0, neg = 0;
    for (int k = 0; k < 4; k++) {
      cv::Point2f e = q[(k + 1) % 4] - q[k], v = p - q[k];
      float cr = e.x * v.y - e.y * v.x;
      pos += cr > 0;
      neg += cr < 0;
    }
    return pos == 0 || neg == 0;
  }

  static int perimeter(const cv::Point2f *a) {
    int sum = 0;
    for (int i = 0; i < 4; i++)
      sum += cv::norm(a[i] - a[(i + 1) % 4]);
    return sum;
  }

  // ---- inner corners (FractalMarkerDetector::detect(img, p3d, p2d)) ----
  //
  // Same algorithm as the reference with one deliberate difference: FAST runs
  // only on the bounding box of the projected inner corners (plus a margin that
  // makes every keypoint within the 10 px match radius identical to a
  // full-frame run). So kfilter's response threshold -- 20% of the response
  // range -- is taken over the marker's neighbourhood instead of the whole
  // scene. Everything else is exact: kfilter, the radius search and the corner
  // classes return what the reference would for the same keypoints.
  void inner_points(const std::vector<Detection> &det,
                    std::vector<cv::Point3f> &p3d,
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
    const cv::Mat H = cv::findHomography(objpoints, imgpoints);

    // project every marker's keypoints; decide which markers are usable
    const int W = gray_.cols, Hh = gray_.rows;
    proj_.clear();
    float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
    for (const auto &fm : set_.fractalMarkerCollection) {
      const std::vector<cv::KeyPoint> &okp = fm.second.keypts;
      Projected pr;
      pr.id = fm.first;
      pr.okp = &okp;
      objPts_.resize(okp.size());
      for (size_t i = 0; i < okp.size(); i++)
        objPts_[i] = okp[i].pt;
      cv::perspectiveTransform(objPts_, pr.img, H);
      pr.obj = objPts_;
      pr.consider = !has_close_pair(pr.img, 150.0);
      if (pr.consider)
        for (const cv::Point2f &q : pr.img)
          if (q.x > 0 && q.x < W && q.y > 0 && q.y < Hh) {
            x0 = std::min(x0, q.x);
            y0 = std::min(y0, q.y);
            x1 = std::max(x1, q.x);
            y1 = std::max(y1, q.y);
          }
      proj_.push_back(std::move(pr));
    }

    kpts_.clear();
    if (x0 <= x1) {
      const int margin = 15;
      const cv::Rect roi =
          cv::Rect(cv::Point((int)std::floor(x0) - margin,
                             (int)std::floor(y0) - margin),
                   cv::Point((int)std::ceil(x1) + margin + 1,
                             (int)std::ceil(y1) + margin + 1)) &
          cv::Rect(0, 0, W, Hh);
      cv::FAST(gray_(roi), kpts_, 10, true, cv::FastFeatureDetector::TYPE_9_16);
      for (auto &k : kpts_) {
        k.pt.x += roi.x;
        k.pt.y += roi.y;
      }
    }
    kfilter(kpts_, params.kfilter_min_dist * params.kfilter_min_dist);
    // (the reference returns here when the whole frame has no FAST keypoint;
    // an empty ROI says nothing about the rest of the frame, so carry on: the
    // outer-corner fallback below still applies)
    if (!kpts_.empty())
      grid_.build(kpts_, 10.f);
    cls_.assign(kpts_.size(), kUnknownClass);

    for (const Projected &pr : proj_) {
      const std::vector<cv::KeyPoint> &okp = *pr.okp;
      if (pr.consider) {
        for (size_t idx = 0; idx < pr.img.size(); idx++) {
          const cv::Point2f q = pr.img[idx];
          if (!(q.x > 0 && q.x < W && q.y > 0 && q.y < Hh))
            continue;
          const int hit =
              kpts_.empty() ? -1 : grid_.unique_within(kpts_, q, 100.0);
          if (hit >= 0 && keypoint_class(hit) == okp[idx].class_id) {
            p2d.push_back(kpts_[hit].pt);
            p3d.push_back(cv::Point3f(pr.obj[idx].x, pr.obj[idx].y, 0));
          }
        }
      } else {
        for (const Detection &m : det)
          if (m.id == pr.id) {
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
      cv::cornerSubPix(
          gray_, p2d, cv::Size(sp, sp), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                           12, 0.005));
  }

  // true when two points are closer than sqrt(lim2) (the reference's O(n^2)
  // "consider" test, same double arithmetic, via a sort-and-sweep on x)
  bool has_close_pair(const std::vector<cv::Point2f> &pts, double lim2) {
    order_.resize(pts.size());
    for (size_t i = 0; i < pts.size(); i++)
      order_[i] = (int)i;
    std::sort(order_.begin(), order_.end(),
              [&](int a, int b) { return pts[a].x < pts[b].x; });
    const double lim = std::sqrt(lim2) + 1.0; // conservative sweep window
    for (size_t a = 0; a < order_.size(); a++) {
      const cv::Point2f &p = pts[order_[a]];
      for (size_t b = a + 1; b < order_.size(); b++) {
        const cv::Point2f &q = pts[order_[b]];
        if ((double)q.x - (double)p.x > lim)
          break;
        const double dx = p.x - q.x, dy = p.y - q.y;
        if (dx * dx + dy * dy < lim2)
          return true;
      }
    }
    return false;
  }

  static constexpr int kUnknownClass = -1000;
  int keypoint_class(int i) {
    if (cls_[i] == kUnknownClass) {
      one_.assign(1, kpts_[i]);
      nanofractal::_private::assignClass(gray_, one_);
      cls_[i] = one_[0].class_id;
    }
    return cls_[i];
  }

  // nanofractal::_private::kfilter with the O(n^2) duplicate search replaced
  // by a grid; identical result for the same input (see comments).
  void kfilter(std::vector<cv::KeyPoint> &kp, float min_dist_sq) {
    if (kp.empty())
      return;
    float minResp = kp[0].response, maxResp = kp[0].response;
    for (auto &p : kp) {
      p.size = 40;
      if (p.response < minResp)
        minResp = p.response;
      if (p.response > maxResp)
        maxResp = p.response;
    }
    const float thr = (maxResp - minResp) * 0.20f + minResp;
    const int n = (int)kp.size();
    if (!(min_dist_sq > 0)) { // nothing is a duplicate
      for (auto &p : kp)
        if (p.response < thr)
          p.size = -1;
    } else {
      // The reference compares kp[xi] (whose position changes whenever it
      // absorbs a stronger duplicate) with every later kp[xj], whose positions
      // are still the original ones. Candidates for xi are gathered from a grid
      // around xi's original position with a 3r reach and visited in index
      // order; once xi has drifted far enough that a point outside that reach
      // could qualify, the remaining indices are scanned exhaustively.
      const float r = std::sqrt(min_dist_sq);
      const float reach = 3 * r;
      kgrid_.build(kp, reach);
      for (int xi = 0; xi < n; xi++) {
        if (kp[xi].response < thr) {
          kp[xi].size = -1;
          continue;
        }
        const cv::Point2f origin = kp[xi].pt;
        cand_.clear();
        kgrid_.collect(origin, [&](int j) {
          if (j > xi)
            cand_.push_back(j);
        });
        std::sort(cand_.begin(), cand_.end());
        // Points outside the reach are >= reach - drift > r away from xi as
        // long as xi has drifted at most reach - r, so skipping them is exact;
        // xi only moves when it absorbs a candidate, so the drift is re-checked
        // after each one and, once exceeded, every index after the last one
        // examined is compared directly.
        const float max_drift2 = (reach - r) * (reach - r);
        auto drifted = [&] {
          const float ox = kp[xi].pt.x - origin.x, oy = kp[xi].pt.y - origin.y;
          return ox * ox + oy * oy > max_drift2;
        };
        int last = xi;
        bool exhaustive = false;
        for (size_t c = 0; c < cand_.size() && !exhaustive; c++) {
          dedup(kp, xi, cand_[c], min_dist_sq);
          last = cand_[c];
          exhaustive = drifted();
        }
        if (exhaustive)
          for (int xj = last + 1; xj < n; xj++)
            dedup(kp, xi, xj, min_dist_sq);
      }
    }
    kp.erase(std::remove_if(kp.begin(), kp.end(),
                            [](const cv::KeyPoint &k) { return k.size == -1; }),
             kp.end());
  }

  static inline void dedup(std::vector<cv::KeyPoint> &kp, int xi, int xj,
                           float min_dist_sq) {
    const float dx = kp[xi].pt.x - kp[xj].pt.x;
    const float dy = kp[xi].pt.y - kp[xj].pt.y;
    if (dx * dx + dy * dy < min_dist_sq) {
      if (kp[xj].response > kp[xi].response)
        kp[xi] = kp[xj];
      kp[xj].size = -1;
    }
  }

  // uniform grid over keypoint positions
  struct Grid {
    float cell = 1;
    int gx = 0, gy = 0;
    float ox = 0, oy = 0;
    std::vector<int> head, next;
    void build(const std::vector<cv::KeyPoint> &kp, float c) {
      cell = c;
      float mx = 1e30f, my = 1e30f, Mx = -1e30f, My = -1e30f;
      for (auto &k : kp) {
        mx = std::min(mx, k.pt.x);
        my = std::min(my, k.pt.y);
        Mx = std::max(Mx, k.pt.x);
        My = std::max(My, k.pt.y);
      }
      ox = mx;
      oy = my;
      gx = (int)((Mx - mx) / cell) + 1;
      gy = (int)((My - my) / cell) + 1;
      head.assign((size_t)gx * gy, -1);
      next.assign(kp.size(), -1);
      // insert in reverse so each bucket lists indices in increasing order
      for (int i = (int)kp.size() - 1; i >= 0; i--) {
        const int b = bucket(kp[i].pt);
        next[i] = head[b];
        head[b] = i;
      }
    }
    int bucket(const cv::Point2f &p) const {
      int cx = std::min(std::max((int)((p.x - ox) / cell), 0), gx - 1);
      int cy = std::min(std::max((int)((p.y - oy) / cell), 0), gy - 1);
      return cy * gx + cx;
    }
    // every index whose cell is within one cell of p's cell (superset of all
    // points within `cell` of p)
    template <typename F> void collect(const cv::Point2f &p, F &&f) const {
      const int cx = (int)std::floor((p.x - ox) / cell),
                cy = (int)std::floor((p.y - oy) / cell);
      for (int y = std::max(cy - 1, 0); y <= std::min(cy + 1, gy - 1); y++)
        for (int x = std::max(cx - 1, 0); x <= std::min(cx + 1, gx - 1); x++)
          for (int i = head[y * gx + x]; i >= 0; i = next[i])
            f(i);
    }
    // index of the only keypoint with squared distance < lim2 to q (the
    // reference kd-tree: float differences, double sum), else -1
    int unique_within(const std::vector<cv::KeyPoint> &kp, const cv::Point2f &q,
                      double lim2) const {
      int hit = -1, nhits = 0;
      collect(q, [&](int i) {
        const double dx = kp[i].pt.x - q.x;
        const double dy = kp[i].pt.y - q.y;
        if (dx * dx + dy * dy < lim2) {
          hit = i;
          nhits++;
        }
      });
      return nhits == 1 ? hit : -1;
    }
  };

  struct Projected {
    int id;
    const std::vector<cv::KeyPoint> *okp;
    std::vector<cv::Point2f> obj, img;
    bool consider;
  };

  bool multiple_ = false;
  nanofractal::FractalMarkerSet set_;
  std::vector<Group> groups_;
  AdaptiveThreshold thresh_;
  ContourTracer tracer_;
  struct Worker {
    ApproxPolyDP dp;
    std::vector<cv::Point> approx;
  };
  std::vector<Worker> workers_;
  std::vector<std::pair<const cv::Point *, int>> conts_;
  std::vector<Candidate> slots_;
  std::vector<int> nslots_;
  cv::Mat gray_, proc_, bin_, nbm_;
  std::vector<Candidate> cands_;
  std::vector<cv::Point2f> corners_;
  std::vector<cv::KeyPoint> kpts_, one_;
  std::vector<Projected> proj_;
  std::vector<cv::Point2f> objPts_;
  std::vector<int> order_, cand_, cls_;
  Grid grid_, kgrid_;
};

} // namespace nanofractal_fast
