#pragma once

// Tunable parameters for the detection pipeline.
// All fields are optional — defaults reproduce the original hard-coded behaviour
// of each detector so existing code needs no changes.
struct DetectorParams {
    // Minimum contour perimeter (pixels) to consider as a marker candidate.
    // Reduce to detect smaller markers; increase to reject noise.
    // -1 = use detector built-in default (ArUco: 50, Fractal: 120).
    int   min_contour_size    = -1;

    // Adaptive-threshold block size (must be odd, >= 3).
    // -1 = auto-scale with image width (ArUco: 13; Fractal: max(3, 15*w/1920)).
    int   adaptive_block_size = -1;

    // Constant subtracted from the adaptive-threshold local mean.
    // Larger values detect edges more aggressively and reject more background.
    float adaptive_c          = 7.f;

    // Contour approximation rate: epsilon = perimeter * rate.
    // Lower = tighter polygon fit; higher = more permissive (faster but less accurate).
    float approx_poly_rate    = 0.05f;

    // Corner sub-pixel refinement half-window in pixels (Fractal only).
    // -1 = use detector default (4, giving a 9x9 search window).
    //  0 = disable sub-pixel refinement.
    int   subpix_win_size     = -1;

    // Minimum distance (pixels) between FAST keypoints (Fractal only).
    // Keypoints within this radius are merged; higher = fewer, more spread-out points.
    float kfilter_min_dist    = 10.f;

    // Downscale factor for the detection stage (both ArUco and Fractal). The
    // expensive adaptive-threshold + contour + bit-decode stage runs on an image
    // scaled by this factor; detected corners are mapped back and sub-pixel refined
    // at full resolution, so accuracy loss stays sub-pixel as long as markers remain
    // >= ~10 px after downscaling. (For Fractal, inner points are sampled from the
    // full-resolution image afterwards.) 1.0 = full resolution (default); 0.5
    // quarters the pixel count of the dominant stage (~4x faster). Values <=0 or
    // >=1 mean no scaling. min_contour_size stays expressed in original-image pixels.
    float detection_scale     = 1.f;
};
