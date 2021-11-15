#pragma once

#include <assert.h>
#include <cuda_runtime_api.h>
#include <gpu_timer.h>

#include "opencv2/opencv.hpp"

namespace svs {

void lut_stitch(const unsigned char *const src, unsigned char *const dst, int width, int height, int dst_width,
                int dst_height, const float *const geo_lut, const unsigned char *const idx_lut);

void lut_stitch_single(const unsigned char *const src, unsigned char *const dst, int width, int height, int dst_width,
                       int dst_height, const float *const geo_lut, const unsigned char *const idx_lut, int idx);

}   // namespace svs