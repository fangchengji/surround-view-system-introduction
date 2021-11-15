/*
 * preprocess.h
 *
 * Created on: Wed Apr 08 2020
 *     Author: fangcheng.ji
 */

#pragma once

#include <assert.h>
#include <cuda_runtime_api.h>
#include <gpu_timer.h>

#include "opencv2/opencv.hpp"

namespace svs {

/**
 * @brief resize(aspect keep), sub mean, pad the bgr image
 *
 * @param src bgr image
 * @param dst output image
 * @param src_width
 * @param src_height
 * @param channel
 * @param dst_width output image width
 * @param dst_height output image height
 * @param mean_b mean value of b channel
 * @param mean_g mean value of g channel
 * @param mean_r mean value of r channel
 */

void aspect_keep_resize_sub_mean_pad(const unsigned char *const src, float *const dst, const int src_width,
                                     const int src_height, const int channel, const int dst_width, const int dst_height,
                                     const float mean_b, const float mean_g, const float mean_r);

} /* namespace svs*/
