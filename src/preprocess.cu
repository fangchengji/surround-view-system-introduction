/*
 * preprocess.cu
 *
 * Created on: Wed Apr 08 2020
 *     Author: fangcheng.ji
 */

#include <device_launch_parameters.h>
#include <preprocess.h>

namespace svs {

__global__ void resize_sub_mean_pad_kernel(const unsigned char *const src, float *const dst, int channel, int width,
                                           int height, int roi_width, int roi_height, int dst_width, int dst_height,
                                           float scale, float mean_b, float mean_g, float mean_r) {
  // 2D Index of current thread
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dst_x < roi_width && dst_y < roi_height) {
    float src_x = (dst_x + 0.5) * scale - 0.5;
    float src_y = (dst_y + 0.5) * scale - 0.5;
    const int x1 = __float2int_rd(src_x);
    const int y1 = __float2int_rd(src_y);
    const int x2 = x1 + 1;
    const int y2 = y1 + 1;
    const int x1_read = max(0, min(x1, width - 1));
    const int y1_read = max(0, min(y1, height - 1));
    const int x2_read = max(0, min(x2, width - 1));
    const int y2_read = max(0, min(y2, height - 1));
    //(h*width+w)*channel+c
    float src_reg = 0;
    int idx11 = (y1_read * width + x1_read) * channel;
    int idx12 = (y1_read * width + x2_read) * channel;
    int idx21 = (y2_read * width + x1_read) * channel;
    int idx22 = (y2_read * width + x2_read) * channel;

    float out = 0;
    // c = 0, bilinear interpolation
    src_reg = (int)src[idx11];
    out = (x2 - src_x) * (y2 - src_y) * src_reg;

    src_reg = (int)src[idx12];
    out += src_reg * (src_x - x1) * (y2 - src_y);

    src_reg = (int)src[idx21];
    out += src_reg * (x2 - src_x) * (src_y - y1);

    src_reg = (int)src[idx22];
    out += src_reg * (src_x - x1) * (src_y - y1);

    out -= mean_b;
    dst[(dst_y * dst_width + dst_x) * channel] = out;

    // c = 1
    src_reg = (int)src[idx11 + 1];
    out = (x2 - src_x) * (y2 - src_y) * src_reg;

    src_reg = (int)src[idx12 + 1];
    out += src_reg * (src_x - x1) * (y2 - src_y);

    src_reg = (int)src[idx21 + 1];
    out += src_reg * (x2 - src_x) * (src_y - y1);

    src_reg = (int)src[idx22 + 1];
    out += src_reg * (src_x - x1) * (src_y - y1);

    out -= mean_g;
    dst[(dst_y * dst_width + dst_x) * channel + 1] = out;

    // c = 2
    src_reg = (int)src[idx11 + 2];
    out = (x2 - src_x) * (y2 - src_y) * src_reg;

    src_reg = (int)src[idx12 + 2];
    out += src_reg * (src_x - x1) * (y2 - src_y);

    src_reg = (int)src[idx21 + 2];
    out += src_reg * (x2 - src_x) * (src_y - y1);

    src_reg = (int)src[idx22 + 2];
    out += src_reg * (src_x - x1) * (src_y - y1);

    out -= mean_r;
    dst[(dst_y * dst_width + dst_x) * channel + 2] = out;
  } else if (dst_x < dst_width && dst_y < dst_height) {
    // padding 0.0
    dst[(dst_y * dst_width + dst_x) * channel] = 0.0;
    dst[(dst_y * dst_width + dst_x) * channel + 1] = 0.0;
    dst[(dst_y * dst_width + dst_x) * channel + 2] = 0.0;
  }
}

void aspect_keep_resize_sub_mean_pad(const unsigned char *const src, float *const dst, const int src_width,
                                     const int src_height, const int channel, const int dst_width, const int dst_height,
                                     const float mean_b, const float mean_g, const float mean_r) {
  float scale_w = float(dst_width) / float(src_width);
  float scale_h = float(dst_height) / float(src_height);
  float scale = std::min(scale_h, scale_w);

  int new_w = int(scale * src_width + 0.5);
  int new_h = int(scale * src_height + 0.5);
  float dst2src_scale = 1.0f / scale;

  // Specify a reasonable block size
  const dim3 block(32, 32);
  // Calculate grid size to cover the whole image
  const dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);

  resize_sub_mean_pad_kernel<<<grid, block>>>(src, dst, channel, src_width, src_height, new_w, new_h, dst_width,
                                              dst_height, dst2src_scale, mean_b, mean_g, mean_r);
}

} /* namespace svs*/
