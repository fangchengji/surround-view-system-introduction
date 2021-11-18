#include "stitching.h"

namespace svs {

__global__ void lut_stitch_kernel(const unsigned char *const src, unsigned char *const dst, int width, int height,
                                  int dst_width, int dst_height, const float *const geo_lut,
                                  const unsigned char *const idx_lut) {
  // 2D Index of current thread
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  const int geo_channel = 2;
  const int src_channel = 3;
  const int dst_channel = 3;
  if (dst_x < dst_width && dst_y < dst_height) {
    const int img_offset = idx_lut[(dst_y * dst_width + dst_x)] * width * height * src_channel;
    const float src_x = geo_lut[(dst_y * dst_width + dst_x) * geo_channel];
    const float src_y = geo_lut[(dst_y * dst_width + dst_x) * geo_channel + 1];

    // bilinear interpolation
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
    int idx11 = (y1_read * width + x1_read) * src_channel + img_offset;
    int idx12 = (y1_read * width + x2_read) * src_channel + img_offset;
    int idx21 = (y2_read * width + x1_read) * src_channel + img_offset;
    int idx22 = (y2_read * width + x2_read) * src_channel + img_offset;

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
    dst[(dst_y * dst_width + dst_x) * dst_channel] = out;
    // c = 1
    src_reg = (int)src[idx11 + 1];
    out = (x2 - src_x) * (y2 - src_y) * src_reg;
    src_reg = (int)src[idx12 + 1];
    out += src_reg * (src_x - x1) * (y2 - src_y);
    src_reg = (int)src[idx21 + 1];
    out += src_reg * (x2 - src_x) * (src_y - y1);
    src_reg = (int)src[idx22 + 1];
    out += src_reg * (src_x - x1) * (src_y - y1);
    dst[(dst_y * dst_width + dst_x) * dst_channel + 1] = out;
    // c = 2
    src_reg = (int)src[idx11 + 2];
    out = (x2 - src_x) * (y2 - src_y) * src_reg;
    src_reg = (int)src[idx12 + 2];
    out += src_reg * (src_x - x1) * (y2 - src_y);
    src_reg = (int)src[idx21 + 2];
    out += src_reg * (x2 - src_x) * (src_y - y1);
    src_reg = (int)src[idx22 + 2];
    out += src_reg * (src_x - x1) * (src_y - y1);
    dst[(dst_y * dst_width + dst_x) * dst_channel + 2] = out;
  }
}

__global__ void lut_stitch_single_kernel(const unsigned char *const src, unsigned char *const dst, int width,
                                         int height, int dst_width, int dst_height, const float *const geo_lut,
                                         const unsigned char *const idx_lut, int idx) {
  // 2D Index of current thread
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  const int geo_channel = 2;
  const int src_channel = 3;
  const int dst_channel = 3;
  if (dst_x < dst_width && dst_y < dst_height && idx_lut[(dst_y * dst_width + dst_x)] == idx) {
    const float src_x = geo_lut[(dst_y * dst_width + dst_x) * geo_channel];
    const float src_y = geo_lut[(dst_y * dst_width + dst_x) * geo_channel + 1];
    const int src_x_i = int(src_x);
    const int src_y_i = int(src_y);
    if (src_x_i < width && src_y_i < height) {
      dst[(dst_y * dst_width + dst_x) * dst_channel] = src[(src_y_i * width + src_x_i) * src_channel];
      dst[(dst_y * dst_width + dst_x) * dst_channel + 1] = src[(src_y_i * width + src_x_i) * src_channel + 1];
      dst[(dst_y * dst_width + dst_x) * dst_channel + 2] = src[(src_y_i * width + src_x_i) * src_channel + 2];
    }
  }
}

void lut_stitch(const unsigned char *const src, unsigned char *const dst, int width, int height, int dst_width,
                int dst_height, const float *const geo_lut, const unsigned char *const idx_lut) {
  // Specify a reasonable block size
  // block.x * block.y should not exceed 1024
  const dim3 block(32, 32);
  // Calculate grid size to cover the whole image
  const dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);

  lut_stitch_kernel<<<grid, block>>>(src, dst, width, height, dst_width, dst_height, geo_lut, idx_lut);
}

void lut_stitch_single(const unsigned char *const src, unsigned char *const dst, int width, int height, int dst_width,
                       int dst_height, const float *const geo_lut, const unsigned char *const idx_lut, int idx) {
  // Specify a reasonable block size
  // block.x * block.y should not exceed 1024
  const dim3 block(32, 32);
  // Calculate grid size to cover the whole image
  const dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);

  lut_stitch_single_kernel<<<grid, block>>>(src, dst, width, height, dst_width, dst_height, geo_lut, idx_lut, idx);
}

}  // namespace svs