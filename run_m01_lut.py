import pandas as pd
from PIL import Image
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2
from numba import jit

CORRECTW = 1350
CORRECTH = 720

SIDE_DISTANCE = 8000
FRONT_DISTANCE = 6000
REAR_DISTANCE = 8000
W_OFFSET = 50

VEHICLE_L = 5023
VEHICLE_W = 1960
HALF_VEHICLE_L = VEHICLE_L / 2
HALF_VEHICLE_W = VEHICLE_W / 2
AVM_PIXEL_SIZE = 15  # 15mm

# -----------------------distortion--------------
file_name = "4066_distortion.xls"
xl_file = pd.ExcelFile(file_name)
dfs = {
    sheet_name: xl_file.parse(sheet_name)
    for sheet_name in xl_file.sheet_names
}
# print(dfs)

distortion = dfs['distortion'].values
ref_table = distortion[:900, 1]
real_table = distortion[:900, 2]
dist_table = distortion[:900, 3] * 0.01


@jit(nopython=True)
def binary_search(arr, l, r, x):
    while l < r:
        mid = (l + r) // 2
        if x > arr[mid]:
            l = mid + 1
        else:
            r = mid
    return r


@jit(nopython=True)
def find_real_r(ref_r):
    idx = binary_search(ref_table, 0, len(ref_table) - 1, ref_r)
    left_ref = ref_table[idx - 1]
    right_ref = ref_table[idx]
    ratio = (ref_r - left_ref) / (right_ref - left_ref)
    left_dist = dist_table[idx - 1]
    right_dist = dist_table[idx]
    target_dist = (right_dist - left_dist) * ratio + left_dist
    return ref_r * (1 + target_dist)


@jit(nopython=True)
def bilinear_interpolation(x, y, img):
    h, w, _ = img.shape
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = min(w - 1, x0 + 1)
    y1 = min(h - 1, y0 + 1)

    v0 = (x1 - x) * img[y0, x0, :] + (x - x0) * img[y0, x1, :]
    v1 = (x1 - x) * img[y1, x0, :] + (x - x0) * img[y1, x1, :]
    pixel = (y1 - y) * v0 + (y - y0) * v1
    return pixel


def fisheye_undistort(img_size, new_pixel_size):
    # "fish eye distortion"
    pixel_size = 0.0042
    new_pixel_size = new_pixel_size
    height_in, width_in = img_size
    width_out = CORRECTW
    height_out = CORRECTH
    lut_out = np.zeros((height_out, width_out, 2), dtype=float)
    # im_out = Image.new("RGB", (width_out, height_out))
    # im_out = np.zeros((height_out, width_out, 3), dtype=int)
    for i in range(width_out):
        for j in range(height_out):
            #offset to center
            x = i - width_out / 2 + 0.5
            y = j - height_out / 2 + 0.5
            r = math.sqrt(x * x + y * y)  # image height
            ref_r = r * new_pixel_size
            real_r = find_real_r(ref_r)
            origin_r = real_r / pixel_size
            # print(ref_r, real_r)
            if ref_r < 0.00001:
                k = 1
            else:
                k = origin_r / r

            src_x = x * k
            src_y = y * k
            src_x = src_x + width_in / 2
            src_y = src_y + height_in / 2
            if src_x >= 0 and src_x < width_in and src_y >= 0 and src_y < height_in:
                lut_out[j, i] = (src_x, src_y)
    return lut_out


def project(lut, project_matrix, x_start, x_stop, x_step, y_start, y_stop,
            y_step):
    height_in, width_in, _ = lut.shape
    x = np.arange(start=x_start, stop=x_stop, step=x_step)
    y = np.arange(start=y_start, stop=y_stop, step=y_step)

    width_out = len(x)
    height_out = len(y)
    lut_out = np.zeros((height_out, width_out, 2), dtype=float)
    for i in range(width_out):
        for j in range(height_out):
            src = np.dot(project_matrix, np.array([x[i], y[j], 1]).transpose())

            src_x = src[0] / src[2]
            src_y = src[1] / src[2]
            if src_x >= 0 and src_x < width_in and src_y >= 0 and src_y < height_in:
                lut_pt = bilinear_interpolation(src_x, src_y, lut)
                lut_out[j, i] = lut_pt
    return lut_out


def project_front(lut_undist):
    dtoimgh = [
        -7.7701834498851527e-02, -3.1444351316730390e-01,
        6.7667562256659016e+02, -7.5324831312253700e-04,
        -1.3769149770759789e-01, 2.4015947063634835e+02,
        -9.9737542400689613e-06, -4.6841167493419367e-04, 1.
    ]
    dtoimgh = np.array(dtoimgh).reshape(3, 3)
    return project(lut_undist,
                   dtoimgh,
                   x_start=-(SIDE_DISTANCE + HALF_VEHICLE_W) + W_OFFSET,
                   x_stop=SIDE_DISTANCE + HALF_VEHICLE_W + W_OFFSET,
                   x_step=AVM_PIXEL_SIZE,
                   y_start=FRONT_DISTANCE + HALF_VEHICLE_L,
                   y_stop=HALF_VEHICLE_L,
                   y_step=-AVM_PIXEL_SIZE)


def project_rear(lut_undist):
    dtoimgh = [
        9.6078919872715829e-02, 3.7984478412119704e-01, 6.7176811959932411e+02,
        -1.0273559647656078e-04, 1.5443579517981729e-01,
        1.6100076711089713e+02, 7.2171039227399751e-06, 5.6609934109434605e-04,
        1.
    ]
    dtoimgh = np.array(dtoimgh).reshape(3, 3)
    return project(lut_undist,
                   dtoimgh,
                   x_start=-(SIDE_DISTANCE + HALF_VEHICLE_W) + W_OFFSET,
                   x_stop=SIDE_DISTANCE + HALF_VEHICLE_W + W_OFFSET,
                   x_step=AVM_PIXEL_SIZE,
                   y_start=-HALF_VEHICLE_L,
                   y_stop=-(HALF_VEHICLE_L + REAR_DISTANCE),
                   y_step=-AVM_PIXEL_SIZE)


def project_left(lut_undist):
    dtoimgh = [
        -6.4452076979484758e+01, 2.0309867965189003e+01,
        -1.1337185738536151e+04, -2.3134388302146341e+01,
        6.8264561066865570e-01, 2.8003769307996728e+04,
        -9.6754821267343902e-02, 3.9264130206281816e-03, 1.
    ]
    dtoimgh = np.array(dtoimgh).reshape(3, 3)
    return project(lut_undist,
                   dtoimgh,
                   x_start=-(SIDE_DISTANCE + HALF_VEHICLE_W) + W_OFFSET,
                   x_stop=-HALF_VEHICLE_W + W_OFFSET,
                   x_step=AVM_PIXEL_SIZE,
                   y_start=HALF_VEHICLE_L + FRONT_DISTANCE,
                   y_stop=-(HALF_VEHICLE_L + REAR_DISTANCE),
                   y_step=-AVM_PIXEL_SIZE)


def project_right(lut_undist):
    dtoimgh = [
        -2.3861579738693353e+00, 5.9431137723140892e-01,
        2.6688519339014096e+02, -9.3666348357416507e-01,
        -7.1126631549249227e-03, -5.6139650014375593e+02,
        -3.5400364043628061e-03, 1.3720983106096465e-05, 1.
    ]
    dtoimgh = np.array(dtoimgh).reshape(3, 3)
    return project(lut_undist,
                   dtoimgh,
                   x_start=HALF_VEHICLE_W + W_OFFSET,
                   x_stop=SIDE_DISTANCE + HALF_VEHICLE_W + W_OFFSET,
                   x_step=AVM_PIXEL_SIZE,
                   y_start=HALF_VEHICLE_L + FRONT_DISTANCE,
                   y_stop=-(HALF_VEHICLE_L + REAR_DISTANCE),
                   y_step=-AVM_PIXEL_SIZE)


def generate_lut(luts):
    front = luts[0]
    rear = luts[1]
    left = luts[2]
    right = luts[3]
    front_h, front_w, _ = front.shape
    left_h, left_w, _ = left.shape
    rear_h, rear_w, _ = rear.shape
    right_h, right_w, _ = right.shape

    shift_w = front_w - right_w
    shift_h = left_h - rear_h

    width_out = front.shape[1]
    height_out = left.shape[0]

    lut_out = np.zeros((height_out, width_out, 2), dtype=float)
    dir_out = np.zeros((height_out, width_out), dtype=int)
    for i in range(width_out):
        for j in range(height_out):
            # top part
            if j < front_h:
                # 18 degree
                if i < left_w and (front_h - 1 - j) <= (left_w - 1 - i) * 3:
                    lut_out[j, i] = left[j, i]
                    dir_out[j, i] = 2
                elif i >= shift_w and (front_h - 1 - j) <= (i - shift_w) * 3:
                    lut_out[j, i] = right[j, i - shift_w]
                    dir_out[j, i] = 3
                else:
                    lut_out[j, i] = front[j, i]
                    dir_out[j, i] = 0
            # middle part
            elif j >= front_h and j < shift_h:
                if i < left_w:
                    lut_out[j, i] = left[j, i]
                    dir_out[j, i] = 2
                elif i >= shift_w:
                    lut_out[j, i] = right[j, i - shift_w]
                    dir_out[j, i] = 3
            # bottom part
            else:
                # 71 degree
                if i < left_w and (left_w - 1 - i) > (j - shift_h) * 3:
                    lut_out[j, i] = left[j, i]
                    dir_out[j, i] = 2
                elif i >= shift_w and (i - shift_w) > (j - shift_h) * 3:
                    lut_out[j, i] = right[j, i - shift_w]
                    dir_out[j, i] = 3
                else:
                    lut_out[j, i] = rear[j - shift_h, i]
                    dir_out[j, i] = 1
    return lut_out, dir_out


def lut_convert_single(img, lut):
    height_in, width_in, _ = img.shape
    height_out, width_out, _ = lut.shape

    img_out = np.zeros((height_out, width_out, 3), dtype=int)
    for i in range(width_out):
        for j in range(height_out):
            src_x, src_y = lut[j, i, :]
            if src_x >= 0 and src_x < width_in and src_y >= 0 and src_y < height_in:
                pixel = bilinear_interpolation(src_x, src_y, img)
                img_out[j, i] = pixel
    return img_out


def lut_convert(imgs, lut, idx):
    height_in, width_in, _ = imgs[0].shape
    height_out, width_out, _ = lut.shape

    img_out = np.zeros((height_out, width_out, 3), dtype=int)
    for i in range(width_out):
        for j in range(height_out):
            src_x, src_y = lut[j, i, :]
            if src_x >= 0 and src_x < width_in and src_y >= 0 and src_y < height_in:
                pixel = bilinear_interpolation(src_x, src_y, imgs[idx[j, i]])
                img_out[j, i] = pixel
    return img_out


if __name__ == "__main__":
    base_dir = "chessboard/"
    ipm_imgs = []
    luts = []
    imgs = []

    # front ----------------------------
    direction = "front"
    input_name = base_dir + direction + ".bmp"
    undistort_name = base_dir + direction + "_undistort.jpg"
    ipm_name = base_dir + direction + "_ipm.jpg"
    im = cv2.imread(input_name)
    imgs.append(im)
    f = 1.29
    new_pixel_size = 0.00975  # 9.75um
    # new_pixel_size = 0.0042  # 8.4um
    height_in, width_in, _ = im.shape
    lut_undist = fisheye_undistort((height_in, width_in), new_pixel_size)
    lut_geo = eval("project_" + direction)(lut_undist)
    luts.append(lut_geo)

    # # rear ----------------------------
    direction = "rear"
    input_name = base_dir + direction + ".bmp"
    undistort_name = base_dir + direction + "_undistort.jpg"
    ipm_name = base_dir + direction + "_ipm.jpg"
    im = cv2.imread(input_name)
    imgs.append(im)
    height_in, width_in, _ = im.shape
    lut_undist = fisheye_undistort((height_in, width_in), new_pixel_size)

    lut_geo = eval("project_" + direction)(lut_undist)
    luts.append(lut_geo)

    # # left ----------------------------
    direction = "left"
    input_name = base_dir + direction + ".bmp"
    undistort_name = base_dir + direction + "_undistort.jpg"
    ipm_name = base_dir + direction + "_ipm.jpg"
    im = cv2.imread(input_name)
    imgs.append(im)
    height_in, width_in, _ = im.shape
    lut_undist = fisheye_undistort((height_in, width_in), new_pixel_size)

    lut_geo = eval("project_" + direction)(lut_undist)
    luts.append(lut_geo)

    # # right ----------------------------
    direction = "right"
    input_name = base_dir + direction + ".bmp"
    undistort_name = base_dir + direction + "_undistort.jpg"
    ipm_name = base_dir + direction + "_ipm.jpg"
    im = cv2.imread(input_name)
    imgs.append(im)
    height_in, width_in, _ = im.shape
    lut_undist = fisheye_undistort((height_in, width_in), new_pixel_size)
    lut_geo = eval("project_" + direction)(lut_undist)
    luts.append(lut_geo)

    # lut surround ------------------
    lut_final, dir_lut = generate_lut(luts)
    img_res = lut_convert(imgs, lut_final, dir_lut)
    surround_lut_name = base_dir + "surround_lut.jpg"
    cv2.imwrite(surround_lut_name, img_res)

    print("suround view stitching completely, save image to %s" %
          surround_lut_name)
