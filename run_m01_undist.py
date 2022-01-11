import pandas as pd
import math
import numpy as np
import cv2
import argparse
from glob import glob
from itertools import chain
import os

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

# -----------------------distortion table--------------
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


def binary_search(arr, l, r, x):
    while l < r:
        mid = (l + r) // 2
        if x > arr[mid]:
            l = mid + 1
        else:
            r = mid
    return r


def find_real_r(ref_r):
    idx = binary_search(ref_table, 0, len(ref_table) - 1, ref_r)
    left_ref = ref_table[idx - 1]
    right_ref = ref_table[idx]
    ratio = (ref_r - left_ref) / (right_ref - left_ref)
    left_dist = dist_table[idx - 1]
    right_dist = dist_table[idx]
    target_dist = (right_dist - left_dist) * ratio + left_dist
    return ref_r * (1 + target_dist)


def fisheye_undistort_lut(img_size, new_pixel_size):
    # "fish eye distortion"
    f = 1.29
    pixel_size = 0.0042
    new_pixel_size = new_pixel_size
    height_in, width_in = img_size
    width_out = CORRECTW
    height_out = CORRECTH
    lut_out = np.zeros((height_out, width_out, 2), dtype=np.float32)
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
            # theta = math.atan2(src_x * pixel_size, f)
            # src_x = f * theta
            # src_y = src_y * math.cos(theta)

            src_x = src_x + width_in / 2
            src_y = src_y + height_in / 2
            if src_x >= 0 and src_x < width_in and src_y >= 0 and src_y < height_in:
                lut_out[j, i] = (src_x, src_y)
    return lut_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input_dir",
                        default="avm",
                        help="input images dir")
    parser.add_argument("-o",
                        "--output_dir",
                        default=".",
                        help="output images dir")
    args = parser.parse_args()

    img_ends = [".bmp", ".jpg", ".png"]
    imgs_lits = list(
        chain(*[glob(args.input_dir + "/*" + img_end)
                for img_end in img_ends]))

    for i, img_path in enumerate(imgs_lits):
        img = cv2.imread(img_path)
        if i == 0:
            f = 1.29
            # new_pixel_size = 0.00975  # 9.75um
            new_pixel_size = 0.0042  # 4.2um
            height_in, width_in, _ = img.shape
            lut_undist = fisheye_undistort_lut((height_in, width_in),
                                               new_pixel_size)

        undist_img = cv2.remap(img, lut_undist, None, cv2.INTER_LINEAR)

        img_name = os.path.basename(img_path)
        cv2.imwrite(args.output_dir + "/" + img_name, undist_img)
