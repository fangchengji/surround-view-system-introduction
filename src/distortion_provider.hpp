#pragma once

#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>

#include "file.h"

namespace svs {

class DistortionProvider {
public:
  DistortionProvider(const std::string &fp) {
    initialized_ = false;
    fp_ = fp;

    std::vector<std::vector<std::string>> tables;
    if (!svs::read_table_from_file(fp, tables)) {
      printf("DistortionProvider read table from file %s failed!!\n", fp.c_str());
      return;
    }
    if (!parse_table(tables)) {
      printf("DistortionProvider parse table failed!!\n");
      return;
    }

    printf("DistortionProvider init success!!\n");
    initialized_ = true;
  }

  virtual ~DistortionProvider() = default;

  bool get(const float &ref_radius, float& ref_dist) const {
    if (!initialized_) {
      printf("Please init DistortionProvider before use it!!\n");
      return false;
    }

    if (ref_radius > paraxial_heights_.back()) {
      printf("The input ref_radius %f is out of range!!\n", ref_radius);
      return false;
    }

    if (ref_radius <= paraxial_heights_.front()) {
      ref_dist = ref_radius;
      return true;
    }
    // look up table and linear interpolation
    auto low = std::lower_bound(paraxial_heights_.begin(), paraxial_heights_.end(), ref_radius);
    int idx = low - paraxial_heights_.begin();
    assert(idx > 0);
    // float left_ref = paraxial_heights_[idx - 1];
    // float right_ref = paraxial_heights_[idx];
    float ratio = (ref_radius - paraxial_heights_[idx - 1]) / (paraxial_heights_[idx] - paraxial_heights_[idx - 1]);
    // float left_dist = distortions_[idx - 1];
    // float right_dist = distortions_[idx];
    ref_dist = (distortions_[idx] - distortions_[idx - 1]) * ratio + distortions_[idx - 1];
    // res = ref_radius * (1 + target_dist);

    return true;
  }

private:
  bool parse_table(const std::vector<std::vector<std::string>> &tables) {
    if (tables.size() != len_ + 1) {
      return false;
    }

    distortions_.clear();
    paraxial_heights_.clear();
    for (int i = 1; i < valid_len_ + 1; ++i) {
      float paraxial_height = std::stof(tables[i][1]);   //paraxial height
      float dist = std::stof(tables[i][3]) * 0.01;      // distortion ratio

      paraxial_heights_.push_back(paraxial_height);
      distortions_.push_back(dist);
    }

    return distortions_.size() == valid_len_;
  }

private:
  std::string fp_;
  bool initialized_;

  const int len_ = 1000;
  const int valid_len_ = 900;
  // const float max_radius_ = 369.0f;     // max radius, unit mm
  std::vector<float> distortions_;
  std::vector<float> paraxial_heights_;
};

} // namespace svs