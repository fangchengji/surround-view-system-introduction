#pragma once

#include <iostream>

#include <cerrno>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace svs {

inline void split_string(std::string s, std::string delimiter, std::vector<std::string> &res) {
  size_t pos = 0;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    res.push_back(s.substr(0, pos));
    s.erase(0, pos + delimiter.length());
  }
  res.push_back(s);
}

inline bool read_table_from_file(const std::string &fp, std::vector<std::vector<std::string>> &tables) {
  std::ifstream in(fp);
  if (!in.is_open()) {
    std::cout << "Error opening file " << fp << "\n";
    return false;
  }
  tables.clear();

  std::string delimiter = "\t";
  std::string s;
  while (getline(in, s)) {
    std::vector<std::string> words;
    split_string(s, delimiter, words);

    tables.push_back(words);
  }

  in.close();
  return true;
}

}  // namespace svs
