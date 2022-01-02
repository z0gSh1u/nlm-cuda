// Utilities for nlm-cuda.
// ZHUO Xu @ https://github.com/z0gSh1u/nlm-cuda

#pragma once

#ifndef NLM_UTILS_HPP
#define NLM_UTILS_HPP

#include <iostream>
#include <fstream>
#include <sstream>

typedef unsigned char uint8;

// Read file content as std::string.
string readFileText(string filePath) {
  std::ifstream s(filePath);
  std::stringstream buf;
  buf << s.rdbuf();
  return buf.str();
}

// Read binary file.
void readFileBinary(std::string filePath, int elementSize, int elementCount,
                    void *store) {
  FILE *fp;
  fopen_s(&fp, filePath.c_str(), "rb");
  fread((char *)store, elementSize, elementCount, fp);
  fclose(fp);
}

// Write binary file.
void writeFileBinary(void *ptr, int elementSize, int elementCount,
                     std::string filePath) {
  FILE *fp;
  fopen_s(&fp, filePath.c_str(), "wb");
  fwrite((char *)ptr, elementSize, elementCount, fp);
  fclose(fp);
}

// Image data type conversion, uint8 to float.
void uint8ImageToFloat(uint8 *src, float *dst, int H, int W) {
  for (int i = 0; i < H * W; i++) {
    dst[i] = src[i] / 255.0f;
  }
}

// Image data type conversion, float to uint8.
void floatImageToUint8(float *src, uint8 *dst, int H, int W) {
  float max_ = -(FLT_MAX - 1), min_ = FLT_MAX - 1;
  for (int i = 0; i < H * W; i++) {
    max_ = std::max(max_, src[i]);
    min_ = std::min(min_, src[i]);
  }
  for (int i = 0; i < H * W; i++) {
    dst[i] = uint8((src[i] - min_) / (max_ - min_) * 255);
  }
}

#endif