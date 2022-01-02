// Non-Local Means Denoising on CPU
// by z0gSh1u @ 2021-12
// https://github.com/z0gSh1u/nlm-cuda

// TODO Use another name.

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <climits>
#include <string>
#include <cstdlib>
#include <rapidjson/document.h>

#include "../misc/ProgressBar.hpp"
#include "../misc/utils.hpp"

using std::cout;
using std::endl;
using std::string;

// ========= [Parameters to read or calculate] =========
// float-type image during processing
float *src;
float *dst;
// height and width source and destination image
int H, W, dstH, dstW;
// IO
string srcPath;
string dstPath;
// parameters of NLM
int FilterRadius;
int WindowRadius;
float ParamH;
// progress bar
ProgressBar progressBar;
// ========= [Parameters to read or calculate] =========

#define CONFIG_PATH "../NLMConfig.json"
// Load configuration file.
void loadConfig(string configPath) {
  rapidjson::Document d;
  d.Parse(readFileText(configPath).c_str());
  W = d["SourceImageWidth"].GetInt();
  H = d["SourceImageHeight"].GetInt();
  FilterRadius = d["FilterSize"].GetInt() >> 1;
  WindowRadius = d["WindowSize"].GetInt() >> 1;
  ParamH = d["ParamH"].GetFloat();
  srcPath = d["SourceImagePath"].GetString();
  dstPath = d["OutputImagePath"].GetString();
  dstH = H + FilterRadius * 2;
  dstW = W + FilterRadius * 2;
}

// Get pixel index according to row and col.
int index(int r, int c, int W) { return r * W + c; }

// Pad image symmetrically (mirror) with FilterRadius.
void padImageSymmetric() {
  int srcIndex, dstIndex, i, j, fr = FilterRadius;

  // Original Viewport.
  for (i = 0; i < H; i++) {
    for (j = 0; j < W; j++) {
      dst[index(i + fr, j + fr, dstW)] = src[index(i, j, W)];
    }
  }

  // Four Wings.
  dstIndex = fr, srcIndex = 0;
  while (srcIndex++, dstIndex--) {
    for (i = 0; i < H; i++) {
      dst[index(i + fr, dstIndex, dstW)] = src[index(i, srcIndex, W)];
      dst[index(i + fr, dstIndex + W + 2 * srcIndex - 1, dstW)] =
          src[index(i, W - srcIndex, W)];
    }
    for (j = 0; j < W; j++) {
      dst[index(dstIndex, j + fr, dstW)] = src[index(srcIndex, j, W)];
      dst[index(dstIndex + H + 2 * srcIndex - 1, j + fr, dstW)] =
          src[index(H - srcIndex, j, W)];
    }
  }

  // Corners. Pad according to rows of wing.
  dstIndex = srcIndex = fr;
  while (srcIndex++, dstIndex--) {
    for (i = 0; i < fr; i++) {
      dst[index(i, dstIndex, dstW)] = dst[index(i, srcIndex, dstW)];
      dst[index(i, srcIndex + W - 1, dstW)] = dst[index(i, dstIndex + W, dstW)];
      dst[index(i + H + fr, dstIndex, dstW)] =
          dst[index(i + H + fr, srcIndex, dstW)];
      dst[index(i + H + fr, srcIndex + W - 1, dstW)] =
          dst[index(i + H + fr, dstIndex + W, dstW)];
    }
  }
}

void nlm(int ds, int Ds) {
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      // corresponding position in dst
      int r = i + ds, c = j + ds;
      // filter range: r+-ds, c+-ds

      // window range:
      int rmin = std::max(r - Ds, ds + 1), rmax = std::min(r + Ds, H + ds),
          smin = std::max(c - Ds, ds + 1), smax = std::min(c + Ds, W + ds);
    }
  }
}

int main() {
  // Load config file.
  loadConfig(CONFIG_PATH);

  // Read source image.
  uint8 *src8bit = new uint8[H * W];
  readFileBinary(srcPath, 1, H * W, src8bit);
  src = new float[H * W];
  uint8ImageToFloat(src8bit, src, H, W);
  delete src8bit;

  // Initialize destination image.
  dst = new float[dstH * dstW];
  std::fill_n(dst, dstH * dstW, 0.0);
  padImageSymmetric();

  // writeFileBinary(dst, sizeof(float), (H + ds * 2) * (W + ds * 2), dstPath);

  std::cout << "nlm-cpu Done." << std::endl;

  return 0;
}
