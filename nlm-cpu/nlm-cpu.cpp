// Non-Local Means Denoising on CPU
// ZHUO Xu @ https://github.com/z0gSh1u/nlm-cuda

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <climits>
#include <string>
#include <cstdlib>
#include <ctime>
#include <rapidjson/document.h>

#include "../misc/ProgressBar.hpp"
#include "../misc/Utils.hpp"

using std::cout;
using std::endl;
using std::string;

// [r0, r1), [c0, c1) determines a patch in image.
struct Patch {
  int r0, r1, c0, c1;
  Patch(int _r0, int _r1, int _c0, int _c1) { r0 = _r0, r1 = _r1, c0 = _c0, c1 = _c1; }
};

// ========= [Parameters to read or calculate] =========
// Float-type image during processing.
float *src; // source image, input image
float *pad; // padded source image
float *dst; // destination image, output image
// Height and width source and padded souce image.
// Output image has same size as source image.
int srcH, srcW, padH, padW;
// IO path.
string srcPath;
string dstPath;
// Parameters of NLM.
int PatchRadius;  // patch
int WindowRadius; // search window
float ParamH;     // parameter h (sigma)
// Progressbar.
ProgressBar progressBar;
float ProgressPerRow;
// ========= [Parameters to read or calculate] =========

// Load configuration file.
void loadConfig(string configPath) {
  rapidjson::Document d;
  d.Parse(readFileText(configPath).c_str());
  srcW = d["SourceImageWidth"].GetInt();
  srcH = d["SourceImageHeight"].GetInt();
  PatchRadius = d["PatchSize"].GetInt() >> 1;
  WindowRadius = d["WindowSize"].GetInt() >> 1;
  ParamH = d["ParamH"].GetFloat();
  srcPath = d["SourceImagePath"].GetString();
  dstPath = d["OutputImagePath"].GetString();
  padH = srcH + PatchRadius * 2;
  padW = srcW + PatchRadius * 2;
  ProgressPerRow = 1.0 / srcH;
  cout << "\033[32m [NLM-CPU] Config file loaded. \033[0m" << endl;
}

// Get pixel INDEX according to row and col.
#define INDEX(r, c, W) ((r) * (W) + (c))

// Pad image symmetrically (mirror) with PatchRadius.
void padSourceImageSymmetric() {
  int srcIdx, dstIdx, i, j, pr = PatchRadius;
  // Original Viewport.
  for (i = 0; i < srcH; i++) {
    for (j = 0; j < srcW; j++) {
      pad[INDEX(i + pr, j + pr, padW)] = src[INDEX(i, j, srcW)];
    }
  }
  // Four Wings.
  dstIdx = pr, srcIdx = 0;
  while (srcIdx++, dstIdx--) {
    for (i = 0; i < srcH; i++) {
      pad[INDEX(i + pr, dstIdx, padW)] = src[INDEX(i, srcIdx, srcW)];
      pad[INDEX(i + pr, dstIdx + srcW + 2 * srcIdx - 1, padW)] = src[INDEX(i, srcW - srcIdx, srcW)];
    }
    for (j = 0; j < srcW; j++) {
      pad[INDEX(dstIdx, j + pr, padW)] = src[INDEX(srcIdx, j, srcW)];
      pad[INDEX(dstIdx + srcH + 2 * srcIdx - 1, j + pr, padW)] = src[INDEX(srcH - srcIdx, j, srcW)];
    }
  }
  // Corners. Pad according to rows of wing.
  dstIdx = srcIdx = pr;
  while (srcIdx++, dstIdx--) {
    for (i = 0; i < pr; i++) {
      pad[INDEX(i, dstIdx, padW)] = pad[INDEX(i, srcIdx, padW)];
      pad[INDEX(i, srcIdx + srcW - 1, padW)] = pad[INDEX(i, dstIdx + srcW, padW)];
      pad[INDEX(i + srcH + pr, dstIdx, padW)] = pad[INDEX(i + srcH + pr, srcIdx, padW)];
      pad[INDEX(i + srcH + pr, srcIdx + srcW - 1, padW)] =
          pad[INDEX(i + srcH + pr, dstIdx + srcW, padW)];
    }
  }
}

// Calculate L2 Norm distance
float calcPatchDistance(Patch w1, Patch w2) {
  // L2 Norm
  float l2 = 0.0;
  int rowRange1 = w1.r1 - w1.r0, rowRange2 = w2.r1 - w2.r0, colRange1 = w1.c1 - w1.c0,
      colRange2 = w1.c1 - w1.c0;
  ASSERT(rowRange1 == rowRange2 && colRange1 == colRange2, "Windows not of same size.");
  for (int i = 0; i < rowRange1; i++) {
    for (int j = 0; j < colRange1; j++) {
      l2 += pow(pad[INDEX(w1.r0 + i, w1.c0 + j, padW)] - pad[INDEX(w2.r0 + i, w2.c0 + j, padW)], 2);
    }
  }
  l2 /= pow(PatchRadius * 2 + 1, 2); // normalize
  return l2;
}

// Non-local Means denoise.
void NLMDenoise(int pr, int wr) {
  float h2 = pow(ParamH, 2);
  // iterate dst image (reverse mapping) to ensure every pixel has denoised value
  // (i, j) is (row, col) of dst image
  for (int i = 0; i < srcH; i++) {
    progressBar.update(ProgressPerRow);
    for (int j = 0; j < srcW; j++) {
      // corresponding position in pad
      int r = i + pr, c = j + pr;

      // source patch
      Patch SourcePatch(r - pr, r + pr, c - pr, c + pr);
      // search window
      Patch SearchWindow(std::max(r - wr, pr), std::min(r + wr, srcH + pr), std::max(c - wr, pr),
                         std::min(c + wr, srcW + pr));

      float sumWeight = 0.0,  // sum of weight of current two patches' matching
          maxWeight = 0.0,    // maximum weight during weight calculation
          averageValue = 0.0, // average value after weighting to replace the noisy pixel
          distance,           // distance of two patches
          weight;             // weight of current matching

      // iterate the search window
      for (int k = SearchWindow.r0; k < SearchWindow.r1; k++) {
        for (int l = SearchWindow.c0; l < SearchWindow.c1; l++) {
          // compare patch
          Patch ComparePatch(k - pr, k + pr, l - pr, l + pr);
          // calculate distance
          if (k == r && l == c) { // center pixel is a special case, see outside
            continue;
          }
          distance = calcPatchDistance(SourcePatch, ComparePatch); // use L2 Norm as distance
          weight = exp(-distance / h2);                // w = exp(-d/h^2)
          sumWeight += weight;
          maxWeight = std::max(maxWeight, weight);
          averageValue += weight * pad[INDEX(k, l, padW)];
        }
      }

      // Deal with center pixel, use maximum weight for it instead of 1 when distance is 0.
      // Otherwise, 1 is too big to other weights, that noise is preserved.
      sumWeight += maxWeight;
      averageValue += maxWeight * pad[INDEX(r, c, padW)];
      dst[INDEX(i, j, srcW)] = averageValue / sumWeight;
    }
  }
}

int main(int argc, char *argv[]) {
  cout << "\033[46;30m [NLM-CPU] Running... \033[0m" << endl;
  // Load config file.
  ASSERT(argc >= 2, "[Usage] ./nlm-cpu path/to/config.json");
  loadConfig(string(argv[1]));

  // Read source image (8-bit, single channel).
  uint8 *src8bit = new uint8[srcH * srcW];
  readFileBinary(srcPath, 1, srcH * srcW, src8bit);
  src = new float[srcH * srcW];
  // Convert to float type for later computation.
  // no /255 here, since the result is better
  uint8ImageToFloat(src8bit, src, srcH, srcW, false);
  delete src8bit;

  // Pad source image so that every pixel can be denoised.
  pad = new float[padH * padW];
  std::fill_n(pad, padH * padW, 0.0);
  padSourceImageSymmetric();
  delete src;

  // Initialize destination image.
  dst = new float[srcH * srcW];
  std::fill_n(dst, srcH * srcW, 0.0);

  // NLM Denoise.
  time_t tic = time(NULL);
  NLMDenoise(PatchRadius, WindowRadius);
  time_t toc = time(NULL);
  cout << endl << " [NLM-CPU] Time elapsed: " << toc - tic << " secs." << endl;

  // Save.
  uint8 *dst8bit = new uint8[srcH * srcW];
  floatImageToUint8(dst, dst8bit, srcH, srcW);
  writeFileBinary(dst8bit, sizeof(uint8), srcH * srcW, dstPath);
  cout << "\033[32m [NLM-CPU] Done. \033[0m" << endl;
  
  return 0;
}
