// Non-Local Means Denoising on CUDA GPU
// by z0gSh1u @ 2021-12
// https://github.com/z0gSh1u/nlm-cuda

#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <climits>
#include <string>
#include <cstdlib>
#include <rapidjson/document.h>
#include <ctime>

#include "../misc/Utils.hpp"

using std::cout;
using std::endl;
using std::string;

#define MIN(a, b) ((a) > (b)) ? (b) : (a)
#define MAX(a, b) ((a) > (b)) ? (a) : (b)

// ========= [Parameters to read or calculate] =========
// Float-type image during processing.
// In GPU.
float *src; // source image, input image
float *pad; // padded source image
float *dst; // destination image, output image
// In CPU.
float *h_src; // source image, input image
float *h_pad; // padded source image
float *h_dst; // destination image, output image
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
// ========= [Parameters to read or calculate] =========

// Load configuration file.
__host__ void loadConfig(string configPath) {
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
  cout << "\033[32m [NLM-CUDA] Config file loaded. \033[0m" << endl;
}

// Get pixel INDEX according to row and col.
#define INDEX(r, c, W) ((r) * (W) + (c))

// Pad image symmetrically (mirror) with PatchRadius.
__host__ void padSourceImageSymmetric(float *pad, float *src) {
  int srcIdx, dstIdx, i, j, fr = PatchRadius;
  // Original Viewport.
  for (i = 0; i < srcH; i++) {
    for (j = 0; j < srcW; j++) {
      pad[INDEX(i + fr, j + fr, padW)] = src[INDEX(i, j, srcW)];
    }
  }
  // Four Wings.
  dstIdx = fr, srcIdx = 0;
  while (srcIdx++, dstIdx--) {
    for (i = 0; i < srcH; i++) {
      pad[INDEX(i + fr, dstIdx, padW)] = src[INDEX(i, srcIdx, srcW)];
      pad[INDEX(i + fr, dstIdx + srcW + 2 * srcIdx - 1, padW)] = src[INDEX(i, srcW - srcIdx, srcW)];
    }
    for (j = 0; j < srcW; j++) {
      pad[INDEX(dstIdx, j + fr, padW)] = src[INDEX(srcIdx, j, srcW)];
      pad[INDEX(dstIdx + srcH + 2 * srcIdx - 1, j + fr, padW)] = src[INDEX(srcH - srcIdx, j, srcW)];
    }
  }
  // Corners. Pad according to rows of wing.
  dstIdx = srcIdx = fr;
  while (srcIdx++, dstIdx--) {
    for (i = 0; i < fr; i++) {
      pad[INDEX(i, dstIdx, padW)] = pad[INDEX(i, srcIdx, padW)];
      pad[INDEX(i, srcIdx + srcW - 1, padW)] = pad[INDEX(i, dstIdx + srcW, padW)];
      pad[INDEX(i + srcH + fr, dstIdx, padW)] = pad[INDEX(i + srcH + fr, srcIdx, padW)];
      pad[INDEX(i + srcH + fr, srcIdx + srcW - 1, padW)] =
          pad[INDEX(i + srcH + fr, dstIdx + srcW, padW)];
    }
  }
}

// Non-local Means denoise.
__global__ void NLMDenoise(int fr, int wr, int srcH, int srcW, int padH, int padW, int PatchRadius,
                           float ParamH, float *pad, float *dst) {
  // iterate dst image (reverse mapping) to ensure every pixel has denoised value
  int i = (blockIdx.x * blockDim.x) + threadIdx.x, j = (blockIdx.y * blockDim.y) + threadIdx.y;
  // corresponding position in pad
  int r = i + fr, c = j + fr;
  // source patch
  int SourcePatchR0 = r - fr, SourcePatchR1 = r + fr, SourcePatchC0 = c - fr,
      SourcePatchC1 = c + fr;
  // search window
  int SearchWindowR0 = MAX(r - wr, fr), SearchWindowR1 = MIN(r + wr, srcH + fr),
      SearchWindowC0 = MAX(c - wr, fr), SearchWindowC1 = MIN(c + wr, srcW + fr);

  float sumWeight = 0.0,  // sum of weight of current two patches' matching
      maxWeight = 0.0,    // maximum weight during weight calculation
      averageValue = 0.0, // average value after weighting to replace the noisy pixel
      distance,           // distance of two patches
      weight;             // weight of current matching
  // iterate the search window
  for (int k = SearchWindowR0; k < SearchWindowR1; k++) {
    for (int l = SearchWindowC0; l < SearchWindowC1; l++) {
      // compare patch
      int ComparePatchR0 = k - fr, ComparePatchR1 = k + fr, ComparePatchC0 = l - fr,
          ComparePatchC1 = l + fr;
      // calculate distance
      if (k == r && l == c) { // center pixel is a special case, see outside
        continue;
      }
      // ====== [Expansion] [calcPatchDistance] ======
      // L2 Norm
      float l2 = 0.0;
      for (int i = 0; i < SourcePatchR1 - SourcePatchR0; i++) {
        for (int j = 0; j < SourcePatchC1 - SourcePatchC0; j++) {
          l2 += powf(pad[INDEX(SourcePatchR0 + i, SourcePatchC0 + j, padW)] -
                         pad[INDEX(ComparePatchR0 + i, ComparePatchC0 + j, padW)],
                     2);
        }
      }
      l2 /= powf(PatchRadius * 2 + 1, 2);
      distance = l2;
      // ====== [Expansion] [calcPatchDistance] ======
      weight = expf(-distance / powf(ParamH, 2)); // w = exp(-d/h^2)
      sumWeight += weight;
      maxWeight = MAX(maxWeight, weight);
      averageValue += weight * pad[INDEX(k, l, padW)];
    }
  }

  // Deal with center pixel, use maximum weight for it instead of 1 when distance is 0.
  // Otherwise, 1 is too big to other weights, that noise is preserved.
  sumWeight += maxWeight;
  averageValue += maxWeight * pad[INDEX(r, c, padW)];
  dst[INDEX(i, j, srcW)] = averageValue / sumWeight;
}

#include <Windows.h>

int main() {
  cout << "\033[46;30m [NLM-GPU] Running... \033[0m" << endl;
  // Load config file.
  // ASSERT(argc >= 2, "[Usage] ./nlm-cpu path/to/config.json");
  // loadConfig(string(argv[1]));
  loadConfig("F:/nlm-cuda/NLMConfig.json");

  // Read source image (8-bit, single channel).
  uint8 *src8bit = new uint8[srcH * srcW];
  readFileBinary(srcPath, 1, srcH * srcW, src8bit);
  h_src = new float[srcH * srcW];
  // Convert to float type for later computation.
  uint8ImageToFloat(src8bit, h_src, srcH, srcW, false);
  delete src8bit;

  // Pad source image so that every pixel can be denoised.
  h_pad = new float[padH * padW];
  std::fill_n(h_pad, padH * padW, 0.0);
  padSourceImageSymmetric(h_pad, h_src);
  delete h_src;

  // Convery pad to GPU.
  cudaMalloc(&pad, padH * padW * sizeof(float));
  cudaMemcpy(pad, h_pad, padH * padW * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // Initialize destination image.
  cudaMalloc(&dst, srcH * srcW * sizeof(float));

  // Non-Local Means Denoise
  dim3 ThreadDim(8, 8);
  dim3 BlockDim(padH / ThreadDim.x, padW / ThreadDim.y);
  time_t tic = GetTickCount();
  NLMDenoise<<<BlockDim, ThreadDim>>>(PatchRadius, WindowRadius, srcH, srcW, padH, padW,
                                             PatchRadius, ParamH, pad, dst);
  cudaDeviceSynchronize();
  time_t toc = GetTickCount();
  cout << endl << " [NLM-CUDA] Time elapsed: " << toc - tic << " msecs." << endl;

  // Fetch dst to GPU.
  h_dst = new float[srcH * srcW];
  cudaMemcpy(h_dst, dst, srcH * srcW * sizeof(float), cudaMemcpyDeviceToHost);

  // Save.
  writeFileBinary(h_dst, sizeof(float), srcH * srcW, dstPath);
  cout << "\033[32m [NLM-CUDA] Done. \033[0m" << endl;

  return 0;
}