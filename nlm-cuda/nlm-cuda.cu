// Non-Local Means Denoising on CUDA GPU
// ZHUO Xu @ https://github.com/z0gSh1u/nlm-cuda

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

// To support milliseconds timing.
#ifdef _WIN32
#include <Windows.h>
#endif

#include "../misc/Utils.hpp"

using std::cout;
using std::endl;
using std::string;

#define MIN(a, b) ((a) > (b)) ? (b) : (a)
#define MAX(a, b) ((a) > (b)) ? (a) : (b)

// Get pixel INDEX according to row and col.
#define INDEX(r, c, W) ((r) * (W) + (c))

// ========= [Parameters to read or calculate] =========
// Float-type image during processing.
// In GPU.
float *src; // source image, input image
float *pad; // padded source image
float *dst; // destination image, output image
// In CPU. `h_` means host.
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

// Pad image symmetrically (mirror) with PatchRadius.
__host__ void padSourceImageSymmetric(float *pad, float *src) {
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

// Non-local Means denoise.
__global__ void NLMDenoise(float *pad, int PatchRadius, int WindowRadius, int padH, int padW,
                           int srcH, int srcW, float ParamH, float *dst) {
  // iterate dst image (reverse mapping) to ensure every pixel has denoised value
  // (i, j) is (row, col) of dst image
  int i = (blockIdx.x * blockDim.x) + threadIdx.x, j = (blockIdx.y * blockDim.y) + threadIdx.y;
  // check, since 1-to-1 mapping is impossible for large images
  if (i >= srcH || j >= srcW) {
    return;
  }

  // corresponding position in pad
  int pr = PatchRadius, wr = WindowRadius;
  int r = i + pr, c = j + pr;

  // ====== [EXPAND] [struct Patch] ======
  // source patch
  int SourcePatchR0 = r - pr, SourcePatchR1 = r + pr, SourcePatchC0 = c - pr,
      SourcePatchC1 = c + pr;
  // search window
  int SearchWindowR0 = MAX(r - wr, pr), SearchWindowR1 = MIN(r + wr, srcH + pr),
      SearchWindowC0 = MAX(c - wr, pr), SearchWindowC1 = MIN(c + wr, srcW + pr);
  // ====== [EXPAND] [struct Patch] ======

  float sumWeight = 0.0,  // sum of weight of current two patches' matching
      maxWeight = 0.0,    // maximum weight during weight calculation
      averageValue = 0.0, // average value after weighting to replace the noisy pixel
      distance,           // distance of two patches
      weight;             // weight of current matching

  // iterate the search window
  for (int k = SearchWindowR0; k < SearchWindowR1; k++) {
    for (int l = SearchWindowC0; l < SearchWindowC1; l++) {
      // ====== [EXPAND] [struct Patch] ======
      // compare patch
      int ComparePatchR0 = k - pr, ComparePatchR1 = k + pr, ComparePatchC0 = l - pr,
          ComparePatchC1 = l + pr;
      // ====== [EXPAND] [struct Patch] ======

      // calculate distance
      if (k == r && l == c) { // center pixel is a special case, see outside
        continue;
      }

      // ====== [EXPAND] [calcPatchDistance] ======
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
      // ====== [EXPAND] [calcPatchDistance] ======

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

// Get current time in ms.
__host__ void getCurrentTimeMS(bool &available, time_t &value) {
#ifdef _WIN32
  available = true;
  value = GetTickCount();
#else
  available = false;
#endif
}

int main(int argc, char *argv[]) {
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
  // Dont normalize to [0, 1] gets better result.
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
  delete h_pad;

  // Initialize destination image.
  cudaMalloc(&dst, srcH * srcW * sizeof(float));

  // Non-Local Means Denoise
  dim3 Block(16, 16); // use 64 threads concurrent
  dim3 Grid((padH + 15) / Block.x, (padW + 15) / Block.y);
  bool timeAvailable;
  time_t tic, toc;

  fprintf(stderr, "s1 : %s\n", cudaGetErrorString(cudaGetLastError()));

  getCurrentTimeMS(timeAvailable, tic);
  NLMDenoise<<<Grid, Block>>>(pad, PatchRadius, WindowRadius, padH, padW, srcH, srcW, ParamH, dst);
  fprintf(stderr, "s2 : %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaDeviceSynchronize();
  getCurrentTimeMS(timeAvailable, toc);

  cudaFree(pad);

  cout << " [NLM-CUDA] Time elapsed: ";
  if (timeAvailable) {
    cout << toc - tic << " ms." << endl;
  } else {
    cout << "Unavailable." << endl;
  }

  // Fetch dst to CPU.
  h_dst = new float[srcH * srcW];

  cudaError_t err;

  err = cudaMemcpy(h_dst, dst, srcH * srcW * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy : %s\n", cudaGetErrorString(cudaGetLastError()));
  }

  // Save.
  uint8 *dst8bit = new uint8[srcH * srcW];
  floatImageToUint8(h_dst, dst8bit, srcH, srcW);
  writeFileBinary(dst8bit, sizeof(uint8), srcH * srcW, dstPath);
  cout << "\033[32m [NLM-CUDA] Done. \033[0m" << endl;

  return 0;
}