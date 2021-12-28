// Non-Local Means Denoising on CPU
// by z0gSh1u @ 2021-12
// https://github.com/z0gSh1u/nlm-cuda

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <climits>
#include <string>
#include <cstdlib>

typedef unsigned char uint8;
using std::string;

int H, W;
uint8 *_src;
uint8 *_dst;
float *src;
float *dst;

float progress = 0.0;
float progressPerRow = 0.0;

// Implement of a simple progressbar.
// @see https://stackoverflow.com/questions/14539867/
void updateProgressBar(float progress, int barWidth = 40) {
  std::cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; i++) {
    std::cout << (i < pos ? "=" : i == pos ? ">" : " ");
  }
  std::cout << "] " << int(progress * 100.0) << " %\r";
  std::cout.flush();
}

void uint8ImageToFloat(uint8 *src, float *dst) {
  for (int i = 0; i < H * W; i++) {
    dst[i] = src[i] / 255.0f;
  }
}

void floatImageToUint8(float *src, uint8 *dst) {
  float max_ = -(FLT_MAX - 1), min_ = FLT_MAX - 1;
  for (int i = 0; i < H * W; i++) {
    max_ = std::max(max_, src[i]);
    min_ = std::min(min_, src[i]);
  }
  for (int i = 0; i < H * W; i++) {
    dst[i] = uint8((src[i] - min_) / (max_ - min_) * 255);
  }
}

int getIndex(int r, int c) { return r * W + c; }

float calcWeightUnNormalized(int blockStartRow, int blockStartCol,
                             int filterSize, int x, int y, float paramH) {
  int filterRadius = filterSize / 2;
  x = x - filterRadius, y = y - filterRadius;
  float l2Norm = 0.0;
  for (int i = 0; i < filterSize; i++) {
    for (int j = 0; j < filterSize; j++) {
      l2Norm += pow(src[getIndex(blockStartRow + i, blockStartCol + j)] -
                        src[getIndex(x + i, y + j)],
                    2);
    }
  }
  l2Norm /= (filterSize * filterSize);
  return exp(-l2Norm / (paramH * paramH));
}

void NLMDenoise(int filterSize, int windowSize, double paramH) {
  int filterRadius = filterSize / 2;
  int windowRadius = windowSize / 2;

  float weightedPixels[512];
  int weightedPixelCount = 0;

  for (int i = filterRadius; i < H; i++) { // TODO Remove filterRadius?
    updateProgressBar(progress = progress + progressPerRow);

    for (int j = filterRadius; j < W; j++) {
      // determine search window
      int windowStartRow = std::max(i - windowRadius, 0),
          windowStartCol = std::max(j - windowRadius, 0),
          windowEndRow = std::min(i + windowRadius, H - 1),
          windowEndCol = std::min(j + windowRadius, W - 1);
      // block matching
      float totalWeight = 0.0;
      weightedPixelCount = 0;
      for (int k = windowStartRow; k < windowEndRow - filterSize; k++) {
        for (int l = windowStartCol; l < windowEndCol - filterSize; l++) {
          float unNormalizedWeight =
              calcWeightUnNormalized(k, l, filterSize, i, j, paramH);
          weightedPixels[weightedPixelCount++] =
              unNormalizedWeight *
              src[getIndex(k + filterRadius, l + filterRadius)];
          totalWeight += unNormalizedWeight;
        }
      }
      // normalize
      for (int m = 0; m < weightedPixelCount; m++) {
        weightedPixels[m] /= totalWeight;
      }
      dst[getIndex(i, j)] = std::accumulate(
          weightedPixels, weightedPixels + weightedPixelCount, 0.0f);
    }
  }
}

void readFileBinary(string filePath, int elementSize, int elementCount,
                    void *store) {
  FILE *fp;
  fopen_s(&fp, filePath.c_str(), "rb");
  fread((char *)store, elementSize, elementCount, fp);
  fclose(fp);
}

void writeFileBinary(void *ptr, int elementSize, int elementCount,
                     string filePath) {
  FILE *fp;
  fopen_s(&fp, filePath.c_str(), "wb");
  fwrite((char *)ptr, elementSize, elementCount, fp);
  fclose(fp);
}

int main() {
  H = W = 512;
  _src = new uint8[H * W];
  _dst = new uint8[H * W];
  src = new float[H * W];
  dst = new float[H * W];
  progressPerRow = 1.0f / H;

  string imgPath = "F:/nlm-cuda/experiment/noisy_image/lena512_gaussian.raw";
  readFileBinary(imgPath, 1, H * W, _src);
  uint8ImageToFloat(_src, src);

  std::fill_n(dst, H * W, 0.0);
  NLMDenoise(3, 14, 20.0);

  string resPath = "F:/nlm-cuda/temp/res.raw";
  // floatImageToUint8(dst, _dst);
  writeFileBinary(dst, sizeof(float), H * W, resPath);

  std::cout << "Done." << std::endl;

  return 0;
}
