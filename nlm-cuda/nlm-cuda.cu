#include <iostream>
#include <cuda_runtime.h>

__global__ void addTwoNumber(int a, int b, int *c) { *c = a + b; }

int main() {
  int *c;
  int h_c;

  cudaMalloc(&c, sizeof(int));
  addTwoNumber<<<1, 1>>>(2, 3, c);

  cudaMemcpy(&h_c, c, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << h_c << " hello" << std::endl;
  return 0;
}