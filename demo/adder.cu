#include "utils.h"

__global__ void my_kernel() {}

// __global__ void add(int *a, int *b, int *c) { *c = *a + *b; }

// Many blocks with one thread each
// One block with many threads
// block x thread_per_block
__global__ void add(int *a, int *b, int *c, int n) {
  // c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
  // c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
  // c[index] = a[index] + b[index];

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // patch for "non-divide", avoid overflow
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

// populate vectors with random ints
void random_ints(int *a, int len) {
  for (size_t i = 0; i < len; i++)
    a[i] = rand() % 1000;
}

#define BLK 10000
#define TRD 1024
#define NOT_ALIGN 123
#define V_LEN ((BLK) * (TRD) + NOT_ALIGN)

int adder() {
  // my_kernel<<<1, 1>>>();
  int *a, *b, *c;       // host copies of a, b, c
  int *d_a, *d_b, *d_c; // device copies of a, b, c
  int size = V_LEN * sizeof(int);

  a = (int *)malloc(size);
  random_ints(a, V_LEN);
  b = (int *)malloc(size);
  random_ints(b, V_LEN);
  c = (int *)malloc(size);
  print_array(a, V_LEN);
  print_array(b, V_LEN);

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU
  // we ensure full coverage, overflow part will be ignored in function
  add<<<BLK + 1, TRD>>>(d_a, d_b, d_c, V_LEN);

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  print_array(c, V_LEN);

  // Cleanup
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
