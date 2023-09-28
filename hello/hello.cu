#include <iostream>

__global__ void my_kernel() {}

// __global__ void add(int *a, int *b, int *c) { *c = *a + *b; }

// Many blocks with one thread each
// One block with many threads
__global__ void add(int *a, int *b, int *c) {
  // c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

inline void print_array(const int *start, size_t count,
                        bool dont_compress = false) {
  std::cout << "[";
  for (size_t i = 0; i < count; i++) {
    if (!dont_compress && i == 5 && count >= 10) {
      i = count - 5;
      std::cout << "...";
    }
    std::cout << start[i];
    if (i != count - 1)
      std::cout << ", ";
  }
  std::cout << "]\n";
}

// populate vectors with random ints
void random_ints(int *a, int len) {
  for (size_t i = 0; i < len; i++)
    a[i] = rand() % 1000;
}

#define V_LEN 1024

int main(void) {
  // my_kernel<<<1, 1>>>();
  // printf("hello world\n");

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
  // add<<<V_LEN, 1>>>(d_a, d_b, d_c);
  add<<<1, V_LEN>>>(d_a, d_b, d_c);

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
