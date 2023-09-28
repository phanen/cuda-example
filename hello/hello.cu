#include <stdio.h>

__global__ void my_kernel() {}

int main(void) {
  my_kernel<<<1, 1>>>();
  printf("hello world\n");
  return 0;
}
