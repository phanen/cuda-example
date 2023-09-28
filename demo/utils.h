#include <iostream>

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
