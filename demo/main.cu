#include "demo.h"
#include <iostream>

int wrapper(int (*demo)(), const char *name) {
  std::cout << "run " << name << ":" << std::endl;
  demo();
  std::cout << std::endl;
  return 0;
}

int main(void) {
  wrapper(adder, "adder");
}
