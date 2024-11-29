
#include <iostream>

#include "embedded_objs.h"

int main(int argc, char** argv)
{
  (void)argc;
  (void)argv;

  unsigned char* ka = kernelA;
  unsigned char* kb = kernelB;
  if(ka != NULL && kb != NULL)
  {
    std::cout << "loaded ptx files." << std::endl;
    return 0;
  }
  return 1;
}
