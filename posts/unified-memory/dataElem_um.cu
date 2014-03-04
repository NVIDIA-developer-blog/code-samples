#include <string.h>
#include <stdio.h>

struct DataElement
{
  char *name;
  int value;
};

__global__ 
void Kernel(DataElement *elem) {
  printf("On device: name=%s, value=%d\n", elem->name, elem->value);

  elem->name[0] = 'd';
  elem->value++;
}

void launch(DataElement *elem) {
  Kernel<<< 1, 1 >>>(elem);
  cudaDeviceSynchronize();
}

int main(void)
{
  DataElement *e;
  cudaMallocManaged((void**)&e, sizeof(DataElement));

  e->value = 10;
  cudaMallocManaged((void**)&(e->name), sizeof(char) * (strlen("hello") + 1) );
  strcpy(e->name, "hello");

  launch(e);

  printf("On host: name=%s, value=%d\n", e->name, e->value);

  cudaFree(e->name);
  cudaFree(e);

  cudaDeviceReset();
}