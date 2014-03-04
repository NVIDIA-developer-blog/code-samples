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
  DataElement *d_elem;
  char *d_name;

  int namelen = strlen(elem->name) + 1;

  // Allocate storage for struct and text
  cudaMalloc(&d_elem, sizeof(DataElement));
  cudaMalloc(&d_name, namelen);

  // Copy up each piece separately, including new “text” pointer value
  cudaMemcpy(d_elem, elem, sizeof(DataElement), cudaMemcpyHostToDevice);
  cudaMemcpy(d_name, elem->name, namelen, cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_elem->name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);

  // Finally we can launch our kernel, but CPU & GPU use different copies of “elem”
  Kernel<<< 1, 1 >>>(d_elem);

  cudaMemcpy(&(elem->value), &(d_elem->value), sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(elem->name, d_name, namelen, cudaMemcpyDeviceToHost);

  cudaFree(d_name);
  cudaFree(d_elem);
}

int main(void)
{
  DataElement *e;
  e = (DataElement*)malloc(sizeof(DataElement));

  e->value = 10;
  e->name = (char*)malloc(sizeof(char) * (strlen("hello") + 1));
  strcpy(e->name, "hello");

  launch(e);

  printf("On host: name=%s, value=%d\n", e->name, e->value);

  free(e->name);
  free(e);

  cudaDeviceReset();
}