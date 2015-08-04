/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
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