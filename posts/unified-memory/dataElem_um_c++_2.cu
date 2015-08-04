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

// Managed Base Class -- inherit from this to automatically 
// allocate objects in Unified Memory
class Managed 
{
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

// String Class for Managed Memory
class String : public Managed
{
  int length;
  char *data;
public:
  String() : length(0), data(0) {}
  
  // Constructor for C-string initializer
  String(const char *s) : length(0), data(0) { 
    _realloc(strlen(s));
    strcpy(data, s);
  }

  // Copy constructor
  String(const String& s) : length(0), data(0) {
    _realloc(s.length);
    strcpy(data, s.data);
  }
  
  ~String() { cudaFree(data); }

  // Assignment operator
  String& operator=(const char* s) {
    _realloc(strlen(s));
    strcpy(data, s);
    return *this;
  }

  // Element access (from host or device)
  __host__ __device__
  char& operator[](int pos) { return data[pos]; }

  // C-string access
  __host__ __device__
  const char* c_str() const { return data; }

private: 
  void _realloc(int len) {
    cudaFree(data);
    length = len;
    cudaMallocManaged(&data, length+1);
  }
};


struct DataElement : public Managed
{
  String name;
  int value;
};

__global__ 
void Kernel_by_pointer(DataElement *elem) {
  printf("On device by pointer:       name=%s, value=%d\n", elem->name.c_str(), elem->value);

  elem->name[0] = 'p';
  elem->value++;
}

__global__ 
void Kernel_by_ref(DataElement &elem) {
  printf("On device by ref:           name=%s, value=%d\n", elem.name.c_str(), elem.value);

  elem.name[0] = 'r';
  elem.value++;
}

__global__ 
void Kernel_by_value(DataElement elem) {
  printf("On device by value:         name=%s, value=%d\n", elem.name.c_str(), elem.value);

  elem.name[0] = 'v';
  elem.value++;
}

void launch_by_pointer(DataElement *elem) {
  Kernel_by_pointer<<< 1, 1 >>>(elem);
  cudaDeviceSynchronize();
}

void launch_by_ref(DataElement &elem) {
  Kernel_by_ref<<< 1, 1 >>>(elem);
  cudaDeviceSynchronize();
}

void launch_by_value(DataElement elem) {
  Kernel_by_value<<< 1, 1 >>>(elem);
  cudaDeviceSynchronize();
}

int main(void)
{
  DataElement *e = new DataElement;
  
  e->value = 10;
  e->name = "hello";
  
  launch_by_pointer(e);

  printf("On host (after by-pointer): name=%s, value=%d\n", e->name.c_str(), e->value);

  launch_by_ref(*e);

  printf("On host (after by-ref):     name=%s, value=%d\n", e->name.c_str(), e->value);

  launch_by_value(*e);

  printf("On host (after by-value):   name=%s, value=%d\n", e->name.c_str(), e->value);

  //delete e;

  cudaDeviceReset();
}


