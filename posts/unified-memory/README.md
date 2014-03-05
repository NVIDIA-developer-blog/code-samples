Unified Memory Example
======================

This is an example of using Unified Memory from CUDACasts Episode #X. 

There are four examples
* dataElem.cu: original simple data structure example in vanilla CUDA C
* dataElem_um.cu: modified example that uses Unified Memory to simplify data allocation.
* dataElem_um_c++_1.cu: A C++ example that shows how to overload new and delete to use cudaMallocManaged().
* dataElem_um_c++_2.cu: A more complete C++ example that creates a managed String class and uses it to simplify the unified memory management inside our data structure.


