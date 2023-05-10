cmake_minimum_required(VERSION 3.16)
project(error_example CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "60;70;80")

find_package(CUDAToolkit REQUIRED)

add_executable(error_example main.cu )
target_link_libraries(error_example CUDA::cuda_driver)
target_compile_options(error_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       #-G
                       #-g
                       --extended-lambda
                       -Xptxas=-v
                       >)

