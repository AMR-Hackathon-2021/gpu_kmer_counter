cmake_minimum_required(VERSION 3.19)
project(cuda_kmers)
set(CMAKE_CXX_STANDARD 14)
cmake_policy(SET CMP0104 OLD) # Can't get CUDA_ARCHITECTURES to work with NEW

# Variable definitions
set(TARGET_NAME cuda_kmers)
add_compile_definitions(PYTHON_EXT)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
# Set these to profile
# set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")
# set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DUSE_CUDA_PROFILER")

# Set paths for non standard lib/ and include/ locations
if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
    link_directories($ENV{CONDA_PREFIX}/lib/intel64)
    link_directories(/home/shorsfield/software/gpu_kmer_counter/gpu_kmer_counter)
endif()

# Add libraries
find_package(ZLIB)
find_package(pybind11 REQUIRED)

# Check for CUDA and compiles GPU library

enable_language(CUDA)
set(CUDA_OPTS "-Wreorder -Xcompiler -fPIC -Xcompiler -Wall -Xcompiler -fopenmp -Xptxas -dlcm=ca --cudart static --relocatable-device-code=false")
string(APPEND CUDA_OPTS " -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    string(APPEND CUDA_OPTS " -G")
else()
    string(APPEND CUDA_OPTS " --use_fast_math")
endif()
# Set this to profile
# string(APPEND CUDA_OPTS " -pg --generate-line-info")
set(CMAKE_CUDA_FLAGS "${CUDA_OPTS}")

add_library("${TARGET_NAME}" MODULE src/python_bindings.cu)
target_include_directories("${TARGET_NAME}" PRIVATE "${pybind11_INCLUDE_DIRS}")
set_property(TARGET "${TARGET_NAME}"
             PROPERTY POSITION_INDEPENDENT_CODE ON
             CUDA_SEPARABLE_COMPILATION OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS ON   # try and ensure device link with nvcc
             CUDA_VISIBILITY_PRESET "hidden")
set_target_properties("${TARGET_NAME}" PROPERTIES
             CXX_VISIBILITY_PRESET "hidden"
             PREFIX "${PYTHON_MODULE_PREFIX}"
             SUFFIX "${PYTHON_MODULE_EXTENSION}")

target_link_libraries("${TARGET_NAME}" PRIVATE pybind11::module)
if(DEFINED ENV{CONDA_PREFIX} AND (NOT APPLE OR CMAKE_COMPILER_IS_GNUCC))
  target_link_libraries("${TARGET_NAME}" PRIVATE gomp z readfastq)
else()
  target_link_libraries("${TARGET_NAME}" PRIVATE ZLIB::ZLIB)
endif()
