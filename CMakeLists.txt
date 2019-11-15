cmake_minimum_required(VERSION 3.4)
project(ringbuffer C CXX CUDA)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})

if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup(TARGETS)
else()
    message(WARNING "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif()

include(GNUInstallDirs)

option(BUILD_SHARED_LIBS "Build shared library" ON)
option(WITH_CUDA "With CUDA Integration" OFF)
option(WITH_OMP "With OMP Integration" OFF)
option(WITH_NUMA "With NUMA Integration" OFF)
option(ENABLE_FIBERS "Enable Boost Fibers Support" OFF)
option(ENABLE_DEBUG "Enable Debug Output" OFF)
option(ENABLE_TRACE "Enable Tracing" OFF)


set(RINGBUFFER_CUDA_LIBRARIES)
if (WITH_CUDA)
    find_package(CUDA REQUIRED)
    find_package(NVRTC REQUIRED)
    find_package(NVTOOLSEXT REQUIRED)
    SET(RINGBUFFER_CUDA_LIBRARIES
            ${CUDA_LIBRARIES}
            ${CUDA_CUDA_LIBRARY}
            ${CUDA_NVTOOLSEXT_LIB}
            ${CUDA_NVRTC_LIB}
            )
endif()

if (WITH_OMP)
    find_package(OpenMP REQUIRED)
endif()

if (WITH_NUMA)
    find_package(NUMA REQUIRED)
endif()

add_subdirectory(src)
add_subdirectory(tests)
