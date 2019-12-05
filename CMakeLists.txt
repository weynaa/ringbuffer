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
option(WITH_NVTOOLSEXT "With NVTOOLSEXT" OFF)
option(WITH_OMP "With OMP Integration" OFF)
option(WITH_NUMA "With NUMA Integration" OFF)
option(ENABLE_FIBERS "Enable Boost Fibers Support" OFF)
option(ENABLE_DEBUG "Enable Debug Output" OFF)
option(ENABLE_TRACE "Enable Tracing" OFF)

if(UNIX)
    if(APPLE)
        set(RINGBUFFER_TARGET_APPLE 1)
    endif()
    set(RINGBUFFER_TARGET_UNIX 1)
elseif(WIN32)
    set(RINGBUFFER_TARGET_WINDOWS 1)
endif()

# if NVTOOLSEXT then require CUDA
if (WITH_NVTOOLSEXT)
  set(WITH_CUDA ON)
endif()

# setup global config
set(RINGBUFFER_WITH_CUDA ${WITH_CUDA})
set(RINGBUFFER_WITH_NVTOOLSEXTL ${WITH_NVTOOLSEXT})
set(RINGBUFFER_WITH_CUDA ${WITH_CUDA})
set(RINGBUFFER_WITH_OMP ${WITH_OMP})
set(RINGBUFFER_WITH_NUMA ${WITH_NUMA})
set(RINGBUFFER_TRACE ${ENABLE_TRACE})
set(RINGBUFFER_DEBUG ${ENABLE_DEBUG})
set(RINGBUFFER_BOOST_FIBER ${ENABLE_FIBERS})
if (NOT BUILD_SHARED_LIBS)
  set(RINGBUFFER_BUILD_STATIC 1)
endif()

set(RINGBUFFER_CUDA_LIBRARIES)
if (WITH_CUDA)
    find_package(CUDA REQUIRED)
    find_package(NVRTC REQUIRED)
    SET(RINGBUFFER_CUDA_LIBRARIES
            ${CUDA_LIBRARIES}
            ${CUDA_CUDA_LIBRARY}
            ${CUDA_NVTOOLSEXT_LIB}
            )
    if(WITH_NVTOOLSEXT)
    find_package(NVTOOLSEXT REQUIRED)
    SET(RINGBUFFER_CUDA_LIBRARIES
        ${RINGBUFFER_CUDA_LIBRARIES}
        ${CUDA_NVRTC_LIB}
        )        
    endif()
endif()

if (WITH_OMP)
    find_package(OpenMP REQUIRED)
endif()

if (WITH_NUMA)
    find_package(NUMA REQUIRED)
endif()

add_subdirectory(src)
add_subdirectory(tests)
