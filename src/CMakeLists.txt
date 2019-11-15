set(RINGBUFFER_HEADERS
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/common.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/detail/memory.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/detail/cuda.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/detail/trace.h"


#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/cuda/stream.hpp"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/affinity.h"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/array.h"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/Common.hpp"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/Complex.hpp"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/cuda.h"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/proclog.h"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/ring.h"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/Ring.hpp"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/unpack.h"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/array_utils.hpp"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/assert.hpp"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/cuda.hpp"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/cuda.hpp"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/proclog.hpp"
#        "${PROJECT_SOURCE_DIR}/include/ringbuffer/utils.hpp"
        )

set(RINGBUFFER_SOURCES
        common.cpp
        detail/memory.cpp
        detail/cuda.cpp
        detail/trace.cpp

#        affinity.cpp
#        array.cpp
#        gunpack.cu
#        gunpack.cuh
#        IndexArray.cuh
#        proclog.cpp
#        ring.cpp
#        ring_impl.cpp
#        ring_impl.hpp
#        unpack.cpp
        )

add_library(ringbuffer ${RINGBUFFER_SOURCES} ${RINGBUFFER_HEADERS})

set_target_properties(ringbuffer PROPERTIES CXX_STANDARD 11)
set_target_properties(ringbuffer PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(ringbuffer PUBLIC
        CONAN_PKG::Boost
        CONAN_PKG::spdlog
        )

target_include_directories(ringbuffer PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>/include
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${PROJECT_BINARY_DIR}/src
        ${PROJECT_SOURCE_DIR}/src
        )

if (WITH_CUDA)
    set_target_properties(ringbuffer PROPERTIES CUDA_STANDARD 11)
    target_compile_definitions(ringbuffer PUBLIC WITH_CUDA)
    target_include_directories(ringbuffer PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(ringbuffer PUBLIC ${RINGBUFFER_CUDA_LIBRARIES})
endif()

if (WITH_OMP)
    target_compile_definitions(ringbuffer PUBLIC WITH_OMP)
    target_link_libraries(ringbuffer PUBLIC OpenMP::OpenMP_CXX)
endif()

if (WITH_NUMA)
    target_compile_definitions(ringbuffer PUBLIC WITH_NUMA)
    target_link_libraries(ringbuffer PUBLIC NUMA::NUMA)
endif()

if (ENABLE_DEBUG)
    target_compile_definitions(ringbuffer PUBLIC RINGBUFFER_DEBUG)
endif()

if (ENABLE_TRACE)
    target_compile_definitions(ringbuffer PUBLIC RINGBUFFER_TRACE)
endif()

install(TARGETS ringbuffer DESTINATION ${CMAKE_INSTALL_LIBDIR})