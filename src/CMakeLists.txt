# Header
configure_file(${PROJECT_SOURCE_DIR}/include/ringbuffer/config.h.in ${PROJECT_BINARY_DIR}/include/ringbuffer/config.h @ONLY)


set(RINGBUFFER_HEADERS
        "${PROJECT_BINARY_DIR}/include/ringbuffer/config.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/common.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/detail/memory.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/detail/cuda.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/detail/trace.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/detail/affinity.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/detail/ring_realloc_lock.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/detail/ring_state.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/detail/guarantee.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/detail/util.h"

        "${PROJECT_SOURCE_DIR}/include/ringbuffer/ring.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/sequence.h"
        "${PROJECT_SOURCE_DIR}/include/ringbuffer/span.h"
        )

set(RINGBUFFER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detail/memory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detail/cuda.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detail/trace.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detail/affinity.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detail/ring_realloc_lock.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detail/guarantee.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ring.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sequence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/span.cpp
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
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${PROJECT_BINARY_DIR}/src
        ${PROJECT_SOURCE_DIR}/src
        )

if (RINGBUFFER_WITH_CUDA)
    set_target_properties(ringbuffer PROPERTIES CUDA_STANDARD 11)
    target_include_directories(ringbuffer PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(ringbuffer PUBLIC ${RINGBUFFER_CUDA_LIBRARIES})
endif()

if (RINGBUFFER_WITH_OMP)
    target_link_libraries(ringbuffer PUBLIC OpenMP::OpenMP_CXX)
endif()

if (RINGBUFFER_WITH_NUMA)
    target_link_libraries(ringbuffer PUBLIC NUMA::NUMA)
endif()

install(TARGETS ringbuffer DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${PROJECT_BINARY_DIR}/include/ringbuffer/config.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
