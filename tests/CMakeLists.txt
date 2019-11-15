
set(TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_memory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_ring.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_pipeline.cpp
        )

add_executable(ringbuffer_test ${TEST_SOURCES})

target_link_libraries(ringbuffer_test
        ringbuffer
        CONAN_PKG::gtest
        ${RINGBUFFER_CUDA_LIBRARIES}
        )

target_include_directories(ringbuffer_test
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/src
        ${PROJECT_SOURCE_DIR}/src
        )
