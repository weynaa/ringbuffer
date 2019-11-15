
set(TEST_SOURCES
        test_memory.cpp
#        test_main.cpp
#        test_ring.cpp
#        test_array.cpp
#        test_pipeline.cpp
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
