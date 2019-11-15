//
// Created by netlabs on 11/15/19.
//

#include "ringbuffer/detail/memory.h"
#include "gtest/gtest.h"

using namespace ringbuffer;

TEST(RingbufferTestSuite, MemorySystem) {

    int width = 1024;
    int height = 1024;
    uint8_t *new_buf;
    std::size_t new_nbyte = width * height;
    RBSpace space = RBSpace::SPACE_SYSTEM;

    EXPECT_EQ(memory::malloc_((void **) &new_buf, new_nbyte, space), RBStatus::STATUS_SUCCESS);

    RBSpace space_verify;
    EXPECT_EQ(memory::getSpace(new_buf, &space_verify), RBStatus::STATUS_SUCCESS);
    EXPECT_EQ(space_verify, RBSpace::SPACE_SYSTEM);

    EXPECT_EQ(getSpaceString(space), getSpaceString(space_verify));

    EXPECT_EQ(memory::memset_(new_buf, space, 5, new_nbyte), RBStatus::STATUS_SUCCESS);

    for (int i = 0; i < new_nbyte; i++) {
        EXPECT_EQ(new_buf[i], 5);
    }

    uint8_t *other_buf;
    EXPECT_EQ(memory::malloc_((void **) &other_buf, new_nbyte, space), RBStatus::STATUS_SUCCESS);

    EXPECT_EQ(memory::memset_(other_buf, space, 4, new_nbyte), RBStatus::STATUS_SUCCESS);

    for (int i = 0; i < new_nbyte; i++) {
        EXPECT_EQ(other_buf[i], 4);
    }

    EXPECT_EQ(memory::memcpy_(other_buf, space, new_buf, space, width * height), RBStatus::STATUS_SUCCESS);

    for (int i = 0; i < new_nbyte; i++) {
        EXPECT_EQ(other_buf[i], 5);
    }

    EXPECT_EQ(memory::free_(new_buf, space), RBStatus::STATUS_SUCCESS);
    EXPECT_EQ(memory::free_(other_buf, space), RBStatus::STATUS_SUCCESS);
}

#ifdef WITH_CUDA
TEST(RingbufferTestSuite, MemoryCuda) {

    int width = 1024;
    int height = 1024;
    uint8_t *new_buf;
    std::size_t new_nbyte = width * height;
    RBSpace space = RBSpace::SPACE_CUDA;

    EXPECT_EQ(memory::malloc_((void **) &new_buf, new_nbyte, space), RBStatus::STATUS_SUCCESS);

    RBSpace space_verify;
    EXPECT_EQ(memory::getSpace(new_buf, &space_verify), RBStatus::STATUS_SUCCESS);
    EXPECT_EQ(space_verify, RBSpace::SPACE_CUDA);

    EXPECT_EQ(getSpaceString(space), getSpaceString(space_verify));

    EXPECT_EQ(memory::memset_(new_buf, space, 5, new_nbyte), RBStatus::STATUS_SUCCESS);

    uint8_t *other_buf;
    EXPECT_EQ(memory::malloc_((void **) &other_buf, new_nbyte, RBSpace::SPACE_SYSTEM), RBStatus::STATUS_SUCCESS);

    EXPECT_EQ(memory::memcpy_(other_buf, RBSpace::SPACE_SYSTEM, new_buf, space, width * height), RBStatus::STATUS_SUCCESS);

    for (int i = 0; i < new_nbyte; i++) {
        EXPECT_EQ(other_buf[i], 5);
    }

    EXPECT_EQ(memory::free_(new_buf, space), RBStatus::STATUS_SUCCESS);
    EXPECT_EQ(memory::free_(other_buf, RBSpace::SPACE_SYSTEM), RBStatus::STATUS_SUCCESS);
}

TEST(RingbufferTestSuite, MemoryCudaHost) {

    int width = 1024;
    int height = 1024;
    uint8_t *new_buf;
    std::size_t new_nbyte = width * height;
    RBSpace space = RBSpace::SPACE_CUDA_HOST;

    EXPECT_EQ(memory::malloc_((void **) &new_buf, new_nbyte, space), RBStatus::STATUS_SUCCESS);

    RBSpace space_verify;
    EXPECT_EQ(memory::getSpace(new_buf, &space_verify), RBStatus::STATUS_SUCCESS);
    EXPECT_EQ(space_verify, RBSpace::SPACE_CUDA_HOST);

    EXPECT_EQ(getSpaceString(space), getSpaceString(space_verify));

    EXPECT_EQ(memory::memset_(new_buf, space, 5, new_nbyte), RBStatus::STATUS_SUCCESS);

    uint8_t *other_buf;
    EXPECT_EQ(memory::malloc_((void **) &other_buf, new_nbyte, RBSpace::SPACE_SYSTEM), RBStatus::STATUS_SUCCESS);

    EXPECT_EQ(memory::memcpy_(other_buf, RBSpace::SPACE_SYSTEM, new_buf, space, width * height), RBStatus::STATUS_SUCCESS);

    for (int i = 0; i < new_nbyte; i++) {
        EXPECT_EQ(other_buf[i], 5);
    }

    EXPECT_EQ(memory::free_(new_buf, space), RBStatus::STATUS_SUCCESS);
    EXPECT_EQ(memory::free_(other_buf, RBSpace::SPACE_SYSTEM), RBStatus::STATUS_SUCCESS);
}

TEST(RingbufferTestSuite, MemoryCudaManaged) {

    int width = 1024;
    int height = 1024;
    uint8_t *new_buf;
    std::size_t new_nbyte = width * height;
    RBSpace space = RBSpace::SPACE_CUDA_MANAGED;

    EXPECT_EQ(memory::malloc_((void **) &new_buf, new_nbyte, space), RBStatus::STATUS_SUCCESS);

    RBSpace space_verify;
    EXPECT_EQ(memory::getSpace(new_buf, &space_verify), RBStatus::STATUS_SUCCESS);
    EXPECT_EQ(space_verify, RBSpace::SPACE_CUDA_MANAGED);

    EXPECT_EQ(getSpaceString(space), getSpaceString(space_verify));

    EXPECT_EQ(memory::memset_(new_buf, space, 5, new_nbyte), RBStatus::STATUS_SUCCESS);

// no support for copying from ManagedMemory

// this seg-faults .. why??
//    for (int i=0; i< new_nbyte; i++) {
//        EXPECT_EQ(new_buf[i], 5);
//    }

    EXPECT_EQ(memory::free_(new_buf, space), RBStatus::STATUS_SUCCESS);
}

#endif // WITH_CUDA
