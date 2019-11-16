/* **********************************************************************************
#                                                                                   #
# Copyright (c) 2019,                                                               #
# Research group CAMP                                                               #
# Technical University of Munich                                                    #
#                                                                                   #
# All rights reserved.                                                              #
# Ulrich Eck - ulrich.eck@tum.de                                                    #
#                                                                                   #
# Redistribution and use in source and binary forms, with or without                #
# modification, are restricted to the following conditions:                         #
#                                                                                   #
#  * The software is permitted to be used internally only by the research group     #
#    CAMP and any associated/collaborating groups and/or individuals.               #
#  * The software is provided for your internal use only and you may                #
#    not sell, rent, lease or sublicense the software to any other entity           #
#    without specific prior written permission.                                     #
#    You acknowledge that the software in source form remains a confidential        #
#    trade secret of the research group CAMP and therefore you agree not to         #
#    attempt to reverse-engineer, decompile, disassemble, or otherwise develop      #
#    source code for the software or knowingly allow others to do so.               #
#  * Redistributions of source code must retain the above copyright notice,         #
#    this list of conditions and the following disclaimer.                          #
#  * Redistributions in binary form must reproduce the above copyright notice,      #
#    this list of conditions and the following disclaimer in the documentation      #
#    and/or other materials provided with the distribution.                         #
#  * Neither the name of the research group CAMP nor the names of its               #
#    contributors may be used to endorse or promote products derived from this      #
#    software without specific prior written permission.                            #
#                                                                                   #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE            #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   #
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;      #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND       #
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS     #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                      #
#                                                                                   #
*************************************************************************************/

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

#ifdef RINGBUFFER_WITH_CUDA
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

#endif // RINGBUFFER_WITH_CUDA
