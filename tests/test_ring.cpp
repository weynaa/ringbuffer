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



#include "gtest/gtest.h"

#include "ringbuffer/ring.h"
#include "ringbuffer/sequence.h"
#include "ringbuffer/span.h"
#include "ringbuffer/detail/guarantee.h"

TEST(RingbufferTestSuite, RingClass) {
    using namespace ringbuffer;

    setDebugEnabled(true);

    std::string name = "testring1";
    RBSpace space = RBSpace::SPACE_SYSTEM;

    auto ring = Ring::create(name, space);
    EXPECT_EQ(ring->name(), name);
    EXPECT_EQ(ring->space(), space);

//Set our ring variables
    std::size_t nringlets = 1; //dimensionality of our ring->
    std::size_t nbytes = sizeof(float) * 8; //number of bytes we are putting in
    std::size_t buffer_bytes = 4 * nbytes;

//resize ring to fit the data
    ring->resize(nbytes, buffer_bytes, nringlets);

//generate some dummy variables
    std::size_t skip_time_tag = -1;
    std::size_t skip_offset = 0;
    std::size_t my_header_size = 0;
    const char *my_header = "";


//create the data
    float data[8] = {10, -10, 0, 0, 1, -3, 1, 0};

    EXPECT_EQ(sizeof(data), nbytes);

    std::string seq_name = "mysequence";

// Writing to the buffer
    {
        ring->begin_writing();
        {

            //we can find our sequence by this name later on
            bool nonblocking = true;

//open a sequence on the ring->
            WriteSequence write_seq(ring, seq_name, skip_time_tag, my_header_size, my_header, nringlets, skip_offset);
            EXPECT_EQ(write_seq.nringlet(), nringlets);

//reserve a "span" on this sequence to put our data
//point our pointer to the span's allocated memory
            WriteSpan write_span(ring, nbytes, nonblocking);

//create a pointer to pass our data to
            void *data_access = write_span.data();

// copy the data to the ring
            memcpy(data_access, &data, nbytes);

//commit this span to memory
            write_span.commit(nbytes);

        }
        ring->end_writing();
    }

// Reading from the buffer

    {
        auto read_seq = ReadSequence::by_name(ring, seq_name, true);
        ReadSpan read_span(&read_seq, skip_offset, nbytes);

//Access the data from the span with a pointer
        void *data_read = read_span.data();

//Copy the data into a readable format
        float *my_data = static_cast<float *>(data_read);
//print out the ring data
        for (int i = 0; i < 8; i++) {
            EXPECT_EQ(my_data[i], data[i]);
        }
//close up our ring access
    }

//delete the ring from memory

}

TEST(RingbufferTestSuite, RingClassMulti) {
    using namespace ringbuffer;

    setDebugEnabled(true);

    struct TestHeader {
        TestHeader() : priority(0), name("") {}

        int priority{0};
        char name[256];
    };

    std::string name = "testring1";
    RBSpace space = RBSpace::SPACE_SYSTEM;

    auto ring = Ring::create(name, space);
    EXPECT_EQ(ring->name(), name);
    EXPECT_EQ(ring->space(), space);

//Set our ring variables
    std::size_t niter = 10000;
    std::size_t nringlets = 1; //dimensionality of our ring->
    std::size_t nbytes = sizeof(float) * 8; //number of bytes we are putting in
    std::size_t buffer_bytes = niter * nbytes;

//resize ring to fit the data
    ring->resize(nbytes, buffer_bytes, nringlets);

//generate some dummy variables
    std::size_t skip_offset = 0;
    std::size_t my_header_size = sizeof(TestHeader);


//create the data
    float data[8] = {10, -10, 0, 0, 1, -3, 1, 0};

    EXPECT_EQ(sizeof(data), nbytes);

    {
        ring->begin_writing();

        bool nonblocking = true;
        TestHeader my_header;

// Writing to the buffer
        for (std::size_t i = 0; i < niter; i++) {

            {
                my_header.priority = i;

//we can find our sequence by this name later on
                std::string seq_name = "mysequence" + std::to_string(i);

//open a sequence on the ring->
                WriteSequence write_seq(ring, seq_name, i, my_header_size, &my_header, nringlets, skip_offset);
                EXPECT_EQ(write_seq.nringlet(), nringlets);

//reserve a "span" on this sequence to put our data
//point our pointer to the span's allocated memory
                WriteSpan write_span(ring, nbytes, nonblocking);

//create a pointer to pass our data to
                void *data_access = write_span.data();

// copy the data to the ring
                memcpy(data_access, &data, nbytes);

//commit this span to memory
                write_span.commit(nbytes);
            }
        }

        ring->end_writing();
    }

// Reading from the buffer

    {
        auto read_seq = ReadSequence::earliest_or_latest(ring, true, false);
        for (std::size_t i = 0; i < niter - 1; i++) {
            ReadSpan read_span(&read_seq, skip_offset, nbytes);

            auto *data_header = (TestHeader *) read_seq.header();
            EXPECT_EQ(data_header->priority, i);

//Access the data from the span with a pointer
            void *data_read = read_span.data();

//Copy the data into a readable format
            float *my_data = static_cast<float *>(data_read);
//print out the ring data
            for (int j = 0; j < 8; j++) {
                EXPECT_EQ(my_data[j], data[j]);
            }

            read_seq.increment_to_next();
        }
        EXPECT_THROW(read_seq.increment_to_next(), std::exception);
//close up our ring access
    }

//delete the ring from memory

}