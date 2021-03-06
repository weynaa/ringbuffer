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
#include "spdlog/spdlog.h"

#include <thread>
#include <chrono>
#include <iostream>

#include "ringbuffer/ring.h"
#include "ringbuffer/sequence.h"
#include "ringbuffer/span.h"
#include "ringbuffer/detail/memory.h"
#include "ringbuffer/detail/guarantee.h"
#include "ringbuffer/detail/affinity.h"
#include "ringbuffer/detail/cuda.h"

TEST(RingbufferTestSuite, RingbufferThreaded){
    using namespace ringbuffer;
    using namespace std::chrono;

    struct ImageHeader {
        ImageHeader() {}
        uint64_t timestamp{0};
        uint16_t width{0};
        uint16_t height{0};
        // more metadata relevant to the image we're buffering
    };

    std::string name = "depthcamera01";
    RBSpace space = RBSpace::SPACE_SYSTEM;

    auto ring = Ring::create(name, space);
    EXPECT_EQ(ring->name(), name);
    EXPECT_EQ(ring->space(), space);

    // set numa affinity of data
    ring->set_core(1);

    //Set our ring variables
    std::size_t niter = 1000;

    std::size_t nringlets = 1; //dimensionality of our ring->
    std::size_t width = 1024;
    std::size_t height = 1024;
    std::size_t npixels = width*height;
    std::size_t nbytes = sizeof(uint16_t)*npixels; // one frame of depth-data
    std::size_t buffer_bytes = 4*nbytes; // 4x nbytes as ringbuffer size ==> could also be std::size_t(-1) to use default

    //resize ring to fit the data
    ring->resize(nbytes, buffer_bytes, nringlets);

    std::size_t skip_offset = 0;
    std::size_t my_header_size = sizeof(ImageHeader);


    auto create_frame = [](uint16_t* data_ptr, std::size_t w, std::size_t h){

        // write data
        for (std::size_t i=0; i<h; i++) {
            std::size_t s = i*w;
            for (std::size_t j=0; j<w; j++) {
                data_ptr[s+j] = j;
            }
        }
    };

    bool is_running = true;
    std::size_t received_packages{0};
    uint64_t received_bytes{0};

    auto recv_thread = std::thread([&](){
        spdlog::info("start receiving thread");
        
        // set numa affinity of receiver thread to 2
        affinity::affinitySetCore(2);

        bool try_again = true;
        while (try_again) {
            try {

                // can we open a sequence if there is none ??
                auto read_seq = ReadSequence::earliest_or_latest(ring, true, false);
                try_again = false;

                for (std::size_t n=0; n < niter; n++) {

                    // this should block !!!
                    ReadSpan read_span(&read_seq, skip_offset, nbytes);

                    auto* data_header = (ImageHeader*)read_seq.header();
                    // timestamp

                    //Access the data from the span with a pointer
                    void* data_read = read_span.data();

                    //Copy the data into a readable format
                    uint16_t *my_data = static_cast<uint16_t*>(data_read);

                    // check data
                    bool data_ok = true;
                    for (std::size_t i=0; i<data_header->height; i++) {
                        std::size_t s = i*data_header->width;
                        for (std::size_t j=0; j<data_header->width; j++) {
                            if (my_data[s+j] != j) {
                                data_ok = false;
                                break;
                            }
                        }
                    }
                    EXPECT_EQ(data_ok, true);

                    received_packages++;
                    received_bytes += (data_header->height * data_header->width * sizeof(uint16_t));

                    read_seq.increment_to_next();

                    if (!is_running) {
                        spdlog::info("Exiting receiving thread (not running)");
                        return;
                    }
                }
            } catch(const RBException &e) {
                if (e.status() == RBStatus::STATUS_END_OF_DATA) {
                    if (try_again) {
                        std::this_thread::yield();
                    }
                } else {
                    spdlog::error("Receiver RBException: {0}", e.what());
                }
            } catch(const std::runtime_error &e) {
                spdlog::error("Receiver std::runtime_error: {0}", e.what());
            } catch(const std::exception &e) {
                spdlog::error("Receiver std::exception: {0}", e.what());
            }
        }
        spdlog::info("Exiting receiving thread (finished)");

    });



    // writer in main
    {
        spdlog::info("start writing to buffer");

        // set numa affinity of writer thread to 1
        affinity::affinitySetCore(1);

        ring->begin_writing();

        bool nonblocking = false;
        // sequence with no name
        std::string seq_name;
        ImageHeader my_header;

        // Writing to the buffer
        for (std::size_t i=0; i < niter; i++ ){

            // write header
            my_header.timestamp = system_clock::now().time_since_epoch().count();
            my_header.width = width;
            my_header.height = height;

            {
                //open a sequence on the ring->
                WriteSequence write_seq(ring, seq_name, i, my_header_size, &my_header, nringlets, skip_offset);
                EXPECT_EQ(write_seq.nringlet(), nringlets);

                //reserve a "span" on this sequence to put our data
                //point our pointer to the span's allocated memory
                WriteSpan write_span(ring, nbytes, nonblocking);

                //create a pointer to pass our data to
                void *data_access = write_span.data();

                // write the frame data to the ring
                create_frame(static_cast<uint16_t*>(data_access), width, height);

                //stop writing
                //commit this span to memory
                write_span.commit(nbytes);
            }
        }

        ring->end_writing();
    }

    recv_thread.join();
    spdlog::info("received packages: {0}", received_packages);
    spdlog::info("received bytes: {0}", received_bytes);

}
#ifdef RINGBUFFER_WITH_CUDA

TEST(RingbufferTestSuite, RingbufferThreadedCuda){
    using namespace ringbuffer;
    using namespace std::chrono;

    struct ImageHeader {
        ImageHeader() {}
        uint64_t timestamp{0};
        uint16_t width{0};
        uint16_t height{0};
        // more metadata relevant to the image we're buffering
    };

    auto ring_input_cpu = Ring::create("depthcamera01_cpu", RBSpace::SPACE_SYSTEM);
    // set numa affinity of data
    ring_input_cpu->set_core(2);

    auto ring_input_gpu = Ring::create("depthcamera01_gpu", RBSpace::SPACE_CUDA);
    // set numa affinity of data
    ring_input_gpu->set_core(3);

    auto ring_output_gpu= Ring::create("result01_gpu", RBSpace::SPACE_CUDA);
    // set numa affinity of data
    ring_output_gpu->set_core(4);

    auto ring_output_cpu = Ring::create("result01_cpu", RBSpace::SPACE_SYSTEM);
    // set numa affinity of data
    ring_output_cpu->set_core(1);



    //Set our ring variables
    std::size_t niter = 1000;

    std::size_t nringlets = 1; //dimensionality of our ring.
    std::size_t width = 2048;
    std::size_t height = 1536;
    std::size_t npixels = width*height;
    std::size_t nbytes = sizeof(uint32_t)*npixels; // one frame of depth-data
    std::size_t buffer_bytes = 4*nbytes; // 4x nbytes as ringbuffer size ==> could also be std::size_t(-1) to use default

    //resize ring to fit the data
    ring_input_cpu->resize(nbytes, buffer_bytes, nringlets);
    ring_input_gpu->resize(nbytes, buffer_bytes, nringlets);
    ring_output_gpu->resize(nbytes, buffer_bytes, nringlets);
    ring_output_cpu->resize(nbytes, buffer_bytes, nringlets);

    std::size_t skip_offset = 0;
    std::size_t my_header_size = sizeof(ImageHeader);
    bool nonblocking = false;
    std::string seq_name;


    auto create_frame = [](uint32_t* data_ptr, std::size_t w, std::size_t h){

        // write data
        for (std::size_t i=0; i<h; i++) {
            std::size_t s = i*w;
            for (std::size_t j=0; j<w; j++) {
                data_ptr[s+j] = j;
            }
        }
    };

    bool is_running = true;
    std::size_t received_packages{0};
    uint64_t received_bytes{0};

    auto cpu2gpu_thread = std::thread([&](){
        spdlog::info("start cpu2gpu thread");
        affinity::affinitySetCore(2);

        ring_input_gpu->begin_writing();

        bool try_again = true;
        while (try_again) {
            try {
                auto read_seq = ReadSequence::earliest_or_latest(ring_input_cpu, true, false);
                try_again = false;

                for (std::size_t n=0; n < niter; n++) {
                    ReadSpan read_span(&read_seq, skip_offset, nbytes);
                    void* data_read = read_span.data();
                    {
                        WriteSequence write_seq(ring_input_gpu, seq_name, n, read_seq.header_size(), read_seq.header(), read_seq.nringlet(), skip_offset);
                        WriteSpan write_span(ring_input_gpu, nbytes, nonblocking);
                        void *data_access = write_span.data();
                        memory::memcpy_(data_access, ring_input_gpu->space(), data_read, ring_input_cpu->space(), nbytes);
                        write_span.commit(nbytes);
                        cuda::streamSynchronize();
                    }
                    read_seq.increment_to_next();

                    if (!is_running) {
                        spdlog::info("Exiting cpu2gpu thread (not running)");
                        return;
                    }
                }
            } catch(const RBException &e) {
                if (e.status() == RBStatus::STATUS_END_OF_DATA) {
                    if (try_again) {
                        std::this_thread::yield();
                    }
                } else {
                    spdlog::error("cpu2gpu RBException: {0}", e.what());
                }
            } catch(const std::runtime_error &e) {
                spdlog::error("cpu2gpu std::runtime_error: {0}", e.what());
            } catch(const std::exception &e) {
                spdlog::error("cpu2gpu std::exception: {0}", e.what());
            }
        }
        ring_input_gpu->end_writing();
        spdlog::info("Exiting cpu2gpu thread (finished)");
    });



    auto gpu2gpu_thread = std::thread([&](){
        spdlog::info("start gpu2gpu thread");
        affinity::affinitySetCore(3);

        ring_output_gpu->begin_writing();

        bool try_again = true;
        while (try_again) {
            try {
                auto read_seq = ReadSequence::earliest_or_latest(ring_input_gpu, true, false);
                try_again = false;

                for (std::size_t n=0; n < niter; n++) {
                    ReadSpan read_span(&read_seq, skip_offset, nbytes);
                    void* data_read = read_span.data();
                    {
                        WriteSequence write_seq(ring_output_gpu, seq_name, n, read_seq.header_size(), read_seq.header(), read_seq.nringlet(), skip_offset);
                        WriteSpan write_span(ring_output_gpu, nbytes, nonblocking);
                        void *data_access = write_span.data();
                        memory::memcpy_(data_access, ring_output_gpu->space(), data_read, ring_input_gpu->space(), nbytes);
                        write_span.commit(nbytes);
                        cuda::streamSynchronize();
                    }
                    read_seq.increment_to_next();

                    if (!is_running) {
                        spdlog::info("Exiting gpu2gpu thread (not running)");
                        return;
                    }
                }
            } catch(const RBException &e) {
                if (e.status() == RBStatus::STATUS_END_OF_DATA) {
                    if (try_again) {
                        std::this_thread::yield();
                    }
                } else {
                    spdlog::error("gpu2gpu RBException: {0}", e.what());
                }
            } catch(const std::runtime_error &e) {
                spdlog::error("gpu2gpu std::runtime_error: {0}", e.what());
            } catch(const std::exception &e) {
                spdlog::error("gpu2gpu std::exception: {0}", e.what());
            }
        }
        ring_output_gpu->end_writing();
        spdlog::info("Exiting gpu2gpu thread (finished)");
    });

    auto gpu2cpu_thread = std::thread([&](){
        spdlog::info("start gpu2cpu thread");
        affinity::affinitySetCore(4);

        ring_output_cpu->begin_writing();

        bool try_again = true;
        while (try_again) {
            try {
                auto read_seq = ReadSequence::earliest_or_latest(ring_output_gpu, true, false);
                try_again = false;

                for (std::size_t n=0; n < niter; n++) {
                    ReadSpan read_span(&read_seq, skip_offset, nbytes);
                    void* data_read = read_span.data();
                    {
                        WriteSequence write_seq(ring_output_cpu, seq_name, n, read_seq.header_size(), read_seq.header(), read_seq.nringlet(), skip_offset);
                        WriteSpan write_span(ring_output_cpu, nbytes, nonblocking);
                        void *data_access = write_span.data();
                        memory::memcpy_(data_access, ring_output_cpu->space(), data_read, ring_output_gpu->space(), nbytes);
                        write_span.commit(nbytes);
                        cuda::streamSynchronize();
                    }
                    read_seq.increment_to_next();

                    if (!is_running) {
                        spdlog::info("Exiting gpu2cpu thread (not running)");
                        return;
                    }
                }
            } catch(const RBException &e) {
                if (e.status() == RBStatus::STATUS_END_OF_DATA) {
                    if (try_again) {
                        std::this_thread::yield();
                    }
                } else {
                    spdlog::error("gpu2cpu RBException: {0}", e.what());
                }
            } catch(const std::runtime_error &e) {
                spdlog::error("gpu2cpu std::runtime_error: {0}", e.what());
            } catch(const std::exception &e) {
                spdlog::error("gpu2cpu std::exception: {0}", e.what());
            }
        }
        ring_output_cpu->end_writing();
        spdlog::info("Exiting gpu2cpu thread (finished)");
    });

    auto recv_thread = std::thread([&](){
        spdlog::info("start receiving thread");

        // set numa affinity of receiver thread to 2
        affinity::affinitySetCore(2);

        bool try_again = true;
        while (try_again) {
            try {

                // can we open a sequence if there is none ??
                auto read_seq = ReadSequence::earliest_or_latest(ring_output_cpu, true, false);
                try_again = false;

                for (std::size_t n=0; n < niter; n++) {

                    // this should block !!!
                    ReadSpan read_span(&read_seq, skip_offset, nbytes);

                    auto* data_header = (ImageHeader*)read_seq.header();
                    // timestamp

                    //Access the data from the span with a pointer
                    void* data_read = read_span.data();

                    //Copy the data into a readable format
                    uint32_t *my_data = static_cast<uint32_t*>(data_read);

                    // check data
                    bool data_ok = true;
                    for (std::size_t i=0; i<data_header->height; i++) {
                        std::size_t s = i*data_header->width;
                        for (std::size_t j=0; j<data_header->width; j++) {
                            if (my_data[s+j] != j) {
                                data_ok = false;
                                break;
                            }
                        }
                    }
                    EXPECT_EQ(data_ok, true);

                    received_packages++;
                    received_bytes += (data_header->height * data_header->width * sizeof(uint32_t));

                    read_seq.increment_to_next();

                    if (!is_running) {
                        spdlog::info("Exiting receiving thread (not running)");
                        return;
                    }
                }
            } catch(const RBException &e) {
                if (e.status() == RBStatus::STATUS_END_OF_DATA) {
                    if (try_again) {
                        std::this_thread::yield();
                    }
                } else {
                    spdlog::error("Receiver RBException: {0}", e.what());
                }
            } catch(const std::runtime_error &e) {
                spdlog::error("Receiver std::runtime_error: {0}", e.what());
            } catch(const std::exception &e) {
                spdlog::error("Receiver std::exception: {0}", e.what());
            }
        }
        spdlog::info("Exiting receiving thread (finished)");

    });



    // writer in main
    {
        spdlog::info("start writing to buffer");

        // set numa affinity of writer thread to 1
        affinity::affinitySetCore(1);

        ring_input_cpu->begin_writing();

        // sequence with no name
        ImageHeader my_header;

        // Writing to the buffer
        for (std::size_t i=0; i < niter; i++ ){

            // write header
            my_header.timestamp = system_clock::now().time_since_epoch().count();
            my_header.width = width;
            my_header.height = height;

            {
                //open a sequence on the ring.
                WriteSequence write_seq(ring_input_cpu, seq_name, i, my_header_size, &my_header, nringlets, skip_offset);
                EXPECT_EQ(write_seq.nringlet(), nringlets);

                //reserve a "span" on this sequence to put our data
                //point our pointer to the span's allocated memory
                WriteSpan write_span(ring_input_cpu, nbytes, nonblocking);

                //create a pointer to pass our data to
                void *data_access = write_span.data();

                // write the frame data to the ring
                create_frame(static_cast<uint32_t*>(data_access), width, height);

                //stop writing
                //commit this span to memory
                write_span.commit(nbytes);
//                std::cout << "wrote frame: " << i << std::endl;
            }
        }
        ring_input_cpu->end_writing();
    }

    cpu2gpu_thread.join();
    gpu2gpu_thread.join();
    gpu2cpu_thread.join();
    recv_thread.join();
    
    spdlog::info("received packages: {0} ", received_packages);
    spdlog::info("received bytes: {0}", received_bytes);

}

#endif // RINGBUFFER_WITH_CUDA