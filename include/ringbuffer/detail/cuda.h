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

#ifndef RINGBUFFER_CUDA_H
#define RINGBUFFER_CUDA_H

#include "ringbuffer/common.h"
#include <vector>
#include <stdexcept>
#include <map>
#include <type_traits>


#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif // WITH_CUDA

namespace ringbuffer {
    namespace cuda {

#ifdef WITH_CUDA
        extern thread_local cudaStream_t g_cuda_stream;

#if THRUST_VERSION >= 100800 // WAR for old Thrust version (e.g., on TK1)
        // @todo: Also need thrust::cuda::par(allocator).on(stream)
    #define thrust_cuda_par_on(stream) thrust::cuda::par.on(stream)
#else
// @todo: Also need thrust::cuda::par(allocator, stream)
#define thrust_cuda_par_on(stream) thrust::cuda::par(stream)
#endif

#define RB_CHECK_CUDA_EXCEPTION(call, err) \
	do { \
		cudaError_t cuda_ret = call; \
		if( cuda_ret != cudaSuccess ) { \
			RB_DEBUG_PRINT(cudaGetErrorString(cuda_ret)); \
		} \
		RB_ASSERT_EXCEPTION(cuda_ret == cudaSuccess, err); \
	} while(0)

#define RB_CHECK_CUDA(call, err) \
	do { \
		cudaError_t cuda_ret = call; \
		if( cuda_ret != cudaSuccess ) { \
			RB_DEBUG_PRINT(cudaGetErrorString(cuda_ret)); \
		} \
		RB_ASSERT(cuda_ret == cudaSuccess, err); \
	} while(0)

#endif // WITH_CUDA

        /*
         * get CUDA stream that is associated with this thread
         */
        RBStatus streamGet(void*       stream);

        /*
         * associate CUDA stream with this thread
         */
        RBStatus streamSet(void const* stream);

        /*
         * synchronize CUDA stream (blocking call)
         */
        RBStatus streamSynchronize();

        /*
         * get CUDA Device
         */
        RBStatus deviceGet(int* device);

        /*
         * set CUDA Device
         */
        RBStatus deviceSet(int  device);

        /*
         * set CUDA Device by PCI Bus ID
         */
        RBStatus deviceSetById(const std::string& pci_bus_id);

        /*
         * Set CUDA Device option to avoid yielding while setting the device (increases latency)
         * Note: This must be called _before_ initializing any devices in the current process
         */
        RBStatus devicesSetNoSpinCPU();

        /*
         * Section blow is only enabled if ringbuffer is compiled with CUDA
         */

#ifdef WITH_CUDA

        int get_cuda_device_cc();

        /*
         * Helper class for JIT CUDA Kernels
         */

        class CUDAKernel {
            CUmodule                  _module;
            CUfunction                _kernel;
            std::string               _func_name;
            std::string               _ptx;
            std::vector<CUjit_option> _opts;

            void cuda_safe_call(CUresult res);

            void create_module(void** optvals = nullptr);

            void destroy_module();

        public:
            CUDAKernel();
            CUDAKernel(const CUDAKernel& other);

            CUDAKernel(const char*   func_name,
                       const char*   ptx,
                       unsigned int  nopts = 0,
                       CUjit_option* opts = nullptr,
                       void**        optvals = nullptr);

            CUDAKernel& set(const char*   func_name,
                            const char*   ptx,
                            unsigned int  nopts = 0,
                            CUjit_option* opts = nullptr,
                            void**        optvals = nullptr);

            ~CUDAKernel();

            void swap(CUDAKernel& other);

            inline explicit operator CUfunction() const { return _kernel; }

            inline CUDAKernel& operator=(const CUDAKernel& other) {
                CUDAKernel tmp(other);
                this->swap(tmp);
                return *this;
            }

            CUresult launch(dim3 grid, dim3 block,
                            unsigned int smem, CUstream stream,
                            std::vector<void*> arg_ptrs);

#if __cplusplus >= 201103L
            template<typename... Args>
            inline CUresult launch(dim3 grid, dim3 block,
                                   unsigned int smem, CUstream stream,
                                   Args... args) {
                return this->launch(grid, block, smem, stream, {(void*)&args...});
            }
#endif

        };

        /*
         * RAII Wrapper for CUDA Stream
         */

        class stream {
            void destroy();
        protected:
            cudaStream_t _obj;
        public:
            // Not copy-assignable
#if __cplusplus >= 201103L
            stream(const cuda::stream& other) = delete;
            stream& operator=(const cuda::stream& other) = delete;

            // Move semantics
            stream(cuda::stream&& other) noexcept;
            cuda::stream& operator=(cuda::stream&& other) noexcept {
                this->destroy();
                this->swap(other);
                return *this;
            }
#else
            stream(const cuda::stream& other);
            stream& operator=(const cuda::stream& other);
#endif

            explicit stream(int priority=0, unsigned flags=cudaStreamDefault);

            ~stream();
            void swap(cuda::stream& other);
            int priority() const;
            unsigned flags() const;
            bool query() const;
            void synchronize() const;
            void wait(cudaEvent_t event, unsigned flags=0) const;
            void addCallback(cudaStreamCallback_t callback,
                             void* userData=0, unsigned flags=0);
            void attachMemAsync(void* devPtr, size_t length, unsigned flags);

            explicit operator const cudaStream_t&() const { return _obj; }
        };


        // This version automatically calls synchronize() before destruction
        class scoped_stream : public cuda::stream {
            typedef cuda::stream super_type;
        public:
            explicit scoped_stream(int priority=0, unsigned flags=cudaStreamNonBlocking);
            ~scoped_stream();
        };

        // This version automatically syncs with a parent stream on construct/destruct
        class child_stream : public cuda::stream {
            typedef cuda::stream super_type;
            cudaStream_t _parent;
            void sync_streams(cudaStream_t dependent, cudaStream_t dependee);
        public:
            explicit child_stream(cudaStream_t parent,
                                  int          priority=0,
                                  unsigned     flags=cudaStreamNonBlocking);
            ~child_stream();
        };

#else // WITH_CUDA
        #define __host__
    #define __device__
#endif // WITH_CUDA


    } // namespace cuda
} // namespace ringbuffer


#endif //RINGBUFFER_CUDA_H
