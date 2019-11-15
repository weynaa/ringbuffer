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

#include "ringbuffer/detail/cuda.h"

using namespace ringbuffer;
using namespace ringbuffer::cuda;

namespace ringbuffer {
    namespace cuda {

#ifdef WITH_CUDA
        thread_local cudaStream_t g_cuda_stream = cudaStreamPerThread;
#endif

        RBStatus streamGet(void* stream) {
            RB_ASSERT(stream, RBStatus::STATUS_INVALID_POINTER);
#ifdef WITH_CUDA
            *(cudaStream_t*)stream = g_cuda_stream;
#else
            RB_FAIL("Built with CUDA support (rfStreamGet)", RBStatus::STATUS_INVALID_STATE);
#endif
            return RBStatus::STATUS_SUCCESS;
        }

        RBStatus streamSet(void const* stream) {
            RB_ASSERT(stream, RBStatus::STATUS_INVALID_POINTER);
#ifdef WITH_CUDA
            g_cuda_stream = *(cudaStream_t*)stream;
#endif
            return RBStatus::STATUS_SUCCESS;
        }

        RBStatus deviceGet(int* device) {
            RB_ASSERT(device, RBStatus::STATUS_INVALID_POINTER);
#ifdef WITH_CUDA
            RB_CHECK_CUDA(cudaGetDevice(device), RBStatus::STATUS_DEVICE_ERROR);
#else
            *device = -1;
#endif
            return RBStatus::STATUS_SUCCESS;
        }

        RBStatus deviceSet(int device) {
#ifdef WITH_CUDA
            RB_CHECK_CUDA(cudaSetDevice(device), RBStatus::STATUS_DEVICE_ERROR);
#endif
            return RBStatus::STATUS_SUCCESS;
        }

        RBStatus deviceSetById(const std::string& pci_bus_id) {
#ifdef WITH_CUDA
            int device;
            RB_CHECK_CUDA(cudaDeviceGetByPCIBusId(&device, pci_bus_id.c_str()),
                          RBStatus::STATUS_DEVICE_ERROR);
            return cuda::deviceSet(device);
#else
            return RBStatus::STATUS_SUCCESS;
#endif
        }

        RBStatus streamSynchronize() {
#ifdef WITH_CUDA
            RB_CHECK_CUDA(cudaStreamSynchronize(g_cuda_stream),
                          RBStatus::STATUS_DEVICE_ERROR);
#endif
            return RBStatus::STATUS_SUCCESS;
        }

        RBStatus devicesSetNoSpinCPU() {
#ifdef WITH_CUDA
            int old_device;
            RB_CHECK_CUDA(cudaGetDevice(&old_device), RBStatus::STATUS_DEVICE_ERROR);
            int ndevices;
            RB_CHECK_CUDA(cudaGetDeviceCount(&ndevices), RBStatus::STATUS_DEVICE_ERROR);
            for( int d=0; d<ndevices; ++d ) {
                RB_CHECK_CUDA(cudaSetDevice(d), RBStatus::STATUS_DEVICE_ERROR);
                RB_CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync),
                              RBStatus::STATUS_DEVICE_ERROR);
            }
            RB_CHECK_CUDA(cudaSetDevice(old_device), RBStatus::STATUS_DEVICE_ERROR);
#endif
            return RBStatus::STATUS_SUCCESS;
        }

#ifdef WITH_CUDA

        int get_cuda_device_cc() {
            int device;
            cudaGetDevice(&device);
            int cc_major;
            cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device);
            int cc_minor;
            cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device);
            return cc_major*10 + cc_minor;
        }


        /*
         * CUDAKernel implementation
         */

        void CUDAKernel::cuda_safe_call(CUresult res) {
            if( res != CUDA_SUCCESS ) {
                const char* msg;
                cuGetErrorName(res, &msg);
                throw std::runtime_error(msg);
            }
        }

        void CUDAKernel::create_module(void** optvals) {
            cuda_safe_call(cuModuleLoadDataEx(&_module, _ptx.c_str(),
                                              _opts.size(), &_opts[0], optvals));
            cuda_safe_call(cuModuleGetFunction(&_kernel, _module,
                                               _func_name.c_str()));
        }

        void CUDAKernel::destroy_module() {
            if( _module ) {
                cuModuleUnload(_module);
            }
        }

        CUDAKernel::CUDAKernel() : _module(0), _kernel(0) {}

        CUDAKernel::CUDAKernel(const CUDAKernel& other) : _module(0), _kernel(0) {
            if( other._module ) {
                _func_name = other._func_name;
                _ptx       = other._ptx;
                _opts      = other._opts;
                this->create_module();
            }
        }

        CUDAKernel::CUDAKernel(const char*   func_name,
                               const char*   ptx,
                               unsigned int  nopts,
                               CUjit_option* opts,
                               void**        optvals) {
            _func_name = func_name;
            _ptx = ptx;
            _opts.assign(opts, opts + nopts);
            this->create_module(optvals);
        }

        CUDAKernel& CUDAKernel::set(const char*   func_name,
                                    const char*   ptx,
                                    unsigned int  nopts,
                                    CUjit_option* opts,
                                    void**        optvals) {
            this->destroy_module();
            _func_name = func_name;
            _ptx = ptx;
            _opts.assign(opts, opts + nopts);
            this->create_module(optvals);
            return *this;
        }

        void CUDAKernel::swap(CUDAKernel& other) {
            std::swap(_func_name, other._func_name);
            std::swap(_ptx, other._ptx);
            std::swap(_opts, other._opts);
            std::swap(_module, other._module);
            std::swap(_kernel, other._kernel);
        }

        CUDAKernel::~CUDAKernel() {
            this->destroy_module();
        }

        CUresult CUDAKernel::launch(dim3 grid, dim3 block,
                                    unsigned int smem, CUstream stream,
                                    std::vector<void*> arg_ptrs) {
            //void* arg_ptrs[]) {
            // Note: This returns "INVALID_ARGUMENT" if 'args' do not match what is
            //         expected (e.g., too few args, wrong types)
            return cuLaunchKernel(_kernel,
                                  grid.x, grid.y, grid.z,
                                  block.x, block.y, block.z,
                                  smem, stream,
                                  &arg_ptrs[0], NULL);
        }


        /*
         * RAII wrapper for CUDA Streams
         */

        void stream::destroy() {
            if (_obj) {
                cudaStreamDestroy(_obj);
                _obj = 0;
            }
        }

#if __cplusplus >= 201103L
        stream::stream(cuda::stream &&other) noexcept : _obj(0) { this->swap(other); }
#endif

        stream::stream(int priority, unsigned flags) : _obj(0) {
            if (priority > 0) {
                int least_priority;
                int greatest_priority;
                cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
                RB_CHECK_CUDA_EXCEPTION(cudaStreamCreateWithPriority(&_obj, flags, greatest_priority),
                                        RBStatus::STATUS_DEVICE_ERROR);
            } else {
                RB_CHECK_CUDA_EXCEPTION(cudaStreamCreateWithFlags(&_obj, flags), RBStatus::STATUS_DEVICE_ERROR);
            }
        }

        stream::~stream() { this->destroy(); }

        void stream::swap(cuda::stream &other) { std::swap(_obj, other._obj); }

        int stream::priority() const {
            int val;
            RB_CHECK_CUDA_EXCEPTION(cudaStreamGetPriority(_obj, &val), RBStatus::STATUS_DEVICE_ERROR);
            return val;
        }

        unsigned stream::flags() const {
            unsigned val;
            RB_CHECK_CUDA_EXCEPTION(cudaStreamGetFlags(_obj, &val), RBStatus::STATUS_DEVICE_ERROR);
            return val;
        }

        bool stream::query() const {
            cudaError_t ret = cudaStreamQuery(_obj);
            if (ret == cudaErrorNotReady) {
                return false;
            } else {
                RB_CHECK_CUDA_EXCEPTION(ret, RBStatus::STATUS_DEVICE_ERROR);
                return true;
            }
        }

        void stream::synchronize() const {
            cudaStreamSynchronize(_obj);
            RB_CHECK_CUDA_EXCEPTION(cudaGetLastError(), RBStatus::STATUS_DEVICE_ERROR);
        }

        void stream::wait(cudaEvent_t event, unsigned flags) const {
            RB_CHECK_CUDA_EXCEPTION(cudaStreamWaitEvent(_obj, event, flags), RBStatus::STATUS_DEVICE_ERROR);
        }

        void stream::addCallback(cudaStreamCallback_t callback, void *userData, unsigned flags) {
            RB_CHECK_CUDA_EXCEPTION(cudaStreamAddCallback(_obj, callback, userData, flags),
                                    RBStatus::STATUS_DEVICE_ERROR);
        }

        void stream::attachMemAsync(void *devPtr, size_t length, unsigned flags) {
            RB_CHECK_CUDA_EXCEPTION(cudaStreamAttachMemAsync(_obj, devPtr, length, flags),
                                    RBStatus::STATUS_DEVICE_ERROR);
        }

        // This version automatically calls synchronize() before destruction
        scoped_stream::scoped_stream(int priority, unsigned flags)
                : super_type(priority, flags) {}

        scoped_stream::~scoped_stream() { this->synchronize(); }


        // This version automatically syncs with a parent stream on construct/destruct
        void child_stream::sync_streams(cudaStream_t dependent, cudaStream_t dependee) {
            // Record event in dependee and make dependent wait for it
            cudaEvent_t event;
            RB_CHECK_CUDA_EXCEPTION(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                                    RBStatus::STATUS_DEVICE_ERROR);
            RB_CHECK_CUDA_EXCEPTION(cudaEventRecord(event, dependee), RBStatus::STATUS_DEVICE_ERROR);
            RB_CHECK_CUDA_EXCEPTION(cudaStreamWaitEvent(dependent, event, 0), RBStatus::STATUS_DEVICE_ERROR);
            RB_CHECK_CUDA_EXCEPTION(cudaEventDestroy(event), RBStatus::STATUS_DEVICE_ERROR);
        }

        child_stream::child_stream(cudaStream_t parent, int priority, unsigned flags)
                : super_type(priority, flags), _parent(parent) {
            sync_streams(this->_obj, _parent);
        }

        child_stream::~child_stream() {
            sync_streams(_parent, this->_obj);
        }

#endif // WITH_CUDA

    } // namespace cuda
} // namespace ringbuffer

