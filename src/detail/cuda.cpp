//
// Created by netlabs on 11/14/19.
//

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

#endif // WITH_CUDA

    } // namespace cuda
} // namespace ringbuffer

