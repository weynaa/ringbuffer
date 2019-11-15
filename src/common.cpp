//
// Created by netlabs on 11/14/19.
//


#include "ringbuffer/common.h"
#include <sstream>

static thread_local bool g_debug_enabled = true;

namespace ringbuffer {

    std::string getStatusString(RBStatus status) {
#define STATUS_STRING_CASE(x) case x: return std::string(#x);
        switch( status ) {
            STATUS_STRING_CASE(RBStatus::STATUS_SUCCESS);
            STATUS_STRING_CASE(RBStatus::STATUS_END_OF_DATA);
            STATUS_STRING_CASE(RBStatus::STATUS_WOULD_BLOCK);
            STATUS_STRING_CASE(RBStatus::STATUS_INVALID_POINTER);
            STATUS_STRING_CASE(RBStatus::STATUS_INVALID_HANDLE);
            STATUS_STRING_CASE(RBStatus::STATUS_INVALID_ARGUMENT);
            STATUS_STRING_CASE(RBStatus::STATUS_INVALID_STATE);
            STATUS_STRING_CASE(RBStatus::STATUS_INVALID_SPACE);
            STATUS_STRING_CASE(RBStatus::STATUS_INVALID_SHAPE);
            STATUS_STRING_CASE(RBStatus::STATUS_INVALID_STRIDE);
            STATUS_STRING_CASE(RBStatus::STATUS_INVALID_DTYPE);
            STATUS_STRING_CASE(RBStatus::STATUS_MEM_ALLOC_FAILED);
            STATUS_STRING_CASE(RBStatus::STATUS_MEM_OP_FAILED);
            STATUS_STRING_CASE(RBStatus::STATUS_UNSUPPORTED);
            STATUS_STRING_CASE(RBStatus::STATUS_UNSUPPORTED_SPACE);
            STATUS_STRING_CASE(RBStatus::STATUS_UNSUPPORTED_SHAPE);
            STATUS_STRING_CASE(RBStatus::STATUS_UNSUPPORTED_STRIDE);
            STATUS_STRING_CASE(RBStatus::STATUS_UNSUPPORTED_DTYPE);
            STATUS_STRING_CASE(RBStatus::STATUS_FAILED_TO_CONVERGE);
            STATUS_STRING_CASE(RBStatus::STATUS_INSUFFICIENT_STORAGE);
            STATUS_STRING_CASE(RBStatus::STATUS_DEVICE_ERROR);
            STATUS_STRING_CASE(RBStatus::STATUS_INTERNAL_ERROR);
            default: {
                std::stringstream ss;
                ss << "Invalid status code: " << static_cast<int>(status);
                return ss.str();
            }
        }
#undef STATUS_STRING_CASE
    }

    std::string getSpaceString(RBSpace space) {
        switch( space ) {
            case RBSpace::SPACE_AUTO:         return "auto";
            case RBSpace::SPACE_SYSTEM:       return "system";
            case RBSpace::SPACE_CUDA:         return "cuda";
            case RBSpace::SPACE_CUDA_HOST:    return "cuda_host";
            case RBSpace::SPACE_CUDA_MANAGED: return "cuda_managed";
            case RBSpace::SPACE_SHM:          return "shm";
            default: return "unknown";
        }
    }

    void requireSuccess(RBStatus status) {
        if( status != RBStatus ::STATUS_SUCCESS ) {
            throw RBException(status);
        }
    }

    bool failOnError(RBStatus status) {
//    RBStatus::DISABLE_DEBUG();
        switch( status ) {
            case RBStatus::STATUS_MEM_ALLOC_FAILED:
            case RBStatus::STATUS_MEM_OP_FAILED:
            case RBStatus::STATUS_DEVICE_ERROR:
            case RBStatus::STATUS_INTERNAL_ERROR:
                ringbuffer::requireSuccess(status);
                return false; // Unreachable
            default: return status == RBStatus::STATUS_SUCCESS;
        }
    }


    bool should_report_error(RBStatus err) {
        return (err != RBStatus::STATUS_END_OF_DATA &&
                err != RBStatus::STATUS_WOULD_BLOCK);
    }

    bool getDebugEnabled() {
#if RINGBUFFER_DEBUG
        return g_debug_enabled;
#else
        return false;
#endif
    }

    RBStatus setDebugEnabled(bool b) {
#if !RINGBUFFER_DEBUG
        return RBStatus ::STATUS_INVALID_STATE;
#else
        g_debug_enabled = b;
        return RBStatus ::STATUS_SUCCESS;
#endif
    }

    bool getCudaEnabled() {
#ifdef WITH_CUDA
        return true;
#else
        return false;
#endif
    }

    bool getShmEnabled() {
        return false; // for now ..
    }

} // namespace ringbuffer

