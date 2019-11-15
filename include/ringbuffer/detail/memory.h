//
// Created by netlabs on 11/14/19.
//

#ifndef RINGFLOW_MEMORY_H
#define RINGFLOW_MEMORY_H

#include "ringbuffer/common.h"

#ifndef RINGBUFFER_ALIGNMENT
    #define RINGBUFFER_ALIGNMENT 4096//512
#endif


namespace ringbuffer {
    namespace memory {

        /*
         * allocate memory of size in given space
         */
        RBStatus malloc_(void** ptr, std::size_t size, RBSpace space);

        /*
         * free allocated memory from given space
         */
        RBStatus free_(void* ptr, RBSpace space);

        /*
         * get space associated with given address
         */
        RBStatus getSpace(const void* ptr, RBSpace* space);

        /*
         * copy memory from one space to another
         * Note: This is sync wrt host but async wrt device
         */
        RBStatus memcpy_(void*       dst,
                         RBSpace     dst_space,
                         const void* src,
                         RBSpace     src_space,
                         std::size_t count);

        /*
         * copy 2D array from one space to another
         */
        RBStatus memcpy2D(void*       dst,
                          std::size_t dst_stride,
                          RBSpace     dst_space,
                          const void* src,
                          std::size_t src_stride,
                          RBSpace     src_space,
                          std::size_t width,
                          std::size_t height);

        /*
         * set memory to given value
         * Note: only works for byte types
         */
        RBStatus memset_(void*        ptr,
                         RBSpace      space,
                         int          value,
                         std::size_t  count);

        /*
         * set memory of 2D array to given value
         * Note: only works for byte types
         */
        RBStatus memset2D(void*       ptr,
                          std::size_t stride,
                          RBSpace     space,
                          int         value,
                          std::size_t width,
                          std::size_t height);

        /*
         * get global alignment
         */
        std::size_t getAlignment();

    }
}

#endif //RINGFLOW_MEMORY_H
