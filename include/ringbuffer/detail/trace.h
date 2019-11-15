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

#ifndef RINGBUFFER_TRACE_H
#define RINGBUFFER_TRACE_H

#include "ringbuffer/detail/cuda.h"

#include <map>
#include <queue>
#include <string>
#include <cstring>

#ifdef WITH_CUDA
#include <nvToolsExt.h>
#endif //WITH_CUDA

namespace ringbuffer {
    namespace trace {
#if RINGBUFFER_TRACE
        // Note: __PRETTY_FUNCTION__ is GCC-specific
//       __FUNCSIG__ is the equivalent in MSVC
#define RB_TRACE()                         ringbuffer::trace::ScopedTracer _rb_tracer(__PRETTY_FUNCTION__)
#define RB_TRACE_NAME(name)                ringbuffer::trace::ScopedTracer _rb_tracer(name)
#define RB_TRACE_STREAM(stream)            ringbuffer::trace::ScopedTracer _rb_stream_tracer(__PRETTY_FUNCTION__, stream)
#define RB_TRACE_NAME_STREAM(name, stream) ringbuffer::trace::ScopedTracer _rb_stream_tracer(name, stream)
#else // not RINGBUFFER_TRACE
#define RB_TRACE()
#define RB_TRACE_NAME(name)
#define RB_TRACE_STREAM(stream)
#define RB_TRACE_NAME_STREAM(name, stream)
#endif // RINGBUFFER_TRACE

        namespace profile_detail {
            inline unsigned simple_hash(const char* c);
            inline uint32_t get_color(unsigned hash);
        } // namespace profile_detail

#ifdef WITH_CUDA

        namespace nvtx {

            class AsyncTracer {
                cudaStream_t          _stream;
                nvtxRangeId_t         _id;
                std::string           _msg;
                nvtxEventAttributes_t _attrs;
                static void range_start_callback(cudaStream_t stream, cudaError_t status, void* userData);
                static void range_end_callback(cudaStream_t stream, cudaError_t status, void* userData);
            public:
                explicit AsyncTracer(cudaStream_t stream);
                void start(const char* msg, uint32_t color, uint32_t category);
                void end();
            };

            typedef std::map<cudaStream_t,std::queue<AsyncTracer*> > TracerStreamMap;
            extern thread_local TracerStreamMap g_nvtx_streams;

        } // namespace nvtx

#endif // WITH_CUDA

        class ScopedTracer {
            std::string _name;
            uint32_t _color;
            uint32_t _category;
#ifdef WITH_CUDA
            cudaStream_t _stream;
            void build_attrs(nvtxEventAttributes_t *attrs);
#endif
        public:
            // Not copy-assignable
            ScopedTracer(ScopedTracer const &) = delete;
            ScopedTracer &operator=(ScopedTracer const &) = delete;

#ifdef WITH_CUDA
            explicit ScopedTracer(const std::string& name, cudaStream_t stream = nullptr);
            ~ScopedTracer();
#else
            explicit ScopedTracer(std::string name) : _name(std::move(name)) {}
            ~ScopedTracer() {};
#endif
        };
    }
}

#endif //RINGBUFFER_TRACE_H
