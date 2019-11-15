//
// Created by netlabs on 11/15/19.
//

#include "ringbuffer/detail/trace.h"

namespace ringbuffer {
    namespace trace {
        namespace profile_detail {

            unsigned simple_hash(const char* c) {
                enum { M = 33 };
                unsigned hash = 5381;
                while( *c ) { hash = hash*M + *c++; }
                return hash;
            }

            uint32_t get_color(unsigned hash) {
                const uint32_t colors[] = {
                        0x00aedb, 0xa200ff, 0xf47835, 0xd41243, 0x8ec127,
                        0xffb3ba, 0xffdfba, 0xffffba, 0xbaffc9, 0xbae1ff,
                        0xbbcbdb, 0x9ebd9e, 0xdd855c, 0xf1e8ca, 0x745151,
                        0x2e4045, 0x83adb5, 0xc7bbc9, 0x5e3c58, 0xbfb5b2,
                        0xff77aa, 0xaaff77, 0x77aaff, 0xffffff, 0x000000
                };
                const int ncolor = sizeof(colors) / sizeof(uint32_t);
                return colors[hash % ncolor];
            }

        } // namespace profile_detail

#ifdef WITH_CUDA

        namespace nvtx {

            thread_local TracerStreamMap g_nvtx_streams;

            void AsyncTracer::range_start_callback(cudaStream_t stream, cudaError_t status, void* userData) {
                auto* range = (AsyncTracer*)userData;
                range->_id = nvtxRangeStartEx(&range->_attrs);
            }

            void AsyncTracer::range_end_callback(cudaStream_t stream, cudaError_t status, void* userData) {
                auto* range = (AsyncTracer*)userData;
                nvtxRangeEnd(range->_id);
                range->_id = 0;
                delete range;
            }

            AsyncTracer::AsyncTracer(cudaStream_t stream) : _stream(stream), _id(0), _attrs() {}

            void AsyncTracer::start(const char* msg, uint32_t color, uint32_t category) {
                _msg = msg;
                _attrs.version       = NVTX_VERSION;
                _attrs.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                _attrs.colorType     = NVTX_COLOR_ARGB;
                _attrs.color         = color;
                _attrs.messageType   = NVTX_MESSAGE_TYPE_ASCII;
                _attrs.message.ascii = _msg.c_str();
                _attrs.category      = category;
                cudaStreamAddCallback(_stream, range_start_callback, (void*)this, 0);
            }

            void AsyncTracer::end() {
                cudaStreamAddCallback(_stream, range_end_callback, (void*)this, 0);
            }

        } // namespace nvtx


        void ScopedTracer::build_attrs(nvtxEventAttributes_t* attrs) {
            ::memset(attrs, 0, sizeof(*attrs));
            attrs->version       = NVTX_VERSION;
            attrs->size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attrs->colorType     = NVTX_COLOR_ARGB;
            attrs->color         = _color;
            attrs->messageType   = NVTX_MESSAGE_TYPE_ASCII;
            attrs->message.ascii = _name.c_str();
            attrs->category      = _category;
        }

        ScopedTracer::ScopedTracer(const std::string& name, cudaStream_t stream)
                : _name(name),
                  _color(profile_detail::get_color(profile_detail::simple_hash(name.c_str()))),
                  _category(123),
                  _stream(stream) {
            if( _stream ) {
                nvtx::g_nvtx_streams[_stream].push(new nvtx::AsyncTracer(stream));
                nvtx::g_nvtx_streams[_stream].back()->start(("[G]"+_name).c_str(),
                                                            _color, _category);
            } else {
                nvtxEventAttributes_t attrs;
                this->build_attrs(&attrs);
                nvtxRangePushEx(&attrs);
            }
        }

        ScopedTracer::~ScopedTracer() {
            if( _stream ) {
                nvtx::g_nvtx_streams[_stream].front()->end();
                nvtx::g_nvtx_streams[_stream].pop();
            } else {
                nvtxRangePop();
            }
        }

#endif // WITH_CUDA
    } // namespace trace
} // namespace ringbuffer

