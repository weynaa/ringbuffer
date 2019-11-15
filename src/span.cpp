//
// Created by netlabs on 11/15/19.
//

#include "ringbuffer/span.h"
#include "ringbuffer/ring.h"
#include "ringbuffer/sequence.h"

namespace ringbuffer {

    void Span::set_base_size(std::size_t size) { m_size = size; }

    Span::Span(Ring* ring, std::size_t size) : m_ring(ring), m_size(size) {}
    Span::~Span() {}

    Ring* Span::ring() const { return m_ring; }
    std::size_t Span::size()     const { return m_size; }
    std::size_t Span::stride()   const { return m_ring->current_stride(); }
    std::size_t Span::nringlet() const { return m_ring->current_nringlet(); }


    WriteSpan::WriteSpan(Ring* ring, std::size_t size, bool nonblocking)
            : Span(ring, size), m_begin(0), m_commit_size(size), m_data(nullptr) {
        this->ring()->reserve_span(size, &m_begin, &m_data, nonblocking);
    }

    WriteSpan* WriteSpan::commit(std::size_t size) {
        RB_ASSERT_EXCEPTION(size <= this->size(), RBStatus::STATUS_INVALID_ARGUMENT);
        m_commit_size = size;
        return this;
    }

    WriteSpan::~WriteSpan() {
        this->ring()->commit_span(m_begin, this->size(), m_commit_size);
    }

    void*       WriteSpan::data()     const { return m_data; }
    std::size_t WriteSpan::offset()   const { return m_begin; }


    ReadSpan::ReadSpan(ReadSequence*   sequence,
                       std::size_t    offset, // Relative to sequence beg
                       std::size_t    requested_size)
            : Span(sequence->ring(), requested_size),
              m_sequence(sequence), m_begin(0), m_data(nullptr) {
        std::size_t returned_size = requested_size;
        this->ring()->acquire_span(sequence, offset, &returned_size, &m_begin, &m_data);
        this->set_base_size(returned_size);
    }

    ReadSpan::~ReadSpan() {
        this->ring()->release_span(m_sequence, m_begin, this->size());
    }

    std::size_t ReadSpan::size_overwritten() const {
        if( m_sequence->guarantee() ) {
            return 0;
        }
        const auto* ring = this->ring();
        std::size_t tail = ring->current_tail_offset();
        return std::max(std::min(delta_type(tail - m_begin),
                                 delta_type(this->size())),
                        delta_type(0));
    }

    void*       ReadSpan::data()     const { return m_data; }
    std::size_t ReadSpan::offset()   const { return m_begin - m_sequence->begin(); }

}
