//
// Created by netlabs on 11/15/19.
//

#ifndef RINGBUFFER_SPAN_H
#define RINGBUFFER_SPAN_H

#include "ringbuffer/common.h"
#include "ringbuffer/types.h"


namespace ringbuffer {

    class Span {
        Ring*        m_ring;
        std::size_t  m_size;

    protected:
        // WAR for awkwardness in subclass constructors
        void set_base_size(std::size_t size);

    public:
        // No copy or move
        Span(Span const& )            = delete;
        Span& operator=(Span const& ) = delete;
        Span(Span&& )                 = delete;
        Span& operator=(Span&& )      = delete;

        Span(Ring* ring, std::size_t size);
        virtual ~Span();

        Ring*           ring() const;
        std::size_t     size() const;
        // Note: These two are only safe to read while a span is open (preventing resize)
        std::size_t     stride() const;
        std::size_t     nringlet() const;

        virtual void*          data()     const = 0;
        virtual std::size_t    offset()   const = 0;
    };
    
    
    class WriteSpan : public Span {
        std::size_t     m_begin;
        std::size_t     m_commit_size;
        void*           m_data;
    public:
        // No copy or move
        WriteSpan(WriteSpan const& )            = delete;
        WriteSpan& operator=(WriteSpan const& ) = delete;
        WriteSpan(WriteSpan&& )                 = delete;
        WriteSpan& operator=(WriteSpan&& )      = delete;

        WriteSpan(Ring* ring, std::size_t size, bool nonblocking);
        ~WriteSpan() override;

        WriteSpan* commit(std::size_t size);

        void*           data() const override;
        // Note: This is the offset relative to the beginning of the ring,
        //         as wspans aren't firmly associated with a sequence.
        // @todo: This is likely to be confusing compared to ReadSpan::offset
        //         Can't easily change the name though because it's a shared API
        std::size_t     offset() const override;
    };
    
    
    class ReadSpan : public Span {
        ReadSequence*   m_sequence;
        std::size_t     m_begin;
        void*           m_data;

    public:
        // No copy or move
        ReadSpan(ReadSpan const& )            = delete;
        ReadSpan& operator=(ReadSpan const& ) = delete;
        ReadSpan(ReadSpan&& )                 = delete;
        ReadSpan& operator=(ReadSpan&& )      = delete;

        ReadSpan(ReadSequence* sequence,
                 std::size_t  offset,
                 std::size_t  size);
        ~ReadSpan();

        std::size_t size_overwritten() const;

        void*       data()     const override;
        // Note: This is the offset relative to the beginning of the sequence
        std::size_t offset()   const override;
    };
    
    

}

#endif //RINGBUFFER_SPAN_H
