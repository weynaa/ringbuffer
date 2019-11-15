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


#ifndef RINGBUFFER_SEQUENCE_H
#define RINGBUFFER_SEQUENCE_H

#include "ringbuffer/common.h"
#include "ringbuffer/types.h"

namespace ringbuffer {



    class Sequence {
        friend class Ring;
        enum { RF_SEQUENCE_OPEN = (std::size_t)-1 };
        Ring*          m_ring;
        std::string    m_name;
        time_tag_type  m_time_tag;
        std::size_t    m_nringlet;
        std::size_t    m_begin;
        std::size_t    m_end;
        header_type    m_header;
        SequencePtr    m_next;
        std::size_t    m_readrefcount; // ever used ??

    public:
        Sequence(Ring*         ring,
                 std::string   name,
                 time_tag_type time_tag,
                 std::size_t   header_size,
                 const void*   header,
                 std::size_t   nringlet,
                 std::size_t   begin);

// not implemented in bifrost ??
//        void finish(std::size_t offset_from_head=0);
//        void close();

        void set_next(SequencePtr next);

        inline bool          is_finished() const { return m_end != RF_SEQUENCE_OPEN; }
        inline Ring*         ring()              { return m_ring; }
        inline std::string   name()        const { return m_name; }
        inline time_tag_type time_tag()    const { return m_time_tag; }
        inline const void*   header()      const { return m_header.size() ? &m_header[0] : nullptr; }
        inline std::size_t   header_size() const { return m_header.size(); }
        inline std::size_t   nringlet()    const { return m_nringlet; }
        inline std::size_t   begin()       const { return m_begin; }
        inline std::size_t   end()         const { return m_end; }
    };
    

    class SequenceWrapper {
    protected:
        SequencePtr _sequence;
    public:
        inline explicit SequenceWrapper(SequencePtr sequence) : _sequence(sequence) {}
        inline SequencePtr   sequence()    const { return _sequence; }
        inline bool          is_finished() const { return _sequence->is_finished(); }
        inline Ring*         ring()              { return _sequence->ring(); }
        inline std::string   name()        const { return _sequence->name(); }
        inline time_tag_type time_tag()    const { return _sequence->time_tag(); }
        inline const void*   header()      const { return _sequence->header(); }
        inline std::size_t   header_size() const { return _sequence->header_size(); }
        inline std::size_t   nringlet()    const { return _sequence->nringlet(); }
        inline std::size_t   begin()       const { return _sequence->begin(); }
    };


    class ReadSequence : public SequenceWrapper {
        std::unique_ptr<state::Guarantee> _guarantee;
    public:
        // TODO: See if can make these function bodies a bit more concise
        static ReadSequence earliest_or_latest(Ring* ring, bool with_guarantee, bool latest);
        static ReadSequence by_name(Ring* ring, const std::string& name, bool with_guarantee);
        static ReadSequence at(Ring* ring, time_tag_type time_tag, bool with_guarantee);
        ReadSequence(SequencePtr sequence, std::unique_ptr<state::Guarantee>& guarantee);

        void increment_to_next();

        inline std::unique_ptr<state::Guarantee>&       guarantee()       { return _guarantee; }
        inline std::unique_ptr<state::Guarantee> const& guarantee() const { return _guarantee; }
        /*
          // TODO: This is needed for rfRingSequenceOpenSame, but it's not clear
          //         that that API is really needed. Also need to delete
          //         assignment and move constructors if this is implemented.
        // Copy constructor points to same underlying RFsequence_impl object, but
        //   creates its own guarantee.
        ReadSequence(ReadSequence const& other)
            : SequenceWrapper(other.sequence()),
              _guarantee(new Guarantee(*other._guarantee)) {}
        */
    };


    class WriteSequence : public SequenceWrapper {
        std::size_t _end_offset_from_head;
    public:
        WriteSequence(WriteSequence const& )            = delete;
        WriteSequence& operator=(WriteSequence const& ) = delete;
        WriteSequence(WriteSequence&& )                 = delete;
        WriteSequence& operator=(WriteSequence&& )      = delete;

        inline WriteSequence(Ring* ring, const std::string& name,
                             time_tag_type time_tag, std::size_t header_size,
                             const void* header, std::size_t nringlet,
                             std::size_t offset_from_head=0);

        ~WriteSequence();

        void set_end_offset_from_head(std::size_t end_offset_from_head);
    };
    
}


#endif //RINGBUFFER_SEQUENCE_H
