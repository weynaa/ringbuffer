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

#include "ringbuffer/detail/guarantee.h"
#include "ringbuffer/ring.h"

namespace ringbuffer {
    namespace state {

        void Guarantee::create(std::size_t offset) {
            m_offset = offset;
            if (m_ring) {
                m_ring->_add_guarantee(m_offset);
            }
        }

        void Guarantee::destroy() {
            if (m_ring) {
                m_ring->_remove_guarantee(m_offset);
            }
        }

        Guarantee::Guarantee(const std::shared_ptr<Ring>& ring)
                : m_ring(ring), m_offset(0) {
            if (m_ring) {
                auto& state = m_ring->get_state();
                state::lock_guard_type lock(state.mutex);
                this->create(state.tail);
            }
        }

        Guarantee::~Guarantee() {
            if (m_ring) {
                auto& state = m_ring->get_state();
                state::lock_guard_type lock(state.mutex);
                this->destroy();
            }
        }

		Guarantee::Guarantee(Guarantee&&) = default;
		Guarantee& Guarantee::operator=(Guarantee&&) = default;

        void Guarantee::move_nolock(std::size_t offset) {
            this->destroy();
            this->create(offset);
        }

        std::size_t Guarantee::offset() const { return m_offset; }

    }
}
