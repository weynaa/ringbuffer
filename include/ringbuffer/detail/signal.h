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


#ifndef RINGBUFFER_SIGNAL_H
#define RINGBUFFER_SIGNAL_H

#include <functional>
#include <map>

namespace ringbuffer {
    namespace detail {

        // A signal object may call multiple slots with the
        // same signature. You can connect functions to the signal
        // which will be called when the dispatch() method on the
        // signal object is invoked. Any argument passed to dispatch()
        // will be passed to the given functions.

        template <typename... Args>
        class Signal {

        public:

            Signal() : current_id_(0) {}

            // copy creates new signal
            Signal(Signal const& other) : current_id_(0) {}

            // connects a member function to this Signal
            template <typename T>
            int connect_member(T *inst, void (T::*func)(Args...)) {
                return connect([=](Args... args) {
                    (inst->*func)(args...);
                });
            }

            // connects a const member function to this Signal
            template <typename T>
            int connect_member(T *inst, void (T::*func)(Args...) const) {
                return connect([=](Args... args) {
                    (inst->*func)(args...);
                });
            }

            // connects a std::function to the signal. The returned
            // value can be used to disconnect the function again
            int connect(std::function<void(Args...)> const& slot) const {
                slots_.insert(std::make_pair(++current_id_, slot));
                return current_id_;
            }

            // disconnects a previously connected function
            void disconnect(int id) const {
                slots_.erase(id);
            }

            // disconnects all previously connected functions
            void disconnect_all() const {
                slots_.clear();
            }

            // calls all connected functions
            void dispatch(Args... p) {
                for(auto it : slots_) {
                    it.second(p...);
                }
            }

            // assignment creates new Signal
            Signal& operator=(Signal const& other) {
                disconnect_all();
            }

            std::size_t subscriberCount() {
                return slots_.size();
            }

        private:
            mutable std::map<int, std::function<void(Args...)>> slots_;
            mutable int current_id_;
        };



    }
}

#endif //RINGBUFFER_SIGNAL_H
