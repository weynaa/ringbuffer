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

#include "ringbuffer/detail/affinity.h"


/*
 * This part should use the hwloc library, maybe changing the interface to support logical cores and numa nodes
*/
#if defined __linux__ && __linux__
#include <pthread.h>
#include <unistd.h>
//#include <sched.h>
#endif
#include <errno.h>

#ifdef RINGBUFFER_WITH_OMP
#include <omp.h>
#endif // RINGBUFFER_WITH_OMP

namespace ringbuffer {
    namespace affinity {


        // Note: Pass core_id = -1 to unbind
        RBStatus affinitySetCore(int core) {
#if defined __linux__ && __linux__
            // Check for valid core
            int ncore = sysconf(_SC_NPROCESSORS_ONLN);
            RB_ASSERT(core >= -1 && core < ncore, RBStatus::STATUS_INVALID_ARGUMENT);
            // Create core mask
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            if( core >= 0 ) {
                // Set specified core
                CPU_SET(core, &cpuset);
            }
            else {
                // Set all cores (i.e., 'un-bind')
                for( int c=0; c<ncore; ++c ) {
                    CPU_SET(c, &cpuset);
                }
            }
            // Apply to current thread
            pthread_t tid = pthread_self();
            // Set affinity (note: non-portable)
            int ret = pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
            //int ret = sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);
            if( ret == 0 ) {
                return RBStatus::STATUS_SUCCESS;
            }
            else {
                return RBStatus::STATUS_INVALID_ARGUMENT;
            }
#else
            //#warning CPU core binding/affinity not supported on this OS
            return RBStatus::STATUS_UNSUPPORTED;
#endif
        }
        
        
        RBStatus affinityGetCore(int* core) {
#if defined __linux__ && __linux__
			RB_ASSERT(core, RBStatus::STATUS_INVALID_POINTER);
            pthread_t tid = pthread_self();
            cpu_set_t cpuset;
            RB_ASSERT(!pthread_getaffinity_np(tid, sizeof(cpu_set_t), &cpuset),
                      RBStatus::STATUS_INTERNAL_ERROR);
            if( CPU_COUNT(&cpuset) > 1 ) {
                // Return -1 if more than one core is set
                // @todo: Should really check if all cores are set, otherwise fail
                *core = -1;
                return RBStatus::STATUS_SUCCESS;
            }
            else {
                int ncore = sysconf(_SC_NPROCESSORS_ONLN);
                for( int c=0; c<ncore; ++c ) {
                    if( CPU_ISSET(c, &cpuset) ) {
                        *core = c;
                        return RBStatus::STATUS_SUCCESS;
                    }
                }
            }
            // No cores are set! (Not sure if this is possible)
            return RBStatus::STATUS_INVALID_STATE;
#else
			//#warning CPU core binding / affinity not supported on this OS
			return RBStatus::STATUS_UNSUPPORTED;
#endif
		}

        
        RBStatus affinitySetOpenMPCores(std::size_t nthread, const int* thread_cores) {
#ifdef RINGBUFFER_WITH_OMP
            int host_core = -1;
            // @todo: Check these for errors
            affinityGetCore(&host_core);
            affinitySetCore(-1); // Unbind host core to unconstrain OpenMP threads
            omp_set_num_threads(nthread);
#pragma omp parallel for schedule(static, 1)
            for( std::size_t t=0; t<nthread; ++t ) {
                int tid = omp_get_thread_num();
                affinitySetCore(thread_cores[tid]);
            }
            return affinitySetCore(host_core);
#else
            return RBStatus::STATUS_UNSUPPORTED;
#endif
        }

    }
}