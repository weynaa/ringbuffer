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

// from: https://raw.githubusercontent.com/nemequ/portable-snippets/master/debug-trap/debug-trap.h

#ifndef RINGBUFFER_DEBUG_TRAP_H
#define RINGBUFFER_DEBUG_TRAP_H

#if !defined(RINGBUFFER_NDEBUG) && defined(NDEBUG) && !defined(RINGBUFFER_DEBUG)
#  define RINGBUFFER_NDEBUG 1
#endif

#if defined(__has_builtin) && !defined(__ibmxl__)
#  if __has_builtin(__builtin_debugtrap)
#    define ringbuffer_trap() __builtin_debugtrap()
#  elif __has_builtin(__debugbreak)
#    define ringbuffer_trap() __debugbreak()
#  endif
#endif
#if !defined(ringbuffer_trap)
#  if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#    define ringbuffer_trap() __debugbreak()
#  elif defined(__ARMCC_VERSION)
#    define ringbuffer_trap() __breakpoint(42)
#  elif defined(__ibmxl__) || defined(__xlC__)
#    include <builtins.h>
#    define ringbuffer_trap() __trap(42)
#  elif defined(__DMC__) && defined(_M_IX86)
static inline void ringbuffer_trap(void) { __asm int 3h; }
#  elif defined(__i386__) || defined(__x86_64__)
static inline void ringbuffer_trap(void) { __asm__ __volatile__("int $03"); }
#  elif defined(__thumb__)
static inline void ringbuffer_trap(void) { __asm__ __volatile__(".inst 0xde01"); }
#  elif defined(__aarch64__)
     static inline void ringbuffer_trap(void) { __asm__ __volatile__(".inst 0xd4200000"); }
#  elif defined(__arm__)
     static inline void ringbuffer_trap(void) { __asm__ __volatile__(".inst 0xe7f001f0"); }
#  elif defined (__alpha__) && !defined(__osf__)
     static inline void ringbuffer_trap(void) { __asm__ __volatile__("bpt"); }
#  elif defined(_54_)
     static inline void ringbuffer_trap(void) { __asm__ __volatile__("ESTOP"); }
#  elif defined(_55_)
     static inline void ringbuffer_trap(void) { __asm__ __volatile__(";\n .if (.MNEMONIC)\n ESTOP_1\n .else\n ESTOP_1()\n .endif\n NOP"); }
#  elif defined(_64P_)
     static inline void ringbuffer_trap(void) { __asm__ __volatile__("SWBP 0"); }
#  elif defined(_6x_)
     static inline void ringbuffer_trap(void) { __asm__ __volatile__("NOP\n .word 0x10000000"); }
#  elif defined(__STDC_HOSTED__) && (__STDC_HOSTED__ == 0) && defined(__GNUC__)
#    define ringbuffer_trap() __builtin_trap()
#  else
#    include <signal.h>
#    if defined(SIGTRAP)
#      define ringbuffer_trap() raise(SIGTRAP)
#    else
#      define ringbuffer_trap() raise(SIGABRT)
#    endif
#  endif
#endif

#if defined(HEDLEY_LIKELY)
#  define RINGBUFFER_DBG_LIKELY(expr) HEDLEY_LIKELY(expr)
#elif defined(__GNUC__) && (__GNUC__ >= 3)
#  define RINGBUFFER_DBG_LIKELY(expr) __builtin_expect(!!(expr), 1)
#else
#  define RINGBUFFER_DBG_LIKELY(expr) (!!(expr))
#endif

#if !defined(RINGBUFFER_NDEBUG) || (RINGBUFFER_NDEBUG == 0)
#  define ringbuffer_dbg_assert(expr) do { \
    if (!RINGBUFFER_DBG_LIKELY(expr)) { \
      ringbuffer_trap(); \
    } \
  } while (0)
#else
#  define ringbuffer_dbg_assert(expr)
#endif


#endif //RINGBUFFER_DEBUG_TRAP_H
