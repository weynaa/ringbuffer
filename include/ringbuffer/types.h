//
// Created by netlabs on 11/15/19.
//

#ifndef RINGBUFFER_TYPES_H
#define RINGBUFFER_TYPES_H

#include <memory>
#include <map>
#include <mutex>
#include <condition_variable>

#ifdef RINGBUFFER_BOOST_FIBER
#include <boost/fiber/all.hpp>
#include <boost/fiber/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/lock_guard.hpp>
#include <boost/thread/lock_traits.hpp>
#endif

namespace ringbuffer {

    // Forward Declarations
    class Ring;

    class Sequence;
    class SequenceWrapper;
    class ReadSequence;
    class WriteSequence;

    class Span;
    class ReadSpan;
    class WriteSpan;

    // type definitions
    typedef std::shared_ptr<Sequence> SequencePtr;
    typedef uint8_t*             pointer;
    typedef uint8_t const*       const_pointer;
    typedef std::vector<char>    header_type;
    typedef unsigned long long   time_tag_type;
    typedef   signed long long   delta_type;

    typedef std::map<std::size_t,std::size_t> guarantee_set; // offset-->count


    namespace state {
        // Forward Declarations
        class RingState;
        class RingReallocLock;
        class Guarantee;

        // type definitions
#ifdef RINGBUFFER_BOOST_FIBER
        typedef boost::fibers::mutex              mutex_type;
        typedef boost::lock_guard<mutex_type>     lock_guard_type;
        typedef std::unique_lock<mutex_type>      unique_lock_type;
        typedef boost::fibers::condition_variable condition_type;
        typedef RingReallocLock                   realloc_lock_type;
#else
        typedef std::mutex                   mutex_type;
        typedef std::lock_guard<mutex_type>  lock_guard_type;
        typedef std::unique_lock<mutex_type> unique_lock_type;
        typedef std::condition_variable      condition_type;
        typedef RingReallocLock              realloc_lock_type;
#endif
    }

}

#endif //RINGBUFFER_TYPES_H
