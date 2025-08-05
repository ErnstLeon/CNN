#ifndef LOOP_UNROLL_HPP
#define LOOP_UNROLL_HPP

#if defined(__GNUC__)
  #define UNROLL_PRAGMA _Pragma("GCC unroll 4")
#elif defined(__clang__)
  #define UNROLL_PRAGMA _Pragma("clang loop unroll(enable)")
#else
  #define UNROLL_PRAGMA
#endif

#endif // LOOP_UNROLL_HPP