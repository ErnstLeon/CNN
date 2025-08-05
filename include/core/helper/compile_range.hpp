#ifndef COMPILE_RANGE_HPP
#define COMPILE_RANGE_HPP

namespace CNN
{

template<size_t STOP, size_t START = 0, size_t STRIDE = 1, typename Func>
requires ((START + STRIDE <= STOP) || START == STOP) && ((STOP - START) % STRIDE == 0)
inline void compile_range(Func&& f)
{
    if constexpr (START == STOP) {
        f.template operator()<START>();
    }
    else{
        f.template operator()<START>();
        compile_range<STOP, START + STRIDE, STRIDE, Func>(std::forward<Func>(f));
    }
}

}

#endif // COMPILE_RANGE_HPP