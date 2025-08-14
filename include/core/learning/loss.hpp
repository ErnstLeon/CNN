#ifndef LOSS_H
#define LOSS_H

#include <cmath>
#include <vector>

namespace CNN{
template<typename Input_Type, typename Target_Type>
requires(std::remove_reference_t<Input_Type>::size == std::remove_reference_t<Target_Type>::size)
inline auto cross_entropy_loss(const Input_Type& output, const Target_Type& target) {
    
    using T = std::remove_reference_t<Input_Type>::type;
    constexpr size_t size = std::remove_reference_t<Input_Type>::size;

    T loss = 0;
    
    UNROLL_PRAGMA
    for (size_t i = 0; i < size; ++i) {
        loss -= target[i] * std::log(output[i] + 1e-12);
    }
    return loss;
}

template<typename T>
inline T cross_entropy_loss(const std::vector<T>& output, const std::vector<T>& target) {

    T loss = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        loss -= target[i] * std::log(output[i] + 1e-12);
    }
    return loss;
}
}

#endif //LOSS_H