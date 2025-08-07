#ifndef LOSS_H
#define LOSS_H

#include <cmath>
#include <vector>

namespace CNN{
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