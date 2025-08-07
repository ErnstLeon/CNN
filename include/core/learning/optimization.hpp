#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <algorithm>
#include <cmath>
#include <vector>

namespace CNN::Optimizer
{
template<typename T>
class Gradient_Descent{
private:
    T learning_rate = static_cast<T>(0.01);

public:
    Gradient_Descent() = default;

    Gradient_Descent(T learning_rate) 
    : learning_rate{learning_rate} {};

    void update(std::vector<T> & current, const std::vector<T> & gradient)
    {
        std::transform(current.begin(), current.end(), gradient.begin(), current.begin(),
                    [&](T val, T grad) { return val - learning_rate * grad;});
    }
};

template<typename T>
class Adam_Optimizer {
private:
    T learning_rate = static_cast<T>(0.001);
    T beta1 = static_cast<T>(0.9);
    T beta2 = static_cast<T>(0.999);
    T epsilon = static_cast<T>(1e-8);
    int timestep = 0;

    std::vector<T> m;
    std::vector<T> v;

public:
    Adam_Optimizer() = default;

    Adam_Optimizer(T learning_rate, 
                T beta1 = static_cast<T>(0.9),   
                T beta2 = static_cast<T>(0.999),
                T epsilon = static_cast<T>(1e-8)) 
    : learning_rate{learning_rate}, beta1{beta1}, beta2{beta2}, 
    epsilon{epsilon } {}


    void update(std::vector<T>& current, const std::vector<T>& gradient) 
    {
        if (m.empty()) {
            m.resize(current.size(), static_cast<T>(0));
            v.resize(current.size(), static_cast<T>(0));
        }

        ++timestep;

        T beta1_pow_t = std::pow(beta1, timestep);
        T beta2_pow_t = std::pow(beta2, timestep);

        for (size_t i = 0; i < current.size(); ++i) {
            T g = gradient[i];

            m[i] = beta1 * m[i] + (static_cast<T>(1) - beta1) * g;
            v[i] = beta2 * v[i] + (static_cast<T>(1) - beta2) * g * g;

            T m_hat = m[i] / (static_cast<T>(1) - beta1_pow_t);
            T v_hat = v[i] / (static_cast<T>(1) - beta2_pow_t);

            current[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }

    void reset() {
        std::fill(m.begin(), m.end(), static_cast<T>(0));
        std::fill(v.begin(), v.end(), static_cast<T>(0));
        timestep = 0;
    }
};
}

#endif // OPTIMIZATION_H