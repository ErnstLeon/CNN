#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <algorithm>
#include <cmath>
#include <vector>

namespace CNN::Optimizer
{

template<typename Tensor>
class Gradient_Descent_ {
private:
    using T = typename Tensor::type;
    static constexpr size_t size = Tensor::size;

    const T learning_rate = static_cast<T>(0.01);
    const T penalty = static_cast<T>(1e-5);

public:
    Gradient_Descent_() = default;
    Gradient_Descent_(T learning_rate, T penalty = static_cast<T>(1e-5)) 
    : learning_rate(learning_rate), penalty(penalty) {}

    void update(Tensor& current, const Tensor& gradient) {
        for(size_t i = 0; i < size; ++i){
            current[i] -= learning_rate * gradient[i];
            current[i] -= learning_rate * penalty * current[i];
        }
    }
};

template<typename T>
class Gradient_Descent {
public:
    const T learning_rate = static_cast<T>(0.01);
    const T penalty = static_cast<T>(1e-5);

    Gradient_Descent() = default;
    Gradient_Descent(T learning_rate, T penalty = static_cast<T>(1e-5)) 
    : learning_rate(learning_rate), penalty(penalty) {}

    template<typename Tensor>
    requires std::is_same_v<typename Tensor::type, T>
    Gradient_Descent_<Tensor> Tensor_Optimizer() const {
        return Gradient_Descent_<Tensor>(learning_rate, penalty);
    }
};

template<typename Tensor>
class Adam_Optimizer_ {
private:
    using T = Tensor::type;
    static constexpr size_t size = Tensor::size;

    const T learning_rate = static_cast<T>(0.001);
    const T penalty = static_cast<T>(1e-5);
    const T beta1 = static_cast<T>(0.9);
    const T beta2 = static_cast<T>(0.999);
    const T epsilon = static_cast<T>(1e-8);

    int timestep = 0;

    Tensor m{};
    Tensor v{};

public:
    Adam_Optimizer_() = default;

    Adam_Optimizer_(T learning_rate, 
                T penalty = static_cast<T>(1e-5),
                T beta1 = static_cast<T>(0.9),   
                T beta2 = static_cast<T>(0.999),
                T epsilon = static_cast<T>(1e-8)) 
    : learning_rate{learning_rate}, penalty{penalty}, 
    beta1{beta1}, beta2{beta2}, epsilon{epsilon} {}


    void update(Tensor& current, const Tensor& gradient) 
    {
        ++timestep;

        T beta1_pow_t = std::pow(beta1, timestep);
        T beta2_pow_t = std::pow(beta2, timestep);

        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) {
            T g = gradient[i];

            m[i] = beta1 * m[i] + (static_cast<T>(1) - beta1) * g;
            v[i] = beta2 * v[i] + (static_cast<T>(1) - beta2) * g * g;

            T m_hat = m[i] / (static_cast<T>(1) - beta1_pow_t);
            T v_hat = v[i] / (static_cast<T>(1) - beta2_pow_t);

            current[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));
            current[i] -= learning_rate * penalty * current[i];
        }
    }

    void reset() {
        m.fill(static_cast<T>(0));
        v.fill(static_cast<T>(0));
        timestep = 0;
    }
};

template<typename T>
class Adam_Optimizer {
public:
    const T learning_rate = static_cast<T>(0.001);
    const T penalty = static_cast<T>(1e-5);
    const T beta1 = static_cast<T>(0.9);
    const T beta2 = static_cast<T>(0.999);
    const T epsilon = static_cast<T>(1e-8);

    Adam_Optimizer() = default;

    Adam_Optimizer(T learning_rate, 
                T penalty = static_cast<T>(1e-5),
                T beta1 = static_cast<T>(0.9),   
                T beta2 = static_cast<T>(0.999),
                T epsilon = static_cast<T>(1e-8)) 
    : learning_rate{learning_rate}, penalty{penalty}, 
    beta1{beta1}, beta2{beta2}, epsilon{epsilon} {}

    template<typename Tensor>
    requires std::is_same_v<typename Tensor::type, T>
    Adam_Optimizer_<Tensor> Tensor_Optimizer() const {
        return Adam_Optimizer_<Tensor>(learning_rate, penalty, beta1, beta2, epsilon);
    }
};

}

#endif // OPTIMIZATION_H