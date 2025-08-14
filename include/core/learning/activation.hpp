#ifndef ACTIVATION_FUNC_HPP
#define ACTIVATION_FUNC_HPP

#include <algorithm>
#include <cmath>
#include <vector>

namespace CNN
{
    
template<typename T>
requires (std::integral<T> || std::floating_point<T>)
class ReLU{
public:
    
    using type = T;
    
    T operator()(T x) const noexcept
    {
        if(x > 0)
        {
            return x;
        }
        else{
            return static_cast<T>(0);
        }
    }

    T derivative(T x) const noexcept
    {
        if(x > 0)
        {
            return 1;
        }
        else{
            return static_cast<T>(0);
        }
    }
};

template<typename T>
requires (std::floating_point<T>)
class Sigmoid{
public:
    using type = T;

    T operator()(T x) const noexcept
    {
        return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    }

    T derivative(T x) const noexcept
    {
        T sx = (*this)(x);
        return sx * (static_cast<T>(1) - sx);
    }
};

template<typename T>
class Softmax{
public:

    using type = T;

    template<typename Tensor>
    requires(std::is_same_v<typename Tensor::type, T>)
    Tensor apply(const Tensor& input) 
    {    
        constexpr size_t size = Tensor::size;
        Tensor output{};

        T max_val = *std::max_element(input.begin(), input.end());

        T sum = 0.0;
        for (size_t i = 0; i < size; ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }

        for (T& val : output) {
            val /= sum;
        }

        return output;
    }

    template<typename Tensor>
    requires(std::is_same_v<typename Tensor::type, T>)
    void apply_inplace(Tensor& input) 
    {
        constexpr size_t size = Tensor::size;

        T max_val = *std::max_element(input.begin(), input.end());

        T sum = 0.0;
        for (size_t i = 0; i < size; ++i) {
            T tmp = input[i];
            input[i] = std::exp(tmp - max_val);
            sum += input[i];
        }

        for (T& val : input) {
            val /= sum;
        }
    }
};

}

#endif // ACTIVATION_FUNC_HPP