#ifndef LAYER_TYPES_HPP
#define LAYER_TYPES_HPP

#include <limits>
#include <random>
#include <vector>

#include "../helper/loop_unroll.hpp"
#include "heap_tensor.hpp"
#include "stack_tensor.hpp"

namespace CNN
{
template<
    size_t CIN, size_t COUT, size_t K, 
    size_t S, size_t P, typename A_FUNC, 
    typename T = A_FUNC::type>
requires(
    callable_with<A_FUNC, T, T> && 
    derivative_callable_with <A_FUNC, T, T>)
struct Convolution_Layer{

    using type = T;

    inline static A_FUNC activation_func{};

    static constexpr size_t input_channels = CIN;
    static constexpr size_t output_channels = COUT;
    static constexpr size_t kernel_size = K;
    static constexpr size_t stride = S;
    static constexpr size_t pooling_size = P;

    StackTensor4D<COUT, CIN, K, K, T> kernels{};
    StackTensor1D<COUT, T> biases{};

    static constexpr size_t kernels_size = decltype(kernels)::size;
    static constexpr size_t biases_size  = decltype(biases)::size;

    Convolution_Layer(bool rnd = true) {
        if(rnd)
        {
            constexpr size_t fan_in = input_channels * kernel_size * kernel_size;

            std::random_device rd;
            std::mt19937 gen(10);

            const T stddev = std::sqrt(T(2) / T(fan_in));
            std::normal_distribution<T> wdist(0.0, stddev);

            UNROLL_PRAGMA
            for (size_t i = 0; i < kernels_size; ++i) 
                kernels[i] = wdist(gen);

            biases.fill(T(0));
        }
    }

    Convolution_Layer(const Convolution_Layer&) = default;
    Convolution_Layer& operator=(const Convolution_Layer&) = default;

    template<
        typename INPUT_Features, 
        typename CONV_OUTPUT_Features,
        typename POOLED_OUTPUT_Features>
    requires(
        INPUT_Features::size_x == CIN && 
        CONV_OUTPUT_Features::size_x == COUT && 
        POOLED_OUTPUT_Features::size_x == COUT &&
        ((INPUT_Features::size_y + S - 1) / S) == CONV_OUTPUT_Features::size_y &&
        ((INPUT_Features::size_z + S - 1) / S) == CONV_OUTPUT_Features::size_z &&
        (((INPUT_Features::size_y + S - 1) / S) + P - 1) / P == POOLED_OUTPUT_Features::size_y &&
        (((INPUT_Features::size_z + S - 1) / S) + P - 1) / P == POOLED_OUTPUT_Features::size_z)
    void apply(const INPUT_Features & input, 
        CONV_OUTPUT_Features & conv_weighted_inputs, 
        POOLED_OUTPUT_Features & pooling_output) noexcept
    {
        constexpr size_t input_stride = INPUT_Features::size_y * INPUT_Features::size_z;
        constexpr size_t conv_h = (INPUT_Features::size_y + S - 1) / S;
        constexpr size_t conv_w = (INPUT_Features::size_z  + S - 1) / S;

        constexpr size_t conv_stride = conv_h * conv_w;
        constexpr size_t pooled_h = (conv_h + P - 1) / P;
        constexpr size_t pooled_w = (conv_w  + P - 1) / P;

        constexpr size_t kernel_half_h = K / 2;
        constexpr size_t kernel_half_w = K / 2;

        CONV_OUTPUT_Features conv_activation_results;

        UNROLL_PRAGMA
        for(size_t out_channel = 0; out_channel < COUT; ++out_channel)
        {
            const size_t kernel_base = out_channel * CIN * K * K;
            const size_t conv_base = out_channel * conv_h * conv_w;
            
            UNROLL_PRAGMA
            for(size_t out_h = 0; out_h < conv_h; ++out_h)
            {
                const size_t out_h_base = conv_base + out_h * conv_w;

                UNROLL_PRAGMA
                for(size_t out_w = 0; out_w < conv_w; ++out_w)
                {
                    T sum = 0;

                    const int input_h_0 = static_cast<int>(out_h * S) - static_cast<int>(kernel_half_h);
                    const int input_w_0 = static_cast<int>(out_w * S) - static_cast<int>(kernel_half_w);

                    UNROLL_PRAGMA
                    for(int k_h = 0; k_h < static_cast<int>(K); ++k_h)
                    {
                        UNROLL_PRAGMA
                        for(int k_w = 0; k_w < static_cast<int>(K); ++k_w)
                        {
                            const int input_h = input_h_0 + k_h;
                            const int input_w = input_w_0 + k_w;

                            const size_t kernel_off = k_h * K + k_w;

                            if (input_h >= 0 && input_w >= 0 &&
                                input_h < static_cast<int>(INPUT_Features::size_y) &&
                                input_w < static_cast<int>(INPUT_Features::size_z))
                            {
                                const size_t input_offset = 
                                    static_cast<size_t>(input_h) * INPUT_Features::size_z +
                                    static_cast<size_t>(input_w);

                                UNROLL_PRAGMA
                                for(size_t input_channel = 0; input_channel < CIN; ++input_channel)
                                {
                                    const size_t kernel_idx =
                                        kernel_base + input_channel * K * K + kernel_off;

                                    const size_t input_idx =
                                        input_channel * input_stride + input_offset;

                                    sum += input[input_idx] * kernels[kernel_idx];
                                }
                            }
                        }
                    }

                    sum += biases[out_channel];
                    conv_weighted_inputs[out_h_base + out_w] = sum;
                    conv_activation_results[out_h_base + out_w] = activation_func(sum);
                }
            }
        }

        UNROLL_PRAGMA
        for(size_t out_channel = 0; out_channel < COUT; ++out_channel)
        {
            const size_t pooled_base = out_channel * pooled_h * pooled_w;
            const size_t conv_base = out_channel * conv_stride;
            
            UNROLL_PRAGMA
            for(size_t out_h = 0; out_h < pooled_h; ++out_h)
            {
                const size_t out_h_base = pooled_base + out_h * pooled_w;

                UNROLL_PRAGMA
                for(size_t out_w = 0; out_w < pooled_w; ++out_w)
                {
                    T sum = 0;

                    const size_t input_h_0 = out_h * P;
                    const size_t input_w_0 = out_w * P;

                    UNROLL_PRAGMA
                    for(size_t p_h = 0; p_h < P; ++p_h)
                    {
                        UNROLL_PRAGMA
                        for(size_t p_w = 0; p_w < P; ++p_w)
                        {
                            const size_t input_h = input_h_0 + p_h;
                            const size_t input_w = input_w_0 + p_w;

                            if (input_h < conv_h && input_w < conv_w)
                            {
                                const size_t conv_idx =
                                    conv_base + input_h * conv_w + input_w;

                                sum += conv_activation_results[conv_idx];
                            }
                        }
                    }

                    sum /= static_cast<T>(P * P);
                    pooling_output[out_h_base + out_w] = sum;
                }
            }
        }
    }

    template<
        typename INPUT_Features, 
        typename OUTPUT_Features>
    requires(
        INPUT_Features::size_x == COUT && OUTPUT_Features::size_x == CIN &&
        ((OUTPUT_Features::size_y + S - 1) / S) == INPUT_Features::size_y &&
        ((OUTPUT_Features::size_z + S - 1) / S) == INPUT_Features::size_z)
    void apply_backwards(const INPUT_Features & input, OUTPUT_Features & output) noexcept
    {
        constexpr size_t output_channels = OUTPUT_Features::size_x;
        constexpr size_t output_height = OUTPUT_Features::size_y;
        constexpr size_t output_width = OUTPUT_Features::size_z;

        constexpr size_t input_channels = INPUT_Features::size_x;
        constexpr size_t input_height = INPUT_Features::size_y;
        constexpr size_t input_width = INPUT_Features::size_z;

        UNROLL_PRAGMA
        for(size_t out_channel = 0; out_channel < output_channels; ++out_channel)
        {   
            UNROLL_PRAGMA
            for(size_t out_h = 0; out_h < output_height; ++out_h)
            {
                UNROLL_PRAGMA
                for(size_t out_w = 0; out_w < output_width; ++out_w)
                {
                    T sum = 0;

                    UNROLL_PRAGMA
                    for(size_t kernel_h = 0; kernel_h < K; ++kernel_h)
                    {
                        UNROLL_PRAGMA
                        for(size_t kernel_w = 0; kernel_w < K; ++kernel_w)
                        {
                            const int ch = static_cast<int>(out_h) +
                                (static_cast<int>((K - 1) / 2) - static_cast<int>(kernel_h)) ;
                            const int cw = static_cast<int>(out_w) +
                                (static_cast<int>((K - 1) / 2) - static_cast<int>(kernel_w));
                            
                            if (ch % S == 0 && cw % S == 0 && ch >= 0 && 
                                cw >= 0 && ch < static_cast<int>(output_height) && 
                                cw < static_cast<int>(output_width))
                            {
                                const int sh = ch / static_cast<int>(S);
                                const int sw = cw / static_cast<int>(S);

                                UNROLL_PRAGMA
                                for(size_t in_channel = 0; in_channel < input_channels; ++in_channel)
                                {
                                    const size_t kernel_idx = 
                                        in_channel * output_channels * K * K +
                                        out_channel * K * K +
                                        kernel_h * K + kernel_w;

                                    const size_t input_idx =
                                        in_channel * input_height * input_width
                                        + sh * input_width + sw;

                                    sum += input[input_idx] * kernels[kernel_idx];
                                }
                            }

                        }
                    }

                    const size_t output_idx =
                        out_channel * output_height * output_width +
                        output_width * out_h + out_w;

                    output[output_idx] = sum;
                }
            }
        }
    }
};

template<
    size_t INPUT_NEURONS, 
    size_t OUTPUT_NEURONS, 
    typename A_FUNC, 
    typename T = A_FUNC::type>
//requires(
//    callable_with<A_FUNC, T, T> && 
//    derivative_callable_with <A_FUNC, T, T>)
struct Neural_Layer{

    using type = T;

    inline static A_FUNC activation_func{};

    static constexpr size_t input_neurons = INPUT_NEURONS;
    static constexpr size_t output_neurons = OUTPUT_NEURONS;

    StackTensor2D<OUTPUT_NEURONS, INPUT_NEURONS, T> weights{};
    StackTensor1D<OUTPUT_NEURONS, T> biases{};

    static constexpr size_t weights_size = decltype(weights)::size;
    static constexpr size_t biases_size  = decltype(biases)::size;

    Neural_Layer(bool rnd = true) {
        if(rnd)
        {
            constexpr size_t fan_in = input_neurons;

            std::random_device rd;
            std::mt19937 gen(10);

            const T stddev = std::sqrt(T(2) / T(fan_in));
            std::normal_distribution<T> wdist(0.0, stddev);

            UNROLL_PRAGMA
            for (size_t i = 0; i < weights_size; ++i)
                weights[i] = wdist(gen);

            biases.fill(T(0));
        }
    }

    Neural_Layer(const Neural_Layer&) = default;
    Neural_Layer& operator=(const Neural_Layer&) = default;

    template<
        typename INPUT_Features, 
        typename OUTPUT_Features>
    requires(
        INPUT_Features::size == INPUT_NEURONS && 
        OUTPUT_Features::size == OUTPUT_NEURONS)
    void apply(const INPUT_Features & input, 
        OUTPUT_Features & output_weighted_inputs,
        OUTPUT_Features & output_activation_results) noexcept
    {
        UNROLL_PRAGMA
        for(size_t output_neuron = 0; output_neuron < output_neurons; ++output_neuron)
        {
            const size_t weights_base = output_neuron * input_neurons;
            T sum = 0;

            UNROLL_PRAGMA
            for(size_t input_neuron = 0; input_neuron < input_neurons; ++input_neuron)
            {
                sum += weights[weights_base + input_neuron] * input[input_neuron];
            }

            T tmp = sum + biases[output_neuron];

            output_weighted_inputs[output_neuron] = tmp;
            output_activation_results[output_neuron] = activation_func(tmp);
        }
    }

    template<
        typename INPUT_Features, 
        typename OUTPUT_Features>
    requires(
        INPUT_Features::size == OUTPUT_NEURONS && 
        OUTPUT_Features::size == INPUT_NEURONS)
    void apply_backwards(const INPUT_Features & input, OUTPUT_Features & output) noexcept
    {
        constexpr size_t input_size = OUTPUT_NEURONS;
        constexpr size_t output_size = INPUT_NEURONS;

        UNROLL_PRAGMA
        for(size_t output_neuron = 0; output_neuron < output_size; ++output_neuron)
        {
            T sum = 0;

            UNROLL_PRAGMA
            for(size_t input_neuron = 0; input_neuron < input_size; ++input_neuron)
            {
                sum += weights[output_neuron + input_neuron * output_size] * input[input_neuron];
            }
            output[output_neuron] = sum;
        }
    }
};

template<
    size_t INPUT_NEURONS, 
    size_t OUTPUT_NEURONS,
    typename T>
struct Neural_Layer<INPUT_NEURONS, OUTPUT_NEURONS, Softmax<T>, T>{

    using type = T;

    inline static Softmax<T> activation_func{};

    static constexpr size_t input_neurons = INPUT_NEURONS;
    static constexpr size_t output_neurons = OUTPUT_NEURONS;

    StackTensor2D<OUTPUT_NEURONS, INPUT_NEURONS, T> weights{};
    StackTensor1D<OUTPUT_NEURONS, T> biases{};

    static constexpr size_t weights_size = decltype(weights)::size;
    static constexpr size_t biases_size  = decltype(biases)::size;

    Neural_Layer(bool rnd = true) {
        if(rnd)
        {
            constexpr size_t fan_in  = input_neurons;

            std::random_device rd;
            std::mt19937 gen(10);

            const T r = std::sqrt(T(6) / T(fan_in));
            std::uniform_real_distribution<T> wdist(-r, r);

            UNROLL_PRAGMA
            for (size_t i = 0; i < weights_size; ++i) 
                weights[i] = wdist(gen);

            biases.fill(T(0));
        }
    }

    Neural_Layer(const Neural_Layer&) = default;
    Neural_Layer& operator=(const Neural_Layer&) = default;

    template<
        typename INPUT_Features, 
        typename OUTPUT_Features>
    requires(
        INPUT_Features::size == INPUT_NEURONS && 
        OUTPUT_Features::size == OUTPUT_NEURONS)
    void apply(const INPUT_Features & input, 
        OUTPUT_Features & output_weighted_inputs,
        OUTPUT_Features & output_activation_results) noexcept
    {
        UNROLL_PRAGMA
        for(size_t output_neuron = 0; output_neuron < output_neurons; ++output_neuron)
        {
            const size_t weights_base = output_neuron * input_neurons;
            T sum = 0;

            UNROLL_PRAGMA
            for(size_t input_neuron = 0; input_neuron < input_neurons; ++input_neuron)
            {
                sum += weights[weights_base + input_neuron] * input[input_neuron];
            }

            T tmp = sum + biases[output_neuron];

            output_weighted_inputs[output_neuron] = tmp;
        }

        output_activation_results = activation_func.apply(output_weighted_inputs);
    }

    template<
        typename INPUT_Features, 
        typename OUTPUT_Features>
    requires(
        INPUT_Features::size == OUTPUT_NEURONS && 
        OUTPUT_Features::size == INPUT_NEURONS)
    void apply_backwards(const INPUT_Features & input, OUTPUT_Features & output) noexcept
    {
        constexpr size_t input_size = OUTPUT_NEURONS;
        constexpr size_t output_size = INPUT_NEURONS;

        UNROLL_PRAGMA
        for(size_t output_neuron = 0; output_neuron < output_size; ++output_neuron)
        {
            T sum = 0;

            UNROLL_PRAGMA
            for(size_t input_neuron = 0; input_neuron < input_size; ++input_neuron)
            {
                sum += weights[output_neuron + input_neuron * output_size] * input[input_neuron];
            }
            output[output_neuron] = sum;
        }
    }
};

}

#endif // LAYER_TYPES_HPP