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
            constexpr size_t fan_in  = input_channels  * kernel_size * kernel_size;

            std::mt19937 gen(100);

            const T r = std::sqrt(T(6) / T(fan_in));
            std::uniform_real_distribution<T> wdist(-r, r);

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
        typename OUTPUT_Features>
    requires(
        INPUT_Features::size_x == CIN && OUTPUT_Features::size_x == COUT &&
        (((INPUT_Features::size_y + S - 1) / S) + P - 1) / P == OUTPUT_Features::size_y &&
        (((INPUT_Features::size_z + S - 1) / S) + P - 1) / P == OUTPUT_Features::size_z)
    void apply(const INPUT_Features & input, 
        OUTPUT_Features & output_weighted_inputs,
        OUTPUT_Features & output_activation_results) noexcept
    {
        constexpr size_t conv_height = (INPUT_Features::size_y + S - 1) / S;
        constexpr size_t conv_width  = (INPUT_Features::size_z  + S - 1) / S;
        constexpr size_t output_height = (conv_height + P - 1) / P;
        constexpr size_t output_width  = (conv_width  + P - 1) / P;

        constexpr size_t kernel_half_h = K / 2;
        constexpr size_t kernel_half_w = K / 2;

        compile_range<COUT>([&]<size_t oc>{
            constexpr size_t kernel_base = oc * CIN * K * K;
            constexpr size_t output_base = oc * output_height * output_width;
            
            compile_range<output_height>([&]<size_t oh>{
                constexpr size_t oh_base = output_base + oh * output_width;

                compile_range<output_width>([&]<size_t ow>{

                    T sum = 0;
                
                    compile_range<P>([&]<size_t ph>{
                        
                        compile_range<P>([&]<size_t pw>{

                            constexpr size_t ch = oh * P + ph;
                            constexpr size_t cw = ow * P + pw;

                            if constexpr (ch < conv_height && cw < conv_width){

                                constexpr int input_base_h =  static_cast<int>(ch * S) - static_cast<int>(kernel_half_h);
                                constexpr int input_base_w =  static_cast<int>(cw * S) - static_cast<int>(kernel_half_w);

                                compile_range<K * K>([&]<size_t kk>{

                                    constexpr size_t kh = kk / K;
                                    constexpr size_t kw = kk % K;

                                    constexpr int ih = static_cast<int>(input_base_h + kh);
                                    constexpr int iw = static_cast<int>(input_base_w + kw);

                                    if constexpr (ih >= 0 && iw >= 0 &&
                                        ih < static_cast<int>(INPUT_Features::size_y) &&
                                        iw < static_cast<int>(INPUT_Features::size_z))
                                    {
                                        compile_range<CIN>([&]<size_t ic>{
                                            constexpr size_t kernel_idx =
                                                kernel_base + ic * K * K + kh * K + kw;

                                            constexpr size_t input_base =
                                                ic * INPUT_Features::size_y * INPUT_Features::size_z;

                                            constexpr size_t input_idx =
                                                input_base + static_cast<size_t>(ih) * INPUT_Features::size_z +
                                                static_cast<size_t>(iw);

                                            sum += input[input_idx] * kernels[kernel_idx];
                                        });
                                    }
                                });
                            }
                        });
                    });

                    sum /= static_cast<T>(P * P);
                    T tmp = sum + biases[oc];
                    
                    constexpr size_t output_idx = oh_base + ow;

                    output_weighted_inputs[output_idx] = tmp;
                    output_activation_results[output_idx] = activation_func(tmp);
                });
            });
        });
    }

    template<
        typename INPUT_Features, 
        typename OUTPUT_Features>
    requires(
        INPUT_Features::size_x == COUT && OUTPUT_Features::size_x == CIN &&
        (((OUTPUT_Features::size_y + S - 1) / S) + P - 1) / P == INPUT_Features::size_y &&
        (((OUTPUT_Features::size_z + S - 1) / S) + P - 1) / P == INPUT_Features::size_z)
    void apply_backwards(const INPUT_Features & input, OUTPUT_Features & output) noexcept
    {
        constexpr size_t output_channels = OUTPUT_Features::size_x;
        constexpr size_t output_height = OUTPUT_Features::size_y;
        constexpr size_t output_width = OUTPUT_Features::size_z;

        constexpr size_t input_channels = INPUT_Features::size_x;
        constexpr size_t input_height = INPUT_Features::size_y;
        constexpr size_t input_width = INPUT_Features::size_z;

        compile_range<output_channels>([&]<size_t oc>{
            
            compile_range<output_height>([&]<size_t oh>{

                compile_range<output_width>([&]<size_t ow>{

                    T sum = 0;

                    compile_range<K>([&]<size_t kernel_h>{
                    
                        compile_range<K>([&]<size_t kernel_w>{

                            constexpr int ch = static_cast<int>(oh) +
                                (static_cast<int>((K - 1) / 2) - static_cast<int>(kernel_h)) ;
                            constexpr int cw = static_cast<int>(ow) +
                                (static_cast<int>((K - 1) / 2) - static_cast<int>(kernel_w));
                            
                            if constexpr (ch % S == 0 && cw % S == 0 && ch >= 0 && 
                                cw >= 0 && ch < static_cast<int>(output_height) && 
                                cw < static_cast<int>(output_width))
                            {
                                constexpr int sh = ch / static_cast<int>(S);
                                constexpr int sw = cw / static_cast<int>(S);
        
                                constexpr int ph = sh / static_cast<int>(P);
                                constexpr int pw = sw / static_cast<int>(P);

                                compile_range<input_channels>([&]<size_t ic>{

                                    constexpr size_t kernel_idx = 
                                        ic * output_channels * K * K +
                                        oc * K * K +
                                        kernel_h * K + kernel_w;

                                    constexpr size_t input_idx =
                                        ic * input_height * input_width
                                        + ph * input_width + pw;

                                    sum += input[input_idx] * kernels[kernel_idx];
                                });
                            }

                        });
                    });

                    constexpr size_t output_idx =
                        oc * output_height * output_width +
                        output_width * oh +
                        ow;

                    output[output_idx] = sum / static_cast<T>(P * P);
                });
            });
        });
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
            constexpr size_t fan_in  = input_neurons;

            std::mt19937 gen(100);

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
        compile_range<output_neurons>([&]<size_t output_neuron>{

            constexpr size_t weights_base = output_neuron * input_neurons;
            T sum = 0;

            compile_range<input_neurons>([&]<size_t input_neuron>{

                sum += weights[weights_base + input_neuron] * input[input_neuron];
            });

            T tmp = sum + biases[output_neuron];

            output_weighted_inputs[output_neuron] = tmp;
            output_activation_results[output_neuron] = activation_func(tmp);
        });
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

        compile_range<output_size>([&]<size_t output_neuron>{

            T sum = 0;

            compile_range<input_size>([&]<size_t input_neuron>{

                sum += weights[output_neuron + input_neuron * output_size] * input[input_neuron];
            });

            output[output_neuron] = sum;
        });
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

            std::mt19937 gen(100);

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
        compile_range<output_neurons>([&]<size_t output_neuron>{

            constexpr size_t weights_base = output_neuron * input_neurons;
            T sum = 0;

            compile_range<input_neurons>([&]<size_t input_neuron>{

                sum += weights[weights_base + input_neuron] * input[input_neuron];
            });

            T tmp = sum + biases[output_neuron];

            output_weighted_inputs[output_neuron] = tmp;
        });

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

        compile_range<output_size>([&]<size_t output_neuron>{

            T sum = 0;

            compile_range<input_size>([&]<size_t input_neuron>{

                sum += weights[output_neuron + input_neuron * output_size] * input[input_neuron];
            });

            output[output_neuron] = sum;
        });
    }
};

}

#endif // LAYER_TYPES_HPP