#ifndef NETWORK_TYPES_HPP
#define NETWORK_TYPES_HPP

#include <vector>
#include <random>

#include "helper/loop_unroll.hpp"

namespace CNN
{

template<size_t C, size_t H, size_t W, typename T = float>
struct Convolution_FeatureMap{

    using type = T;
    static constexpr size_t channels = C;
    static constexpr size_t height = H;
    static constexpr size_t width = W;

    std::vector<T> features;

    Convolution_FeatureMap(const Convolution_FeatureMap & orig) 
    : features{orig.features} {}

    Convolution_FeatureMap(Convolution_FeatureMap && orig) noexcept 
    : features{std::move(orig.features)} {}

    Convolution_FeatureMap(const std::vector<T>& input_features)
    { 
        if (input_features.size() != channels * width * height) {
            throw std::runtime_error("Feature vector size does not match dimensions.");
        }
        features = input_features;
    }

    Convolution_FeatureMap(std::vector<T>&& input_features)
    { 
        if (input_features.size() != channels * width * height) {
            throw std::runtime_error("Feature vector size does not match dimensions.");
        }
        features = std::move(input_features);
    }

    Convolution_FeatureMap(T default_val = static_cast<T>(0)) 
    : features(channels * width * height, default_val) {}

    Convolution_FeatureMap & operator=(const Convolution_FeatureMap & orig)
    {
        if(this != &orig){
            features = orig.features;
        }
        return *this;
    }

    Convolution_FeatureMap & operator=(Convolution_FeatureMap && orig) noexcept 
    {
        if(this != &orig){
            features = std::move(orig.features);
        }
        return *this;
    }
};

template<size_t CIN, size_t COUT, size_t K, 
size_t S, size_t P, typename A_FUNC, typename T = A_FUNC::type>
requires (callable_with<A_FUNC, T, T> && derivative_callable_with <A_FUNC, T, T>)
struct Convolution_Layer{

    using type = T;

    const A_FUNC activation_func{};

    static constexpr size_t stride = S;
    static constexpr size_t kernel_size = K;
    static constexpr size_t input_channels = CIN;
    static constexpr size_t output_channels = COUT;
    static constexpr size_t pooling_size = P;

    std::vector<T> kernels;

    Convolution_Layer(T default_value) 
    : kernels(output_channels * input_channels * kernel_size * kernel_size, default_value) {}

    Convolution_Layer() 
    : kernels(output_channels * input_channels * kernel_size * kernel_size) 
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(0.0, 1.0);

        for (auto& val : kernels) val = dist(gen);
    }

    Convolution_Layer(const std::vector<T>& input_kernels)
    { 
        if (input_kernels.size() != output_channels * input_channels * kernel_size * kernel_size) {
            throw std::runtime_error("Kernel size does not match dimensions.");
        }
        kernels = input_kernels;
    }

    Convolution_Layer(std::vector<T>&& input_kernels)
    { 
        if (input_kernels.size() != output_channels * input_channels * kernel_size * kernel_size) {
            throw std::runtime_error("Kernel size does not match dimensions.");
        }
        kernels = std::move(input_kernels);
    }

    template<typename INPUT_FeatureMap, typename OUTPUT_FeatureMap>
    requires (INPUT_FeatureMap::channels == CIN && OUTPUT_FeatureMap::channels == COUT &&
    (((INPUT_FeatureMap::height + S - 1) / S) + P - 1) / P == OUTPUT_FeatureMap::height &&
    (((INPUT_FeatureMap::width + S - 1) / S) + P - 1) / P == OUTPUT_FeatureMap::width)
    void apply(const INPUT_FeatureMap & input, OUTPUT_FeatureMap & output)
    {
        constexpr size_t conv_height = (INPUT_FeatureMap::height + S - 1) / S;
        constexpr size_t conv_width = (INPUT_FeatureMap::width + S - 1) / S;
        constexpr size_t pool_height = (conv_height + P - 1) / P;
        constexpr size_t pool_width = (conv_width + P - 1) / P;

        constexpr size_t kernel_height_shift = K / 2;
        constexpr size_t kernel_width_shift = K / 2;

        UNROLL_PRAGMA
        for (size_t output_channel = 0; output_channel < COUT; ++output_channel) {
            UNROLL_PRAGMA
            for (size_t ph = 0; ph < pool_height; ++ph) {
                UNROLL_PRAGMA
                for (size_t pw = 0; pw < pool_width; ++pw) {

                    T max_val = std::numeric_limits<T>::lowest();
                    UNROLL_PRAGMA
                    for (size_t inner_ph = 0; inner_ph < P; ++inner_ph) {
                        UNROLL_PRAGMA
                        for (size_t inner_pw = 0; inner_pw < P; ++inner_pw) {

                            const size_t ch = ph * P + inner_ph;
                            const size_t cw = pw * P + inner_pw;

                            if (ch >= conv_height || cw >= conv_width)
                                continue;

                            T sum = 0;

                            const size_t kernel_base = output_channel * CIN * K * K;
                            const size_t input_base_h = ch * S;
                            const size_t input_base_w = cw * S;

                            UNROLL_PRAGMA
                            for (size_t input_channel = 0; input_channel < CIN; ++input_channel) {

                                const size_t input_base = input_channel * INPUT_FeatureMap::height * 
                                                        INPUT_FeatureMap::width;

                                UNROLL_PRAGMA
                                for (size_t kernel_h = 0; kernel_h < K; ++kernel_h) {
                                    UNROLL_PRAGMA
                                    for (size_t kernel_w = 0; kernel_w < K; ++kernel_w) {

                                        const size_t kernel_idx =
                                            kernel_base + input_channel * K * K +
                                            kernel_h * K + kernel_w;

                                        int ih = static_cast<int>(input_base_h + kernel_h) - 
                                                static_cast<int>(kernel_height_shift);
                                        int iw = static_cast<int>(input_base_w + kernel_w) - 
                                                static_cast<int>(kernel_width_shift);

                                        if (ih < 0 || iw < 0 || ih >= static_cast<int>(INPUT_FeatureMap::height) 
                                            || iw >= static_cast<int>(INPUT_FeatureMap::width)) continue;

                                        const size_t input_idx =
                                            input_base + ih * INPUT_FeatureMap::width +
                                            iw;

                                        sum += input.features[input_idx] * kernels[kernel_idx];
                                    }
                                }
                            }
                            max_val = std::max(max_val, activation_func(sum));
                        }
                    }

                    const size_t output_idx =
                        output_channel * pool_height * pool_width +
                        ph * pool_width +
                        pw;

                    output.features[output_idx] = max_val;
                }
            }
        }
    }
};

template<size_t SIZE, typename T = float>
struct Neural_FeatureMap{

    using type = T;

    static constexpr size_t size = SIZE;

    std::vector<T> features;

    Neural_FeatureMap(const Neural_FeatureMap & orig) 
    : features{orig.features} {}

    Neural_FeatureMap(Neural_FeatureMap && orig) noexcept 
    : features{std::move(orig.features)} {}

    Neural_FeatureMap(const std::vector<T>& input_features) 
    { 
        if (input_features.size() != size) {
            throw std::runtime_error("Feature vector size does not match dimensions.");
        }
        features = input_features;
    };

    Neural_FeatureMap(std::vector<T>&& input_features)
    { 
        if (input_features.size() != size) {
            throw std::runtime_error("Feature vector size does not match dimensions.");
        }
        features = std::move(input_features);
    };

    Neural_FeatureMap(T default_val = static_cast<T>(0)) 
    : features(size, default_val) {}

    Neural_FeatureMap & operator=(const Neural_FeatureMap & orig)
    {
        if(this != &orig){
            features = orig.features;
        }
        return *this;
    }

    Neural_FeatureMap & operator=(Neural_FeatureMap && orig) noexcept 
    {
        if(this != &orig){
            features = std::move(orig.features);
        }
        return *this;
    }
};

template<size_t INPUT_NEURONS, size_t OUTPUT_NEURONS, typename T = float>
struct Neural_Layer{

    using type = T;

    static constexpr size_t input_neurons = INPUT_NEURONS;
    static constexpr size_t output_neurons = OUTPUT_NEURONS;

    std::vector<T> weights;

    Neural_Layer() 
    : weights(input_neurons * output_neurons)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 1.0);

        for (auto& val : weights) val = dist(gen);
    }

    Neural_Layer(T default_value) 
    : weights(input_neurons * output_neurons, default_value) {}

    template<typename INPUT_FeatureMap, typename OUTPUT_FeatureMap>
    requires (INPUT_FeatureMap::size == INPUT_NEURONS && OUTPUT_FeatureMap::size == OUTPUT_NEURONS)
    void apply(const INPUT_FeatureMap & input, OUTPUT_FeatureMap & output)
    {
        UNROLL_PRAGMA
        for (size_t output_neuron = 0; output_neuron < output_neurons; ++output_neuron) {

            size_t weights_base = output_neuron * input_neurons;
            T sum = 0;

            UNROLL_PRAGMA
            for (size_t input_neuron = 0; input_neuron < input_neurons; ++input_neuron) {
                sum += weights[weights_base + input_neuron] * input[input_neuron];
            }
            output[output_neuron] = sum;
        }
    }
};

}

#endif // NETWORK_TYPES_HPP