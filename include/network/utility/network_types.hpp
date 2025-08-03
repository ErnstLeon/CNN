#ifndef NETWORK_TYPES_HPP
#define NETWORK_TYPES_HPP

#include <vector>
#include <random>

namespace CNN::Network{

    template<typename T>
    struct Convolution_Layer{

        using type = T;

        size_t stride;
        size_t kernel_size;
        size_t input_channels;
        size_t output_channels;
        size_t pooling_size;

        std::vector<T> kernel;

        Convolution_Layer(size_t stride, size_t kernel_size, size_t output_channels, size_t pooling_size)
            : stride{stride}, 
            kernel_size{kernel_size}, 
            output_channels{output_channels}, 
            pooling_size{pooling_size},
            kernel(input_channels * output_channels * kernel_size * kernel_size)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dist(0.0, 1.0);

                for (auto& val : kernel) val = dist(gen);
            }

       // void apply(const std::vector<>)
    };

    template<typename T>
    struct Neural_Layer{

        using type = T;

        size_t input_neurons;
        size_t output_neurons;

        std::vector<T> weights;

        Neural_Layer(size_t input_neurons, size_t output_neurons) 
            : input_neurons{input_neurons}, 
            output_neurons{output_neurons},
            weights(input_neurons, output_neurons)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dist(0.0, 1.0);

                for (auto& val : weights) val = dist(gen);
            }
    };

}

#endif // NETWORK_TYPES_HPP