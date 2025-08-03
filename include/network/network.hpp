#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "utility/compile_range.hpp"
#include "utility/network_types.hpp"

namespace CNN::Network{

    template<typename T, size_t NUM_CONV_LAYERS, size_t NUM_NEURAL_LAYERS>
    class Network {
    private:
        std::array<Convolution_Layer, NUM_CONV_LAYERS> conv_layers{};
        std::array<Neural_Layer, NUM_NEURAL_LAYERS> neural_layers{};

        std::array<std::vector<T>, NUM_CONV_LAYERS> conv_filters{};
        std::array<std::vector<T>, NUM_CONV_LAYERS - 1> neural_weights{};
    
    public:
        
        Network(const std::array<Convolution_Layer, NUM_CONV_LAYERS> & conv_layers,
        const std::array<Neural_Layer, NUM_NEURAL_LAYERS> & neural_layers) : 
        conv_layers{conv_layers}, neural_layers{neural_layers} {

            // Random number generator setup
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(0.0, 1.0);

            for(size_t con_layer = 0; con_layer < NUM_CONV_LAYERS; ++con_layer){
                
                auto kernel_size = conv_layers[con_layer].kernel_size;
                auto filter_size = kernel_size * kernel_size;

                conv_filters[con_layer].resize(filter_size);
                for (auto& val : conv_filters[con_layer]) val = dist(gen);
            }

            for(size_t neural_layer = 0; neural_layer < NUM_NEURAL_LAYERS - 1; ++neural_layer){
                
                auto weights_size = neural_layers[neural_layer].num_neurons * 
                                    neural_layers[neural_layer + 1].num_neurons;

                neural_weights[neural_layer].resize(weights_size);
                for (auto& val : neural_weights[neural_layer]) val = dist(gen);
            }
        };

        template<typename... Layer_Args>
        requires (sizeof...(Layer_Args) == NUM_CONV_LAYERS + NUM_NEURAL_LAYERS)
        Network(Layer_Args&&... layers) {
            
            std::tuple<Layer_Args...> layers_{std::forward<Layer_Args>(layers)...};
            
            compile_range<NUM_CONV_LAYERS - 1>([&]<size_t I>(){
               conv_layers[I] = std::move(std::get<I>(layers_));
            });

            compile_range<NUM_CONV_LAYERS + NUM_NEURAL_LAYERS - 1, NUM_CONV_LAYERS>([&]<size_t I>(){
               neural_layers[I] = std::move(std::get<I>(layers_));
            });

            // Random number generator setup
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(0.0, 1.0);

            for(size_t con_layer = 0; con_layer < NUM_CONV_LAYERS; ++con_layer){
                
                auto kernel_size = conv_layers[con_layer].kernel_size;
                
                conv_filters[con_layer].resize(kernel_size * kernel_size);

                for (auto& val : conv_filters[con_layer]) val = dist(gen);
            }

            for(size_t neural_layer = 0; neural_layer < NUM_NEURAL_LAYERS - 1; ++neural_layer){
                
                auto weights_size = neural_layers[neural_layer].num_neurons * 
                                    neural_layers[neural_layer + 1].num_neurons;

                neural_weights[neural_layer].resize(weights_size);

                for (auto& val : neural_weights[neural_layer]) val = dist(gen);
            }

        }



    };

}



#endif // NETWORK_HPP