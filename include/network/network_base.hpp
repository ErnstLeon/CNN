#ifndef NETWORK_BASE_HPP
#define NETWORK_BASE_HPP

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "../core.hpp"

namespace CNN::Network
{

template<typename Conv_Layer_Tuple, typename Neural_Layer_Tuple, 
    typename Conv_Feature_Tuple, typename Neural_Feature_Tuple>
requires (std::tuple_size_v<Conv_Layer_Tuple> == std::tuple_size_v<Conv_Feature_Tuple> - 1 
    && std::tuple_size_v<Neural_Layer_Tuple> == std::tuple_size_v<Neural_Feature_Tuple> - 1)
class Network {
public:

    Conv_Layer_Tuple conv_layers;
    Neural_Layer_Tuple neural_layers;

    static constexpr size_t num_conv_layers = std::tuple_size_v<Conv_Layer_Tuple>;
    static constexpr size_t num_neural_layers = std::tuple_size_v<Neural_Layer_Tuple>;

    Network(const Conv_Layer_Tuple& conv_layers, const Neural_Layer_Tuple& neural_layers) 
    : conv_layers{conv_layers},
    neural_layers{neural_layers} {};

    Network(Conv_Layer_Tuple&& conv_layers, Neural_Layer_Tuple&& neural_layers) 
    : conv_layers{std::move(conv_layers)},
    neural_layers{std::move(neural_layers)} {};

private: 

    void foward_propagate(
        const std::vector<typename std::tuple_element_t<0, Conv_Feature_Tuple>::type> &, 
        Conv_Feature_Tuple &, 
        Conv_Feature_Tuple &, 
        Neural_Feature_Tuple &, 
        Neural_Feature_Tuple &);

    void backward_propagate(
        const std::vector<typename std::tuple_element_t<0, Conv_Feature_Tuple>::type> &, 
        const Conv_Feature_Tuple &, 
        const Conv_Feature_Tuple &, 
        const Neural_Feature_Tuple &, 
        const Neural_Feature_Tuple &,
        Conv_Layer_Tuple &, 
        Neural_Layer_Tuple &);

public:
    
    void compute_gradient(const std::vector<typename std::tuple_element_t<0, Conv_Feature_Tuple>::type> & input){
        Conv_Feature_Tuple conv_weighted_inputs;
        Conv_Feature_Tuple conv_activation_outputs;
        Neural_Feature_Tuple neural_weighted_inputs;
        Neural_Feature_Tuple neutal_activation_outputs;
        foward_propagate(input, 
            conv_weighted_inputs, 
            conv_activation_outputs, 
            neural_weighted_inputs, 
            neutal_activation_outputs);
        
        Conv_Layer_Tuple conv_gradients;
        Neural_Layer_Tuple neural_gradients;

        backward_propagate(std::vector<float>(200,0), 
            conv_weighted_inputs, 
            conv_activation_outputs, 
            neural_weighted_inputs, 
            neutal_activation_outputs,
            conv_gradients,
            neural_gradients);
    }
};

template<typename Conv_Layer_Tuple, typename Neural_Layer_Tuple, 
        typename Conv_Feature_Tuple, typename Neural_Feature_Tuple>
requires (std::tuple_size_v<Conv_Layer_Tuple> == std::tuple_size_v<Conv_Feature_Tuple> - 1 
        && std::tuple_size_v<Neural_Layer_Tuple> == std::tuple_size_v<Neural_Feature_Tuple> - 1)
void Network<Conv_Layer_Tuple, Neural_Layer_Tuple, 
        Conv_Feature_Tuple, Neural_Feature_Tuple>::foward_propagate(
    const std::vector<typename std::tuple_element_t<0, Conv_Feature_Tuple>::type> & input, 
    Conv_Feature_Tuple & conv_weighted_inputs, 
    Conv_Feature_Tuple & conv_activation_results, 
    Neural_Feature_Tuple & neural_weighted_inputs, 
    Neural_Feature_Tuple & neural_activation_results)
{
    constexpr std::size_t Num_Conv_Layers = std::tuple_size_v<Conv_Layer_Tuple>;
    constexpr std::size_t Num_Neural_Layers = std::tuple_size_v<Neural_Layer_Tuple>;

    std::get<0>(conv_activation_results) = input;

    compile_range<Num_Conv_Layers>([&]<size_t I>(){
        (std::get<I>(conv_layers)).apply(std::get<I>(conv_activation_results), 
            std::get<I + 1>(conv_weighted_inputs), std::get<I + 1>(conv_activation_results));
    });

    std::get<0>(neural_activation_results).features = 
        std::get<Num_Conv_Layers>(conv_activation_results).features;

    compile_range<Num_Neural_Layers>([&]<size_t I>(){
        (std::get<I>(neural_layers)).apply(std::get<I>(neural_activation_results), 
            std::get<I + 1>(neural_weighted_inputs), std::get<I + 1>(neural_activation_results));
    });
}

template<typename Conv_Layer_Tuple, typename Neural_Layer_Tuple, 
        typename Conv_Feature_Tuple, typename Neural_Feature_Tuple>
requires (std::tuple_size_v<Conv_Layer_Tuple> == std::tuple_size_v<Conv_Feature_Tuple> - 1 
        && std::tuple_size_v<Neural_Layer_Tuple> == std::tuple_size_v<Neural_Feature_Tuple> - 1)
void Network<Conv_Layer_Tuple, Neural_Layer_Tuple, 
        Conv_Feature_Tuple, Neural_Feature_Tuple>::backward_propagate(
    const std::vector<typename std::tuple_element_t<0, Conv_Feature_Tuple>::type> & output, 
    const Conv_Feature_Tuple & conv_weighted_inputs, 
    const Conv_Feature_Tuple & conv_activation_results, 
    const Neural_Feature_Tuple & neural_weighted_inputs, 
    const Neural_Feature_Tuple & neural_activation_results,
    Conv_Layer_Tuple & conv_layer_deriv, 
    Neural_Layer_Tuple & neural_layer_deriv)
{
    constexpr std::size_t Num_Conv_Layers = std::tuple_size_v<Conv_Layer_Tuple>;
    constexpr std::size_t Num_Neural_Layers = std::tuple_size_v<Neural_Layer_Tuple>;

    std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases = output;

    compile_range<std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::output_neurons>(
    [&]<size_t J>()
    {
        std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases[J] -= 
            std::get<Num_Neural_Layers>(neural_activation_results).features[J];
        
        compile_range<std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::input_neurons>(
        [&]<size_t K>()
        {
            std::get<Num_Neural_Layers - 1>(neural_layer_deriv).weights[
                J * std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::input_neurons + K] =
            std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases[J] *
            std::get<Num_Neural_Layers - 1>(neural_activation_results).features[K];
        });
    });

    compile_range<Num_Neural_Layers - 1, 1>(
    [&]<size_t I>()
    {
        std::get<Num_Neural_Layers - I>(neural_layers).apply_backwards(
            std::get<Num_Neural_Layers - I>(neural_layer_deriv).biases, 
            std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).biases);

        compile_range<std::tuple_element_t<Num_Neural_Layers - I - 1, Neural_Layer_Tuple>::output_neurons>(
        [&]<size_t J>()
        {
            std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).biases[J] *= 
            std::get<Num_Neural_Layers - I - 1>(neural_layers).activation_func.derivative(
                std::get<Num_Neural_Layers - I>(neural_weighted_inputs).features[J]);
            
            compile_range<std::tuple_element_t<Num_Neural_Layers - I - 1, Neural_Layer_Tuple>::input_neurons>(
            [&]<size_t K>()
            {
                std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).weights[
                    J * std::tuple_element_t<Num_Neural_Layers - I - 1, Neural_Layer_Tuple>::input_neurons + K] =
                std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).biases[J] *
                std::get<Num_Neural_Layers - I - 1>(neural_activation_results).features[K];
            });
        });
    });

    Conv_Feature_Tuple layer_delta;

    // construct delta L for the convolution backward propagation
    std::get<0>(neural_layers).apply_backwards(
        std::get<0>(neural_layer_deriv).biases, 
        std::get<Num_Conv_Layers>(layer_delta));

    compile_range<std::tuple_element_t<0, Neural_Layer_Tuple>::input_neurons>(
    [&]<size_t I>()
    {
        std::get<Num_Conv_Layers>(layer_delta)[I] *= 
        std::get<Num_Conv_Layers - 1>(conv_layers).activation_func.derivative(
            std::get<Num_Conv_Layers>(conv_weighted_inputs).features[I]);
    });

    // based on delta L, compule the kernel and bias derivatives for L
    compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::output_channels>(
    [&]<size_t OC>()
    {
        compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::input_channels>(
        [&]<size_t IC>()
        {
            compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size>(
            [&]<size_t KH>()
            {
                compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size>(
                [&]<size_t KW>()
                {
                    typename std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::type sum = 0;

                    compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::height>(
                    [&]<size_t H>()
                    {
                        compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::width>(
                        [&]<size_t W>()
                        {   
                            constexpr size_t delta_id = 
                                IC * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::height *
                                H * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::width +
                                W;

                            constexpr size_t pool_h = H * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size;
                            constexpr size_t pool_w = W * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size;

                            compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size>(
                            [&]<size_t PW>()
                            {
                                compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size>(
                                [&]<size_t PH>()
                                {
                                    constexpr size_t activation_height = 
                                        (pool_h + PH) * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::stride - 
                                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size / 2 + KH;

                                    constexpr size_t activation_width = 
                                        (pool_w + PW) * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::stride - 
                                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size / 2 + KW;

                                    if constexpr (activation_height >= 0 && activation_width >= 0 &&
                                        activation_height < std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::height &&
                                        activation_width < std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::width)
                                    {
                                        constexpr size_t activation_id = 
                                            OC * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::height *
                                            std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::width +
                                            activation_height * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::width +
                                            activation_width;

                                        sum += std::get<Num_Conv_Layers>(layer_delta)[delta_id] * 
                                            std::get<Num_Conv_Layers - 1>(conv_activation_results).features[activation_id];
                                    }
                                });
                            });
                        });
                    });
                    
                    constexpr size_t kernel_deriv_id = 
                        OC * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::input_channels *
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size *
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size +
                        IC * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size *
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size +
                        KW * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size + 
                        KH;

                    std::get<Num_Conv_Layers - 1>(conv_layer_deriv).kernels[kernel_deriv_id] = sum;
                });
            });
        });
    });

    compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::output_channels>(
    [&]<size_t OC>()
    {   
        typename std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::type sum = 0;

        compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::height>(
        [&]<size_t H>()
        {
            compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::width>(
            [&]<size_t W>()
            {   
                constexpr size_t delta_id = 
                    OC * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::height *
                    H * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::width +
                    W;
                
                sum += std::get<Num_Conv_Layers>(layer_delta)[delta_id];
            });
        });

        std::get<Num_Conv_Layers - 1>(conv_layer_deriv).biases[OC] = sum;
    });

    compile_range<Num_Conv_Layers - 1, 1>(
    [&]<size_t I>()
    {
        std::get<Num_Conv_Layers - I>(conv_layers).apply_backwards(
            std::get<Num_Conv_Layers - I + 1>(layer_delta), 
            std::get<Num_Conv_Layers - I>(layer_delta));

        // based on delta L, compule the kernel and bias derivatives for L
        compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::output_channels>(
        [&]<size_t OC>()
        {
            compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::input_channels>(
            [&]<size_t IC>()
            {
                compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size>(
                [&]<size_t KH>()
                {
                    compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size>(
                    [&]<size_t KW>()
                    {
                        typename std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::type sum = 0;

                        compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::height>(
                        [&]<size_t H>()
                        {
                            compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::width>(
                            [&]<size_t W>()
                            {   
                                constexpr size_t delta_id = 
                                    IC * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::height *
                                    H * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::width +
                                    W;

                                constexpr size_t pool_h = H * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size;
                                constexpr size_t pool_w = W * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size;

                                compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size>(
                                [&]<size_t PW>()
                                {
                                    compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size>(
                                    [&]<size_t PH>()
                                    {
                                        constexpr size_t activation_height = 
                                            (pool_h + PH) * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::stride - 
                                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size / 2 + KH;

                                        constexpr size_t activation_width = 
                                            (pool_w + PW) * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::stride - 
                                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size / 2 + KW;

                                        if constexpr (activation_height >= 0 && activation_width >= 0 &&
                                            activation_height < std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::height &&
                                            activation_width < std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::width)
                                        {
                                            constexpr size_t activation_id = 
                                                OC * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::height *
                                                std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::width +
                                                activation_height * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::width +
                                                activation_width;

                                            sum += std::get<Num_Conv_Layers - I>(layer_delta)[delta_id] * 
                                                std::get<Num_Conv_Layers - I - 1>(conv_activation_results).features[activation_id];
                                        }
                                    });
                                });
                            });
                        });
                        
                        constexpr size_t kernel_deriv_id = 
                            OC * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::input_channels *
                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size *
                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size +
                            IC * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size *
                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size +
                            KW * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size + 
                            KH;

                        std::get<Num_Conv_Layers - I - 1>(conv_layer_deriv).kernels[kernel_deriv_id] = sum;
                    });
                });
            });
        });

        compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::output_channels>(
        [&]<size_t OC>()
        {   
            typename std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::type sum = 0;

            compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::height>(
            [&]<size_t H>()
            {
                compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::width>(
                [&]<size_t W>()
                {   
                    constexpr size_t delta_id = 
                        OC * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::height *
                        H * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::width +
                        W;
                    
                    sum += std::get<Num_Conv_Layers - I>(layer_delta)[delta_id];
                });
            });

            std::get<Num_Conv_Layers - I - 1>(conv_layer_deriv).biases[OC] = sum;
        });
    });
}

template<size_t C, size_t H, size_t W,
        size_t NUM_CONV_LAYERS, size_t NUM_NEURAL_LAYERS, typename... Layer_Args>
requires (NUM_CONV_LAYERS + NUM_NEURAL_LAYERS == sizeof...(Layer_Args) &&
        NUM_CONV_LAYERS > 1 && NUM_NEURAL_LAYERS > 1)
inline auto network(Layer_Args&&... args){

    auto [conv_layers, neural_layers] = split_tuple<NUM_CONV_LAYERS - 1>(
        std::tuple<Layer_Args...>{std::forward<Layer_Args>(args)...});

    auto conv_featureMaps = featureMaps_from_layer<C, H, W>(conv_layers);

    using Conv_Layers_t = decltype(conv_layers);
    using Neural_Layers_t = decltype(neural_layers);
    using Conv_FeatureMap_t = decltype(conv_featureMaps);
    using LastConvFeatureMap = std::tuple_element_t<NUM_CONV_LAYERS, Conv_FeatureMap_t>;
    using FirstNeuralLayer = std::tuple_element_t<0, Neural_Layers_t>;

    if constexpr (flat_size<LastConvFeatureMap> != FirstNeuralLayer::input_neurons)
    {
        auto extended_neural_layers = add_tuple_begin(
            std::move(neural_layers), Neural_Layer<flat_size<LastConvFeatureMap>, 
            FirstNeuralLayer::input_neurons, decltype(FirstNeuralLayer::activation_func)>{});
            
        auto extended_neural_featureMaps = featureMaps_from_layer(extended_neural_layers);

        return Network<Conv_Layers_t, decltype(extended_neural_layers),
            decltype(conv_featureMaps), decltype(extended_neural_featureMaps)>{ 
            std::move(conv_layers), std::move(extended_neural_layers)};
    }
    else
    {
        auto neural_featureMaps = featureMaps_from_layer(neural_layers);

        return Network<Conv_Layers_t, Neural_Layers_t,
            Conv_FeatureMap_t, decltype(neural_featureMaps)>{ 
            std::move(conv_layers), std::move(neural_layers)};
    }
}

}



#endif // NETWORK_BASE_HPP