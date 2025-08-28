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

    using input_type = typename std::tuple_element_t<0, Conv_Feature_Tuple>::type;
    using output_type = typename std::tuple_element_t<std::tuple_size_v<Neural_Feature_Tuple> - 1, 
        Neural_Feature_Tuple>::type;

    using conv_layer_tuple = Conv_Layer_Tuple;
    using neural_layer_tuple = Neural_Layer_Tuple;

    using conv_feature_tuple = Conv_Feature_Tuple;
    using neural_feature_tuple = Neural_Feature_Tuple;

    static constexpr size_t input_channels = std::tuple_element_t<0, Conv_Feature_Tuple>::size_x;
    static constexpr size_t input_height = std::tuple_element_t<0, Conv_Feature_Tuple>::size_y;
    static constexpr size_t input_width = std::tuple_element_t<0, Conv_Feature_Tuple>::size_z;
    static constexpr size_t output_neurons = std::tuple_element_t<std::tuple_size_v<Neural_Feature_Tuple> - 1, 
        Neural_Feature_Tuple>::size;

    static constexpr size_t num_conv_layers = std::tuple_size_v<Conv_Layer_Tuple>;
    static constexpr size_t num_neural_layers = std::tuple_size_v<Neural_Layer_Tuple>;

    Conv_Layer_Tuple conv_layers;
    Neural_Layer_Tuple neural_layers;

    Network(const Conv_Layer_Tuple& conv_layers, const Neural_Layer_Tuple& neural_layers) 
    : conv_layers{conv_layers},
    neural_layers{neural_layers} {};

    Network(Conv_Layer_Tuple&& conv_layers, Neural_Layer_Tuple&& neural_layers) 
    : conv_layers{std::move(conv_layers)},
    neural_layers{std::move(neural_layers)} {};

public: 

    void foward_propagate(
        const HeapTensor3D<input_channels, input_height, input_width, input_type> &, 
        Conv_Feature_Tuple &, 
        Conv_Feature_Tuple &, 
        Neural_Feature_Tuple &, 
        Neural_Feature_Tuple &);

    void backward_propagate(
        const HeapTensor1D<output_neurons, output_type> &, 
        const Conv_Feature_Tuple &, 
        const Conv_Feature_Tuple &, 
        const Neural_Feature_Tuple &, 
        const Neural_Feature_Tuple &,
        Conv_Layer_Tuple &, 
        Neural_Layer_Tuple &);

    void compute_gradient(
        const HeapTensor3D<input_channels, input_height, input_width, input_type> &,
        const HeapTensor1D<output_neurons, output_type> &,
        Conv_Layer_Tuple &, 
        Neural_Layer_Tuple &);  

public:

    template<typename Optimizer>
    void train(
        const std::vector<
        std::pair<HeapTensor3D<input_channels, input_height, input_width, input_type>, 
        HeapTensor1D<output_neurons, output_type>>> &, Optimizer optimizer, 
        size_t, 
        size_t num_epochs = 1000);

    HeapTensor1D<output_neurons, output_type> evaluate(
        const HeapTensor3D<input_channels, input_height, input_width, input_type> &);

};

/*
    Constructs a CNN from the given layers and image dimensions.

    Template Parameters:
        C  : Number of image channels
        H  : Image height
        W  : Image width
        NUM_CONV_LAYERS   : Number of convolutional layers
        NUM_NEURAL_LAYERS : Number of fully-connected (neural) layers
        Layer_Args        : Types of the input layers (deduced)

    Function Arguments:
        args : The layers of the CNN

    Description:
        - Splits the provided layers into two tuples: convolutional layers and neural layers.
        - Computes, at compile time, the feature sizes for both convolutional and neural layers,
            ensuring compatibility between image dimensions and network design.
        - If the output of the last convolutional layer does not match the input size of the
            first neural layer, an additional fully-connected layer with ReLU activation is inserted
            to bridge the connection.
        - Finally, assembles the CNN from the prepared tuples.
*/
template<
    size_t C, size_t H, size_t W,
    size_t NUM_CONV_LAYERS, 
    size_t NUM_NEURAL_LAYERS, 
    typename... Layer_Args>
requires(
    NUM_CONV_LAYERS + NUM_NEURAL_LAYERS == sizeof...(Layer_Args) &&
    NUM_CONV_LAYERS > 0 && NUM_NEURAL_LAYERS > 0)
inline auto network(Layer_Args&&... args)
{
    auto [conv_layers, neural_layers] = split_tuple<NUM_CONV_LAYERS - 1>(
        std::tuple<Layer_Args...>{std::forward<Layer_Args>(args)...});

    using Conv_Layers_t = decltype(conv_layers);
    using Neural_Layers_t = decltype(neural_layers);

    using Conv_Features_Tuple_t = decltype(features_from_layer<C, H, W, Conv_Layers_t>());
    using Last_Conv_Features_t = std::tuple_element_t<NUM_CONV_LAYERS, Conv_Features_Tuple_t>;
    using First_Neural_Layer_t = std::tuple_element_t<0, Neural_Layers_t>;
    using Last_Neural_Layer_t = std::tuple_element_t<NUM_NEURAL_LAYERS - 1, Neural_Layers_t>;

    static_assert(output_eq_input_channels<Conv_Layers_t>(), "Output and input channel size does not match.");
    static_assert(output_eq_input_neurons<Neural_Layers_t>(), "Output and input neuron size does not match.");
    static_assert(std::is_same_v<decltype(Last_Neural_Layer_t::activation_func), 
        Softmax<typename Last_Neural_Layer_t::type>>, "Last activation function must be softmax.");

    if constexpr (Last_Conv_Features_t::size != First_Neural_Layer_t::input_neurons)
    {
        auto extended_neural_layers = add_tuple_begin(
            std::move(neural_layers), Neural_Layer<Last_Conv_Features_t::size, 
            First_Neural_Layer_t::input_neurons, ReLU<typename First_Neural_Layer_t::type>>{});

        using Extended_Neural_Layers_t = decltype(extended_neural_layers);
        using Neural_Features_t = decltype(features_from_layer<Extended_Neural_Layers_t>());

        return Network<Conv_Layers_t, Extended_Neural_Layers_t,
            Conv_Features_Tuple_t, Neural_Features_t>{ 
            std::move(conv_layers), std::move(extended_neural_layers)};
    }
    else
    {
        using Neural_Features_t = decltype(features_from_layer<Neural_Layers_t>());

        return Network<Conv_Layers_t, Neural_Layers_t,
            Conv_Features_Tuple_t, Neural_Features_t>{ 
            std::move(conv_layers), std::move(neural_layers)};
    }
}

}



#endif // NETWORK_BASE_HPP