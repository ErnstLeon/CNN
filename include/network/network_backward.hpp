#ifndef NETWORK_BACKWARD_HPP
#define NETWORK_BACKWARD_HPP

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "../core.hpp"
#include "network_base.hpp"

namespace CNN::Network
{

/*
    Performs backpropagation through the CNN, computing derivatives of weights and biases.

    Function Arguments:
        true_output              : True output (HeapTensor1D with correct shape).
        conv_weighted_inputs     : Post-activation outputs of convolution layers.
        pooling_results          : Outputs of pooling layers.
        neural_weighted_inputs   : Pre-activation outputs of fully-connected layers.
        neural_activation_results: Post-activation outputs of fully-connected layers.
        conv_layer_deriv         : Stores computed gradients (kernels and biases) for convolutional layers.
        neural_layer_deriv       : Stores computed gradients (weights and biases) for fully-connected layers.

    Description:
        - Initializes the last fully-connected layer's deltas from the network output and the true labels.
        - Iteratively propagates deltas backwards through the fully-connected layers, applying activation derivatives
          and computing weight and bias gradients.
        - Propagates the error signal from the neural part into the convolutional stack.
        - For each convolutional layer (starting from the deepest):
            * Backpropagates pooled deltas to the previous feature maps.
            * Applies the activation function derivative elementwise.
            * Computes convolution kernel gradients by multiplying deltas with input activations.
            * Accumulates bias gradients as the sum of deltas.
        - Gradients are written into conv_layer_deriv and neural_layer_deriv, aligned with the
          network parameter structure.
*/
template<
    typename Conv_Layer_Tuple, 
    typename Neural_Layer_Tuple, 
    typename Conv_Feature_Tuple, 
    typename Pooled_Feature_Tuple, 
    typename Neural_Feature_Tuple>
requires(
    std::tuple_size_v<Conv_Layer_Tuple> == std::tuple_size_v<Conv_Feature_Tuple> - 1 
    && std::tuple_size_v<Neural_Layer_Tuple> == std::tuple_size_v<Neural_Feature_Tuple> - 1
    && std::tuple_size_v<Conv_Feature_Tuple> == std::tuple_size_v<Pooled_Feature_Tuple>)
void Network<
    Conv_Layer_Tuple, Neural_Layer_Tuple, 
    Conv_Feature_Tuple, Pooled_Feature_Tuple, Neural_Feature_Tuple>::backward_propagate(
    const HeapTensor1D<output_neurons, output_type> & true_output, 
    const Conv_Feature_Tuple & conv_weighted_inputs, 
    const Pooled_Feature_Tuple & pooling_results, 
    const Neural_Feature_Tuple & neural_weighted_inputs, 
    const Neural_Feature_Tuple & neural_activation_results,
    Conv_Layer_Tuple & conv_layer_deriv, 
    Neural_Layer_Tuple & neural_layer_deriv)
{
    constexpr std::size_t Num_Conv_Layers   = std::tuple_size_v<Conv_Layer_Tuple>;
    constexpr std::size_t Num_Neural_Layers = std::tuple_size_v<Neural_Layer_Tuple>;

    // Output layer references
    auto& delta_last     = std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases;
    auto& gradW_last     = std::get<Num_Neural_Layers - 1>(neural_layer_deriv).weights;
    const auto& act_prev = std::get<Num_Neural_Layers - 1>(neural_activation_results);
    const auto& act_out  = std::get<Num_Neural_Layers>(neural_activation_results);

    constexpr size_t output_neurons_last = 
        std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::output_neurons;
    constexpr size_t input_neurons_last = 
        std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::input_neurons;

    // Initialize output layer delta for backpropagation
    // Assumes the last layer uses Softmax activation with Cross-Entropy loss
    // In this case, the derivative simplifies to (predicted_output - true_output)
    delta_last = act_out;
    
    // Compute the gradient of the loss with respect to the last layer's weights. 
    // For each weight connecting input neuron K to output neuron J, the gradient is: 
    // dL/dW[J,K] = delta[J] * activation_previous[J]
    UNROLL_PRAGMA
    for (size_t j = 0; j < output_neurons_last; ++j) 
    {
        delta_last[j] -= true_output[j];

        UNROLL_PRAGMA
        for (size_t k = 0; k < input_neurons_last; ++k) 
        {
            gradW_last[j * input_neurons_last + k] =
                delta_last[j] * act_prev[k];
        }
    }

    compile_range<Num_Neural_Layers, 1>(
    [&]<size_t I>()
    {
        constexpr size_t output_neurons =
            std::tuple_element_t<Num_Neural_Layers - I - 1, Neural_Layer_Tuple>::output_neurons;
        constexpr size_t input_neurons =
            std::tuple_element_t<Num_Neural_Layers - I - 1, Neural_Layer_Tuple>::input_neurons;

        // Shorthand references
        const auto& delta_last = std::get<Num_Neural_Layers - I>(neural_layer_deriv).biases;
        auto& delta  = std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).biases;
        auto& gradW  = std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).weights;
        const auto& z   = std::get<Num_Neural_Layers - I>(neural_weighted_inputs);
        const auto& act_prev = std::get<Num_Neural_Layers - I - 1>(neural_activation_results);

        // Apply the transposed weight matrix of the current layer to the current delta. 
        // This propagates the error signal backward to the previous layer
        std::get<Num_Neural_Layers - I>(neural_layers).apply_backwards(delta_last, delta);

        // Multiply elementwise by the derivative of the activation function of the previous layer. 
        // This gives the delta for the previous layer
        UNROLL_PRAGMA
        for (size_t j = 0; j < output_neurons; ++j) 
        {
            delta[j] *= std::get<Num_Neural_Layers - I - 1>(neural_layers).activation_func.derivative(z[j]);

            // Compute the gradient of the weights for the previous layer. 
            // Each weight connecting neuron K to neuron J: 
            // dL/dW[J,K] = delta[J] * activation_previous[K]
            UNROLL_PRAGMA
            for (size_t k = 0; k < input_neurons; ++k) 
            {
                gradW[j * input_neurons + k] = delta[j] * act_prev[k];
            }
        }
    });

    // Initialize storage for convolutional and pooling layer deltas
    Pooled_Feature_Tuple pooled_layer_delta;
    Conv_Feature_Tuple conv_layer_delta;

    // Propagate the error from the first fully-connected (neural) layer
    // back into the last convolutional feature maps.
    // This applies the transposed weights of the first fully-connected layer to the output delta
    std::get<0>(neural_layers).apply_backwards(
        std::get<0>(neural_layer_deriv).biases, 
        std::get<Num_Conv_Layers>(pooled_layer_delta));

    // Unpool the pooled delta back to pre-pooled size
    constexpr size_t channels_last =
        std::tuple_element_t<Num_Conv_Layers, Pooled_Feature_Tuple>::size_x;
    constexpr size_t pooled_h_last =
        std::tuple_element_t<Num_Conv_Layers, Pooled_Feature_Tuple>::size_y;
    constexpr size_t pooled_w_last =
        std::tuple_element_t<Num_Conv_Layers, Pooled_Feature_Tuple>::size_z;
    constexpr size_t pre_pooled_h_last =
        std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_y;
    constexpr size_t pre_pooled_w_last =
        std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z;
    constexpr size_t pool_size_last =
        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size;
    using T_last = typename
        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::type;

    const T_last scale_last = T_last(1) / (pool_size_last * pool_size_last);

    UNROLL_PRAGMA
    for (size_t c = 0; c < channels_last; ++c) {
        UNROLL_PRAGMA
        for (size_t ph = 0; ph < pooled_h_last; ++ph) {
            UNROLL_PRAGMA
            for (size_t pw = 0; pw < pooled_w_last; ++pw) {

                const T_last delta_val = std::get<Num_Conv_Layers>(pooled_layer_delta)(c, ph, pw) * scale_last;

                UNROLL_PRAGMA
                for (size_t kh = 0; kh < pool_size_last; ++kh) {
                    UNROLL_PRAGMA
                    for (size_t kw = 0; kw < pool_size_last; ++kw) {
                        const size_t h = ph * pool_size_last + kh;
                        const size_t w = pw * pool_size_last + kw;

                        if (h < pre_pooled_h_last && w < pre_pooled_w_last) {
                            std::get<Num_Conv_Layers>(conv_layer_delta)(c, h, w) = delta_val;
                        }
                    }
                }
            }
        }
    }

    // Apply elementwise multiplication by the derivative of the convolution layer's activation function.
    // This serves as the starting point for backpropagation through the convolutional layers.
    constexpr size_t conv_size_last = 
        std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size;

    UNROLL_PRAGMA
    for(size_t i = 0; i < conv_size_last; ++i)
    {
        std::get<Num_Conv_Layers>(conv_layer_delta)[i] *= 
            std::get<Num_Conv_Layers - 1>(conv_layers).activation_func.derivative(
            std::get<Num_Conv_Layers>(conv_weighted_inputs)[i]);
    }

    // Compute gradients of the convolution kernels for the last convolutional layer.
    //
    // Outer loops iterate over:
    //   OC = output channels (kernels) of the current conv layer
    //   IC = input channels feeding into each kernel
    //   KH, KW = kernel height and width indices
    //
    // Using stride and kernel offset compute the corresponding
    // activation position in the input feature maps (activation_height, activation_width).
    // Multiply this input activation by the unpooled_delta at (OC, H, W) and 
    // sum contributions to obtain the kernel gradient.
    //
    // Mathematically:
    //     dL/dK[OC, IC, KH, KW] = Σ_H Σ_W delta_unpooled[OC, H, W] * ∂Z[OC, H, W] / ∂K[OC, IC, KH, KW]
    // where ∂Z/∂K represents the input activation corresponding to the kernel element.
    //
    // Effectively, each kernel gradient element is the sum of deltas multiplied by the input values
    // that contributed to the respective output positions, weighted by the kernel.
    constexpr size_t output_channels_last =
        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::output_channels;
    constexpr size_t input_channels_last =
        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::input_channels;
    constexpr size_t kernel_size_last =
        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size;
    constexpr size_t stride_last =
        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::stride;
    constexpr size_t pre_conv_h_last =
        std::tuple_element_t<Num_Conv_Layers - 1, Pooled_Feature_Tuple>::size_y;
    constexpr size_t pre_conv_w_last =
        std::tuple_element_t<Num_Conv_Layers - 1, Pooled_Feature_Tuple>::size_z;
    constexpr size_t post_conv_h_last =
        std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_y;
    constexpr size_t post_conv_w_last =
        std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z;
    
    UNROLL_PRAGMA
    for(size_t out_channel = 0; out_channel < output_channels_last; ++out_channel)
    {
        UNROLL_PRAGMA
        for(size_t in_channel = 0; in_channel < input_channels_last; ++in_channel)
        {
            UNROLL_PRAGMA
            for(size_t k_h = 0; k_h < kernel_size_last; ++k_h)
            {
                UNROLL_PRAGMA
                for(size_t k_w = 0; k_w < kernel_size_last; ++k_w)
                {
                    typename std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::type sum = 0;

                    UNROLL_PRAGMA
                    for(size_t h = 0; h < post_conv_h_last; ++h)
                    {
                        UNROLL_PRAGMA
                        for(size_t w = 0; w < post_conv_w_last; ++w)
                        { 
                            const size_t delta_id = 
                                out_channel * post_conv_h_last * post_conv_w_last +
                                h * post_conv_w_last + w;

                            const size_t activation_height = 
                                h * stride_last - kernel_size_last / 2 + k_h;

                            const size_t activation_width = 
                                w * stride_last - kernel_size_last / 2 + k_w;

                            if (activation_height >= 0 && activation_width >= 0 &&
                                activation_height < pre_conv_h_last && activation_width < pre_conv_w_last)
                            {
                                const size_t activation_id = 
                                    in_channel * pre_conv_h_last * pre_conv_w_last +
                                    activation_height * pre_conv_w_last +
                                    activation_width;

                                sum += std::get<Num_Conv_Layers>(conv_layer_delta)[delta_id] * 
                                    std::get<Num_Conv_Layers - 1>(pooling_results)[activation_id];
                            }
                        }
                    }
                    
                    const size_t kernel_deriv_id = 
                        out_channel * input_channels_last * kernel_size_last * kernel_size_last +
                        in_channel * kernel_size_last * kernel_size_last +
                        k_h * kernel_size_last + k_w;

                    std::get<Num_Conv_Layers - 1>(conv_layer_deriv).kernels[kernel_deriv_id] = sum;
                }
            }
        }
    }

    // Compute the gradient of the biases for the last convolutional layer.
    // For each output channel (OC), sum the delta values over all spatial positions (H, W)
    // in the corresponding feature map. This gives:
    //     dL/db[OC] = Σ_H Σ_W delta[OC, H, W]
    UNROLL_PRAGMA
    for(size_t out_channel = 0; out_channel < output_channels_last; ++out_channel)
    { 
        typename std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::type sum = 0;

        UNROLL_PRAGMA
        for(size_t h = 0; h < pre_pooled_h_last; ++h)
        {
            UNROLL_PRAGMA
            for(size_t w = 0; w < pre_pooled_w_last; ++w)
            { 
                const size_t delta_id = 
                    out_channel * pre_pooled_h_last * pre_pooled_w_last +
                    h * pre_pooled_w_last + w;
                
                sum += std::get<Num_Conv_Layers>(conv_layer_delta)[delta_id];
            }
        }
        std::get<Num_Conv_Layers - 1>(conv_layer_deriv).biases[out_channel] = sum;
    }

    // Backpropagation through each convolutional layer:
    //
    // Propagate the delta from the next layer to the current layer using the layer's backward function.
    //       delta_current = layer.apply_backwards(delta_next)
    //
    // Multiply elementwise by the derivative of the activation function after pooling
    //       delta_current[i] *= f'(z_current[i])
    //
    // After this step, layer_delta contains the correctly scaled delta,
    // which is then used in the previously described kernel and bias gradient computations.
    compile_range<Num_Conv_Layers, 1>(
    [&]<size_t I>()
    {
        std::get<Num_Conv_Layers - I>(conv_layers).apply_backwards(
            std::get<Num_Conv_Layers - I + 1>(conv_layer_delta), 
            std::get<Num_Conv_Layers - I>(pooled_layer_delta));

        // Unpool the pooled delta back to pre-pooled size
        constexpr size_t channels = 
            std::tuple_element_t<Num_Conv_Layers - I, Pooled_Feature_Tuple>::size_x;
        constexpr size_t pooled_h = 
            std::tuple_element_t<Num_Conv_Layers - I, Pooled_Feature_Tuple>::size_y;
        constexpr size_t pooled_w = 
            std::tuple_element_t<Num_Conv_Layers - I, Pooled_Feature_Tuple>::size_z;
        constexpr size_t pre_pooled_h = 
            std::tuple_element_t<Num_Conv_Layers - I - 1, Pooled_Feature_Tuple>::size_y;
        constexpr size_t pre_pooled_w = 
            std::tuple_element_t<Num_Conv_Layers - I - 1, Pooled_Feature_Tuple>::size_z;
        constexpr size_t pool_size = 
            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size;
        using T = typename
            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::type;

        const T scale = T(1) / (pool_size * pool_size);

        UNROLL_PRAGMA
        for (size_t c = 0; c < channels; ++c) {
            UNROLL_PRAGMA
            for (size_t ph = 0; ph < pooled_h; ++ph) {
                UNROLL_PRAGMA
                for (size_t pw = 0; pw < pooled_w; ++pw) {

                    const T delta_val = std::get<Num_Conv_Layers - I>(pooled_layer_delta)(c, ph, pw) * scale;

                    UNROLL_PRAGMA
                    for (size_t kh = 0; kh < pool_size; ++kh) {
                        UNROLL_PRAGMA
                        for (size_t kw = 0; kw < pool_size; ++kw) {
                            const size_t h = ph * pool_size + kh;
                            const size_t w = pw * pool_size + kw;

                            if(h < pre_pooled_h && w < pre_pooled_w)
                                std::get<Num_Conv_Layers - I>(conv_layer_delta)(c, h, w) = delta_val;
                        }
                    }
                }
            }
        }

        constexpr size_t conv_size = 
            std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size;

        UNROLL_PRAGMA
        for(size_t i = 0; i < conv_size; ++i)
        {
            std::get<Num_Conv_Layers - I>(conv_layer_delta)[i] *= 
                std::get<Num_Conv_Layers - I - 1>(conv_layers).activation_func.derivative(
                std::get<Num_Conv_Layers - I>(conv_weighted_inputs)[i]);
        }

        // As described above, use the current layer's delta to compute the kernel and bias gradients.
        constexpr size_t output_channels =
            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::output_channels;
        constexpr size_t input_channels =
            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::input_channels;
        constexpr size_t kernel_size =
            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size;
        constexpr size_t stride =
            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::stride;
        constexpr size_t pre_conv_h =
            std::tuple_element_t<Num_Conv_Layers - I - 1, Pooled_Feature_Tuple>::size_y;
        constexpr size_t pre_conv_w =
            std::tuple_element_t<Num_Conv_Layers - I - 1, Pooled_Feature_Tuple>::size_z;
        constexpr size_t post_conv_h =
            std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y;
        constexpr size_t post_conv_w =
            std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z;

        UNROLL_PRAGMA
        for(size_t out_channel = 0; out_channel < output_channels; ++out_channel)
        {
            UNROLL_PRAGMA
            for(size_t in_channel = 0; in_channel < input_channels; ++in_channel)
            {
                UNROLL_PRAGMA
                for(size_t k_h = 0; k_h < kernel_size; ++k_h)
                {
                    UNROLL_PRAGMA
                    for(size_t k_w = 0; k_w < kernel_size; ++k_w)
                    {
                        typename std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::type sum = 0;

                        UNROLL_PRAGMA
                        for(size_t h = 0; h < post_conv_h; ++h)
                        {
                            UNROLL_PRAGMA
                            for(size_t w = 0; w < post_conv_w; ++w)
                            {
                                const size_t delta_id = 
                                    out_channel * post_conv_h * post_conv_w +
                                    h * post_conv_w + w;

                                const size_t activation_height = 
                                    h * stride - kernel_size / 2 + k_h;

                                const size_t activation_width = 
                                    w * stride - kernel_size / 2 + k_w;

                                if (activation_height >= 0 && activation_width >= 0 &&
                                    activation_height < pre_conv_h && activation_width < pre_conv_w)
                                {
                                    const size_t activation_id = 
                                        in_channel * pre_conv_h * pre_conv_w +
                                        activation_height * pre_conv_w +
                                        activation_width;

                                    sum += std::get<Num_Conv_Layers - I>(conv_layer_delta)[delta_id] * 
                                        std::get<Num_Conv_Layers - I - 1>(pooling_results)[activation_id];
                                }
                            }
                        }
                        
                        const size_t kernel_deriv_id = 
                            out_channel * input_channels * kernel_size * kernel_size +
                            in_channel * kernel_size * kernel_size +
                            k_h * kernel_size + k_w;

                        std::get<Num_Conv_Layers - I - 1>(conv_layer_deriv).kernels[kernel_deriv_id] = sum;
                    }
                }
            }
        }

        UNROLL_PRAGMA
        for(size_t out_channel = 0; out_channel < output_channels; ++out_channel)
        { 
            typename std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::type sum = 0;
            
            UNROLL_PRAGMA
            for(size_t h = 0; h < post_conv_h; ++h)
            {
                UNROLL_PRAGMA
                for(size_t w = 0; w < post_conv_w; ++w)
                {
                    const size_t delta_id = 
                        out_channel * post_conv_h * post_conv_w +
                        h * post_conv_w + w;
                    
                    sum += std::get<Num_Conv_Layers - I>(conv_layer_delta)[delta_id];
                }
            }

            std::get<Num_Conv_Layers - I - 1>(conv_layer_deriv).biases[out_channel] = sum;
        }
    });
}

}

#endif // NETWORK_BACKWARD_HPP