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

    static constexpr size_t input_channels = std::tuple_element_t<0, Conv_Feature_Tuple>::size_x;
    static constexpr size_t input_height = std::tuple_element_t<0, Conv_Feature_Tuple>::size_y;
    static constexpr size_t input_width = std::tuple_element_t<0, Conv_Feature_Tuple>::size_z;

    static constexpr size_t output_neurons = std::tuple_element_t<std::tuple_size_v<Neural_Feature_Tuple> - 1, 
        Neural_Feature_Tuple>::size;

    using conv_layer_tuple = Conv_Layer_Tuple;
    using neural_layer_tuple = Neural_Layer_Tuple;

    using conv_feature_tuple = Conv_Feature_Tuple;
    using neural_feature_tuple = Neural_Feature_Tuple;

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

};

template<
    typename Conv_Layer_Tuple, typename Neural_Layer_Tuple, 
    typename Conv_Feature_Tuple, typename Neural_Feature_Tuple>
requires(
    std::tuple_size_v<Conv_Layer_Tuple> == std::tuple_size_v<Conv_Feature_Tuple> - 1 && 
    std::tuple_size_v<Neural_Layer_Tuple> == std::tuple_size_v<Neural_Feature_Tuple> - 1)
template<typename Optimizer>
void Network<
    Conv_Layer_Tuple, Neural_Layer_Tuple, 
    Conv_Feature_Tuple, Neural_Feature_Tuple>::train(
    const std::vector<std::pair<HeapTensor3D<input_channels, input_height, input_width, input_type>, 
    HeapTensor1D<output_neurons, output_type>>> & dataset_orig, 
    Optimizer optimizer, size_t batch_size, size_t num_epochs)
{
    using T = input_type;

    static constexpr size_t num_conv_layers = std::tuple_size_v<Conv_Layer_Tuple>;
    static constexpr size_t num_neural_layers = std::tuple_size_v<Neural_Layer_Tuple>;

    auto dataset = dataset_orig;

    std::mt19937 gen(100);
    size_t dataset_size = dataset.size();
    size_t num_batches = (dataset_size + batch_size - 1) / batch_size;

    Conv_Layer_Tuple conv_deriv;
    Neural_Layer_Tuple neural_deriv;

    auto conv_kernels_optimizers = kernels_optimizer<Conv_Layer_Tuple>(optimizer);
    auto conv_biases_optimizers = biases_optimizer<Conv_Layer_Tuple>(optimizer);

    auto neural_weights_optimizers = weights_optimizer<Neural_Layer_Tuple>(optimizer);
    auto neural_biases_optimizers = biases_optimizer<Neural_Layer_Tuple>(optimizer);

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) 
    {
        std::shuffle(dataset.begin(), dataset.end(), gen);

        for (size_t b = 0; b < num_batches; ++b) 
        {
            size_t current_batch_size = std::min(batch_size, dataset_size - b * batch_size);

            // set derivates to zero for accumulation over samples
            compile_range<num_conv_layers>([&]<size_t I>(){
                std::get<I>(conv_deriv).kernels.fill(0);
                std::get<I>(conv_deriv).biases.fill(0);
            });

            compile_range<num_neural_layers>([&]<size_t I>(){
                std::get<I>(neural_deriv).weights.fill(0);
                std::get<I>(neural_deriv).biases.fill(0);
            });

            T error = 0;

            #pragma omp parallel shared(conv_deriv, neural_deriv) reduction(+ : error)
            {
                Conv_Layer_Tuple thread_conv_deriv;
                Neural_Layer_Tuple thread_neural_deriv;

                // set derivates to zero for accumulation over samples
                compile_range<num_conv_layers>([&]<size_t I>(){
                    std::get<I>(thread_conv_deriv).kernels.fill(0);
                    std::get<I>(thread_conv_deriv).biases.fill(0);
                });

                compile_range<num_neural_layers>([&]<size_t I>(){
                    std::get<I>(thread_neural_deriv).weights.fill(0);
                    std::get<I>(thread_neural_deriv).biases.fill(0);
                });

                Conv_Layer_Tuple local_conv_deriv;
                Neural_Layer_Tuple local_neural_deriv;

                #pragma omp for nowait
                for (size_t sample = 0; sample < current_batch_size; ++sample) 
                {
                    size_t sample_index = b * batch_size + sample;

                    const auto & input = dataset[sample_index].first;
                    const auto & true_output = dataset[sample_index].second;

                    compute_gradient(input, true_output, local_conv_deriv, local_neural_deriv);

                    compile_range<num_conv_layers>(
                    [&]<size_t I>(){
                        std::get<I>(thread_conv_deriv).kernels += std::get<I>(local_conv_deriv).kernels;
                        std::get<I>(thread_conv_deriv).biases += std::get<I>(local_conv_deriv).biases;
                    });

                    compile_range<num_neural_layers>(
                    [&]<size_t I>(){
                        std::get<I>(thread_neural_deriv).weights += std::get<I>(local_neural_deriv).weights;
                        std::get<I>(thread_neural_deriv).biases += std::get<I>(local_neural_deriv).biases;
                    });

                    Conv_Feature_Tuple a;
                    Conv_Feature_Tuple b;
                    Neural_Feature_Tuple c; 
                    Neural_Feature_Tuple d;

                    foward_propagate(input, a,b,c,d);
                    error += cross_entropy_loss(std::get<num_neural_layers>(d), true_output);
                }

                #pragma omp critical
                {
                    compile_range<num_conv_layers>(
                    [&]<size_t I>(){
                        std::get<I>(conv_deriv).kernels += std::get<I>(thread_conv_deriv).kernels;
                        std::get<I>(conv_deriv).biases += std::get<I>(thread_conv_deriv).biases;
                    });

                    compile_range<num_neural_layers>(
                    [&]<size_t I>(){
                        std::get<I>(neural_deriv).weights += std::get<I>(thread_neural_deriv).weights;
                        std::get<I>(neural_deriv).biases += std::get<I>(thread_neural_deriv).biases;
                    });
                }
            }

            T inv_sample_size = T{1} / static_cast<T>(current_batch_size);

            compile_range<num_conv_layers>(
            [&]<size_t I>(){
                std::get<I>(conv_deriv).kernels *= inv_sample_size;
                std::get<I>(conv_deriv).biases *= inv_sample_size;
            });

            compile_range<num_neural_layers>(
            [&]<size_t I>(){
                std::get<I>(neural_deriv).weights *= inv_sample_size;
                std::get<I>(neural_deriv).biases *= inv_sample_size;
            });

            std::cout << std::get<0>(conv_layers).kernels[0] << std::endl;
            std::cout << std::get<0>(conv_deriv).kernels[0] << std::endl;

            compile_range<num_conv_layers>(
            [&]<size_t I>(){
                std::get<I>(conv_kernels_optimizers).update(std::get<I>(conv_layers).kernels, std::get<I>(conv_deriv).kernels);
                std::get<I>(conv_biases_optimizers).update(std::get<I>(conv_layers).biases, std::get<I>(conv_deriv).biases);
            });

            std::cout << std::get<0>(conv_layers).kernels[0] << std::endl;

            compile_range<num_neural_layers>(
            [&]<size_t I>(){
                std::get<I>(neural_weights_optimizers).update(std::get<I>(neural_layers).weights, std::get<I>(neural_deriv).weights);
                std::get<I>(neural_biases_optimizers).update(std::get<I>(neural_layers).biases, std::get<I>(neural_deriv).biases);
            });
/*
            for (int i = 0; i < 64 * 3; ++i) std::cout << std::get<0>(neural_layers).weights[i] << std::endl;
            for (int i = 0; i < 64 * 3; ++i) std::cout << std::get<0>(neural_deriv).weights[i] << std::endl;

            for (int i = 0; i < 64 * 64; ++i) std::cout << std::get<1>(neural_layers).weights[i] << std::endl;
            for (int i = 0; i < 64 * 64; ++i) std::cout << std::get<1>(neural_deriv).weights[i] << std::endl;

            for (int i = 0; i < 64 * 200; ++i) std::cout << std::get<2>(neural_layers).weights[i] << std::endl;
            for (int i = 0; i < 64 * 200; ++i) std::cout << std::get<2>(neural_deriv).weights[i] << std::endl;
*/
            std::cout << "done batch: " << b << ", error: " << error/current_batch_size << std::endl;
        }
    }

}

template<
    typename Conv_Layer_Tuple, typename Neural_Layer_Tuple, 
    typename Conv_Feature_Tuple, typename Neural_Feature_Tuple>
requires(
    std::tuple_size_v<Conv_Layer_Tuple> == std::tuple_size_v<Conv_Feature_Tuple> - 1 && 
    std::tuple_size_v<Neural_Layer_Tuple> == std::tuple_size_v<Neural_Feature_Tuple> - 1)
void Network<
    Conv_Layer_Tuple, Neural_Layer_Tuple, 
    Conv_Feature_Tuple, Neural_Feature_Tuple>::compute_gradient(
    const HeapTensor3D<input_channels, input_height, input_width, input_type> & input, 
    const HeapTensor1D<output_neurons, output_type> & output, 
    Conv_Layer_Tuple & conv_gradients,
    Neural_Layer_Tuple & neural_gradients)
{
    Conv_Feature_Tuple conv_weighted_inputs{};
    Conv_Feature_Tuple conv_activation_outputs{};
    Neural_Feature_Tuple neural_weighted_inputs{};
    Neural_Feature_Tuple neutal_activation_outputs{};
    
    foward_propagate(input, 
        conv_weighted_inputs, 
        conv_activation_outputs, 
        neural_weighted_inputs, 
        neutal_activation_outputs);

    backward_propagate(output, 
        conv_weighted_inputs, 
        conv_activation_outputs, 
        neural_weighted_inputs, 
        neutal_activation_outputs,
        conv_gradients,
        neural_gradients);
}

template<
    typename Conv_Layer_Tuple, typename Neural_Layer_Tuple, 
    typename Conv_Feature_Tuple, typename Neural_Feature_Tuple>
requires(
    std::tuple_size_v<Conv_Layer_Tuple> == std::tuple_size_v<Conv_Feature_Tuple> - 1 && 
    std::tuple_size_v<Neural_Layer_Tuple> == std::tuple_size_v<Neural_Feature_Tuple> - 1)
void Network<
    Conv_Layer_Tuple, Neural_Layer_Tuple, 
    Conv_Feature_Tuple, Neural_Feature_Tuple>::foward_propagate(
    const HeapTensor3D<input_channels, input_height, input_width, input_type> & input, 
    Conv_Feature_Tuple & conv_weighted_inputs, 
    Conv_Feature_Tuple & conv_activation_results, 
    Neural_Feature_Tuple & neural_weighted_inputs, 
    Neural_Feature_Tuple & neural_activation_results)
{
    constexpr std::size_t Num_Conv_Layers = std::tuple_size_v<Conv_Layer_Tuple>;
    constexpr std::size_t Num_Neural_Layers = std::tuple_size_v<Neural_Layer_Tuple>;

    std::get<0>(conv_activation_results) = input;

    compile_range<Num_Conv_Layers>([&]<size_t I>(){
        std::get<I>(conv_layers).apply(std::get<I>(conv_activation_results), 
            std::get<I + 1>(conv_weighted_inputs), std::get<I + 1>(conv_activation_results));
    });

    std::get<0>(neural_activation_results) = 
        std::get<Num_Conv_Layers>(conv_activation_results);
    
    compile_range<Num_Neural_Layers>([&]<size_t I>(){
        std::get<I>(neural_layers).apply(std::get<I>(neural_activation_results), 
            std::get<I + 1>(neural_weighted_inputs), std::get<I + 1>(neural_activation_results));
    });
}

template<
    typename Conv_Layer_Tuple, 
    typename Neural_Layer_Tuple, 
    typename Conv_Feature_Tuple, 
    typename Neural_Feature_Tuple>
requires(
    std::tuple_size_v<Conv_Layer_Tuple> == std::tuple_size_v<Conv_Feature_Tuple> - 1 &&
    std::tuple_size_v<Neural_Layer_Tuple> == std::tuple_size_v<Neural_Feature_Tuple> - 1)
void Network<
    Conv_Layer_Tuple, Neural_Layer_Tuple, 
    Conv_Feature_Tuple, Neural_Feature_Tuple>::backward_propagate(
    const HeapTensor1D<output_neurons, output_type> & true_output, 
    const Conv_Feature_Tuple & conv_weighted_inputs, 
    const Conv_Feature_Tuple & conv_activation_results, 
    const Neural_Feature_Tuple & neural_weighted_inputs, 
    const Neural_Feature_Tuple & neural_activation_results,
    Conv_Layer_Tuple & conv_layer_deriv, 
    Neural_Layer_Tuple & neural_layer_deriv)
{
    constexpr std::size_t Num_Conv_Layers = std::tuple_size_v<Conv_Layer_Tuple>;
    constexpr std::size_t Num_Neural_Layers = std::tuple_size_v<Neural_Layer_Tuple>;

    std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases = 
        std::get<Num_Neural_Layers>(neural_activation_results);

    compile_range<std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::output_neurons>(
    [&]<size_t J>()
    {
        std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases[J] -= true_output[J];
        
        compile_range<std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::input_neurons>(
        [&]<size_t K>()
        {
            std::get<Num_Neural_Layers - 1>(neural_layer_deriv).weights[
                J * std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::input_neurons + K] =
            std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases[J] *
            std::get<Num_Neural_Layers - 1>(neural_activation_results)[K];
        });
    });

    compile_range<Num_Neural_Layers, 1>(
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
                std::get<Num_Neural_Layers - I>(neural_weighted_inputs)[J]);
            
            compile_range<std::tuple_element_t<Num_Neural_Layers - I - 1, Neural_Layer_Tuple>::input_neurons>(
            [&]<size_t K>()
            {
                std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).weights[
                    J * std::tuple_element_t<Num_Neural_Layers - I - 1, Neural_Layer_Tuple>::input_neurons + K] =
                std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).biases[J] *
                std::get<Num_Neural_Layers - I - 1>(neural_activation_results)[K];
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
            std::get<Num_Conv_Layers>(conv_weighted_inputs)[I]);
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

                    compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_y>(
                    [&]<size_t H>()
                    {
                        compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z>(
                        [&]<size_t W>()
                        {   
                            constexpr size_t delta_id = 
                                OC * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_y *
                                std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z +
                                H * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z +
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
                                        activation_height < std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::size_y &&
                                        activation_width < std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::size_z)
                                    {
                                        constexpr size_t activation_id = 
                                            IC * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::size_y *
                                            std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::size_z +
                                            activation_height * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::size_z +
                                            activation_width;

                                        sum += std::get<Num_Conv_Layers>(layer_delta)[delta_id] * 
                                            std::get<Num_Conv_Layers - 1>(conv_activation_results)[activation_id];
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
                        KH * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size + 
                        KW;

                    std::get<Num_Conv_Layers - 1>(conv_layer_deriv).kernels[kernel_deriv_id] = sum / (
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size *
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size);
                });
            });
        });
    });

    compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::output_channels>(
    [&]<size_t OC>()
    {   
        typename std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::type sum = 0;

        compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_y>(
        [&]<size_t H>()
        {
            compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z>(
            [&]<size_t W>()
            {   
                constexpr size_t delta_id = 
                    OC * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_y *
                    std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z +
                    H * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z +
                    W;
                
                sum += std::get<Num_Conv_Layers>(layer_delta)[delta_id];
            });
        });

        std::get<Num_Conv_Layers - 1>(conv_layer_deriv).biases[OC] = sum;
    });

    compile_range<Num_Conv_Layers, 1>(
    [&]<size_t I>()
    {
        std::get<Num_Conv_Layers - I>(conv_layers).apply_backwards(
            std::get<Num_Conv_Layers - I + 1>(layer_delta), 
            std::get<Num_Conv_Layers - I>(layer_delta));

        CNN::compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::output_channels>(
        [&]<size_t OC>()
        {
            compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y>(
            [&]<size_t H>()
            {
                compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z>(
                [&]<size_t W>()
                {   
                    constexpr size_t delta_id = 
                        OC * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y *
                        std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z + 
                        H * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z + W;

                    std::get<Num_Conv_Layers - I>(layer_delta)[delta_id] *= 
                        std::get<Num_Conv_Layers - I - 1>(conv_layers).activation_func.derivative(
                        std::get<Num_Conv_Layers - I>(conv_weighted_inputs)[delta_id]);
                });
            });
        });

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

                        compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y>(
                        [&]<size_t H>()
                        {
                            compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z>(
                            [&]<size_t W>()
                            {   
                                constexpr size_t delta_id = 
                                    OC * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y *
                                    std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z +
                                    H * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z +
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
                                            activation_height < std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::size_y &&
                                            activation_width < std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::size_z)
                                        {
                                            constexpr size_t activation_id = 
                                                IC * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::size_y *
                                                std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::size_z +
                                                activation_height * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::size_z +
                                                activation_width;

                                            sum += std::get<Num_Conv_Layers - I>(layer_delta)[delta_id] * 
                                                std::get<Num_Conv_Layers - I - 1>(conv_activation_results)[activation_id];
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
                            KH * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size + 
                            KW;

                        std::get<Num_Conv_Layers - I - 1>(conv_layer_deriv).kernels[kernel_deriv_id] = sum /(
                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size *
                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size);
                    });
                });
            });
        });

        compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::output_channels>(
        [&]<size_t OC>()
        {   
            typename std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::type sum = 0;

            compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y>(
            [&]<size_t H>()
            {
                compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z>(
                [&]<size_t W>()
                {   
                    constexpr size_t delta_id = 
                        OC * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y *
                        std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z + 
                        H * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z +
                        W;
                    
                    sum += std::get<Num_Conv_Layers - I>(layer_delta)[delta_id];
                });
            });

            std::get<Num_Conv_Layers - I - 1>(conv_layer_deriv).biases[OC] = sum;
        });
    });
}

template<
    size_t C, size_t H, size_t W,
    size_t NUM_CONV_LAYERS, 
    size_t NUM_NEURAL_LAYERS, 
    typename... Layer_Args>
requires(
    NUM_CONV_LAYERS + NUM_NEURAL_LAYERS == sizeof...(Layer_Args))
inline auto network(Layer_Args&&... args)
{
    auto [conv_layers, neural_layers] = split_tuple<NUM_CONV_LAYERS - 1>(
        std::tuple<Layer_Args...>{std::forward<Layer_Args>(args)...});

    using Conv_Layers_t = decltype(conv_layers);
    using Neural_Layers_t = decltype(neural_layers);

    static_assert(output_eq_input_channels<Conv_Layers_t>(), "Output and input channel size does not match.");
    static_assert(output_eq_input_neurons<Neural_Layers_t>(), "Output and input neuron size does not match.");

    auto conv_features = features_from_layer<C, H, W, Conv_Layers_t>();

    using Conv_Features_Tuple_t = decltype(conv_features);
    using Last_Conv_Features_t = std::tuple_element_t<NUM_CONV_LAYERS, Conv_Features_Tuple_t>;
    using First_Neural_Layer_t = std::tuple_element_t<0, Neural_Layers_t>;

    if constexpr (Last_Conv_Features_t::size != First_Neural_Layer_t::input_neurons)
    {
        auto extended_neural_layers = add_tuple_begin(
            std::move(neural_layers), Neural_Layer<Last_Conv_Features_t::size, 
            First_Neural_Layer_t::input_neurons, decltype(First_Neural_Layer_t::activation_func)>{});

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