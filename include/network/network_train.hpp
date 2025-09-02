#ifndef NETWORK_TRAIN_HPP
#define NETWORK_TRAIN_HPP

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "../core.hpp"
#include "network_base.hpp"

namespace CNN::Network
{

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

    auto dataset = dataset_orig;

    std::random_device rd;
    std::mt19937 gen(rd());
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

            #pragma omp parallel shared(conv_deriv, neural_deriv)
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

            compile_range<num_conv_layers>(
            [&]<size_t I>(){
                std::get<I>(conv_kernels_optimizers).update(std::get<I>(conv_layers).kernels, std::get<I>(conv_deriv).kernels);
                std::get<I>(conv_biases_optimizers).update(std::get<I>(conv_layers).biases, std::get<I>(conv_deriv).biases);
            });

            compile_range<num_neural_layers>(
            [&]<size_t I>(){
                std::get<I>(neural_weights_optimizers).update(std::get<I>(neural_layers).weights, std::get<I>(neural_deriv).weights);
                std::get<I>(neural_biases_optimizers).update(std::get<I>(neural_layers).biases, std::get<I>(neural_deriv).biases);
            });
        }

        auto [loss, error] = assess(dataset_orig);

        std::cout << "done epoch: " << epoch << "/" << num_epochs << std::endl;
        std::cout << "average loss: " << loss << " average error: " << error << std::endl;
    }
}
}

#endif // NETWORK_TRAIN_HPP