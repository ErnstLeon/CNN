#ifndef NETWORK_ASSESS_HPP
#define NETWORK_ASSESS_HPP

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
std::pair<
    typename Network<Conv_Layer_Tuple, Neural_Layer_Tuple, Conv_Feature_Tuple, Neural_Feature_Tuple>::output_type, 
    typename Network<Conv_Layer_Tuple, Neural_Layer_Tuple, Conv_Feature_Tuple, Neural_Feature_Tuple>::output_type> 
Network<
    Conv_Layer_Tuple, Neural_Layer_Tuple, 
    Conv_Feature_Tuple, Neural_Feature_Tuple>::assess(
    const std::vector<std::pair<HeapTensor3D<input_channels, input_height, input_width, input_type>, 
    HeapTensor1D<output_neurons, output_type>>> & dataset)
{
    output_type avg_loss = 0;
    output_type avg_error = 0;

    auto dataset_size = dataset.size();

    #pragma omp parallel for reduction(+ : avg_loss, avg_error)
    for(size_t i = 0; i < dataset_size; ++i){

        const auto & sample = dataset[i];
        const auto model_output = evaluate(std::get<0>(sample));

        auto max_model = std::distance(
            model_output.begin(), 
            std::max_element(model_output.begin(), model_output.end())
        );

        auto max_true = std::distance(
            std::get<1>(sample).begin(), 
            std::max_element(std::get<1>(sample).begin(), std::get<1>(sample).end())
        );

        output_type loss = cross_entropy_loss(model_output, std::get<1>(sample));
        output_type error = (max_model != max_true) ? 1 : 0;

        avg_loss += loss;
        avg_error += error;
    }
    
    return {
        avg_loss / static_cast<output_type>(dataset_size), 
        avg_error / static_cast<output_type>(dataset_size)};
}

}

#endif // NETWORK_ASSESS_HPP