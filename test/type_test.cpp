#include <gtest/gtest.h>

#include "core.hpp"
#include "network.hpp"

TEST(NeuralLayerTest, ApplyComputesCorrectly) {

    CNN::Neural_Layer<2, 1, CNN::ReLU<float>> layer;

    layer.weights[0] = 2;
    layer.weights[1] = 3;
    layer.biases[0] = 1;

    CNN::HeapTensor1D<2, float> input(std::vector<float>{1, 2});
    CNN::HeapTensor1D<1, float> output_conv;
    CNN::HeapTensor1D<1, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    EXPECT_FLOAT_EQ(output_conv[0], 2*1 + 3*2 + 1);
}

TEST(ConvolutionLayerTest, ZeroApply) {

    constexpr size_t CIN = 1, COUT = 1, K = 3, S = 1, P = 1;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(false);

    CNN::HeapTensor3D<CIN, 5, 5, float> input(1);
    CNN::HeapTensor3D<COUT, 5, 5, float> output_conv;
    CNN::HeapTensor3D<COUT, 5, 5, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    for (float val : output_conv) {
        EXPECT_FLOAT_EQ(val, static_cast<float>(0));
    }
}

TEST(ConvolutionLayerTest, IdentApply) {

    constexpr size_t CIN = 1, COUT = 1, K = 3, S = 1, P = 1;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(false);

    layer.kernels[K / 2 * K + K / 2] = 1;

    CNN::HeapTensor3D<CIN, 5, 5, float> input(2);
    CNN::HeapTensor3D<COUT, 5, 5, float> output_conv;
    CNN::HeapTensor3D<COUT, 5, 5, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    for (size_t i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(input[i], output_conv[i]);
    }
}

TEST(ConvolutionLayerTest, PoolingApply) {

    constexpr size_t CIN = 1, COUT = 1, K = 3, S = 1, P = 2;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(false);

    layer.kernels[K / 2 * K + K / 2] = 1;

    CNN::HeapTensor3D<CIN, 5, 5, float> input(2);
    CNN::HeapTensor3D<COUT, 3, 3, float> output_conv;
    CNN::HeapTensor3D<COUT, 3, 3, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    for (size_t i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(2), output_conv[i]);
    }
    EXPECT_FLOAT_EQ(static_cast<float>(1), output_conv[2]);

    for (size_t i = 3; i < 5; ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(2), output_conv[i]);
    }
    EXPECT_FLOAT_EQ(static_cast<float>(1), output_conv[5]);

    for (size_t i = 6; i < 8; ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(1), output_conv[i]);
    }
    EXPECT_FLOAT_EQ(static_cast<float>(0.5), output_conv[8]);
}

TEST(ConvolutionLayerTest, StridingApply) {

    constexpr size_t CIN = 1, COUT = 1, K = 3, S = 3, P = 1;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(false);

    layer.kernels[K / 2 * K + K / 2] = 1;

    CNN::HeapTensor3D<CIN, 5, 5, float> input(0);
    CNN::HeapTensor3D<COUT, 2, 2, float> output_conv;
    CNN::HeapTensor3D<COUT, 2, 2, float> output_activ;

    input[0] = static_cast<float>(2);
    input[3] = static_cast<float>(1);

    input[15] = static_cast<float>(2);
    input[18] = static_cast<float>(1);

    layer.apply(input, output_conv, output_activ);

    EXPECT_FLOAT_EQ(static_cast<float>(2), output_conv[0]);
    EXPECT_FLOAT_EQ(static_cast<float>(1), output_conv[1]);
    EXPECT_FLOAT_EQ(static_cast<float>(2), output_conv[2]);
    EXPECT_FLOAT_EQ(static_cast<float>(1), output_conv[3]);
}

TEST(ConvolutionLayerTest, OnesApply) {

    constexpr size_t CIN = 1, COUT = 1, K = 3, S = 1, P = 1;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(false);

    for(auto & k: layer.kernels) k = 1;

    CNN::HeapTensor3D<CIN, 5, 5, float> input(2);
    CNN::HeapTensor3D<COUT, 5, 5, float> output_conv;
    CNN::HeapTensor3D<COUT, 5, 5, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    for(size_t i = 0; i < 5; ++i){
        for(size_t j = 0; j < 5; ++j){
            if (i == 0 || i == 4){
                if(j == 0 || j == 4){
                    EXPECT_FLOAT_EQ(output_conv[i * 5 + j], 8);
                }
                else{
                    EXPECT_FLOAT_EQ(output_conv[i * 5 + j], 12);
                }
            }
            else if (j == 0 || j == 4){
                EXPECT_FLOAT_EQ(output_conv[i * 5 + j], 12);
            }
            else{
                EXPECT_FLOAT_EQ(output_conv[i * 5 + j], 18);
            }
        }
    } 
}

TEST(ConvolutionLayerTest, InputChannelAddition) {

    constexpr size_t CIN = 3, COUT = 1, K = 3, S = 1, P = 1;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(false);

    layer.kernels[K / 2 * K + K / 2] = 1;
    layer.kernels[K * K + K / 2 * K + K / 2] = 1;
    layer.kernels[2 * K * K + K / 2 * K + K / 2] = 1;

    CNN::HeapTensor3D<3, 5, 5, float> input(1);

    CNN::HeapTensor3D<COUT, 5, 5, float> output_conv;
    CNN::HeapTensor3D<COUT, 5, 5, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    for (size_t i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(output_conv[i], static_cast<float>(3));
    }
}

TEST(NetworkTest, InputForwarding) {

    constexpr size_t K = 3;

    CNN::Convolution_Layer<3, 1, 3, 1, 1, CNN::ReLU<float>> layer_1(false);
    layer_1.kernels[K / 2 * K + K / 2] = 1;
    layer_1.kernels[K * K + K / 2 * K + K / 2] = 1;
    layer_1.kernels[2 * K * K + K / 2 * K + K / 2] = 1;

    CNN::Convolution_Layer<1, 1, 3, 1, 1, CNN::ReLU<float>> layer_2(false);
    layer_2.kernels[K / 2 * K + K / 2] = 3;
    layer_2.biases[0] = 1;

    CNN::Neural_Layer<25, 25, CNN::ReLU<float>> layer_3(false);
    for(size_t i= 0; i < 25; ++i) layer_3.weights[i * 25 + i] = 2;

    auto network = CNN::Network::network<3, 5, 5, 2, 1>(layer_1, layer_2, layer_3);

    typename decltype(network)::conv_feature_tuple conv_weighted_inputs;
    typename decltype(network)::conv_feature_tuple conv_activation_outputs;

    typename decltype(network)::neural_feature_tuple neural_weighted_inputs;
    typename decltype(network)::neural_feature_tuple neural_activation_outputs;

    std::vector<float> input(75, 1);

    network.foward_propagate(input, 
        conv_weighted_inputs, 
        conv_activation_outputs, 
        neural_weighted_inputs, 
        neural_activation_outputs);

    for (size_t i = 0; i < 75; ++i) {
        EXPECT_FLOAT_EQ(std::get<0>(conv_weighted_inputs)[i], static_cast<float>(0));
        EXPECT_FLOAT_EQ(std::get<0>(conv_activation_outputs)[i], static_cast<float>(1));
    }

    for (size_t i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(std::get<1>(conv_weighted_inputs)[i], static_cast<float>(3));
        EXPECT_FLOAT_EQ(std::get<1>(conv_activation_outputs)[i], static_cast<float>(3));
    }

    for (size_t i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(std::get<2>(conv_weighted_inputs)[i], static_cast<float>(10));
        EXPECT_FLOAT_EQ(std::get<2>(conv_activation_outputs)[i], static_cast<float>(10));
    }

    for (size_t i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(std::get<0>(neural_weighted_inputs)[i], static_cast<float>(0));
        EXPECT_FLOAT_EQ(std::get<0>(neural_activation_outputs)[i], static_cast<float>(10));
    }

    for (size_t i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(std::get<1>(neural_weighted_inputs)[i], static_cast<float>(20));
        EXPECT_FLOAT_EQ(std::get<1>(neural_activation_outputs)[i], static_cast<float>(20));
    }
    
}

TEST(NetworkBuildTest, Correct_LayerShapeDeduction)
{
    constexpr size_t CHANNELS = 3;
    constexpr size_t IMG_HEIGHT = 1;
    constexpr size_t IMG_WIDTH = 25;

    CNN::Convolution_Layer<3, 3, 3, 4, 2, CNN::ReLU<float>> con_layer_1;
    CNN::Convolution_Layer<3, 9, 3, 2, 2, CNN::ReLU<float>> con_layer_2;

    CNN::Neural_Layer<64, 64, CNN::ReLU<float>> neural_layer_1;
    CNN::Neural_Layer<64, 200, CNN::ReLU<float>> neural_layer_2;

    auto network = CNN::Network::network<CHANNELS, IMG_HEIGHT, IMG_WIDTH, 2, 2>(
        con_layer_1, con_layer_2, neural_layer_1, neural_layer_2);

    using Conv_Layers_Tuple = decltype(network.conv_layers);
    using Neural_Layers_Tuple = decltype(network.neural_layers);

    using Conv_Features_Tuple = decltype(
        CNN::features_from_layer<CHANNELS, IMG_HEIGHT, IMG_WIDTH, Conv_Layers_Tuple>());
    using Neural_Features_Tuple = decltype(
        CNN::features_from_layer<Neural_Layers_Tuple>());
    
    static_assert(std::tuple_size_v<Conv_Layers_Tuple> == 2, 
        "Number of Conv layers in Network does not match input.");

    static_assert(std::tuple_size_v<Neural_Layers_Tuple> == 3, 
        "Number of Neural layers in Network does not match input.");

    static_assert(std::tuple_element_t<0, Conv_Features_Tuple>::size_y == 1, 
        "Height of first Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<0, Conv_Features_Tuple>::size_z == 25, 
        "Width of first Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<0, Conv_Features_Tuple>::size_x == 3, 
        "Channels of first Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<1, Conv_Features_Tuple>::size_y == 1, 
        "Height of second Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<1, Conv_Features_Tuple>::size_z == 4, 
        "Width of second Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<1, Conv_Features_Tuple>::size_x == 3, 
        "Channels of second Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<2, Conv_Features_Tuple>::size_y == 1, 
        "Height of third Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<2, Conv_Features_Tuple>::size_z == 1, 
        "Width of third Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<2, Conv_Features_Tuple>::size_x == 9, 
        "Channels of third Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<0, Neural_Features_Tuple>::size == 9, 
        "Size of first Neural layer in Network does not match input.");
    
    static_assert(std::tuple_element_t<1, Neural_Features_Tuple>::size == 64, 
        "Size of second Neural layer in Network does not match input.");
    
    static_assert(std::tuple_element_t<2, Neural_Features_Tuple>::size == 64, 
        "Size of third Neural layer in Network does not match input.");

    static_assert(std::tuple_element_t<3, Neural_Features_Tuple>::size == 200, 
        "Size of fourth Neural layer in Network does not match input.");
}
