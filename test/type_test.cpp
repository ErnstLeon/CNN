#include <gtest/gtest.h>

#include "core.hpp"
#include "network.hpp"

TEST(NeuralLayerTest, ApplyComputesCorrectly) {

    CNN::Neural_Layer<2, 1, CNN::ReLU<float>> layer;

    layer.weights = {2, 3};
    layer.biases = {1};

    CNN::Neural_FeatureMap<2, float> input(std::vector<float>{1, 2});
    CNN::Neural_FeatureMap<1, float> output_conv;
    CNN::Neural_FeatureMap<1, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    EXPECT_FLOAT_EQ(output_conv.features[0], 2*1 + 3*2 + 1);
}

TEST(ConvolutionLayerTest, ZeroApply) {

    constexpr size_t CIN = 1, COUT = 1, K = 3, S = 1, P = 1;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(0);

    CNN::Convolution_FeatureMap<CIN, 5, 5, float> input(1);
    CNN::Convolution_FeatureMap<COUT, 5, 5, float> output_conv;
    CNN::Convolution_FeatureMap<COUT, 5, 5, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    for (float val : output_conv.features) {
        EXPECT_FLOAT_EQ(val, static_cast<float>(0));
    }
}

TEST(ConvolutionLayerTest, IdentApply) {

    constexpr size_t CIN = 1, COUT = 1, K = 3, S = 1, P = 1;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(0);

    layer.kernels[K / 2 * K + K / 2] = 1;

    CNN::Convolution_FeatureMap<CIN, 5, 5, float> input(2);
    CNN::Convolution_FeatureMap<COUT, 5, 5, float> output_conv;
    CNN::Convolution_FeatureMap<COUT, 5, 5, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    for (size_t i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(input.features[i], output_conv.features[i]);
    }
}

TEST(ConvolutionLayerTest, PoolingApply) {

    constexpr size_t CIN = 1, COUT = 1, K = 3, S = 1, P = 2;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(0);

    layer.kernels[K / 2 * K + K / 2] = 1;

    CNN::Convolution_FeatureMap<CIN, 5, 5, float> input(2);
    CNN::Convolution_FeatureMap<COUT, 3, 3, float> output_conv;
    CNN::Convolution_FeatureMap<COUT, 3, 3, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    for (size_t i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(2), output_conv.features[i]);
    }
    EXPECT_FLOAT_EQ(static_cast<float>(1), output_conv.features[2]);

    for (size_t i = 3; i < 5; ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(2), output_conv.features[i]);
    }
    EXPECT_FLOAT_EQ(static_cast<float>(1), output_conv.features[5]);

    for (size_t i = 6; i < 8; ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(1), output_conv.features[i]);
    }
    EXPECT_FLOAT_EQ(static_cast<float>(0.5), output_conv.features[8]);
}

TEST(ConvolutionLayerTest, StridingApply) {

    constexpr size_t CIN = 1, COUT = 1, K = 3, S = 3, P = 1;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(0);

    layer.kernels[K / 2 * K + K / 2] = 1;

    CNN::Convolution_FeatureMap<CIN, 5, 5, float> input(0);
    CNN::Convolution_FeatureMap<COUT, 2, 2, float> output_conv;
    CNN::Convolution_FeatureMap<COUT, 2, 2, float> output_activ;

    input.features[0] = static_cast<float>(2);
    input.features[3] = static_cast<float>(1);

    input.features[15] = static_cast<float>(2);
    input.features[18] = static_cast<float>(1);

    layer.apply(input, output_conv, output_activ);

    EXPECT_FLOAT_EQ(static_cast<float>(2), output_conv.features[0]);
    EXPECT_FLOAT_EQ(static_cast<float>(1), output_conv.features[1]);
    EXPECT_FLOAT_EQ(static_cast<float>(2), output_conv.features[2]);
    EXPECT_FLOAT_EQ(static_cast<float>(1), output_conv.features[3]);
}

TEST(ConvolutionLayerTest, OnesApply) {

    constexpr size_t CIN = 1, COUT = 1, K = 3, S = 1, P = 1;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(0);

    for(auto & k: layer.kernels) k = 1;

    CNN::Convolution_FeatureMap<CIN, 5, 5, float> input(2);
    CNN::Convolution_FeatureMap<COUT, 5, 5, float> output_conv;
    CNN::Convolution_FeatureMap<COUT, 5, 5, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    for(size_t i = 0; i < 5; ++i){
        for(size_t j = 0; j < 5; ++j){
            if (i == 0 || i == 4){
                if(j == 0 || j == 4){
                    EXPECT_FLOAT_EQ(output_conv.features[i * 5 + j], 8);
                }
                else{
                    EXPECT_FLOAT_EQ(output_conv.features[i * 5 + j], 12);
                }
            }
            else if (j == 0 || j == 4){
                EXPECT_FLOAT_EQ(output_conv.features[i * 5 + j], 12);
            }
            else{
                EXPECT_FLOAT_EQ(output_conv.features[i * 5 + j], 18);
            }
        }
    } 
}

TEST(ConvolutionLayerTest, InputChannelAddition) {

    constexpr size_t CIN = 3, COUT = 1, K = 3, S = 1, P = 1;
    CNN::Convolution_Layer<CIN, COUT, K, S, P, CNN::ReLU<float>> layer(0);

    layer.kernels[K / 2 * K + K / 2] = 1;
    layer.kernels[K * K + K / 2 * K + K / 2] = 1;
    layer.kernels[2 * K * K + K / 2 * K + K / 2] = 1;

    CNN::Convolution_FeatureMap<3, 5, 5, float> input(1);

    CNN::Convolution_FeatureMap<COUT, 5, 5, float> output_conv;
    CNN::Convolution_FeatureMap<COUT, 5, 5, float> output_activ;

    layer.apply(input, output_conv, output_activ);

    for (size_t i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(output_conv.features[i], static_cast<float>(3));
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
        CNN::featureMaps_from_layer<CHANNELS, IMG_HEIGHT, IMG_WIDTH>(network.conv_layers));
    using Neural_Features_Tuple = decltype(
        CNN::featureMaps_from_layer(network.neural_layers));
    
    static_assert(std::tuple_size_v<Conv_Layers_Tuple> == 2, 
        "Number of Conv layers in Network does not match input.");

    static_assert(std::tuple_size_v<Neural_Layers_Tuple> == 3, 
        "Number of Neural layers in Network does not match input.");

    static_assert(std::tuple_element_t<0, Conv_Features_Tuple>::height == 1, 
        "Height of first Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<0, Conv_Features_Tuple>::width == 25, 
        "Width of first Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<0, Conv_Features_Tuple>::channels == 3, 
        "Channels of first Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<1, Conv_Features_Tuple>::height == 1, 
        "Height of second Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<1, Conv_Features_Tuple>::width == 4, 
        "Width of second Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<1, Conv_Features_Tuple>::channels == 3, 
        "Channels of second Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<2, Conv_Features_Tuple>::height == 1, 
        "Height of third Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<2, Conv_Features_Tuple>::width == 1, 
        "Width of third Conv layer in Network does not match input.");

    static_assert(std::tuple_element_t<2, Conv_Features_Tuple>::channels == 9, 
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
