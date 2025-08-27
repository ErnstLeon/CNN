#include <gtest/gtest.h>

#include "core.hpp"
#include "network.hpp"

TEST(LearningTest, CrossEntropyLossTest) {
    std::vector<double> output = {0.7, 0.2, 0.1}; 
    std::vector<double> target = {1.0, 0.0, 0.0};

    double expected = - std::log(0.7 + 1e-12);
    double actual = CNN::cross_entropy_loss(output, target);

    EXPECT_NEAR(actual, expected, 1e-6);
}

TEST(LearningTest, ReLUActivationTest) {
    CNN::ReLU<double> relu;

    EXPECT_EQ(relu(3.14), 3.14);
    EXPECT_EQ(relu(-2.0), -0.02);
    EXPECT_EQ(relu(0.0), 0.0);

    EXPECT_EQ(relu.derivative(5.0), 1.0);
    EXPECT_EQ(relu.derivative(-1.0), 0.01);
    EXPECT_EQ(relu.derivative(0.0), 1.0);
}

TEST(LearningTest, SigmoidActivationTest) {
    CNN::Sigmoid<double> sigmoid;

    EXPECT_NEAR(sigmoid(0.0), 0.5, 1e-6);
    EXPECT_NEAR(sigmoid(15.0), 1.0, 1e-4);
    EXPECT_NEAR(sigmoid(-15.0), 0.0, 1e-4);

    double s = sigmoid(0.0);
    EXPECT_NEAR(sigmoid.derivative(0.0), s * (1 - s), 1e-6);
}

TEST(LearningTest, SoftmaxFunctionTest) {
    CNN::Softmax<double> softmax;

    CNN::HeapTensor1D<3, double> input(std::vector<double>{1.0, 2.0, 3.0});
    CNN::HeapTensor1D<3, double> output;
    output = softmax.apply(input);

    double sum = 0.0;
    for (double val : output) {
        EXPECT_GE(val, 0.0);
        EXPECT_LE(val, 1.0);
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);
    EXPECT_GT(output[2], output[1]);
    EXPECT_GT(output[1], output[0]);
}

TEST(LearningTest, SoftmaxInplaceFunctionTest) {
    CNN::Softmax<double> softmax;

    CNN::HeapTensor1D<3, double> input(std::vector<double>{1.0, 2.0, 3.0});
    softmax.apply_inplace(input);

    double sum = 0.0;
    for (double val : input) {
        EXPECT_GE(val, 0.0);
        EXPECT_LE(val, 1.0);
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);
    EXPECT_GT(input[2], input[1]);
    EXPECT_GT(input[1], input[0]);
}

TEST(LearningTest, GDOptimizerConverges) {

    CNN::HeapTensor1D<1, double> value(1);
    CNN::HeapTensor1D<1, double> gradient(1);
    
    CNN::Optimizer::Gradient_Descent_<CNN::HeapTensor1D<1, double>> optimizer{0.1};

    // function (x - 3)^2, derivate: 2(x - 3)
    value[0] = static_cast<double>(0.0);
    for (int step = 0; step < 1000; ++step) {
        gradient[0] = 2 * (value[0] - 3);
        optimizer.update(value, gradient);
    }

    EXPECT_NEAR(value[0], 3.0, 1e-4);
}

TEST(LearningTest, AdamOptimizerConverges) {
    
    CNN::HeapTensor1D<1, double> value(1);
    CNN::HeapTensor1D<1, double> gradient(1);
    
    CNN::Optimizer::Adam_Optimizer_<CNN::HeapTensor1D<1, double>> optimizer{0.1};

    // function (x - 3)^2, derivate: 2(x - 3)
    value[0] = static_cast<double>(0.0);
    for (int step = 0; step < 1000; ++step) {
        gradient[0] = 2 * (value[0] - 3);
        optimizer.update(value, gradient);
    }

    EXPECT_NEAR(value[0], 3.0, 1e-4);
}

TEST(GradientTest, SingelNeuralLayerGradientCorrect) {
    
    CNN::Neural_Layer<16, 2, CNN::Softmax<float>> neural_layer;

    CNN::HeapTensor1D<16, float> input{std::vector<float>{
        0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79}};

    CNN::HeapTensor1D<2, float> output{std::vector<float>{0.43,0.57}};

    CNN::HeapTensor1D<2, float> neural_weighted_output{};
    CNN::HeapTensor1D<2, float> neural_activ_output{};

    neural_layer.apply(input, neural_weighted_output, neural_activ_output);

    auto neural_biases_deriv = neural_layer.biases;
    auto neural_weights_deriv = neural_layer.weights;

    neural_biases_deriv = neural_activ_output;

    for(size_t i = 0; i < 2; ++i){
        neural_biases_deriv[i] -= output[i];
    }
    
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 16; ++j){
            neural_weights_deriv[i * 16 + j] =
                neural_biases_deriv[i] * input[j];
        }
    }
 
    CNN::Neural_Layer<16, 2, CNN::Softmax<float>> neural_layer_eps;

    neural_layer_eps.weights = neural_layer.weights;
    neural_layer_eps.biases = neural_layer.biases;

    CNN::HeapTensor1D<2, float> neural_weighted_output_eps{};
    CNN::HeapTensor1D<2, float> neural_activ_output_eps{};

    // check biases derivate
    for(size_t i = 0; i < 2; ++i){
        neural_layer_eps.biases[i] = neural_layer.biases[i] + 1e-3;
        neural_layer_eps.apply(input, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_p = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        neural_layer_eps.biases[i] = neural_layer.biases[i] - 1e-3;
        neural_layer_eps.apply(input, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_m = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = neural_biases_deriv[i];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));

        EXPECT_TRUE(rel_diff < 0.1);
    }

    // check weights derivate
    for(size_t i = 0; i < 32; ++i){
        neural_layer_eps.weights[i] = neural_layer.weights[i] + 1e-3;
        neural_layer_eps.apply(input, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_p = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        neural_layer_eps.weights[i] = neural_layer.weights[i] - 1e-3;
        neural_layer_eps.apply(input, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_m = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = neural_weights_deriv[i];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);
    }
}

TEST(GradientTest, DoubleNeuralLayerGradientCorrect) {
    
    CNN::Neural_Layer<16, 8, CNN::ReLU<float>> neural_layer_1;
    CNN::Neural_Layer<8, 2, CNN::Softmax<float>> neural_layer_2;

    CNN::HeapTensor1D<16, float> input{std::vector<float>{
        0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79}};

    CNN::HeapTensor1D<2, float> output{std::vector<float>{0.43,0.57}};

    CNN::HeapTensor1D<8, float> neural_weighted_output_1{};
    CNN::HeapTensor1D<8, float> neural_activ_output_1{};

    CNN::HeapTensor1D<2, float> neural_weighted_output_2{};
    CNN::HeapTensor1D<2, float> neural_activ_output_2{};

    neural_layer_1.apply(input, neural_weighted_output_1, neural_activ_output_1);
    neural_layer_2.apply(neural_activ_output_1, neural_weighted_output_2, neural_activ_output_2);

    // backpropagate
    auto neural_biases_deriv_2 = neural_layer_2.biases;
    auto neural_weights_deriv_2 = neural_layer_2.weights;

    neural_biases_deriv_2 = neural_activ_output_2;

    for(size_t i = 0; i < 2; ++i){
        neural_biases_deriv_2[i] -= output[i];
    }
    
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 8; ++j){
            neural_weights_deriv_2[i * 8 + j] =
                neural_biases_deriv_2[i] * neural_activ_output_1[j];
        }
    }

    auto neural_biases_deriv_1 = neural_layer_1.biases;
    auto neural_weights_deriv_1 = neural_layer_1.weights;

    neural_layer_2.apply_backwards(neural_biases_deriv_2, neural_biases_deriv_1);

    for(size_t i = 0; i < 8; ++i){
        neural_biases_deriv_1[i] *= neural_layer_1.activation_func.derivative(
            neural_weighted_output_1[i]);
    }

    for(size_t i = 0; i < 8; ++i){
        for(size_t j = 0; j < 16; ++j){
            neural_weights_deriv_1[i * 16 + j] =
                neural_biases_deriv_1[i] * input[j];
        }
    }
    
    // compute deriv numeric with epsilon
    CNN::Neural_Layer<16, 8, CNN::ReLU<float>> neural_layer_1_eps;

    neural_layer_1_eps.weights = neural_layer_1.weights;
    neural_layer_1_eps.biases = neural_layer_1.biases;

    CNN::HeapTensor1D<8, float> neural_weighted_output_eps_1{};
    CNN::HeapTensor1D<8, float> neural_activ_output_eps_1{};

    CNN::HeapTensor1D<2, float> neural_weighted_output_eps_2{};
    CNN::HeapTensor1D<2, float> neural_activ_output_eps_2{};

    // check biases derivate
    for(size_t i = 0; i < 8; ++i){
        neural_layer_1_eps.biases[i] = neural_layer_1.biases[i] + 1e-3;
        
        neural_layer_1_eps.apply(input, neural_weighted_output_eps_1, neural_activ_output_eps_1);
        neural_layer_2.apply(neural_activ_output_eps_1, neural_weighted_output_eps_2, neural_activ_output_eps_2);

        auto l_p = CNN::cross_entropy_loss(neural_activ_output_eps_2, output);

        neural_layer_1_eps.biases[i] = neural_layer_1.biases[i] - 1e-3;

        neural_layer_1_eps.apply(input, neural_weighted_output_eps_1, neural_activ_output_eps_1);
        neural_layer_2.apply(neural_activ_output_eps_1, neural_weighted_output_eps_2, neural_activ_output_eps_2);

        auto l_m = CNN::cross_entropy_loss(neural_activ_output_eps_2, output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = neural_biases_deriv_1[i];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

    }

    // check weights derivate
    for(size_t i = 0; i < 16 * 8; ++i){

        neural_layer_1_eps.weights[i] = neural_layer_1.weights[i] + 1e-3;
        
        neural_layer_1_eps.apply(input, neural_weighted_output_eps_1, neural_activ_output_eps_1);
        neural_layer_2.apply(neural_activ_output_eps_1, neural_weighted_output_eps_2, neural_activ_output_eps_2);

        auto l_p = CNN::cross_entropy_loss(neural_activ_output_eps_2, output);

        neural_layer_1_eps.weights[i] = neural_layer_1.weights[i] - 1e-3;

        neural_layer_1_eps.apply(input, neural_weighted_output_eps_1, neural_activ_output_eps_1);
        neural_layer_2.apply(neural_activ_output_eps_1, neural_weighted_output_eps_2, neural_activ_output_eps_2);

        auto l_m = CNN::cross_entropy_loss(neural_activ_output_eps_2, output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = neural_weights_deriv_1[i];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

    }
}


TEST(GradientTest, ConvolutionLayerGradientCorrect) {
    
    CNN::Convolution_Layer<2, 1, 3, 1, 1, CNN::ReLU<float>> con_layer;
    CNN::Neural_Layer<16, 2, CNN::Softmax<float>> neural_layer;

    CNN::HeapTensor3D<2, 4, 4, float> input{std::vector<float>{
        0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,
        0.41,0.23,0.25,0.79,0.1,0.234,0.35,0.79,0.21,0.23,0.25,0.79,0.1,0.23,0.5,0.79}};

    CNN::HeapTensor1D<2, float> output{std::vector<float>{0.43,0.57}};

    CNN::HeapTensor3D<1, 4, 4, float> conv_weighted_output{};
    CNN::HeapTensor3D<1, 4, 4, float> conv_activ_output{};

    CNN::HeapTensor1D<2, float> neural_weighted_output{};
    CNN::HeapTensor1D<2, float> neural_activ_output{};

    con_layer.apply(input, conv_weighted_output, conv_activ_output);
    neural_layer.apply(conv_activ_output, neural_weighted_output, neural_activ_output);

    auto neural_biases_deriv = neural_layer.biases;
    auto neural_weights_deriv = neural_layer.weights;

    auto con_biases_deriv = con_layer.biases;
    auto con_weights_deriv = con_layer.kernels;

    neural_biases_deriv = neural_activ_output;

    for(size_t i = 0; i < 2; ++i){
        neural_biases_deriv[i] -= output[i];
    }
    
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 16; ++j){
            neural_weights_deriv[i * 16 + j] =
                neural_biases_deriv[i] * conv_activ_output[j];
        }
    }

    // construct delta L for the convolution backward propagation
    CNN::HeapTensor1D<16, float> layer_delta{};
    neural_layer.apply_backwards(neural_biases_deriv, layer_delta);

    for(size_t j = 0; j < 16; ++j){
        layer_delta[j] *= con_layer.activation_func.derivative(conv_weighted_output[j]);
    }

    // based on delta L, compule the kernel and bias derivatives for L
    for(size_t OC = 0; OC < 1; ++OC)
    {
        for(size_t IC = 0; IC < 2; ++IC)
        {
            for(size_t KH = 0; KH < 3; ++KH)
            {
                for(size_t KW = 0; KW < 3; ++KW)
                {
                    float sum = 0;

                    for(size_t H = 0; H < 4; ++H)
                    {
                        for(size_t W = 0; W < 4; ++W)
                        { 
                            size_t delta_id = 
                                OC * 4 * 4 + H * 4 + W;

                            size_t activation_height = H - 1 + KH;

                            size_t activation_width = W - 1 + KW;

                            if (activation_height >= 0 && activation_width >= 0 &&
                                activation_height < 4 &&
                                activation_width < 4)
                            {
                                size_t activation_id = 
                                    IC * 4 * 4 +
                                    activation_height * 4 +
                                    activation_width;

                                sum += layer_delta[delta_id] * input[activation_id];
                            }
                        }
                    }
                    
                    size_t kernel_deriv_id = 
                        OC * 2 * 3 * 3 +
                        IC * 3 * 3 + KH * 3 + 
                        KW;

                    con_weights_deriv[kernel_deriv_id] = sum;
                }
            }
        }
    }

    CNN::compile_range<1>(
    [&]<size_t OC>()
    {   
        float sum = 0;

        CNN::compile_range<4>(
        [&]<size_t H>()
        {
            CNN::compile_range<4>(
            [&]<size_t W>()
            {   
                constexpr size_t delta_id = 
                    OC * 4 * 4 + H * 4 + W;
                
                sum += layer_delta[delta_id];
            });
        });

        con_biases_deriv[OC] = sum;
    });

    CNN::Convolution_Layer<2, 1, 3, 1, 1, CNN::ReLU<float>> con_layer_eps; 

    con_layer_eps.kernels = con_layer.kernels;
    con_layer_eps.biases = con_layer.biases;

    CNN::HeapTensor3D<1, 4, 4, float> conv_weighted_output_eps{};
    CNN::HeapTensor3D<1, 4, 4, float> conv_activ_output_eps{};

    CNN::HeapTensor1D<2, float> neural_weighted_output_eps{};
    CNN::HeapTensor1D<2, float> neural_activ_output_eps{};

    for(size_t j = 0; j < 18; ++j){

        con_layer_eps.kernels[j] = con_layer.kernels[j] + 1e-3;
        con_layer_eps.apply(input, conv_weighted_output_eps, conv_activ_output_eps);
        neural_layer.apply(conv_activ_output_eps, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_p = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        con_layer_eps.kernels[j] = con_layer.kernels[j] - 1e-3;
        con_layer_eps.apply(input, conv_weighted_output_eps, conv_activ_output_eps);
        neural_layer.apply(conv_activ_output_eps, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_m = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = con_weights_deriv[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

    }

    for(size_t j = 0; j < 1; ++j){

        con_layer_eps.biases[j] = con_layer.biases[j] + 1e-3;
        con_layer_eps.apply(input, conv_weighted_output_eps, conv_activ_output_eps);
        neural_layer.apply(conv_activ_output_eps, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_p = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        con_layer_eps.biases[j] = con_layer.biases[j] - 1e-3;
        con_layer_eps.apply(input, conv_weighted_output_eps, conv_activ_output_eps);
        neural_layer.apply(conv_activ_output_eps, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_m = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = con_biases_deriv[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

    }

}

TEST(GradientTest, DoubleConvolutionLayerGradientCorrect) {
    
    CNN::Convolution_Layer<2, 1, 3, 2, 2, CNN::ReLU<float>> con_layer_1;
    CNN::Convolution_Layer<1, 2, 3, 1, 1, CNN::ReLU<float>> con_layer_2;
    CNN::Neural_Layer<8, 2, CNN::Softmax<float>> neural_layer;

    CNN::HeapTensor3D<2, 8, 8, float> input{std::vector<float>{
        0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,
        0.41,0.23,0.25,0.79,0.1,0.234,0.35,0.79,0.21,0.23,0.25,0.79,0.1,0.23,0.5,0.79,
        0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,
        0.41,0.23,0.25,0.79,0.1,0.234,0.35,0.79,0.21,0.23,0.25,0.79,0.1,0.23,0.5,0.79,
        0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,
        0.41,0.23,0.25,0.79,0.1,0.234,0.35,0.79,0.21,0.23,0.25,0.79,0.1,0.23,0.5,0.79,
        0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,
        0.41,0.23,0.25,0.79,0.1,0.234,0.35,0.79,0.21,0.23,0.25,0.79,0.1,0.23,0.5,0.79}};

    CNN::HeapTensor1D<2, float> output{std::vector<float>{0.43,0.57}};

    CNN::HeapTensor3D<1, 2, 2, float> conv_weighted_output_1{};
    CNN::HeapTensor3D<1, 2, 2, float> conv_activ_output_1{};

    CNN::HeapTensor3D<2, 2, 2, float> conv_weighted_output_2{};
    CNN::HeapTensor3D<2, 2, 2, float> conv_activ_output_2{};

    CNN::HeapTensor1D<2, float> neural_weighted_output{};
    CNN::HeapTensor1D<2, float> neural_activ_output{};

    con_layer_1.apply(input, conv_weighted_output_1, conv_activ_output_1);
    con_layer_2.apply(conv_activ_output_1, conv_weighted_output_2, conv_activ_output_2);
    neural_layer.apply(conv_activ_output_2, neural_weighted_output, neural_activ_output);

    auto neural_biases_deriv = neural_layer.biases;
    auto neural_weights_deriv = neural_layer.weights;

    auto con_biases_deriv_1 = con_layer_1.biases;
    auto con_weights_deriv_1 = con_layer_1.kernels;

    auto con_biases_deriv_2 = con_layer_2.biases;
    auto con_weights_deriv_2 = con_layer_2.kernels;

    neural_biases_deriv = neural_activ_output;

    for(size_t i = 0; i < 2; ++i){
        neural_biases_deriv[i] -= output[i];
    }
    
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 8; ++j){
            neural_weights_deriv[i * 8 + j] =
                neural_biases_deriv[i] * conv_activ_output_2[j];
        }
    }

    // construct delta L for the convolution backward propagation
    CNN::HeapTensor3D<2, 2, 2, float> layer_delta_2{};
    neural_layer.apply_backwards(neural_biases_deriv, layer_delta_2);

    for(size_t j = 0; j < 8; ++j){
        layer_delta_2[j] *= con_layer_2.activation_func.derivative(conv_weighted_output_2[j]);
    }

    // based on delta L, compule the kernel and bias derivatives for L
    for(size_t OC = 0; OC < 2; ++OC)
    {
        for(size_t IC = 0; IC < 1; ++IC)
        {
            for(size_t KH = 0; KH < 3; ++KH)
            {
                for(size_t KW = 0; KW < 3; ++KW)
                {
                    float sum = 0;

                    for(size_t H = 0; H < 2; ++H)
                    {
                        for(size_t W = 0; W < 2; ++W)
                        { 
                            size_t delta_id = 
                                OC * 2 * 2 + H * 2 + W;

                            size_t activation_height = H - 1 + KH;

                            size_t activation_width = W - 1 + KW;

                            if (activation_height >= 0 && activation_width >= 0 &&
                                activation_height < 2 &&
                                activation_width < 2)
                            {
                                size_t activation_id = 
                                    IC * 2 * 2 +
                                    activation_height * 2 +
                                    activation_width;

                                sum += layer_delta_2[delta_id] * conv_activ_output_1[activation_id];
                            }
                        }
                    }
                    
                    size_t kernel_deriv_id = 
                        OC * 1 * 3 * 3 +
                        IC * 3 * 3 + KH * 3 + 
                        KW;

                    con_weights_deriv_2[kernel_deriv_id] = sum;
                }
            }
        }
    }

    CNN::compile_range<2>(
    [&]<size_t OC>()
    {   
        float sum = 0;

        CNN::compile_range<2>(
        [&]<size_t H>()
        {
            CNN::compile_range<2>(
            [&]<size_t W>()
            {   
                constexpr size_t delta_id = 
                    OC * 2 * 2 + H * 2 + W;
                
                sum += layer_delta_2[delta_id];
            });
        });

        con_biases_deriv_2[OC] = sum;
    });

    CNN::HeapTensor3D<1, 2, 2, float> layer_delta_1{};

    con_layer_2.apply_backwards(layer_delta_2, layer_delta_1);

    CNN::compile_range<1>(
    [&]<size_t OC>()
    {
        CNN::compile_range<2>(
        [&]<size_t H>()
        {
            CNN::compile_range<2>(
            [&]<size_t W>()
            {   
                constexpr size_t delta_id = 
                    OC * 2 * 2 + H * 2 + W;

                layer_delta_1[delta_id] *= 
                    con_layer_1.activation_func.derivative(conv_weighted_output_1[delta_id]);
            });
        });
    });

    // based on delta L, compule the kernel and bias derivatives for L
    CNN::compile_range<1>(
    [&]<size_t OC>()
    {
        CNN::compile_range<2>(
        [&]<size_t IC>()
        {
            CNN::compile_range<3>(
            [&]<size_t KH>()
            {
                CNN::compile_range<3>(
                [&]<size_t KW>()
                {
                    float sum = 0;

                    CNN::compile_range<2>(
                    [&]<size_t H>()
                    {
                        CNN::compile_range<2>(
                        [&]<size_t W>()
                        {   
                            constexpr size_t delta_id = 
                                OC * 2 * 2 + H * 2 + W;

                            constexpr size_t pool_h = H * 2;
                            constexpr size_t pool_w = W * 2;

                            CNN::compile_range<2>(
                            [&]<size_t PW>()
                            {
                                CNN::compile_range<2>(
                                [&]<size_t PH>()
                                {
                                    constexpr size_t activation_height = 
                                        (pool_h + PH) * 2 - 1 + KH;

                                    constexpr size_t activation_width = 
                                        (pool_w + PW) * 2 - 1 + KW;

                                    if constexpr (activation_height >= 0 && activation_width >= 0 &&
                                        activation_height < 8 && activation_width < 8)
                                    {
                                        constexpr size_t activation_id = 
                                            IC * 8 * 8 +
                                            activation_height * 8 +
                                            activation_width;

                                        sum += layer_delta_1[delta_id] * input[activation_id];
                                    }
                                });
                            });
                        });
                    });
                        
                    constexpr size_t kernel_deriv_id = 
                        OC * 2 * 3 * 3 +
                        IC * 3 * 3 +
                        KH * 3 + KW;

                    con_weights_deriv_1[kernel_deriv_id] = sum / (4);
                });
            });
        });
    });

    CNN::compile_range<1>(
    [&]<size_t OC>()
    {   
        float sum = 0;

        CNN::compile_range<2>(
        [&]<size_t H>()
        {
            CNN::compile_range<2>(
            [&]<size_t W>()
            {   
                constexpr size_t delta_id = 
                    OC * 2 * 2 +
                    H * 2 +
                    W;
                
                sum += layer_delta_1[delta_id];
            });
        });

        con_biases_deriv_1[OC] = sum;
    });

    CNN::Convolution_Layer<2, 1, 3, 2, 2, CNN::ReLU<float>> con_layer_1_eps; 

    con_layer_1_eps.kernels = con_layer_1.kernels;
    con_layer_1_eps.biases = con_layer_1.biases;

    CNN::HeapTensor3D<1, 2, 2, float> conv_weighted_output_1_eps{};
    CNN::HeapTensor3D<1, 2, 2, float> conv_activ_output_1_eps{};

    CNN::HeapTensor3D<2, 2, 2, float> conv_weighted_output_2_eps{};
    CNN::HeapTensor3D<2, 2, 2, float> conv_activ_output_2_eps{};

    CNN::HeapTensor1D<2, float> neural_weighted_output_eps{};
    CNN::HeapTensor1D<2, float> neural_activ_output_eps{};

    for(size_t j = 0; j < 18; ++j){

        con_layer_1_eps.kernels[j] = con_layer_1.kernels[j] + 1e-3;
        con_layer_1_eps.apply(input, conv_weighted_output_1_eps, conv_activ_output_1_eps);
        con_layer_2.apply(conv_activ_output_1_eps, conv_weighted_output_2_eps, conv_activ_output_2_eps);
        neural_layer.apply(conv_activ_output_2_eps, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_p = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        con_layer_1_eps.kernels[j] = con_layer_1.kernels[j] - 1e-3;
        con_layer_1_eps.apply(input, conv_weighted_output_1_eps, conv_activ_output_1_eps);
        con_layer_2.apply(conv_activ_output_1_eps, conv_weighted_output_2_eps, conv_activ_output_2_eps);
        neural_layer.apply(conv_activ_output_2_eps, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_m = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = con_weights_deriv_1[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4); 

    }

    for(size_t j = 0; j < 1; ++j){

        con_layer_1_eps.biases[j] = con_layer_1.biases[j] + 1e-3;
        con_layer_1_eps.apply(input, conv_weighted_output_1_eps, conv_activ_output_1_eps);
        con_layer_2.apply(conv_activ_output_1_eps, conv_weighted_output_2_eps, conv_activ_output_2_eps);
        neural_layer.apply(conv_activ_output_2_eps, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_p = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        con_layer_1_eps.biases[j] = con_layer_1.biases[j] - 1e-3;
        con_layer_1_eps.apply(input, conv_weighted_output_1_eps, conv_activ_output_1_eps);
        con_layer_2.apply(conv_activ_output_1_eps, conv_weighted_output_2_eps, conv_activ_output_2_eps);
        neural_layer.apply(conv_activ_output_2_eps, neural_weighted_output_eps, neural_activ_output_eps);

        auto l_m = CNN::cross_entropy_loss(neural_activ_output_eps, output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = con_biases_deriv_1[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);
    }

}

TEST(GradientTest, NetworkGradientCorrect) {
    
    CNN::Convolution_Layer<2, 1, 3, 2, 2, CNN::ReLU<float>> con_layer_1;
    CNN::Convolution_Layer<1, 1, 3, 1, 1, CNN::ReLU<float>> con_layer_2;
    CNN::Convolution_Layer<1, 2, 3, 1, 1, CNN::ReLU<float>> con_layer_3;
    CNN::Neural_Layer<8, 8, CNN::ReLU<float>> neural_layer_1;
    CNN::Neural_Layer<8, 2, CNN::Softmax<float>> neural_layer_2;

    CNN::HeapTensor3D<2, 8, 8, float> input{std::vector<float>{
        0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,
        0.41,0.23,0.25,0.79,0.1,0.234,0.35,0.79,0.21,0.23,0.25,0.79,0.1,0.23,0.5,0.79,
        0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,
        0.41,0.23,0.25,0.79,0.1,0.234,0.35,0.79,0.21,0.23,0.25,0.79,0.1,0.23,0.5,0.79,
        0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,
        0.41,0.23,0.25,0.79,0.1,0.234,0.35,0.79,0.21,0.23,0.25,0.79,0.1,0.23,0.5,0.79,
        0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,0.1,0.23,0.5,0.79,
        0.41,0.23,0.25,0.79,0.1,0.234,0.35,0.79,0.21,0.23,0.25,0.79,0.1,0.23,0.5,0.79}};

    CNN::HeapTensor1D<2, float> output{std::vector<float>{0.9,0.1}};

    auto network = CNN::Network::network<2, 8, 8, 3, 2>(
                con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

    decltype(network)::conv_layer_tuple conv_gradients;
    decltype(network)::neural_layer_tuple neural_gradients;

    decltype(network)::conv_feature_tuple conv_weighted_outputs;
    decltype(network)::conv_feature_tuple conv_activ_outputs;

    decltype(network)::neural_feature_tuple neural_weighted_outputs;
    decltype(network)::neural_feature_tuple neural_activ_outputs;

    network.compute_gradient(input, output, conv_gradients, neural_gradients);

    for(size_t j = 0; j < 18; ++j){

        con_layer_1.kernels[j] += 1e-3;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_1.kernels[j] -= 2 * 1e-3;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = std::get<0>(conv_gradients).kernels[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

    }

    for(size_t j = 0; j < 9; ++j){

        con_layer_2.kernels[j] += 1e-3;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_2.kernels[j] -= 2 * 1e-3;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = std::get<1>(conv_gradients).kernels[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

    }

    for(size_t j = 0; j < 18; ++j){

        con_layer_3.kernels[j] += 1e-3;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_3.kernels[j] -= 2 * 1e-3;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = std::get<2>(conv_gradients).kernels[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

    }

    for(size_t j = 0; j < 64; ++j){

        neural_layer_1.weights[j] += 1e-3;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        neural_layer_1.weights[j] -= 2 * 1e-3;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = std::get<0>(neural_gradients).weights[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

    }

    for(size_t j = 0; j < 16; ++j){

        neural_layer_2.weights[j] += 1e-3;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        neural_layer_2.weights[j] -= 2 * 1e-3;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = std::get<1>(neural_gradients).weights[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

    }

    for(size_t j = 0; j < 1; ++j){

        con_layer_1.biases[j] += 1e-2;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_1.biases[j] -= 2 * 1e-2;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-2);
        auto analytic_div = std::get<0>(conv_gradients).biases[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

    }

    for(size_t j = 0; j < 1; ++j){

        con_layer_2.biases[j] += 1e-4;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_2.biases[j] -= 2 * 1e-4;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-4);
        auto analytic_div = std::get<1>(conv_gradients).biases[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);
    }

    for(size_t j = 0; j < 2; ++j){

        con_layer_3.biases[j] += 1e-4;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_3.biases[j] -= 2 * 1e-4;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-4);
        auto analytic_div = std::get<2>(conv_gradients).biases[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);
    }
/*
    for(size_t j = 0; j < 8; ++j){

        neural_layer_1.biases[j] += 1e-3;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        neural_layer_1.biases[j] -= 2 * 1e-3;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = std::get<0>(neural_gradients).biases[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

        std::cout << j << " " << numeric_div << " " << analytic_div << " " << rel_diff << " " << abs_diff <<std::endl;

    }
*/
    for(size_t j = 0; j < 2; ++j){

        neural_layer_2.biases[j] += 1e-3;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        neural_layer_2.biases[j] -= 2 * 1e-3;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.foward_propagate(input, 
        conv_weighted_outputs, 
        conv_activ_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = std::get<1>(neural_gradients).biases[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-4);

    }
}