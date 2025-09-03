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

TEST(GradientTest, NetworkGradientCorrect) {
    
    CNN::Convolution_Layer<2, 1, 3, 1, 2, CNN::ReLU<float>> con_layer_1;
    CNN::Convolution_Layer<1, 1, 3, 2, 1, CNN::ReLU<float>> con_layer_2;
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
    decltype(network)::pooled_feature_tuple pooling_outputs;

    decltype(network)::neural_feature_tuple neural_weighted_outputs;
    decltype(network)::neural_feature_tuple neural_activ_outputs;

    network.compute_gradient(input, output, conv_gradients, neural_gradients);

    for(size_t j = 0; j < 18; ++j){

        con_layer_1.kernels[j] += 1e-2;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_1.kernels[j] -= 2 * 1e-2;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-2);
        auto analytic_div = std::get<0>(conv_gradients).kernels[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.5 || abs_diff < 1e-2);

        std::cout << j << " " << numeric_div << " " << analytic_div << " " << rel_diff << " " << abs_diff <<std::endl;
    }

    for(size_t j = 0; j < 9; ++j){

        con_layer_2.kernels[j] += 1e-2;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_2.kernels[j] -= 2 * 1e-2;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-2);
        auto analytic_div = std::get<1>(conv_gradients).kernels[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.5 || abs_diff < 1e-2);

        std::cout << j << " " << numeric_div << " " << analytic_div << " " << rel_diff << " " << abs_diff <<std::endl;
    }

    for(size_t j = 0; j < 18; ++j){

        con_layer_3.kernels[j] += 1e-3;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_3.kernels[j] -= 2 * 1e-3;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = std::get<2>(conv_gradients).kernels[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.5 || abs_diff < 1e-2);

        std::cout << j << " " << numeric_div << " " << analytic_div << " " << rel_diff << " " << abs_diff <<std::endl;

    }

    for(size_t j = 0; j < 64; ++j){

        neural_layer_1.weights[j] += 1e-3;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        neural_layer_1.weights[j] -= 2 * 1e-3;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-3);
        auto analytic_div = std::get<0>(neural_gradients).weights[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.5 || abs_diff < 1e-2);

        std::cout << j << " " << numeric_div << " " << analytic_div << " " << rel_diff << " " << abs_diff <<std::endl;

    }

    for(size_t j = 0; j < 16; ++j){

        neural_layer_2.weights[j] += 1e-2;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        neural_layer_2.weights[j] -= 2 * 1e-2;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-2);
        auto analytic_div = std::get<1>(neural_gradients).weights[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.5 || abs_diff < 1e-2);

        std::cout << j << " " << numeric_div << " " << analytic_div << " " << rel_diff << " " << abs_diff <<std::endl;

    }

    for(size_t j = 0; j < 1; ++j){

        con_layer_1.biases[j] += 1e-2;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs,
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_1.biases[j] -= 2 * 1e-2;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs,
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-2);
        auto analytic_div = std::get<0>(conv_gradients).biases[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.5 || abs_diff < 1e-2);

        std::cout << j << " " << numeric_div << " " << analytic_div << " " << rel_diff << " " << abs_diff <<std::endl;

    }

    for(size_t j = 0; j < 1; ++j){

        con_layer_2.biases[j] += 1e-2;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs,
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_2.biases[j] -= 2 * 1e-2;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs,
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-2);
        auto analytic_div = std::get<1>(conv_gradients).biases[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.5 || abs_diff < 1e-2);

        std::cout << j << " " << numeric_div << " " << analytic_div << " " << rel_diff << " " << abs_diff <<std::endl;
    }

    for(size_t j = 0; j < 2; ++j){

        con_layer_3.biases[j] += 1e-2;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs,
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        con_layer_3.biases[j] -= 2 * 1e-2;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs,
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-2);
        auto analytic_div = std::get<2>(conv_gradients).biases[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.5 || abs_diff < 1e-2);

        std::cout << j << " " << numeric_div << " " << analytic_div << " " << rel_diff << " " << abs_diff <<std::endl;
    }

    for(size_t j = 0; j < 8; ++j){

        neural_layer_1.biases[j] += 1e-4;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs,
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        neural_layer_1.biases[j] -= 2 * 1e-4;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-4);
        auto analytic_div = std::get<0>(neural_gradients).biases[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        EXPECT_TRUE(rel_diff < 0.5 || abs_diff < 1e-2);

        std::cout << j << " " << numeric_div << " " << analytic_div << " " << rel_diff << " " << abs_diff <<std::endl;
    }
/*
    for(size_t j = 0; j < 2; ++j){

        neural_layer_2.biases[j] += 1e-4;

        auto network_eps_p = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

        network_eps_p.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_p = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        neural_layer_2.biases[j] -= 2 * 1e-4;

        auto network_eps_m = CNN::Network::network<2, 8, 8, 3, 2>(
            con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);
        
        network_eps_m.forward_propagate(input, 
        conv_weighted_outputs, 
        pooling_outputs, 
        neural_weighted_outputs, 
        neural_activ_outputs);

        auto l_m = CNN::cross_entropy_loss(std::get<2>(neural_activ_outputs), output);

        auto numeric_div = (l_p - l_m)/float(2 * 1e-4);
        auto analytic_div = std::get<1>(neural_gradients).biases[j];

        auto rel_diff = std::fabs(analytic_div - numeric_div) / std::max(std::fabs(analytic_div), std::fabs(numeric_div));
        auto abs_diff = std::fabs(analytic_div - numeric_div);

        std::cout << rel_diff << " " << abs_diff << std::endl;

        EXPECT_TRUE(rel_diff < 0.1 || abs_diff < 1e-2);

    }
*/
}

TEST(LearningTest, NetworkGradientCorrect) {
    CNN::HeapTensor3D<1,2,2> input1{1,0,0,1};
    CNN::HeapTensor3D<1,2,2> input2{0,1,1,0};
    CNN::HeapTensor3D<1,2,2> input3{0,5,5,0};
    CNN::HeapTensor3D<1,2,2> input4{2,0,0,1};

    CNN::HeapTensor1D<2> target1{1,0};
    CNN::HeapTensor1D<2> target2{0,1};
    CNN::HeapTensor1D<2> target3{0,1};
    CNN::HeapTensor1D<2> target4{1,0};

    CNN::Convolution_Layer<1, 2, 2, 1, 2, CNN::ReLU<float>> con_layer_1;
    CNN::Convolution_Layer<2, 2, 2, 1, 1, CNN::ReLU<float>> con_layer_2;
    CNN::Neural_Layer<2, 8, CNN::ReLU<float>> neural_layer_1;
    CNN::Neural_Layer<8, 2, CNN::Softmax<float>> neural_layer_2;

    auto network = CNN::Network::network<1, 2, 2, 2, 2>(
        con_layer_1, con_layer_2, neural_layer_1, neural_layer_2);

    CNN::Optimizer::Adam_Optimizer<float> opt(0.1);

    std::vector<std::pair<CNN::HeapTensor3D<1, 2, 2, float>, CNN::HeapTensor1D<2, float>>> training_data{
        {input1, target1}, {input2, target2}, {input3, target3}, {input4, target4}
    };

    network.train(training_data, opt, 4, 200); 

    auto result1 = network.evaluate(input1);
    auto result2 = network.evaluate(input2);
    auto result3 = network.evaluate(input3);
    auto result4 = network.evaluate(input4);

    CNN::HeapTensor3D<1,2,2> test{9,0,0,5};
    auto result_test = network.evaluate(test);

    EXPECT_TRUE(result1[0] > 0.5);
    EXPECT_TRUE(result2[1] > 0.5);
    EXPECT_TRUE(result3[1] > 0.5);
    EXPECT_TRUE(result4[0] > 0.5);

    EXPECT_TRUE(result_test[0] > 0.5);
}
