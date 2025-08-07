#include <gtest/gtest.h>

#include "core.hpp"

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
    EXPECT_EQ(relu(-2.0), 0.0);
    EXPECT_EQ(relu(0.0), 0.0);

    EXPECT_EQ(relu.derivative(5.0), 1.0);
    EXPECT_EQ(relu.derivative(-1.0), 0.0);
    EXPECT_EQ(relu.derivative(0.0), 0.0);
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
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> output = CNN::softmax(input);

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
    std::vector<double> input = {1.0, 2.0, 3.0};
    CNN::softmax_inplace(input);

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
    
    CNN::Optimizer::Gradient_Descent<double> optimizer{0.1};

    std::vector<double> value(1);
    std::vector<double> gradient(1);

    // function (x - 3)^2, derivate: 2(x - 3)
    value[0] = static_cast<double>(0.0);
    for (int step = 0; step < 1000; ++step) {
        gradient[0] = 2 * (value[0] - 3);
        optimizer.update(value, gradient);
    }

    EXPECT_NEAR(value[0], 3.0, 1e-4);
}

TEST(LearningTest, AdamOptimizerConverges) {
    
    CNN::Optimizer::Adam_Optimizer<double> optimizer{0.1};

    std::vector<double> value(1);
    std::vector<double> gradient(1);

    // function (x - 3)^2, derivate: 2(x - 3)
    value[0] = static_cast<double>(0.0);
    for (int step = 0; step < 1000; ++step) {
        gradient[0] = 2 * (value[0] - 3);
        optimizer.update(value, gradient);
    }

    EXPECT_NEAR(value[0], 3.0, 1e-4);
}