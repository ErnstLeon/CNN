#include <iostream>

#include "core.hpp"
#include "utility.hpp"
#include "network.hpp"

int main(int argc, char ** argv){

  using T = float;

  // Image shape of data
  // If shape is not met, we use openCV rescale
  constexpr size_t CHANNELS = 3;
  constexpr size_t IMG_HEIGHT = 64;
  constexpr size_t IMG_WIDTH = 64;

  // Number of catergories for classification
  constexpr size_t CATEGORIES = 200;

  // Load tiny-imagenet-200 categories and data
  auto category_map = CNN::Utility::load_tiny_imag_categories("data/tiny-imagenet-200/wnids.txt");
  auto training_data = CNN::Utility::load_tiny_imag_train<CHANNELS, IMG_HEIGHT, IMG_WIDTH, CATEGORIES, T>("data/tiny-imagenet-200/train", category_map);
  auto test_data = CNN::Utility::load_tiny_imag_test<CHANNELS, IMG_HEIGHT, IMG_WIDTH, CATEGORIES, T>("data/tiny-imagenet-200/val", category_map);
  
  // CNN design
  //  - Convolution_Layer : 
  //      Input channels, Output channels, Kernel size, Stride, Pooling size
  //  - Neural_Layer : 
  //      Input neurons, Output neurons
  CNN::Convolution_Layer<3, 8, 3, 1, 2, CNN::ReLU<T>> con_layer_1;
  CNN::Convolution_Layer<8, 16, 3, 1, 2, CNN::ReLU<T>> con_layer_2;
  CNN::Convolution_Layer<16, 32, 3, 1, 2, CNN::ReLU<T>> con_layer_3;
  CNN::Neural_Layer<128, 200, CNN::Softmax<T>> neural_layer_1;

  // CNN initialization
  auto network = CNN::Network::network<CHANNELS, IMG_HEIGHT, IMG_WIDTH, 3, 1>(
      con_layer_1, con_layer_2, con_layer_3, neural_layer_1);

  // CNN optimization with Adam and step size and decay
  CNN::Optimizer::Adam_Optimizer<T> opt(0.001, 0.001);
  network.train(training_data, opt, 64, 100); 

  // Get error of trained network on test data
  auto [loss, error] = network.assess(test_data);
  std::cout << "Test loss: " << loss << " test error: " << error << std::endl;
  
  // Get network output category for known image
  auto test_image = CNN::Utility::img_vec_convert<CHANNELS, IMG_HEIGHT, IMG_WIDTH>("data/banana.jpg");
  auto test_output = network.evaluate(test_image);
  
  auto test_category = std::distance(test_output.begin(), std::max_element(test_output.begin(), test_output.end()));
  std::cout << "Category of test image: " << test_category << std::endl;
}
