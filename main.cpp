#include <iostream>

#include "core.hpp"
#include "utility.hpp"
#include "network.hpp"

int main(int argc, char ** argv){

  using T = float;

  // image shape of training data;
  // if shape is not met, we use openCV rescale
  constexpr size_t CHANNELS = 3;
  constexpr size_t IMG_HEIGHT = 64;
  constexpr size_t IMG_WIDTH = 64;

  constexpr size_t CATEGORIES = 200;

  auto category_map = CNN::Utility::load_tiny_imag_categories("data/tiny-imagenet-200/wnids.txt");
  auto training_data = CNN::Utility::load_tiny_imag_train<CHANNELS, IMG_HEIGHT, IMG_WIDTH, CATEGORIES, T>("data/tiny-imagenet-200/train", category_map);
 // auto test_data = CNN::Utility::load_tiny_imag_test<CHANNELS, IMG_HEIGHT, IMG_WIDTH, CATEGORIES, T>("data/tiny-imagenet-200/val", category_map);

  CNN::Convolution_Layer<3, 9, 5, 1, 2, CNN::ReLU<T>> con_layer_1;
  CNN::Convolution_Layer<9, 16, 5, 2, 2, CNN::ReLU<T>> con_layer_2;
  CNN::Convolution_Layer<16, 32, 5, 2, 2, CNN::ReLU<T>> con_layer_3;
  CNN::Neural_Layer<64, 64, CNN::ReLU<T>> neural_layer_1;
  CNN::Neural_Layer<64, 200, CNN::Softmax<T>> neural_layer_2;

  auto network = CNN::Network::network<CHANNELS, IMG_HEIGHT, IMG_WIDTH, 3, 2>(
      con_layer_1, con_layer_2, con_layer_3, neural_layer_1, neural_layer_2);

  CNN::Optimizer::Adam_Optimizer<T> opt(0.1);
  
  network.train(training_data, opt, 128, 100); 
}