#include <iostream>

#include "core.hpp"
#include "utility.hpp"
#include "network/network.hpp"

int main(int argc, char ** argv){

  using T = float;

  // image shape of training data;
  // if shape is not met, we use openCV rescale
  constexpr size_t CHANNELS = 3;
  constexpr size_t IMG_HEIGHT = 28;
  constexpr size_t IMG_WIDTH = 28;

  auto category_map = CNN::Utility::load_tiny_imag_categories("data/tiny-imagenet-200/wnids.txt");
 // auto training_data = CNN::Utility::load_tiny_imag_train<CHANNELS, IMG_HEIGHT, IMG_WIDTH, T>("data/tiny-imagenet-200/train", category_map);
 // auto test_data = CNN::Utility::load_tiny_imag_test<CHANNELS, IMG_HEIGHT, IMG_WIDTH, T>("data/tiny-imagenet-200/val", category_map);

  CNN::Convolution_Layer<3, 3, 3, 2, 2, CNN::ReLU<T>> con_layer_1;
  CNN::Convolution_Layer<3, 9, 3, 2, 2, CNN::ReLU<T>> con_layer_2;
  CNN::Neural_Layer<64, 64, T> neural_layer_1;
  CNN::Neural_Layer<64, 200, T> neural_layer_2;

  auto network = CNN::Network::network<CHANNELS, IMG_HEIGHT, IMG_WIDTH, 2, 2>(con_layer_1, con_layer_2, neural_layer_1, neural_layer_2);

}