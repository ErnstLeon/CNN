#include <iostream>

#include "utility/utility.hpp"
#include "network/network.hpp"

int main(int argc, char ** argv){

    using T = float;

  //  auto category_map = CNN::Utility::load_tiny_imag_categories("data/tiny-imagenet-200/wnids.txt");
  //  auto training_data = CNN::Utility::load_tiny_imag_train<T>("data/tiny-imagenet-200/train", category_map);
  //  auto test_data = CNN::Utility::load_tiny_imag_test<T>("data/tiny-imagenet-200/val", category_map);

    CNN::Network::Convolution_Layer con_layer_1{1, 3, 2};
    CNN::Network::Neural_Layer neural_layer_1{64};
    CNN::Network::Neural_Layer neural_layer_2{200};

    CNN::Network::Network<T, 1, 2>(con_layer_1, neural_layer_1, neural_layer_2);

}