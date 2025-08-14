#ifndef IMAGE_PROCESSING_HPP
#define IMAGE_PROCESSING_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../core.hpp"

namespace CNN::Utility{

template<size_t C, size_t H, size_t W, typename T = float>
static std::vector<T> img_vec_convert(const std::string &filename) {

    cv::Mat img = cv::imread(filename);
    if (img.empty()) {
        throw std::runtime_error("Could not load image: " + filename);
    }

    if (img.channels() != C) {
        throw std::runtime_error("Unexpected number of channels in image: " + filename);
    }

    cv::resize(img, img, cv::Size(W, H));

    std::vector<T> vec(H * W * C);

    for (size_t channel = 0; channel < C; ++channel) {
        for (size_t row = 0; row < H; ++row) {
            for (size_t col = 0; col < W; ++col) {
                T raw_val = static_cast<T>(img.at<cv::Vec3b>(row, col)[channel]);
                vec[channel * H * W + row * W + col] = raw_val / static_cast<T>(255);
            }
        }
    }

    return vec;
}

template<typename T>
static void print_ascii(const std::vector<T>& vec, int rows, int cols, int channels) {
    const std::string levels = " .:-=+*#%@";

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float r_val = vec[0 * rows * cols + r * cols + c];
            float g_val = vec[1 * rows * cols + r * cols + c];
            float b_val = vec[2 * rows * cols + r * cols + c];
            float gray = (r_val + g_val + b_val) / static_cast<T>(3);

            int idx = static_cast<int>(gray * (levels.size() - 1));
            std::cout << levels[idx];
        }
        std::cout << '\n';
    }
}

template<size_t C, size_t H, size_t W, typename T = float>
static void print_ascii(const HeapTensor3D<C, H, W, T>& input) {
    const std::string levels = " .:-=+*#%@";

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            float r_val = input.features[0 * H * W + r * W + c];
            float g_val = input.features[1 * H * W + r * W + c];
            float b_val = input.features[2 * H * W + r * W + c];
            float gray = (r_val + g_val + b_val) / static_cast<T>(3);

            int idx = static_cast<int>(gray * (levels.size() - 1));
            std::cout << levels[idx];
        }
        std::cout << '\n';
    }
}

template<size_t C, size_t H, size_t W, typename T = float>
static void print_ascii(const StackTensor3D<C, H, W, T>& input) {
    const std::string levels = " .:-=+*#%@";

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            float r_val = input.features[0 * H * W + r * W + c];
            float g_val = input.features[1 * H * W + r * W + c];
            float b_val = input.features[2 * H * W + r * W + c];
            float gray = (r_val + g_val + b_val) / static_cast<T>(3);

            int idx = static_cast<int>(gray * (levels.size() - 1));
            std::cout << levels[idx];
        }
        std::cout << '\n';
    }
}

}
#endif // IMAGE_PROCESSING_HPP