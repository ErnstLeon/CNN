#ifndef IMAGE_PROCESSING_HPP
#define IMAGE_PROCESSING_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace CNN::Utility{
template<typename T>
static std::vector<T> img_vec_convert(const std::string & filename){

    cv::Mat img = cv::imread(filename);
    if (img.empty()) {
        throw std::runtime_error("Could not load image: " + filename);
    }

    int num_rows = img.rows, num_cols = img.cols, num_channels = img.channels(); 

    std::vector<T> vec(num_rows * num_cols * num_channels);

    for(int channel = 0; channel < num_channels; ++channel)
    {
        for(int row = 0; row < num_rows; ++row)
        {
            for(int col = 0; col < num_cols; ++col)
            {
                T raw_val = static_cast<T>(img.at<cv::Vec3b>(row, col)[channel]);
                vec[channel * num_rows * num_cols + row * num_cols + col] = raw_val / static_cast<T>(255);
            }
        }
    }

    return vec;
}

template<typename T>
static void print_ascii_image(const std::vector<T>& vec, int rows, int cols, int channels) {
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
}
#endif // IMAGE_PROCESSING_HPP