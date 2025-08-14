#ifndef TINY_IMAGE_LOADER_HPP
#define TINY_IMAGE_LOADER_HPP

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../core.hpp"

namespace CNN::Utility{

static std::unordered_map<std::string, int> load_tiny_imag_categories(const std::string & filename){

    std::ifstream category_file(filename);

    if(!category_file){
        throw std::runtime_error("Could not open Category file: " + filename);
    }

    std::unordered_map<std::string, int> category_map;

    std::string line;
    int counter = 0;

    while (std::getline(category_file, line)) {
        if (!line.empty()) {
            category_map[line] = counter++;
        }
    }

    return category_map;
}

template<size_t C, size_t H, size_t W, size_t Cat, typename T = float>
static std::vector<std::pair<HeapTensor3D<C, H, W, T>, HeapTensor1D<Cat, T>>> load_tiny_imag_train(
    const std::string & dirname, const std::unordered_map<std::string, int> & category_map){

    if(Cat != category_map.size()){
        throw std::runtime_error("Category map is not of given size.");
    }

    std::vector<std::pair<HeapTensor3D<C, H, W, T>, HeapTensor1D<Cat, T>>> training_data{};
    const std::filesystem::path base_directory(dirname);

    for (const auto& subdir : std::filesystem::directory_iterator(base_directory)) {
        if (!subdir.is_directory()) continue;

        const auto& category_name = subdir.path().filename().string();

        auto category_citer = category_map.find(category_name);
        if(category_citer == category_map.end()){
            throw std::runtime_error("Category not found in category_map: " + category_name);
        }

        for (const auto& image_file : std::filesystem::directory_iterator(subdir.path() / "images")){
            if (!image_file.is_regular_file()) continue;

            HeapTensor3D<C, H, W, T> imag_vec = 
                        img_vec_convert<C, H, W, T>(image_file.path().string());
            HeapTensor1D<Cat, T> category_vec(static_cast<T>(0));

            category_vec[category_citer -> second] = static_cast<T>(1);

            training_data.emplace_back(std::move(imag_vec), std::move(category_vec));
        }
    }
    return training_data;
}

template<size_t C, size_t H, size_t W, size_t Cat, typename T = float>
static std::vector<std::pair<HeapTensor3D<C, H, W, T>, HeapTensor1D<Cat, T>>> load_tiny_imag_test(
    const std::string & dirname, const std::unordered_map<std::string, int> & category_map){

    if(Cat != category_map.size()){
        throw std::runtime_error("Category map is not of given size.");
    }

    std::ifstream annotations_file(dirname + "/val_annotations.txt");
    std::unordered_map<std::string, int> annotations;

    std::string line;
    while(std::getline(annotations_file, line)){
        std::stringstream ss(line);

        std::string image_name;
        std::string category_name;

        std::getline(ss, image_name, '\t');
        std::getline(ss, category_name, '\t');

        auto category_citer = category_map.find(category_name);
        if(category_citer == category_map.end()){
            throw std::runtime_error("Category not found in category_map: " + category_name);
        }
        annotations[image_name] = category_citer -> second;
    }

    std::vector<std::pair<HeapTensor3D<C, H, W, T>, HeapTensor1D<Cat, T>>> training_data{};
    const std::filesystem::path base_directory(dirname + "/images");

    for (const auto& image_file : std::filesystem::directory_iterator(base_directory)){
        if (!image_file.is_regular_file()) continue;

        HeapTensor3D<C, H, W, T> imag_vec = 
                    img_vec_convert<C, H, W, T>(image_file.path().string());
        HeapTensor1D<Cat, T> category_vec(category_map.size(), static_cast<T>(0));

        category_vec[annotations[image_file.path().filename().string()]] = static_cast<T>(1);

        training_data.emplace_back(std::move(imag_vec), std::move(category_vec));
    }

    return training_data;
}

}
#endif // TINY_IMAGE_LOADER_HPP