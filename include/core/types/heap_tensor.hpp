#ifndef HEAP_TENSOR_HPP
#define HEAP_TENSOR_HPP

#include <limits>
#include <random>
#include <vector>

#include "../helper/loop_unroll.hpp"

namespace CNN
{

template<size_t Size, typename T = float>
struct HeapTensor1D {

    using type = T;
    static constexpr size_t size = Size;

    std::vector<T> contents;

    HeapTensor1D(T default_val = static_cast<T>(0)) : contents(size, default_val) {}

    HeapTensor1D(const std::vector<T>& input) : contents(input) {
        if (input.size() != size) throw std::runtime_error("Size mismatch");
    }

    HeapTensor1D(std::vector<T>&& input) : contents(std::move(input)) {
        if (contents.size() != size) throw std::runtime_error("Size mismatch");
    }

    HeapTensor1D(const HeapTensor1D& other) : contents(other.contents) {}

    HeapTensor1D(HeapTensor1D&& other) noexcept : contents(std::move(other.contents)) {}

    HeapTensor1D(std::initializer_list<T> input) : contents(input) {
        if (input.size() != size)
            throw std::runtime_error("Size mismatch");
    }

    HeapTensor1D& operator=(const HeapTensor1D& other) { 
        contents = other.contents; 
        return *this; 
    }

    HeapTensor1D& operator=(HeapTensor1D&& other) noexcept {
        contents = std::move(other.contents); return *this; 
    }

    template<typename OtherTensor>
    requires (std::remove_reference_t<OtherTensor>::size == size)
    HeapTensor1D& operator=(const OtherTensor& other) {
        std::copy(other.contents.begin(), other.contents.end(), contents.begin());
        return *this;
    }

    HeapTensor1D& fill(T val) noexcept {
        std::fill(contents.begin(), contents.end(), val); 
        return *this; 
    }

    HeapTensor1D& operator+=(const HeapTensor1D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] += rhs.contents[i]; 
        return *this; 
    }

    HeapTensor1D& operator*=(const HeapTensor1D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= rhs.contents[i]; 
        return *this; 
    }

    HeapTensor1D& operator*=(T val) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= val; 
        return *this; 
    }

    constexpr T& operator()(size_t i) { 
        return contents[i]; 
    }

    constexpr const T& operator()(size_t i) const {
        return contents[i]; 
    }

    constexpr T& operator[](size_t id) { 
        return contents[id]; 
    }

    constexpr const T& operator[](size_t id) const { 
        return contents[id]; 
    }

    constexpr T* data() noexcept { return contents.data(); }
    constexpr const T* data() const noexcept { return contents.data(); }

    constexpr auto begin() noexcept { return contents.begin(); }
    constexpr auto begin() const noexcept { return contents.cbegin(); }

    constexpr auto end() noexcept { return contents.end(); }
    constexpr auto end() const noexcept { return contents.cend(); }
};

template<size_t SizeX, size_t SizeY, typename T = float>
struct HeapTensor2D {

    using type = T;
    static constexpr size_t size_x = SizeX;
    static constexpr size_t size_y = SizeY;
    static constexpr size_t size = SizeX * SizeY;

    std::vector<T> contents;

    HeapTensor2D(T default_val = static_cast<T>(0)) : contents(size, default_val) {}

    HeapTensor2D(const std::vector<T>& input) : contents(input) { 
        if (input.size() != size) throw std::runtime_error("Size mismatch"); 
    }

    HeapTensor2D(std::vector<T>&& input) : contents(std::move(input)) {
        if (contents.size() != size) throw std::runtime_error("Size mismatch"); 
    }

    HeapTensor2D(const HeapTensor2D& other) : contents(other.contents) {}

    HeapTensor2D(HeapTensor2D&& other) noexcept : contents(std::move(other.contents)) {}

    HeapTensor2D(std::initializer_list<T> input) : contents(input) {
        if (input.size() != size)
            throw std::runtime_error("Size mismatch");
    }

    HeapTensor2D& operator=(const HeapTensor2D& other) { 
        contents = other.contents; return *this; 
    }

    HeapTensor2D& operator=(HeapTensor2D&& other) noexcept {
        contents = std::move(other.contents); 
        return *this; 
    }

    template<typename OtherTensor>
    requires (std::remove_reference_t<OtherTensor>::size == size)
    HeapTensor2D& operator=(const OtherTensor& other) {
        std::copy(other.contents.begin(), other.contents.end(), contents.begin());
        return *this;
    }

    HeapTensor2D& fill(T val) noexcept { 
        std::fill(contents.begin(), contents.end(), val); 
        return *this; 
    }

    HeapTensor2D& operator+=(const HeapTensor2D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] += rhs.contents[i]; 
        return *this; 
    }

    HeapTensor2D& operator*=(const HeapTensor2D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= rhs.contents[i]; 
        return *this; 
    }

    HeapTensor2D& operator*=(T val) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= val; 
        return *this; 
    }

    constexpr T& operator()(size_t x, size_t y) { 
        return contents[x * size_y + y]; 
    }

    constexpr const T& operator()(size_t x, size_t y) const { 
        return contents[x * size_y + y]; 
    }

    constexpr T& operator[](size_t id) { 
        return contents[id]; 
    }

    constexpr const T& operator[](size_t id) const { 
        return contents[id]; 
    }

    constexpr T* data() noexcept { return contents.data(); }
    constexpr const T* data() const noexcept { return contents.data(); }

    constexpr auto begin() noexcept { return contents.begin(); }
    constexpr auto begin() const noexcept { return contents.cbegin(); }

    constexpr auto end() noexcept { return contents.end(); }
    constexpr auto end() const noexcept { return contents.cend(); }
};

template<size_t SizeX, size_t SizeY, size_t SizeZ, typename T = float>
struct HeapTensor3D {

    using type = T;
    static constexpr size_t size_x = SizeX;
    static constexpr size_t size_y = SizeY;
    static constexpr size_t size_z = SizeZ;
    static constexpr size_t size = SizeX * SizeY * SizeZ;

    std::vector<T> contents;

    HeapTensor3D(T default_val = static_cast<T>(0)) : contents(size, default_val) {}

    HeapTensor3D(const std::vector<T>& input) : contents(input) { 
        if (input.size() != size) throw std::runtime_error("Size mismatch"); 
    }

    HeapTensor3D(std::vector<T>&& input) : contents(std::move(input)) { 
        if (contents.size() != size) throw std::runtime_error("Size mismatch"); 
    }

    HeapTensor3D(const HeapTensor3D& other) : contents(other.contents) {}

    HeapTensor3D(HeapTensor3D&& other) noexcept : contents(std::move(other.contents)) {}

    HeapTensor3D(std::initializer_list<T> input) : contents(input) {
        if (input.size() != size)
            throw std::runtime_error("Size mismatch");
    }

    HeapTensor3D& operator=(const HeapTensor3D& other) { 
        contents = other.contents; 
        return *this; 
    }

    HeapTensor3D& operator=(HeapTensor3D&& other) noexcept { 
        contents = std::move(other.contents); 
        return *this; 
    }

    template<typename OtherTensor>
    requires (std::remove_reference_t<OtherTensor>::size == size)
    HeapTensor3D& operator=(const OtherTensor& other) {
        std::copy(other.contents.begin(), other.contents.end(), contents.begin());
        return *this;
    }

    HeapTensor3D& fill(T val) noexcept { 
        std::fill(contents.begin(), contents.end(), val); 
        return *this; 
    }

    HeapTensor3D& operator+=(const HeapTensor3D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] += rhs.contents[i]; 
        return *this; 
    }

    HeapTensor3D& operator*=(const HeapTensor3D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= rhs.contents[i]; 
        return *this; 
    }

    HeapTensor3D& operator*=(T val) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= val; 
        return *this; 
    }

    constexpr T& operator()(size_t x, size_t y, size_t z) { 
        return contents[x * size_y * size_z + y * size_z + z]; 
    }

    constexpr const T& operator()(size_t x, size_t y, size_t z) const { 
        return contents[x * size_y * size_z + y * size_z + z]; 
    }

    constexpr T& operator[](size_t id) { 
        return contents[id]; 
    }

    constexpr const T& operator[](size_t id) const { 
        return contents[id]; 
    }

    constexpr T* data() noexcept { return contents.data(); }
    constexpr const T* data() const noexcept { return contents.data(); }

    constexpr auto begin() noexcept { return contents.begin(); }
    constexpr auto begin() const noexcept { return contents.cbegin(); }

    constexpr auto end() noexcept { return contents.end(); }
    constexpr auto end() const noexcept { return contents.cend(); }
};

template<size_t SizeW, size_t SizeX, size_t SizeY, size_t SizeZ, typename T = float>
struct HeapTensor4D {

    using type = T;
    static constexpr size_t size_w = SizeW;
    static constexpr size_t size_x = SizeX;
    static constexpr size_t size_y = SizeY;
    static constexpr size_t size_z = SizeZ;
    static constexpr size_t size = SizeW * SizeX * SizeY * SizeZ;

    std::vector<T> contents;

    HeapTensor4D(T default_val = static_cast<T>(0)) : contents(size, default_val) {}

    HeapTensor4D(const std::vector<T>& input) : contents(input) { 
        if (input.size() != size) throw std::runtime_error("Size mismatch"); 
    }

    HeapTensor4D(std::vector<T>&& input) : contents(std::move(input)) { 
        if (contents.size() != size) throw std::runtime_error("Size mismatch"); 
    }

    HeapTensor4D(const HeapTensor4D& other) : contents(other.contents) {}

    HeapTensor4D(HeapTensor4D&& other) noexcept : contents(std::move(other.contents)) {}

    HeapTensor4D(std::initializer_list<T> input) : contents(input) {
        if (input.size() != size)
            throw std::runtime_error("Size mismatch");
    }

    HeapTensor4D& operator=(const HeapTensor4D& other) { 
        contents = other.contents; return *this; 
    }

    HeapTensor4D& operator=(HeapTensor4D&& other) noexcept { 
        contents = std::move(other.contents); 
        return *this; 
    }

    template<typename OtherTensor>
    requires (std::remove_reference_t<OtherTensor>::size == size)
    HeapTensor4D& operator=(const OtherTensor& other) {
        std::copy(other.contents.begin(), other.contents.end(), contents.begin());
        return *this;
    }

    HeapTensor4D& fill(T val) noexcept { 
        std::fill(contents.begin(), contents.end(), val); return *this; 
    }

    HeapTensor4D& operator+=(const HeapTensor4D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] += rhs.contents[i]; 
        return *this; 
    }

    HeapTensor4D& operator*=(const HeapTensor4D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= rhs.contents[i]; 
        return *this; 
    }

    HeapTensor4D& operator*=(T val) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= val; 
        return *this; 
    }

    constexpr T& operator()(size_t w, size_t x, size_t y, size_t z) { 
        return contents[w * size_x * size_y * size_z + x * size_y * size_z + y * size_z + z]; 
    }
    constexpr const T& operator()(size_t w, size_t x, size_t y, size_t z) const { 
        return contents[w * size_x * size_y * size_z + x * size_y * size_z + y * size_z + z]; 
    }

    constexpr T& operator[](size_t id) { 
        return contents[id]; 
    }

    constexpr const T& operator[](size_t id) const { 
        return contents[id]; 
    }

    constexpr T* data() noexcept { return contents.data(); }
    constexpr const T* data() const noexcept { return contents.data(); }

    constexpr auto begin() noexcept { return contents.begin(); }
    constexpr auto begin() const noexcept { return contents.cbegin(); }

    constexpr auto end() noexcept { return contents.end(); }
    constexpr auto end() const noexcept { return contents.cend(); }
};

}

#endif // HEAP_TENSOR_HPP