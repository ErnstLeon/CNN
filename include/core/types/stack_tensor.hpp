#ifndef STACK_TENSOR_HPP
#define STACK_TENSOR_HPP

#include <limits>
#include <random>
#include <vector>

#include "../helper/loop_unroll.hpp"

namespace CNN
{

template<size_t Size, typename T = float>
struct StackTensor1D {

    using type = T;
    static constexpr size_t size = Size;

    std::array<T, Size> contents{};

    constexpr StackTensor1D() = default;
    constexpr StackTensor1D(const std::array<T, Size>& input) : contents(input) {}
    constexpr StackTensor1D(const StackTensor1D& other) = default;
    constexpr StackTensor1D(StackTensor1D&& other) = default;
    constexpr StackTensor1D& operator=(const StackTensor1D& other) = default;
    constexpr StackTensor1D& operator=(StackTensor1D&& other) = default;

    template<typename OtherTensor>
    requires (std::remove_reference_t<OtherTensor>::size == size)
    StackTensor1D& operator=(const OtherTensor& other) {
        std::copy(other.contents.begin(), other.contents.end(), contents.begin());
        return *this;
    }

    StackTensor1D& operator=(const std::vector<T> & other) {
        if (other.size() != size) throw std::runtime_error("Size mismatch");
        std::copy(other.begin(), other.end(), contents.begin());
        return *this;
    }

    constexpr void fill(T val) noexcept {
        contents.fill(val); 
    }

    constexpr StackTensor1D& operator+=(const StackTensor1D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] += rhs.contents[i]; 
        return *this; 
    }

    constexpr StackTensor1D& operator*=(const StackTensor1D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= rhs.contents[i]; 
        return *this; 
    }

    constexpr StackTensor1D& operator*=(T val) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= val; 
        return *this; 
    }

    constexpr T& operator[](size_t id) { 
        return contents[id]; 
    }

    constexpr const T& operator[](size_t id) const { 
        return contents[id]; 
    }

    constexpr T& operator()(size_t x) { 
        return contents[x]; 
    }

    constexpr const T& operator()(size_t x) const { 
        return contents[x]; 
    }

    constexpr T* data() noexcept { return contents.data(); }
    constexpr const T* data() const noexcept { return contents.data(); }

    constexpr auto begin() noexcept { return contents.begin(); }
    constexpr auto begin() const noexcept { return contents.cbegin(); }

    constexpr auto end() noexcept { return contents.end(); }
    constexpr auto end() const noexcept { return contents.cend(); }
};

template<size_t SizeX, size_t SizeY, typename T = float>
struct StackTensor2D {

    using type = T;
    static constexpr size_t size_x = SizeX;
    static constexpr size_t size_y = SizeY;
    static constexpr size_t size = SizeX * SizeY;

    std::array<T, size> contents{};

    constexpr StackTensor2D() = default;
    constexpr StackTensor2D(const std::array<T, size>& input) : contents(input) {}
    constexpr StackTensor2D(const StackTensor2D& other) = default;
    constexpr StackTensor2D(StackTensor2D&& other) = default;
    constexpr StackTensor2D& operator=(const StackTensor2D& other) = default;
    constexpr StackTensor2D& operator=(StackTensor2D&& other) = default;

    template<typename OtherTensor>
    requires (std::remove_reference_t<OtherTensor>::size == size)
    StackTensor2D& operator=(const OtherTensor& other) {
        std::copy(other.contents.begin(), other.contents.end(), contents.begin());
        return *this;
    }

    StackTensor2D& operator=(const std::vector<T> & other) {
        if (other.size() != size) throw std::runtime_error("Size mismatch");
        std::copy(other.begin(), other.end(), contents.begin());
        return *this;
    }

    constexpr void fill(T val) noexcept { 
        contents.fill(val); 
    }

    constexpr StackTensor2D& operator+=(const StackTensor2D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] += rhs.contents[i]; 
        return *this; 
    }

    constexpr StackTensor2D& operator*=(const StackTensor2D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= rhs.contents[i]; 
        return *this; 
    }

    constexpr StackTensor2D& operator*=(T val) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= val; 
        return *this; 
    }

    constexpr T& operator[](size_t id) { 
        return contents[id]; 
    }

    constexpr const T& operator[](size_t id) const { 
        return contents[id]; 
    }

    constexpr T& operator()(size_t x, size_t y) { 
        return contents[x * size_y + y]; 
    }

    constexpr const T& operator()(size_t x, size_t y) const { 
        return contents[x * size_y + y]; 
    }

    constexpr T* data() noexcept { return contents.data(); }
    constexpr const T* data() const noexcept { return contents.data(); }

    constexpr auto begin() noexcept { return contents.begin(); }
    constexpr auto begin() const noexcept { return contents.cbegin(); }

    constexpr auto end() noexcept { return contents.end(); }
    constexpr auto end() const noexcept { return contents.cend(); }
};

template<size_t SizeX, size_t SizeY, size_t SizeZ, typename T = float>
struct StackTensor3D {

    using type = T;
    static constexpr size_t size_x = SizeX;
    static constexpr size_t size_y = SizeY;
    static constexpr size_t size_z = SizeZ;
    static constexpr size_t size = SizeX * SizeY * SizeZ;

    std::array<T, size> contents{};

    constexpr StackTensor3D() = default;
    constexpr StackTensor3D(const std::array<T, size>& input) : contents(input) {}
    constexpr StackTensor3D(const StackTensor3D& other) = default;
    constexpr StackTensor3D(StackTensor3D&& other) = default;
    constexpr StackTensor3D& operator=(const StackTensor3D& other) = default;
    constexpr StackTensor3D& operator=(StackTensor3D&& other) = default;

    template<typename OtherTensor>
    requires (std::remove_reference_t<OtherTensor>::size == size)
    StackTensor3D& operator=(const OtherTensor& other) {
        std::copy(other.contents.begin(), other.contents.end(), contents.begin());
        return *this;
    }

    StackTensor3D& operator=(const std::vector<T> & other) {
        if (other.size() != size) throw std::runtime_error("Size mismatch");
        std::copy(other.begin(), other.end(), contents.begin());
        return *this;
    }

    constexpr void fill(T val) noexcept { 
        contents.fill(val); 
    }

    constexpr StackTensor3D& operator+=(const StackTensor3D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] += rhs.contents[i]; 
        return *this; 
    }

    constexpr StackTensor3D& operator*=(const StackTensor3D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= rhs.contents[i]; 
        return *this; 
    }

    constexpr StackTensor3D& operator*=(T val) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= val; 
        return *this; 
    }
    
    constexpr T& operator[](size_t id) { 
        return contents[id]; 
    }

    constexpr const T& operator[](size_t id) const { 
        return contents[id]; 
    }

    constexpr T& operator()(size_t x, size_t y, size_t z) { 
        return contents[x * size_y * size_z + y * size_z + z]; 
    }

    constexpr const T& operator()(size_t x, size_t y, size_t z) const { 
        return contents[x * size_y * size_z + y * size_z + z]; 
    }

    constexpr T* data() noexcept { return contents.data(); }
    constexpr const T* data() const noexcept { return contents.data(); }

    constexpr auto begin() noexcept { return contents.begin(); }
    constexpr auto begin() const noexcept { return contents.cbegin(); }

    constexpr auto end() noexcept { return contents.end(); }
    constexpr auto end() const noexcept { return contents.cend(); }
};

template<size_t SizeW, size_t SizeX, size_t SizeY, size_t SizeZ, typename T = float>
struct StackTensor4D {

    using type = T;
    static constexpr size_t size_w = SizeW;
    static constexpr size_t size_x = SizeX;
    static constexpr size_t size_y = SizeY;
    static constexpr size_t size_z = SizeZ;
    static constexpr size_t size = SizeW * SizeX * SizeY * SizeZ;

    std::array<T, size> contents{};

    constexpr StackTensor4D() = default;
    constexpr StackTensor4D(const std::array<T, size>& input) : contents(input) {}
    constexpr StackTensor4D(const StackTensor4D& other) = default;
    constexpr StackTensor4D(StackTensor4D&& other) = default;
    constexpr StackTensor4D& operator=(const StackTensor4D& other) = default;
    constexpr StackTensor4D& operator=(StackTensor4D&& other) = default;

    template<typename OtherTensor>
    requires (std::remove_reference_t<OtherTensor>::size == size)
    StackTensor4D& operator=(const OtherTensor& other) {
        std::copy(other.contents.begin(), other.contents.end(), contents.begin());
        return *this;
    }

    StackTensor4D& operator=(const std::vector<T> & other) {
        if (other.size() != size) throw std::runtime_error("Size mismatch");
        std::copy(other.begin(), other.end(), contents.begin());
        return *this;
    }

    constexpr void fill(T val) noexcept { 
        contents.fill(val); 
    }

    constexpr StackTensor4D& operator+=(const StackTensor4D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] += rhs.contents[i]; 
        return *this; 
    }

    constexpr StackTensor4D& operator*=(const StackTensor4D& rhs) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= rhs.contents[i]; 
        return *this; 
    }

    constexpr StackTensor4D& operator*=(T val) noexcept { 
        UNROLL_PRAGMA
        for (size_t i = 0; i < size; ++i) contents[i] *= val; 
        return *this; 
    }

    constexpr T& operator[](size_t id) { 
        return contents[id]; 
    }

    constexpr const T& operator[](size_t id) const { 
        return contents[id]; 
    }

    constexpr T& operator()(size_t w, size_t x, size_t y, size_t z) { 
        return contents[w * size_x * size_y * size_z + x * size_y * size_z + y * size_z + z]; 
    }

    constexpr const T& operator()(size_t w, size_t x, size_t y, size_t z) const { 
        return contents[w * size_x * size_y * size_z + x * size_y * size_z + y * size_z + z]; 
    }

    constexpr T* data() noexcept { return contents.data(); }
    constexpr const T* data() const noexcept { return contents.data(); }

    constexpr auto begin() noexcept { return contents.begin(); }
    constexpr auto begin() const noexcept { return contents.cbegin(); }

    constexpr auto end() noexcept { return contents.end(); }
    constexpr auto end() const noexcept { return contents.cend(); }
};

}

#endif // STACK_TENSOR_HPP