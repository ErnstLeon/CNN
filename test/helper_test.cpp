#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "core.hpp"

TEST(TupleUtilityTest, AddTupleBegin_TypeCheck){

    std::tuple<double, int> original{2, 3};

    auto result = CNN::add_tuple_begin(original, static_cast<float>(1));

    using ResultType = decltype(result);

    static_assert(std::tuple_size_v<ResultType> == 3, "Tuple size after begin addition is incorrect");
    static_assert(std::is_same_v<std::tuple_element_t<0, ResultType>, float>, "First element type should be float");
    static_assert(std::is_same_v<std::tuple_element_t<1, ResultType>, double>, "Second element type should be double");
    static_assert(std::is_same_v<std::tuple_element_t<2, ResultType>, int>, "Third element type should be int");
}

TEST(TupleUtilityTest, AddTupleBegin_ValueCheck){

    std::tuple<double, int> original{2, 3};

    auto result = CNN::add_tuple_begin(original, static_cast<float>(1));

    EXPECT_EQ(std::get<0>(result), static_cast<float>(1));
    EXPECT_EQ(std::get<1>(result), static_cast<double>(2));
    EXPECT_EQ(std::get<2>(result), static_cast<double>(3));
}

TEST(TupleUtilityTest, AddTupleEnd_TypeCheck){

    std::tuple<double, int> original{1, 2};

    auto result = CNN::add_tuple_end(original, static_cast<float>(3));

    using ResultType = decltype(result);

    static_assert(std::tuple_size_v<ResultType> == 3, "Tuple size after end addition is incorrect");
    static_assert(std::is_same_v<std::tuple_element_t<0, ResultType>, double>, "First element type should be float");
    static_assert(std::is_same_v<std::tuple_element_t<1, ResultType>, int>, "Second element type should be double");
    static_assert(std::is_same_v<std::tuple_element_t<2, ResultType>, float>, "Third element type should be int");
}

TEST(TupleUtilityTest, AddTupleEnd_ValueCheck){

    std::tuple<double, int> original{1, 2};

    auto result = CNN::add_tuple_end(original, static_cast<float>(3));

    EXPECT_EQ(std::get<0>(result), static_cast<double>(1));
    EXPECT_EQ(std::get<1>(result), static_cast<int>(2));
    EXPECT_EQ(std::get<2>(result), static_cast<float>(3));
}

TEST(TupleUtilityTest, SplitTuple_TypeCheck){

    std::tuple<double, int, std::string> original{1, 2, "test"};

    auto [left, right] = CNN::split_tuple<1>(original);

    using LeftType = decltype(left);
    using RightType = decltype(right);

    static_assert(std::tuple_size_v<LeftType> == 2, "Left tuple size after split is incorrect");
    static_assert(std::tuple_size_v<RightType> == 1, "Right tuple size after split is incorrect");

    static_assert(std::is_same_v<std::tuple_element_t<0, LeftType>, double>, "First element type should be float");
    static_assert(std::is_same_v<std::tuple_element_t<1, LeftType>, int>, "Second element type should be double");
    static_assert(std::is_same_v<std::tuple_element_t<0, RightType>, std::string>, "Third element type should be int");
}

TEST(TupleUtilityTest, SplitTupleEnd_ValueCheck){

std::tuple<double, int, std::string> original{1, 2, "test"};

    auto [left, right] = CNN::split_tuple<1>(original);

    using LeftType = decltype(left);
    using RightType = decltype(right);

    static_assert(std::tuple_size_v<LeftType> == 2, "Left tuple size after split is incorrect");
    static_assert(std::tuple_size_v<RightType> == 1, "Right tuple size after split is incorrect");

    EXPECT_EQ(std::get<0>(left), static_cast<double>(1));
    EXPECT_EQ(std::get<1>(left), static_cast<int>(2));
    EXPECT_EQ(std::get<0>(right), std::string("test"));
}