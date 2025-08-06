#ifndef SPLIT_TUPLE_HPP
#define SPLIT_TUPLE_HPP

namespace CNN
{

template<size_t R_START, typename Tuple, size_t... Ls, size_t... Rs>
requires (sizeof...(Ls) + sizeof...(Rs) == std::tuple_size_v<std::remove_reference_t<Tuple>>)
inline constexpr auto split_tuple_helper(Tuple&& tuple, std::index_sequence<Ls...>, std::index_sequence<Rs...>) 
{
    return std::make_tuple(
        std::make_tuple(std::move(std::get<Ls>(tuple))...),
        std::make_tuple(std::move(std::get<R_START + Rs>(tuple))...)
    );
}

template<size_t SPLIT_ID, typename Tuple>
requires (SPLIT_ID < std::tuple_size_v<std::remove_reference_t<Tuple>>)
inline constexpr auto split_tuple(Tuple&& tuple) {

    constexpr size_t SIZE = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    constexpr size_t SIZE_LEFT = SPLIT_ID + 1;
    constexpr size_t SIZE_RIGHT = SIZE - SIZE_LEFT;

    using IdL = std::make_index_sequence<SIZE_LEFT>;
    using IdR = std::make_index_sequence<SIZE_RIGHT>;

    if constexpr (SIZE_LEFT == SIZE){
        return std::make_tuple(std::forward<Tuple>(tuple), std::tuple<>{});
    }
    else{
        return split_tuple_helper<SIZE_LEFT>(std::forward<Tuple>(tuple), IdL{}, IdR{});
    }
}

}

#endif // SPLIT_TUPLE_HPP