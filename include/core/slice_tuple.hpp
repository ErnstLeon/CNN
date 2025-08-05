#ifndef SLICE_TUPLE_HPP
#define SLICE_TUPLE_HPP

namespace CNN{

template<size_t START, typename Tuple, size_t... Ids>
constexpr auto slice_tuple_helper(Tuple&& tuple, std::index_sequence<Ids...>) {
    return std::make_tuple(std::get<START + Ids>(std::forward<Tuple>(tuple))...);
}

template<size_t START, size_t STOP, typename Tuple>
requires (START <= STOP && STOP < std::tuple_size_v<std::remove_reference_t<Tuple>>)
constexpr auto slice_tuple(Tuple&& tuple) {
    constexpr size_t N = STOP - START + 1;
    return slice_tuple_helper<START>(std::forward<Tuple>(tuple), std::make_index_sequence<N>{});
}

}

#endif // SLICE_TUPLE_HPP