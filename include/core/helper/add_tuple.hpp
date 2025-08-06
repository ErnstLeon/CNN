#ifndef ADD_TUPLE_HPP
#define ADD_TUPLE_HPP

namespace CNN
{

template<typename Tuple, typename Elem, size_t... Ids>
inline constexpr auto add_tuple_end_helper(Tuple && tuple, Elem && elem, std::index_sequence<Ids...>) {
    return std::make_tuple(std::get<Ids>(std::forward<Tuple>(tuple))..., std::forward<Elem>(elem));
}

template<typename Tuple, typename Elem>
inline constexpr auto add_tuple_end(Tuple && tuple, Elem && elem) {
    return add_tuple_end_helper(std::forward<Tuple>(tuple), std::forward<Elem>(elem), 
            std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

template<typename Tuple, typename Elem, size_t... Ids>
inline constexpr auto add_tuple_begin_helper(Tuple && tuple, Elem && elem, std::index_sequence<Ids...>) {
    return std::make_tuple(std::forward<Elem>(elem), std::get<Ids>(std::forward<Tuple>(tuple))...);
}

template<typename Tuple, typename Elem>
inline constexpr auto add_tuple_begin(Tuple && tuple, Elem && elem) {
    return add_tuple_begin_helper(std::forward<Tuple>(tuple), std::forward<Elem>(elem), 
            std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

}

#endif // ADD_TUPLE_HPP