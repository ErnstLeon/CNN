#ifndef CONCEPTS_HPP
#define CONCEPTS_HPP

template<typename F, typename Arg, typename Ret>
concept callable_with = 
    std::invocable<F, Arg> &&
    std::convertible_to<std::invoke_result_t<F, Arg>, Ret>;

template<typename F, typename Arg, typename Ret>
concept derivative_callable_with = requires(F f, Arg in)
{
    { f.derivative(in) } -> std::convertible_to<Ret>;
};
#endif // CONCEPTS_HPP