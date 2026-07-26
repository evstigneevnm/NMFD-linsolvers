#ifndef __NMFD_SOLVERS_KRYLOV_OPERATOR_APPLY_H__
#define __NMFD_SOLVERS_KRYLOV_OPERATOR_APPLY_H__

#include <functional>
#include <type_traits>
#include <utility>

namespace nmfd
{
namespace solvers
{
namespace krylov
{

template<class Function, class... Args>
bool invoke_status(Function&& function, Args&&... args)
{
    using result_type = std::invoke_result_t<Function, Args...>;
    if constexpr(std::is_void<result_type>::value)
    {
        std::invoke(
            std::forward<Function>(function),
            std::forward<Args>(args)...);
        return true;
    }
    else
    {
        static_assert(
            std::is_convertible<result_type, bool>::value,
            "Krylov operation must return void or a bool-convertible status");
        return static_cast<bool>(std::invoke(
            std::forward<Function>(function),
            std::forward<Args>(args)...));
    }
}

template<class Operator, class InputVector, class OutputVector>
bool apply_operator(
    const Operator& operation,
    const InputVector& input,
    OutputVector& output)
{
    return invoke_status(
        [&operation](const InputVector& source, OutputVector& destination)
            -> decltype(auto)
        {
            return operation.apply(source, destination);
        },
        input,
        output);
}

template<class Operator, class InputVector, class OutputVector>
bool solve_operator(
    const Operator& operation,
    const InputVector& input,
    OutputVector& output)
{
    return invoke_status(
        [&operation](const InputVector& source, OutputVector& destination)
            -> decltype(auto)
        {
            return operation.solve(source, destination);
        },
        input,
        output);
}

template<class Operation, class Vector>
bool apply_in_place(const Operation& operation, Vector& vector)
{
    return invoke_status(
        [&operation](Vector& value) -> decltype(auto)
        {
            return operation.apply(value);
        },
        vector);
}

} // namespace krylov
} // namespace solvers
} // namespace nmfd

#endif
