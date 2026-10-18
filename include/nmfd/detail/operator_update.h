#ifndef NMFD_DETAIL_OPERATOR_UPDATE_H
#define NMFD_DETAIL_OPERATOR_UPDATE_H

#include <memory>
#include <type_traits>
#include <utility>

namespace nmfd
{
namespace detail
{

template <class Algorithm, class Operator, class = void>
struct has_update_operator : std::false_type
{
};

template <class Algorithm, class Operator>
struct has_update_operator<
    Algorithm, Operator,
    std::void_t<
        decltype( std::declval<Algorithm &>().update_operator( std::declval<std::shared_ptr<const Operator>>() ) )>>
    : std::true_type
{
};

template <class Algorithm, class Operator>
void update_operator_or_set( Algorithm &algorithm, std::shared_ptr<const Operator> op )
{
    if constexpr ( has_update_operator<Algorithm, Operator>::value )
        algorithm.update_operator( std::move( op ) );
    else
        algorithm.set_operator( std::move( op ) );
}

} // namespace detail
} // namespace nmfd

#endif
