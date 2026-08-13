// Copyright © 2020-2026 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of NMFD.

// NMFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// NMFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with NMFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __NMFD_ALGO_UTILS_HIERARCHY_FACTORY_H__
#define __NMFD_ALGO_UTILS_HIERARCHY_FACTORY_H__

#include <memory>
#include <type_traits>
#include <utility>

#include "algo_utils_hierarchy.h"

namespace nmfd
{
namespace detail
{

template<class...>
struct algo_utils_hierarchy_factory_dependent_false : std::false_type
{
};

template<class Algorithm, class Backend, class VectorSpace>
typename algo_utils_hierarchy<Algorithm>::type make_algo_utils_hierarchy(
    Backend& backend,
    std::shared_ptr<VectorSpace> vector_space)
{
    using utils_type = typename algo_utils_hierarchy<Algorithm>::type;

    if constexpr(std::is_constructible_v<
                     utils_type,
                     Backend&,
                     std::shared_ptr<VectorSpace>>)
    {
        return utils_type(backend, std::move(vector_space));
    }
    else if constexpr(std::is_constructible_v<
                          utils_type,
                          std::shared_ptr<VectorSpace>>)
    {
        return utils_type(std::move(vector_space));
    }
    else
    {
        static_assert(
            algo_utils_hierarchy_factory_dependent_false<
                Algorithm,
                Backend,
                VectorSpace>::value,
            "Algorithm utilities cannot be constructed from a backend and "
            "vector space; provide the utility hierarchy explicitly");
    }
}

} // namespace detail
} // namespace nmfd

#endif
