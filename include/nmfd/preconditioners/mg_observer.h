#ifndef NMFD_PRECONDITIONERS_MG_OBSERVER_H
#define NMFD_PRECONDITIONERS_MG_OBSERVER_H

#include <cstddef>

namespace nmfd
{
namespace preconditioners
{

enum class mg_cycle_event
{
    cycle_begin_rhs,
    initial_residual,
    pre_smoothing_residual,
    restricted_rhs,
    coarse_solution,
    prolongated_correction,
    post_smoothing_residual,
    cycle_end_solution
};

struct null_mg_observer
{
    template <class VectorSpace, class Vector>
    void on_event( std::size_t, mg_cycle_event, const VectorSpace &, const Vector & ) const noexcept
    {
    }
};

} // namespace preconditioners
} // namespace nmfd

#endif
