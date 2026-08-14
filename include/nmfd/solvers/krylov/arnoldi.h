#ifndef __NMFD_SOLVERS_KRYLOV_ARNOLDI_H__
#define __NMFD_SOLVERS_KRYLOV_ARNOLDI_H__

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "operator_apply.h"
#include "orthogonalization.h"

namespace nmfd
{
namespace solvers
{
namespace krylov
{

enum class arnoldi_status
{
    success,
    happy_breakdown,
    invalid_initial_vector,
    operator_failure,
    non_finite,
    capacity_exceeded,
    invalid_input
};

inline const char *arnoldi_status_name( arnoldi_status status )
{
    switch ( status )
    {
    case arnoldi_status::success:
        return "success";
    case arnoldi_status::happy_breakdown:
        return "happy_breakdown";
    case arnoldi_status::invalid_initial_vector:
        return "invalid_initial_vector";
    case arnoldi_status::operator_failure:
        return "operator_failure";
    case arnoldi_status::non_finite:
        return "non_finite";
    case arnoldi_status::capacity_exceeded:
        return "capacity_exceeded";
    case arnoldi_status::invalid_input:
        return "invalid_input";
    }
    return "unknown";
}

template <class Norm>
struct arnoldi_step_result
{
    arnoldi_status                 status      = arnoldi_status::invalid_input;
    std::size_t                    step        = 0;
    Norm                           subdiagonal = Norm{};
    orthogonalization_result<Norm> orthogonalization;

    bool succeeded() const
    {
        return status == arnoldi_status::success || status == arnoldi_status::happy_breakdown;
    }
};

template <class Norm>
struct arnoldi_factorization_result
{
    arnoldi_status            status          = arnoldi_status::invalid_input;
    std::size_t               completed_steps = 0;
    std::size_t               operator_calls  = 0;
    Norm                      initial_norm    = Norm{};
    arnoldi_step_result<Norm> last_step;

    bool succeeded() const
    {
        return status == arnoldi_status::success || status == arnoldi_status::happy_breakdown;
    }
};

template <class VectorSpace>
arnoldi_status initialize_arnoldi_basis(
    const VectorSpace &vector_space, const typename VectorSpace::vector_type &initial_vector,
    typename VectorSpace::multivector_type &basis, std::size_t basis_capacity,
    typename VectorSpace::vector_type &normalized_vector, typename VectorSpace::norm_type &initial_norm,
    typename VectorSpace::norm_type minimum_norm = static_cast<typename VectorSpace::norm_type>( 10 ) *
                                                   std::numeric_limits<typename VectorSpace::norm_type>::epsilon()
)
{
    using scalar_type  = typename VectorSpace::scalar_type;
    using norm_type    = typename VectorSpace::norm_type;
    using ordinal_type = typename VectorSpace::ordinal_type;

    if ( basis_capacity == 0 )
        return arnoldi_status::capacity_exceeded;

    vector_space.assign( initial_vector, normalized_vector );
    initial_norm = vector_space.norm( normalized_vector );
    if ( !std::isfinite( initial_norm ) )
        return arnoldi_status::non_finite;
    if ( !( initial_norm > minimum_norm ) )
        return arnoldi_status::invalid_initial_vector;

    vector_space.scale( scalar_type( norm_type( 1 ) / initial_norm ), normalized_vector );
    vector_space.assign( normalized_vector, basis, static_cast<ordinal_type>( basis_capacity ), ordinal_type( 0 ) );
    return arnoldi_status::success;
}

template <class VectorSpace, class ApplyOperation, class HessenbergMatrix>
arnoldi_step_result<typename VectorSpace::norm_type> arnoldi_step(
    const VectorSpace &vector_space, ApplyOperation &&apply_operation, typename VectorSpace::multivector_type &basis,
    std::size_t basis_capacity, std::size_t step, typename VectorSpace::vector_type &basis_vector,
    typename VectorSpace::vector_type &candidate, typename VectorSpace::scalar_type *coefficients,
    typename VectorSpace::scalar_type *pass_coefficients, HessenbergMatrix &hessenberg,
    const orthogonalization_options<typename VectorSpace::norm_type> &options = {}
)
{
    using scalar_type  = typename VectorSpace::scalar_type;
    using norm_type    = typename VectorSpace::norm_type;
    using ordinal_type = typename VectorSpace::ordinal_type;

    arnoldi_step_result<norm_type> result;
    result.step = step;
    if ( step + 1 >= basis_capacity )
    {
        result.status = arnoldi_status::capacity_exceeded;
        return result;
    }

    vector_space.assign(
        basis, static_cast<ordinal_type>( basis_capacity ), static_cast<ordinal_type>( step ), basis_vector
    );
    if ( !invoke_status(
             std::forward<ApplyOperation>( apply_operation ),
             static_cast<const typename VectorSpace::vector_type &>( basis_vector ), candidate
         ) )
    {
        result.status = arnoldi_status::operator_failure;
        return result;
    }

    result.orthogonalization = orthogonalize(
        vector_space, basis, basis_capacity, step + 1, candidate, coefficients, pass_coefficients, options
    );
    for ( std::size_t row = 0; row <= step; ++row )
        hessenberg( row, step ) = coefficients[row];

    switch ( result.orthogonalization.status )
    {
    case orthogonalization_status::invalid_input:
        result.status = arnoldi_status::invalid_input;
        return result;
    case orthogonalization_status::non_finite:
        result.status = arnoldi_status::non_finite;
        return result;
    case orthogonalization_status::happy_breakdown:
        result.subdiagonal           = result.orthogonalization.norm_after;
        hessenberg( step + 1, step ) = static_cast<scalar_type>( result.subdiagonal );
        result.status                = arnoldi_status::happy_breakdown;
        return result;
    case orthogonalization_status::success:
        break;
    }

    result.subdiagonal           = result.orthogonalization.norm_after;
    hessenberg( step + 1, step ) = static_cast<scalar_type>( result.subdiagonal );
    vector_space.scale( scalar_type( norm_type( 1 ) / result.subdiagonal ), candidate );
    vector_space.assign(
        candidate, basis, static_cast<ordinal_type>( basis_capacity ), static_cast<ordinal_type>( step + 1 )
    );
    result.status = arnoldi_status::success;
    return result;
}

template <class VectorSpace, class ApplyOperation, class HessenbergMatrix>
arnoldi_factorization_result<typename VectorSpace::norm_type> build_arnoldi_factorization(
    const VectorSpace &vector_space, ApplyOperation &&apply_operation,
    const typename VectorSpace::vector_type &initial_vector, typename VectorSpace::multivector_type &basis,
    std::size_t basis_capacity, std::size_t requested_steps, typename VectorSpace::vector_type &basis_vector,
    typename VectorSpace::vector_type &candidate, typename VectorSpace::scalar_type *coefficients,
    typename VectorSpace::scalar_type *pass_coefficients, HessenbergMatrix &hessenberg,
    const orthogonalization_options<typename VectorSpace::norm_type> &options = {}
)
{
    using norm_type = typename VectorSpace::norm_type;

    arnoldi_factorization_result<norm_type> result;
    if ( requested_steps + 1 > basis_capacity )
    {
        result.status = arnoldi_status::capacity_exceeded;
        return result;
    }

    result.status = initialize_arnoldi_basis(
        vector_space, initial_vector, basis, basis_capacity, basis_vector, result.initial_norm
    );
    if ( result.status != arnoldi_status::success )
        return result;

    for ( std::size_t step = 0; step < requested_steps; ++step )
    {
        result.last_step = arnoldi_step(
            vector_space, apply_operation, basis, basis_capacity, step, basis_vector, candidate, coefficients,
            pass_coefficients, hessenberg, options
        );
        ++result.operator_calls;
        if ( !result.last_step.succeeded() )
        {
            result.status = result.last_step.status;
            return result;
        }

        ++result.completed_steps;
        if ( result.last_step.status == arnoldi_status::happy_breakdown )
        {
            result.status = arnoldi_status::happy_breakdown;
            return result;
        }
    }

    result.status = arnoldi_status::success;
    return result;
}

} // namespace krylov
} // namespace solvers
} // namespace nmfd

#endif
