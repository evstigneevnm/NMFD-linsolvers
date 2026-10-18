#ifndef __NMFD_SOLVERS_KRYLOV_ORTHOGONALIZATION_H__
#define __NMFD_SOLVERS_KRYLOV_ORTHOGONALIZATION_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace nmfd
{
namespace solvers
{
namespace krylov
{

enum class orthogonalization_method
{
    modified_gram_schmidt,
    classical_gram_schmidt
};

enum class reorthogonalization_policy
{
    none,
    always,
    dgks
};

enum class orthogonalization_status
{
    success,
    happy_breakdown,
    non_finite,
    invalid_input
};

inline const char *orthogonalization_method_name( orthogonalization_method method )
{
    switch ( method )
    {
    case orthogonalization_method::modified_gram_schmidt:
        return "mgs";
    case orthogonalization_method::classical_gram_schmidt:
        return "cgs";
    }
    return "unknown";
}

inline const char *reorthogonalization_policy_name( reorthogonalization_policy policy )
{
    switch ( policy )
    {
    case reorthogonalization_policy::none:
        return "none";
    case reorthogonalization_policy::always:
        return "always";
    case reorthogonalization_policy::dgks:
        return "dgks";
    }
    return "unknown";
}

inline orthogonalization_method parse_orthogonalization_method( const std::string &value )
{
    if ( value == "mgs" || value == "modified_gram_schmidt" )
        return orthogonalization_method::modified_gram_schmidt;
    if ( value == "cgs" || value == "classical_gram_schmidt" )
        return orthogonalization_method::classical_gram_schmidt;
    throw std::invalid_argument( "unknown orthogonalization method: " + value );
}

inline reorthogonalization_policy parse_reorthogonalization_policy( const std::string &value )
{
    if ( value == "none" )
        return reorthogonalization_policy::none;
    if ( value == "always" )
        return reorthogonalization_policy::always;
    if ( value == "dgks" || value == "selective" )
        return reorthogonalization_policy::dgks;
    throw std::invalid_argument( "unknown reorthogonalization policy: " + value );
}

template <class Norm>
struct orthogonalization_options
{
    orthogonalization_method   method              = orthogonalization_method::modified_gram_schmidt;
    reorthogonalization_policy reorthogonalization = reorthogonalization_policy::dgks;
    Norm                       dgks_eta            = static_cast<Norm>( 0.717 );
    Norm                       breakdown_absolute  = Norm{};
    Norm                       breakdown_relative  = static_cast<Norm>( 10 ) * std::numeric_limits<Norm>::epsilon();
    unsigned                   max_passes          = 2;
    bool                       compute_orthogonality_error = false;
};

template <class Norm>
struct orthogonalization_result
{
    orthogonalization_status status                     = orthogonalization_status::invalid_input;
    Norm                     norm_before                = Norm{};
    Norm                     norm_after                 = Norm{};
    Norm                     max_relative_inner_product = Norm{};
    unsigned                 passes                     = 0;

    bool succeeded() const
    {
        return status == orthogonalization_status::success || status == orthogonalization_status::happy_breakdown;
    }

    bool breakdown() const
    {
        return status == orthogonalization_status::happy_breakdown;
    }
};

namespace detail
{

template <class Scalar>
auto scalar_abs( const Scalar &value )
{
    using std::abs;
    return abs( value );
}

template <class VectorSpace>
void modified_gram_schmidt_pass(
    const VectorSpace &vector_space, const typename VectorSpace::multivector_type &basis, std::size_t basis_capacity,
    std::size_t basis_size, typename VectorSpace::vector_type &candidate,
    typename VectorSpace::scalar_type *coefficients
)
{
    using scalar_type  = typename VectorSpace::scalar_type;
    using ordinal_type = typename VectorSpace::ordinal_type;

    for ( std::size_t index = 0; index < basis_size; ++index )
    {
        const auto coefficient = vector_space.scalar_prod(
            basis, static_cast<ordinal_type>( basis_capacity ), static_cast<ordinal_type>( index ), candidate
        );
        coefficients[index] += coefficient;
        vector_space.add_lin_comb(
            -coefficient, basis, static_cast<ordinal_type>( basis_capacity ), static_cast<ordinal_type>( index ),
            scalar_type( 1 ), candidate
        );
    }
}

template <class VectorSpace>
void classical_gram_schmidt_pass(
    const VectorSpace &vector_space, const typename VectorSpace::multivector_type &basis, std::size_t basis_capacity,
    std::size_t basis_size, typename VectorSpace::vector_type &candidate,
    typename VectorSpace::scalar_type *coefficients, typename VectorSpace::scalar_type *pass_coefficients
)
{
    using scalar_type  = typename VectorSpace::scalar_type;
    using ordinal_type = typename VectorSpace::ordinal_type;

    for ( std::size_t index = 0; index < basis_size; ++index )
    {
        pass_coefficients[index] = vector_space.scalar_prod(
            basis, static_cast<ordinal_type>( basis_capacity ), static_cast<ordinal_type>( index ), candidate
        );
        coefficients[index] += pass_coefficients[index];
    }
    for ( std::size_t index = 0; index < basis_size; ++index )
    {
        vector_space.add_lin_comb(
            -pass_coefficients[index], basis, static_cast<ordinal_type>( basis_capacity ),
            static_cast<ordinal_type>( index ), scalar_type( 1 ), candidate
        );
    }
}

} // namespace detail

template <class VectorSpace>
orthogonalization_result<typename VectorSpace::norm_type> orthogonalize(
    const VectorSpace &vector_space, const typename VectorSpace::multivector_type &basis, std::size_t basis_capacity,
    std::size_t basis_size, typename VectorSpace::vector_type &candidate,
    typename VectorSpace::scalar_type *coefficients, typename VectorSpace::scalar_type *pass_coefficients,
    const orthogonalization_options<typename VectorSpace::norm_type> &options = {}
)
{
    using scalar_type  = typename VectorSpace::scalar_type;
    using norm_type    = typename VectorSpace::norm_type;
    using ordinal_type = typename VectorSpace::ordinal_type;

    orthogonalization_result<norm_type> result;
    if ( basis_size > basis_capacity || ( basis_size > 0 && coefficients == nullptr ) ||
         ( basis_size > 0 && options.method == orthogonalization_method::classical_gram_schmidt &&
           pass_coefficients == nullptr ) ||
         options.max_passes == 0 || !( options.dgks_eta > norm_type{} ) || !( options.dgks_eta < norm_type( 1 ) ) )
    {
        result.status = orthogonalization_status::invalid_input;
        return result;
    }

    std::fill( coefficients, coefficients + basis_size, scalar_type{} );
    result.norm_before = vector_space.norm( candidate );
    result.norm_after  = result.norm_before;
    if ( !std::isfinite( result.norm_before ) )
    {
        result.status = orthogonalization_status::non_finite;
        return result;
    }

    norm_type norm_before_pass = result.norm_before;
    bool      run_another_pass = basis_size > 0;
    while ( run_another_pass && result.passes < options.max_passes )
    {
        if ( options.method == orthogonalization_method::modified_gram_schmidt )
        {
            detail::modified_gram_schmidt_pass(
                vector_space, basis, basis_capacity, basis_size, candidate, coefficients
            );
        }
        else
        {
            detail::classical_gram_schmidt_pass(
                vector_space, basis, basis_capacity, basis_size, candidate, coefficients, pass_coefficients
            );
        }

        ++result.passes;
        result.norm_after = vector_space.norm( candidate );
        if ( !std::isfinite( result.norm_after ) )
        {
            result.status = orthogonalization_status::non_finite;
            return result;
        }

        switch ( options.reorthogonalization )
        {
        case reorthogonalization_policy::none:
            run_another_pass = false;
            break;
        case reorthogonalization_policy::always:
            run_another_pass = result.passes < options.max_passes;
            break;
        case reorthogonalization_policy::dgks:
            run_another_pass =
                result.passes < options.max_passes && result.norm_after <= options.dgks_eta * norm_before_pass;
            break;
        }
        norm_before_pass = result.norm_after;
    }

    const norm_type breakdown_threshold =
        std::max( options.breakdown_absolute, options.breakdown_relative * result.norm_before );
    result.status = result.norm_after <= breakdown_threshold ? orthogonalization_status::happy_breakdown
                                                             : orthogonalization_status::success;

    if ( options.compute_orthogonality_error && basis_size > 0 && result.norm_after > norm_type{} )
    {
        for ( std::size_t index = 0; index < basis_size; ++index )
        {
            const auto value = vector_space.scalar_prod(
                basis, static_cast<ordinal_type>( basis_capacity ), static_cast<ordinal_type>( index ), candidate
            );
            result.max_relative_inner_product = std::max(
                result.max_relative_inner_product,
                static_cast<norm_type>( detail::scalar_abs( value ) ) / result.norm_after
            );
        }
    }

    return result;
}

} // namespace krylov
} // namespace solvers
} // namespace nmfd

#endif
