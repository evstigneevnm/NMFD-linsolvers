#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include <nmfd/solvers/krylov/orthogonalization.h>

#include "common/cpu_reference_vector_space.h"
#include "common/test_harness.h"

namespace
{

using nmfd::solvers::krylov::orthogonalization_method;
using nmfd::solvers::krylov::orthogonalization_options;
using nmfd::solvers::krylov::orthogonalization_status;
using nmfd::solvers::krylov::reorthogonalization_policy;

template <class Scalar>
void prepare_basis(
    const nmfd::tests::cpu_reference_vector_space<Scalar>                      &vector_space,
    typename nmfd::tests::cpu_reference_vector_space<Scalar>::multivector_type &basis
)
{
    vector_space.init_multivector( basis, 3 );
    vector_space.start_use_multivector( basis, 3 );
    basis[0][0] = Scalar( 1 );
    basis[1][1] = Scalar( 1 );
    basis[2][2] = Scalar( 1 );
}

void test_mgs( nmfd::tests::test_harness &test )
{
    using vector_space_type = nmfd::tests::cpu_reference_vector_space<double>;
    vector_space_type                   vector_space( 4 );
    vector_space_type::multivector_type basis;
    prepare_basis( vector_space, basis );
    vector_space_type::vector_type candidate{ 1.0, 2.0, 3.0, 4.0 };
    double                         coefficients[3]{};
    double                         pass_coefficients[3]{};

    orthogonalization_options<double> options;
    options.method                      = orthogonalization_method::modified_gram_schmidt;
    options.reorthogonalization         = reorthogonalization_policy::none;
    options.compute_orthogonality_error = true;
    const auto result                   = nmfd::solvers::krylov::orthogonalize(
        vector_space, basis, 3, 2, candidate, coefficients, pass_coefficients, options
    );

    test.require( result.status == orthogonalization_status::success, "MGS succeeds" );
    test.require( result.passes == 1, "MGS uses one pass" );
    test.require_close( coefficients[0], 1.0, 1e-14, "MGS coefficient 0" );
    test.require_close( coefficients[1], 2.0, 1e-14, "MGS coefficient 1" );
    test.require_close( candidate[0], 0.0, 1e-14, "MGS removes basis 0" );
    test.require_close( candidate[1], 0.0, 1e-14, "MGS removes basis 1" );
    test.require_close( result.norm_after, 5.0, 1e-14, "MGS residual norm" );
    test.require( result.max_relative_inner_product <= 1e-14, "MGS orthogonality diagnostic" );
}

void test_reorthogonalization( nmfd::tests::test_harness &test )
{
    using vector_space_type = nmfd::tests::cpu_reference_vector_space<double>;
    vector_space_type                   vector_space( 3 );
    vector_space_type::multivector_type basis;
    vector_space.init_multivector( basis, 2 );
    vector_space.start_use_multivector( basis, 2 );
    basis[0][0] = 1.0;
    basis[1][1] = 1.0;
    vector_space_type::vector_type candidate{ 1e8, 1.0, 1.0 };
    double                         coefficients[2]{};
    double                         pass_coefficients[2]{};

    orthogonalization_options<double> options;
    options.method              = orthogonalization_method::modified_gram_schmidt;
    options.reorthogonalization = reorthogonalization_policy::dgks;
    options.max_passes          = 2;
    const auto result           = nmfd::solvers::krylov::orthogonalize(
        vector_space, basis, 2, 1, candidate, coefficients, pass_coefficients, options
    );
    test.require( result.status == orthogonalization_status::success, "DGKS succeeds" );
    test.require( result.passes == 2, "DGKS requests a second pass" );
    test.require_close( candidate[0], 0.0, 1e-14, "DGKS removes basis" );

    candidate                   = { 3.0, -2.0, 4.0 };
    options.method              = orthogonalization_method::classical_gram_schmidt;
    options.reorthogonalization = reorthogonalization_policy::always;
    const auto cgs_result       = nmfd::solvers::krylov::orthogonalize(
        vector_space, basis, 2, 2, candidate, coefficients, pass_coefficients, options
    );
    test.require( cgs_result.status == orthogonalization_status::success, "CGS2 succeeds" );
    test.require( cgs_result.passes == 2, "CGS2 uses two passes" );
    test.require_close( candidate[0], 0.0, 1e-14, "CGS2 removes basis 0" );
    test.require_close( candidate[1], 0.0, 1e-14, "CGS2 removes basis 1" );
}

void test_breakdown_and_errors( nmfd::tests::test_harness &test )
{
    using vector_space_type = nmfd::tests::cpu_reference_vector_space<double>;
    vector_space_type                   vector_space( 2 );
    vector_space_type::multivector_type basis;
    vector_space.init_multivector( basis, 1 );
    vector_space.start_use_multivector( basis, 1 );
    basis[0][0] = 1.0;
    vector_space_type::vector_type candidate{ 2.0, 0.0 };
    double                         coefficient{};
    double                         pass_coefficient{};

    orthogonalization_options<double> options;
    const auto                        breakdown = nmfd::solvers::krylov::orthogonalize(
        vector_space, basis, 1, 1, candidate, &coefficient, &pass_coefficient, options
    );
    test.require(
        breakdown.status == orthogonalization_status::happy_breakdown, "dependent vector reports happy breakdown"
    );

    candidate          = { 1.0, 1.0 };
    options.max_passes = 0;
    const auto invalid = nmfd::solvers::krylov::orthogonalize(
        vector_space, basis, 1, 1, candidate, &coefficient, &pass_coefficient, options
    );
    test.require( invalid.status == orthogonalization_status::invalid_input, "invalid options are rejected" );

    options.max_passes    = 2;
    candidate             = { std::numeric_limits<double>::quiet_NaN(), 1.0 };
    const auto non_finite = nmfd::solvers::krylov::orthogonalize(
        vector_space, basis, 1, 1, candidate, &coefficient, &pass_coefficient, options
    );
    test.require( non_finite.status == orthogonalization_status::non_finite, "non-finite input is rejected" );
}

void test_complex( nmfd::tests::test_harness &test )
{
    using scalar_type       = std::complex<double>;
    using vector_space_type = nmfd::tests::cpu_reference_vector_space<scalar_type>;
    vector_space_type                   vector_space( 3 );
    vector_space_type::multivector_type basis;
    vector_space.init_multivector( basis, 1 );
    vector_space.start_use_multivector( basis, 1 );
    const double inverse_sqrt_two = 1.0 / std::sqrt( 2.0 );
    basis[0][0]                   = scalar_type( inverse_sqrt_two, 0.0 );
    basis[0][1]                   = scalar_type( 0.0, inverse_sqrt_two );
    vector_space_type::vector_type candidate{
        scalar_type( 2.0, -1.0 ), scalar_type( -0.5, 3.0 ), scalar_type( 1.0, 2.0 )
    };
    scalar_type                       coefficient{};
    scalar_type                       pass_coefficient{};
    orthogonalization_options<double> options;
    options.reorthogonalization         = reorthogonalization_policy::always;
    options.compute_orthogonality_error = true;
    const auto result                   = nmfd::solvers::krylov::orthogonalize(
        vector_space, basis, 1, 1, candidate, &coefficient, &pass_coefficient, options
    );
    test.require( result.status == orthogonalization_status::success, "complex orthogonalization succeeds" );
    test.require(
        std::abs( vector_space.scalar_prod( basis[0], candidate ) ) <= 1e-13, "complex inner product uses conjugation"
    );
    test.require( result.max_relative_inner_product <= 1e-13, "complex orthogonality diagnostic" );
}

void test_parsers( nmfd::tests::test_harness &test )
{
    test.require(
        nmfd::solvers::krylov::parse_orthogonalization_method( "mgs" ) ==
            orthogonalization_method::modified_gram_schmidt,
        "parse MGS"
    );
    test.require(
        nmfd::solvers::krylov::parse_orthogonalization_method( "cgs" ) ==
            orthogonalization_method::classical_gram_schmidt,
        "parse CGS"
    );
    test.require(
        nmfd::solvers::krylov::parse_reorthogonalization_policy( "dgks" ) == reorthogonalization_policy::dgks,
        "parse DGKS"
    );

    bool threw = false;
    try
    {
        (void)nmfd::solvers::krylov::parse_orthogonalization_method( "bad" );
    }
    catch ( const std::invalid_argument & )
    {
        threw = true;
    }
    test.require( threw, "unknown method throws" );
}

} // namespace

int main()
{
    nmfd::tests::test_harness test;
    test_mgs( test );
    test_reorthogonalization( test );
    test_breakdown_and_errors( test );
    test_complex( test );
    test_parsers( test );
    return test.finish( "NMFD orthogonalization tests" );
}
