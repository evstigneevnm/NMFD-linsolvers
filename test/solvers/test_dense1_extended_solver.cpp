#include <memory>
#include <scfd/static_mat/mat.h>
#include <scfd/static_vec/vec.h>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/solvers/dense1_extended_solver.h>
#include "common/test_harness.h"

namespace
{

template <class T>
struct matrix_operator
{
    using scalar_type = T;
    using vector_type = scfd::static_vec::vec<T, 3>;
    using matrix_type = scfd::static_mat::mat<T, 3, 3>;
    matrix_type matrix;

    void apply( const vector_type &in, vector_type &out ) const
    {
        out = matrix * in;
    }
};

template <class Operator>
struct inverse_solver
{
    using vector_type = typename Operator::vector_type;
    typename Operator::matrix_type inverse;
    int                            calls        = 0;
    int                            fail_on_call = 0;

    void set_operator( std::shared_ptr<const Operator> op )
    {
        inverse = inv( op->matrix );
    }

    bool solve( const vector_type &rhs, vector_type &out )
    {
        out = inverse * rhs;
        return ++calls != fail_on_call;
    }
};

template <class T>
void test_solver( nmfd::tests::test_harness &test )
{
    using operator_type          = matrix_operator<T>;
    using vector_type            = typename operator_type::vector_type;
    using space_type             = nmfd::operations::static_vector_space<T, 3, vector_type>;
    using original_solver_type   = inverse_solver<operator_type>;
    using extended_operator_type = nmfd::operations::dense1_extended_operator<operator_type, space_type>;
    using solver_type = nmfd::solvers::dense1_extended_solver<original_solver_type, operator_type, space_type>;
    using pair_type   = std::pair<vector_type, typename nmfd::operations::static_vector_space<T, 1>::vector_type>;

    auto space                                          = std::make_shared<space_type>();
    auto original                                       = std::make_shared<original_solver_type>();
    auto op                                             = std::make_shared<operator_type>();
    op->matrix                                          = { 4, 1, 0, -1, 5, 1, 0, -1, 6 };
    auto extended                                       = std::make_shared<extended_operator_type>( space, op );
    extended->u()                                       = { T( 0.5 ), T( -1 ), T( 2 ) };
    extended->v()                                       = { T( 1 ), T( 0.25 ), T( -0.5 ) };
    extended->w()[0]                                    = T( 3 );
    std::weak_ptr<space_type>             weak_space    = space;
    std::weak_ptr<original_solver_type>   weak_original = original;
    std::weak_ptr<extended_operator_type> weak_extended = extended;
    std::weak_ptr<operator_type>          weak_operator = op;
    {
        solver_type solver( space, original );
        solver.set_operator( extended );
        const pair_type exact{ { T( 1 ), T( -2 ), T( 3 ) }, { T( -0.5 ) } };
        const double    tolerance = sizeof( T ) == sizeof( float ) ? 2e-6 : 2e-13;
        for ( int replay = 0; replay < 3; ++replay )
        {
            if ( replay == 1 )
            {
                op               = std::make_shared<operator_type>();
                op->matrix       = { 6, -1, 1, 2, 7, 0, 0, 1, 5 };
                extended         = std::make_shared<extended_operator_type>( space, op );
                extended->u()    = { T( -1 ), T( 2 ), T( 0.5 ) };
                extended->v()    = { T( 0.5 ), T( -1 ), T( 1 ) };
                extended->w()[0] = T( 4 );
                solver.set_operator( extended );
                test.require(
                    weak_extended.expired() && weak_operator.expired(), "rebinding releases the old operator"
                );
                weak_extended = extended;
                weak_operator = op;
            }
            if ( replay == 2 )
            {
                original = std::make_shared<original_solver_type>();
                solver.set_solver( original );
                test.require( weak_original.expired(), "rebinding releases the old solver" );
                weak_original = original;
            }

            scfd::static_mat::mat<T, 4, 4> assembled;
            for ( int i = 0; i < 3; ++i )
            {
                for ( int j = 0; j < 3; ++j )
                    assembled( i, j ) = op->matrix( i, j );
                assembled( i, 3 ) = extended->u()[i];
                assembled( 3, i ) = extended->v()[i];
            }
            assembled( 3, 3 ) = extended->w()[0];
            const scfd::static_vec::vec<T, 4> expected{
                exact.first[0], exact.first[1], exact.first[2], exact.second[0]
            };
            const auto      reference_rhs = assembled * expected;
            const pair_type rhs{ { reference_rhs[0], reference_rhs[1], reference_rhs[2] }, { reference_rhs[3] } };
            pair_type       result{};
            test.require( solver.solve( rhs, result ), "extended solve converges after binding/rebinding" );
            const scfd::static_vec::vec<T, 4> actual{
                result.first[0], result.first[1], result.first[2], result.second[0]
            };
            const auto residual = assembled * actual - reference_rhs;
            for ( int i = 0; i < 4; ++i )
            {
                test.require_close( actual[i], expected[i], tolerance, "assembled reference solution" );
                test.require_close( residual[i], T( 0 ), tolerance, "full bordered true residual" );
            }
        }
        for ( int failed_solve : { 1, 2 } )
        {
            original->calls        = 0;
            original->fail_on_call = failed_solve;
            pair_type result{};
            test.require( !solver.solve( exact, result ), "failure in either underlying solve propagates" );
        }
        space.reset();
        original.reset();
        extended.reset();
        op.reset();
        test.require(
            !weak_space.expired() && !weak_original.expired() && !weak_extended.expired() && !weak_operator.expired(),
            "solver retains its computational dependencies"
        );
    }
    test.require(
        weak_space.expired() && weak_original.expired() && weak_extended.expired() && weak_operator.expired(),
        "dependency owners are released without cycles"
    );
}

} // namespace

int main()
{
    nmfd::tests::test_harness test;
    test_solver<float>( test );
    test_solver<double>( test );
    test_solver<long double>( test );
    return test.finish( "Dense1 extended solver" );
}
