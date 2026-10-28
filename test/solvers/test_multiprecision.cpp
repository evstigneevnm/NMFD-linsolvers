#include <limits>
#include <memory>
#include <stdexcept>
#include <scfd/utils/log.h>
#include <nmfd/operations/cpu_vector_operations_var_prec.h>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/direct_solver.h>
#include <nmfd/solvers/sherman_morrison.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/preconditioners/small_schur_inverse.h>
#include "common/test_harness.h"

namespace
{

template <class Space>
struct diagonal_operator
{
    using T           = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    T first           = T( 2 ) + T( "1e-35" );

    void apply( const vector_type &in, vector_type &out ) const
    {
        out[0] = first * in[0];
        out[1] = T( 3 ) * in[1];
        out[2] = T( 5 ) * in[2];
    }
};

template <class Space, class Operator>
struct diagonal_inverse
{
    using T           = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    std::shared_ptr<const Operator> op;
    bool                            fail           = false;
    bool                            throw_on_apply = false;

    void set_operator( std::shared_ptr<const Operator> value )
    {
        op = std::move( value );
    }
    bool apply( vector_type &in ) const
    {
        if ( throw_on_apply )
            throw std::runtime_error( "injected inverse failure" );
        if ( fail )
            return false;
        in[0] /= op->first;
        in[1] /= T( 3 );
        in[2] /= T( 5 );
        return true;
    }
};

template <unsigned int Precision, template <class, class, class, class, class> class LinearSolver>
void test_solver( nmfd::tests::test_harness &test )
{
    using space_type    = cpu_vector_operations_var_prec<Precision>;
    using T             = typename space_type::scalar_type;
    using math_type     = nmfd::operations::backend::math<void, T>;
    using vector_type   = typename space_type::vector_type;
    using operator_type = diagonal_operator<space_type>;
    using inverse_type  = diagonal_inverse<space_type, operator_type>;
    using log_type      = scfd::utils::log_std;
    using monitor_type  = nmfd::solvers::default_monitor<space_type, log_type>;
    using solver_type =
        nmfd::solvers::sherman_morrison<space_type, monitor_type, log_type, operator_type, inverse_type, LinearSolver>;
    auto                         space         = std::make_shared<space_type>( 3 );
    auto                         op            = std::make_shared<operator_type>();
    auto                         inverse       = std::make_shared<inverse_type>();
    std::weak_ptr<space_type>    weak_space    = space;
    std::weak_ptr<operator_type> weak_operator = op;
    std::weak_ptr<inverse_type>  weak_inverse  = inverse;
    {
        solver_type                   solver( space, op, inverse );
        typename monitor_type::params parameters;
        parameters.rel_tol       = T( "1e-65" );
        parameters.abs_tol       = T( "1e-70" );
        parameters.max_iters_num = 50;
        auto &monitor            = solver.get_linsolver_handle()->monitor();
        monitor.configure( parameters );
        solver.get_linsolver_handle_original()->monitor().configure( parameters );
        solver.is_small_alpha( true );
        vector_type exact{ T( 1 ) + T( "1e-35" ), T( -2 ), T( 3 ) };
        vector_type c{ T( 1 ), T( "0.25" ), T( "-0.5" ) };
        vector_type d{ T( 1 ), T( -1 ), T( 2 ) };
        vector_type b( 3 ), x( 3 ), residual( 3 );
        op->apply( exact, b );
        test.require( solver.solve( *op, b, x ), "multiprecision ordinary solve" );
        space->assign_lin_comb( T( 1 ), x, T( -1 ), exact, residual );
        test.require( space->norm( residual ) < T( "1e-60" ), "ordinary solve preserves sub-double information" );
        for ( const T &alpha : { T( 2 ), T( "1e-20" ), T( "-1e-20" ) } )
        {
            const T scale = math_type::abs( alpha ) < T( "1e-3" ) ? alpha : T( 1 );
            d             = { scale, -scale, T( 2 ) * scale };
            const T gamma( "1.25" );
            op->apply( exact, b );
            space->add_lin_comb( -space->scalar_prod( c, exact ) / alpha, d, gamma, b );
            space->assign_scalar( T( 0 ), x );
            test.require( solver.solve( gamma, *op, alpha, c, d, b, x ), "multiprecision rank-one solve" );
            op->apply( x, residual );
            space->add_lin_comb( -space->scalar_prod( c, x ) / alpha, d, gamma, residual );
            space->add_lin_comb( T( -1 ), b, T( 1 ), residual );
            test.require( space->norm( residual ) < T( "1e-60" ), "rank-one true residual" );

            const T exact_scalar( "0.75" );
            op->apply( exact, b );
            space->add_lin_comb( exact_scalar, d, T( 1 ), b );
            const T beta = space->scalar_prod( c, exact ) + alpha * exact_scalar;
            T       scalar( 0 );
            space->assign_scalar( T( 0 ), x );
            test.require( solver.solve( *op, c, d, alpha, b, beta, x, scalar ), "multiprecision scalar border" );
            op->apply( x, residual );
            space->add_lin_comb( scalar, d, T( 1 ), residual );
            space->add_lin_comb( T( -1 ), b, T( 1 ), residual );
            const T scalar_residual = space->scalar_prod( c, x ) + alpha * scalar - beta;
            test.require(
                math_type::hypot( space->norm( residual ), scalar_residual ) <=
                    parameters.abs_tol + parameters.rel_tol * math_type::hypot( space->norm( b ), beta ),
                "full bordered true residual"
            );
            space->assign_lin_comb( T( 1 ), x, T( -1 ), exact, residual );
            test.require(
                space->norm( residual ) < T( "1e-55" ) && math_type::abs( scalar - exact_scalar ) < T( "1e-55" ),
                "bordered solution matches reference"
            );
            test.require(
                monitor.rel_tol() == parameters.rel_tol && monitor.abs_tol() == parameters.abs_tol,
                "bordered solve restores monitor tolerances"
            );
        }
        test.require( !solver.solve( *op, c, d, T( 0 ), b, T( 1 ), x, exact[0] ), "zero elimination pivot fails" );
        inverse->fail = true;
        test.require( !solver.solve( *op, b, x ), "inverse failure propagates" );
        inverse->fail           = false;
        inverse->throw_on_apply = true;
        bool threw              = false;
        try
        {
            T scalar;
            solver.solve( *op, c, d, T( 1 ), b, T( 1 ), x, scalar );
        }
        catch ( const std::runtime_error & )
        {
            threw = true;
        }
        test.require(
            threw && monitor.rel_tol() == parameters.rel_tol && monitor.abs_tol() == parameters.abs_tol,
            "exception restores monitor tolerances"
        );
        inverse->throw_on_apply = false;
        monitor.set_tolerances( T( "1e1000" ), T( 0 ) );
        test.require( monitor.rel_tol() == T( "1e1000" ), "finite tolerance outside double range is accepted" );
        monitor.configure( parameters );
        bool invalid_tolerance = false;
        try
        {
            monitor.set_tolerances( std::numeric_limits<T>::infinity(), T( 0 ) );
        }
        catch ( const std::invalid_argument & )
        {
            invalid_tolerance = true;
        }
        test.require(
            invalid_tolerance && monitor.rel_tol() == parameters.rel_tol, "nonfinite tolerance rejected transactionally"
        );
        space.reset();
        op.reset();
        inverse.reset();
        test.require(
            !weak_space.expired() && !weak_operator.expired() && !weak_inverse.expired(), "retained dependencies"
        );
    }
    test.require( weak_space.expired() && weak_operator.expired() && weak_inverse.expired(), "no dependency cycles" );
}

void test_gmres_schur( nmfd::tests::test_harness &test )
{
    using space_type    = cpu_vector_operations_var_prec<100>;
    using T             = typename space_type::scalar_type;
    using vector_type   = typename space_type::vector_type;
    using operator_type = diagonal_operator<space_type>;
    using inverse_type  = diagonal_inverse<space_type, operator_type>;
    using solver_type   = nmfd::solvers::gmres<
          space_type, nmfd::solvers::monitor_krylov<space_type, scfd::utils::log_std>, scfd::utils::log_std,
          operator_type, inverse_type>;
    auto                         space = std::make_shared<space_type>( 3 );
    auto                         op    = std::make_shared<operator_type>();
    typename solver_type::params parameters;
    parameters.monitor.abs_tol = T( "1e-70" );
    parameters.monitor.rel_tol = T( "1e-70" );
    solver_type solver( space, nullptr, parameters, std::make_shared<inverse_type>() );
    solver.set_operator( op );
    vector_type exact{ T( 1 ) + T( "1e-35" ), T( 2 ), T( -3 ) }, rhs( 3 ), result( 3, T( 0 ) );
    op->apply( exact, rhs );
    test.require( solver.solve( rhs, result ), "multiprecision GMRES" );
    space->add_lin_comb( T( -1 ), exact, T( 1 ), result );
    test.require( space->norm( result ) < T( "1e-60" ), "GMRES retains sub-double information" );
    nmfd::preconditioners::small_schur_inverse<space_type, operator_type> schur( space );
    test.require( schur.set_operator( op ) && schur.apply( rhs ), "multiprecision Schur preparation and application" );
    space->add_lin_comb( T( -1 ), exact, T( 1 ), rhs );
    test.require( space->norm( rhs ) < T( "1e-60" ), "small Schur solve retains precision" );
    op->first = 0;
    test.require( !schur.set_operator( op ) && !schur.ready(), "failed Schur rebinding invalidates factors" );
}

} // namespace

int main()
{
    nmfd::tests::test_harness test;
    test_gmres_schur( test );
    test_solver<100, nmfd::solvers::cgs>( test );
    test_solver<200, nmfd::solvers::cgs>( test );
    test_solver<100, nmfd::solvers::direct_solver>( test );
    test_solver<200, nmfd::solvers::direct_solver>( test );
    return test.finish( "Multiprecision linear solvers" );
}
