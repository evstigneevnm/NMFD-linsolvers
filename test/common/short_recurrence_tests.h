#ifndef __NMFD_TEST_SHORT_RECURRENCE_TESTS_H__
#define __NMFD_TEST_SHORT_RECURRENCE_TESTS_H__

#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <scfd/utils/log.h>
#include <nmfd/detail/algo_hierarchy_creator.h>
#include <nmfd/solvers/monitor_krylov.h>
#include "analytical_operator.h"
#include "cpu_reference_vector_space.h"
#include "test_harness.h"

namespace nmfd
{
namespace tests
{

template <class Function>
void require_invalid( test_harness &test, Function function, const std::string &message )
{
    bool caught = false;
    try
    {
        function();
    }
    catch ( const std::invalid_argument & )
    {
        caught = true;
    }
    test.require( caught, message );
}

template <class T>
struct counted_space : cpu_reference_vector_space<T>
{
    using parent_t = cpu_reference_vector_space<T>;
    using typename parent_t::vector_type;
    mutable int allocations = 0, frees = 0, activations = 0;
    int         fail_activation = -1;
    explicit counted_space( int size ) : parent_t( size )
    {
    }
    void init_vector( vector_type &v ) const
    {
        parent_t::init_vector( v );
        ++allocations;
    }
    void free_vector( vector_type &v ) const
    {
        parent_t::free_vector( v );
        ++frees;
    }
    void start_use_vector( vector_type &v ) const
    {
        parent_t::start_use_vector( v );
        if ( activations++ == fail_activation )
            throw std::runtime_error( "Injected workspace activation failure" );
    }
};

template <template <class, class, class, class, class> class Solver, bool Polynomial, class T>
void short_recurrence_case( test_harness &test )
{
    using space_type           = cpu_reference_vector_space<T>;
    using operator_type        = analytical_operator<space_type>;
    using prec_type            = diagonal_preconditioner<space_type, operator_type>;
    using log_type             = scfd::utils::log_std;
    using monitor_type         = nmfd::solvers::monitor_krylov<space_type, log_type>;
    using solver_type          = Solver<space_type, monitor_type, log_type, operator_type, prec_type>;
    using norm_type            = typename space_type::norm_type;
    auto                 space = std::make_shared<space_type>( 4 );
    const std::vector<T> matrix{ T( 4 ), T( 1 ),  T( 0 ), T( 0 ), T( -1 ), T( 5 ), T( 1 ),  T( 0 ),
                                 T( 0 ), T( -1 ), T( 6 ), T( 1 ), T( 1 ),  T( 0 ), T( -1 ), T( 7 ) };
    auto                 op = std::make_shared<operator_type>( *space, matrix );
    auto                 prec =
        std::make_shared<prec_type>( *space, std::vector<T>{ T( 0.25 ), T( 0.2 ), T( 1 ) / T( 6 ), T( 1 ) / T( 7 ) } );
    const norm_type              tolerance = norm_type( 500 ) * std::numeric_limits<norm_type>::epsilon();
    typename solver_type::params parameters;
    parameters.monitor.rel_tol                      = tolerance;
    parameters.monitor.abs_tol                      = tolerance * norm_type( 0.01 );
    parameters.monitor.max_iters_num                = 60;
    parameters.monitor.save_convergence_history     = true;
    parameters.monitor.out_min_resid_norm           = true;
    parameters.monitor.divide_out_norms_by_rel_base = false;
    const std::vector<T> exact{ T( 1 ), T( -2 ), T( 0.5 ), T( 3 ) };
    std::vector<T>       rhs( 4 ), x( 4 ), residual( 4 );
    op->apply( exact, rhs );
    for ( bool preconditioned_norm : { false, true } )
        for ( bool real_residual : { false, true } )
            for ( int frequency : { 0, 1, 2, 3 } )
            {
                parameters.use_precond_resid = preconditioned_norm;
                parameters.use_real_resid    = real_residual;
                parameters.resid_recalc_freq = frequency;
                solver_type solver( op, space, nullptr, parameters, prec );
                for ( int attempt = 0; attempt < 2; ++attempt )
                {
                    space->assign_scalar( T( 0.1 ), x );
                    const bool converged = solver.solve( rhs, x );
                    op->apply( x, residual );
                    space->add_lin_comb( T( 1 ), rhs, T( -1 ), residual );
                    test.require(
                        converged,
                        "Repeated preconditioned solve converges: relative residual / tolerance = " +
                            std::to_string( double( space->norm( residual ) / ( tolerance * space->norm( rhs ) ) ) ) +
                            ", preconditioned=" + std::to_string( preconditioned_norm ) +
                            ", real=" + std::to_string( real_residual ) + ", frequency=" + std::to_string( frequency )
                    );
                    test.require(
                        space->norm( residual ) <= norm_type( 2 ) * tolerance * space->norm( rhs ),
                        "Original-system residual"
                    );
                    for ( int i = 0; i < 4; ++i )
                        test.require_close(
                            x[i], exact[i], double( tolerance ) * 10, "Solution agrees with known root"
                        );
                    test.require(
                        solver.monitor().iters_performed() <= parameters.monitor.max_iters_num,
                        "Iteration budget is respected"
                    );
                }
                test.require( !solver.monitor().convergence_history().empty(), "Monitor records residual history" );
            }
    if constexpr ( Polynomial )
    {
        solver_type solver( op, space, nullptr, parameters, prec );
        for ( int degree : { 1, 2, 3, 4 } )
        {
            solver.set_basis_size( degree );
            space->assign_scalar( T( 0 ), x );
            test.require( solver.solve( rhs, x ), "Polynomial degrees and workspace resizing" );
        }
        require_invalid( test, [&solver] { solver.set_basis_size( 0 ); }, "Zero degree is rejected" );
        require_invalid( test, [&solver] { solver.set_basis_size( 31 ); }, "Oversized degree is rejected" );
        require_invalid( test, [&solver] { solver.set_basis_size( -1 ); }, "Negative degree is rejected" );
        space->assign_scalar( T( 0 ), x );
        test.require( solver.solve( rhs, x ), "Invalid degree leaves solver usable" );
    }
}

template <template <class, class, class, class, class> class Solver>
void short_recurrence_true_residual( test_harness &test )
{
    using space_type    = cpu_reference_vector_space<double>;
    using operator_type = analytical_operator<space_type>;
    using prec_type     = diagonal_preconditioner<space_type, operator_type>;
    using monitor_type  = nmfd::solvers::monitor_krylov<space_type, scfd::utils::log_std>;
    using solver_type   = Solver<space_type, monitor_type, scfd::utils::log_std, operator_type, prec_type>;
    auto                         space = std::make_shared<space_type>( 2 );
    auto                         op    = std::make_shared<operator_type>( *space, std::vector<double>{ 2, 0, 0, 3 } );
    auto                         prec  = std::make_shared<prec_type>( *space, std::vector<double>{ 0.5, 1e-10 } );
    typename solver_type::params parameters;
    parameters.monitor.rel_tol       = 1e-8;
    parameters.monitor.abs_tol       = 0;
    parameters.monitor.max_iters_num = 30;
    solver_type         solver( op, space, nullptr, parameters, prec );
    std::vector<double> rhs{ 2, 3 }, x( 2 ), residual( 2 );
    test.require( solver.solve( rhs, x ), "Preconditioned tolerance hit continues until true residual converges" );
    op->apply( x, residual );
    space->add_lin_comb( 1.0, rhs, -1.0, residual );
    test.require(
        space->norm( residual ) <= parameters.monitor.rel_tol * space->norm( rhs ),
        "Strongly scaled preconditioner does not permit false convergence"
    );
    test.require(
        solver.monitor().iters_performed() <= parameters.monitor.max_iters_num,
        "True-residual continuation remains bounded"
    );
    solver.monitor().set_temp_max_iterations( 1 );
    space->assign_scalar( 0.0, x );
    test.require( !solver.solve( rhs, x ), "Unsatisfied true residual at iteration limit returns failure" );
}

template <template <class, class, class, class, class> class Solver, bool Polynomial>
void short_recurrence_failures( test_harness &test )
{
    using space_type    = cpu_reference_vector_space<double>;
    using operator_type = analytical_operator<space_type>;
    using log_type      = scfd::utils::log_std;
    using prec_type     = diagonal_preconditioner<space_type, operator_type>;
    using monitor_type  = nmfd::solvers::monitor_krylov<space_type, log_type>;
    using solver_type   = Solver<space_type, monitor_type, log_type, operator_type, prec_type>;
    auto space          = std::make_shared<space_type>( 3 );
    auto op             = std::make_shared<operator_type>( *space, std::vector<double>{ 2, 0, 0, 0, 3, 0, 0, 0, 4 } );
    auto prec           = std::make_shared<prec_type>( *space, std::vector<double>{ 0.5, 1.0 / 3, 0.25 } );
    typename solver_type::params p;
    p.monitor.rel_tol       = 1e-12;
    p.monitor.abs_tol       = 1e-14;
    p.monitor.max_iters_num = 30;
    solver_type         solver( op, space, nullptr, p, prec );
    std::vector<double> rhs{ 1, 2, 3 }, x( 3 ), residual( 3 );
    op->fail_after( 0 );
    test.require( !solver.solve( rhs, x ), "Operator failure is not convergence" );
    op->allow_all_calls();
    op->reset_calls();
    prec->fail_after( 0 );
    test.require( !solver.solve( rhs, x ), "Preconditioner failure is propagated" );
    prec->allow_all_calls();
    prec->reset_calls();
    test.require( solver.solve( rhs, x ), "Solver recovers on repeated call after failures" );
    test.require(
        solver.solve( rhs, x ) && solver.monitor().iters_performed() == 0, "Exact initial solution uses no iterations"
    );
    space->assign_scalar( 0, rhs );
    space->assign_scalar( 0, x );
    test.require( solver.solve( rhs, x ), "Zero RHS and state are accepted without normalization" );

    auto zero = std::make_shared<operator_type>( *space, std::vector<double>( 9, 0 ) );
    solver.set_operator( zero );
    space->assign_scalar( 1, rhs );
    test.require( !solver.solve( rhs, x ), "Zero operator breakdown returns false" );
    test.require( space->is_valid_number( x ), "Breakdown avoids nonfinite updates" );
    solver.update_operator( op );
    test.require( solver.solve( rhs, x ), "Operator rebinding after breakdown" );
    solver.set_preconditioner_bypass( true );
    space->assign_scalar( 0, x );
    test.require( solver.solve( rhs, x ) && solver.preconditioner_bypass_enabled(), "Preconditioner bypass" );
    solver.set_preconditioner_bypass( false );

    p.monitor.max_iters_num = 1;
    solver_type limited( op, space, nullptr, p, prec );
    limited.set_preconditioner_bypass( true );
    space->assign_scalar( 0, x );
    test.require( !limited.solve( rhs, x ), "Iteration exhaustion returns false" );
    test.require( limited.monitor().iters_performed() == 1, "No polynomial-cycle budget overshoot" );

    auto tiny               = std::make_shared<prec_type>( *space, std::vector<double>( 3, 1e-100 ) );
    p.monitor.max_iters_num = 30;
    p.use_precond_resid     = true;
    solver_type scaled( op, space, nullptr, p, tiny );
    space->assign_scalar( 0, x );
    const bool scaled_converged = scaled.solve( rhs, x );
    op->apply( x, residual );
    space->add_lin_comb( 1.0, rhs, -1.0, residual );
    test.require(
        !scaled_converged || space->norm( residual ) <= p.monitor.abs_tol + p.monitor.rel_tol * space->norm( rhs ),
        "Tiny preconditioned residual cannot hide original residual"
    );
    rhs[0] = std::numeric_limits<double>::quiet_NaN();
    test.require( !solver.solve( rhs, x ), "NaN RHS returns failure" );
    rhs[0] = 1;
    x[0]   = std::numeric_limits<double>::infinity();
    test.require( !solver.solve( rhs, x ), "Nonfinite initial state returns failure" );
    require_invalid(
        test,
        [space, p, prec] {
            auto bad              = p;
            bad.resid_recalc_freq = -1;
            solver_type invalid( space, nullptr, bad, prec );
        },
        "Negative recomputation frequency rejected"
    );
    require_invalid(
        test, [p, prec] { solver_type invalid( nullptr, nullptr, p, prec ); }, "Null vector space rejected"
    );
    require_invalid( test, [&solver] { solver.set_resid_recalc_freq( -1 ); }, "Invalid setter rejected" );
    if constexpr ( Polynomial )
    {
        require_invalid(
            test,
            [space, p, prec] {
                auto bad       = p;
                bad.basis_size = -1;
                solver_type invalid( space, nullptr, bad, prec );
            },
            "Negative constructor degree rejected"
        );
    }
    auto identity = std::make_shared<operator_type>( *space, std::vector<double>{ 1, 0, 0, 0, 1, 0, 0, 0, 1 } );
    solver.set_operator( identity );
    solver.set_preconditioner_bypass( true );
    space->assign_scalar( 0, x );
    test.require( solver.solve( rhs, x ), "Exact half-step convergence avoids zero stabilization denominator" );

    using base_type = nmfd::solvers::iter_solver_base<space_type, monitor_type, log_type, operator_type, prec_type>;
    base_type &base = solver;
    space->assign_scalar( 0, x );
    test.require( base.solve( rhs, x ), "Canonical virtual solver-base entry point" );
    space->assign_scalar( 0, x );
    test.require( base.solve( *op, rhs, x ), "Borrowed operator entry point" );
}

template <template <class, class, class, class, class> class Solver>
void short_recurrence_ownership( test_harness &test )
{
    using space_type    = counted_space<double>;
    using operator_type = analytical_operator<space_type>;
    using log_type      = scfd::utils::log_std;
    using monitor_type  = nmfd::solvers::monitor_krylov<space_type, log_type>;
    using solver_type   = Solver<
          space_type, monitor_type, log_type, operator_type, nmfd::preconditioners::dummy<space_type, operator_type>>;
    for ( int failure : { 0, 1, 3 } )
    {
        auto space             = std::make_shared<space_type>( 3 );
        space->fail_activation = failure;
        bool caught            = false;
        try
        {
            solver_type solver( space );
        }
        catch ( const std::runtime_error & )
        {
            caught = true;
        }
        test.require( caught, "Injected construction failure is reached" );
        test.require( space->allocations == space->frees, "Partial construction frees every workspace" );
    }
    auto                      space  = std::make_shared<space_type>( 3 );
    std::weak_ptr<space_type> weak   = space;
    auto                      solver = std::make_unique<solver_type>( space );
    space.reset();
    test.require( !weak.expired(), "Solver retains its vector space" );
    solver.reset();
    test.require( weak.expired(), "No dependency ownership cycle" );
    auto sole_space  = std::make_shared<space_type>( 3 );
    auto sole_solver = std::make_unique<solver_type>( sole_space );
    sole_space.reset();
    sole_solver.reset();
    test.require( true, "Monitor cleanup while solver is the sole space owner" );
}

struct short_recurrence_backend
{
    scfd::utils::log_std  logger;
    scfd::utils::log_std &log()
    {
        return logger;
    }
};

template <template <class, class, class, class, class> class Solver, bool Polynomial>
int run_short_recurrence_tests( const std::string &name )
{
    test_harness test;
    short_recurrence_case<Solver, Polynomial, float>( test );
    short_recurrence_case<Solver, Polynomial, double>( test );
    short_recurrence_case<Solver, Polynomial, long double>( test );
    short_recurrence_case<Solver, Polynomial, std::complex<double>>( test );
    short_recurrence_failures<Solver, Polynomial>( test );
    short_recurrence_ownership<Solver>( test );
    short_recurrence_true_residual<Solver>( test );

    using space_type    = cpu_reference_vector_space<double>;
    using operator_type = analytical_operator<space_type>;
    using log_type      = scfd::utils::log_std;
    using monitor_type  = nmfd::solvers::monitor_krylov<space_type, log_type>;
    using solver_type   = Solver<
          space_type, monitor_type, log_type, operator_type, nmfd::preconditioners::dummy<space_type, operator_type>>;
    auto                                   space = std::make_shared<space_type>( 3 );
    short_recurrence_backend               backend;
    typename solver_type::utils_hierarchy  utilities( backend, space );
    typename solver_type::params_hierarchy parameters;
    auto solver   = nmfd::detail::algo_hierarchy_creator<solver_type>::get( utilities, parameters );
    auto identity = std::make_shared<operator_type>( *space, std::vector<double>{ 1, 0, 0, 0, 1, 0, 0, 0, 1 } );
    solver->set_operator( identity );
    std::vector<double> rhs{ 1, 2, 3 }, x( 3 );
    test.require( solver->solve( rhs, x ), "Backend and recursive preconditioner hierarchy construction" );
#ifdef NMFD_ENABLE_NLOHMANN
    const auto                             serialized = parameters.to_json();
    typename solver_type::params_hierarchy restored;
    restored.from_json( serialized );
    test.require( serialized == restored.to_json(), "JSON hierarchy round trip" );
#endif
    return test.finish( name );
}

}
}

#endif
