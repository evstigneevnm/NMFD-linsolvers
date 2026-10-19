#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <scfd/utils/log.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/jacobi.h>
#include <nmfd/solvers/monitor_krylov.h>
#include "common/analytical_operator.h"
#include "common/cpu_reference_vector_space.h"
#include "common/test_harness.h"

namespace
{

template <class Function>
void require_invalid( nmfd::tests::test_harness &test, Function function )
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
    test.require( caught, "Invalid construction is rejected" );
}

template <class T, template <class, class> class Monitor>
void test_solve( nmfd::tests::test_harness &test )
{
    using space_type    = nmfd::tests::cpu_reference_vector_space<T>;
    using operator_type = nmfd::tests::analytical_operator<space_type>;
    using prec_type     = nmfd::tests::diagonal_preconditioner<space_type, operator_type>;
    using log_type      = scfd::utils::log_std;
    using monitor_type  = Monitor<space_type, log_type>;
    using solver_type   = nmfd::solvers::jacobi<space_type, operator_type, prec_type, monitor_type, log_type>;
    using norm_type     = typename space_type::norm_type;
    auto space          = std::make_shared<space_type>( 4 );
    auto op             = std::make_shared<operator_type>(
        *space, std::vector<T>{
                    T( 4 ), T( 1 ), T( 0 ), T( 0 ), T( -1 ), T( 5 ), T( 1 ), T( 0 ), T( 0 ), T( -1 ), T( 6 ), T( 1 ),
                    T( 1 ), T( 0 ), T( -1 ), T( 7 )
                }
    );
    auto prec =
        std::make_shared<prec_type>( *space, std::vector<T>{ T( .25 ), T( .2 ), T( 1 ) / T( 6 ), T( 1 ) / T( 7 ) } );
    const norm_type              tolerance = norm_type( 500 ) * std::numeric_limits<norm_type>::epsilon();
    typename solver_type::params p;
    p.monitor.rel_tol                      = tolerance;
    p.monitor.abs_tol                      = tolerance * norm_type( .01 );
    p.monitor.max_iters_num                = 200;
    p.monitor.save_convergence_history     = true;
    p.monitor.divide_out_norms_by_rel_base = false;
    const std::vector<T> exact{ T( 1 ), T( -2 ), T( .5 ), T( 3 ) };
    std::vector<T>       rhs( 4 ), x( 4 ), residual( 4 );
    op->apply( exact, rhs );
    for ( bool minimum : { false, true } )
    {
        p.monitor.out_min_resid_norm = minimum;
        solver_type solver( op, space, nullptr, p, prec );
        for ( int attempt = 0; attempt < 3; ++attempt )
        {
            space->assign_scalar( T( .1 ), x );
            test.require( solver.solve( rhs, x ), "Repeated Jacobi solve" );
            op->apply( x, residual );
            space->add_lin_comb( T( -1 ), rhs, T( 1 ), residual );
            test.require(
                space->norm( residual ) <= tolerance * ( space->norm( rhs ) + norm_type( 1 ) ), "True residual"
            );
            for ( int i = 0; i < 4; ++i )
                test.require_close( x[i], exact[i], double( tolerance ) * 10, "Known solution" );
            test.require( solver.monitor().iters_performed() <= 200, "Bounded iteration count" );
            test.require( !solver.monitor().convergence_history().empty(), "Convergence history" );
        }
        test.require( solver.solve( rhs, x ) && solver.monitor().iters_performed() == 0, "Exact initial state" );
    }
}

void test_failures( nmfd::tests::test_harness &test )
{
    using space_type    = nmfd::tests::cpu_reference_vector_space<double>;
    using operator_type = nmfd::tests::analytical_operator<space_type>;
    using prec_type     = nmfd::tests::diagonal_preconditioner<space_type, operator_type>;
    using log_type      = scfd::utils::log_std;
    using monitor_type  = nmfd::solvers::monitor_krylov<space_type, log_type>;
    using solver_type   = nmfd::solvers::jacobi<space_type, operator_type, prec_type, monitor_type, log_type>;
    using base_type     = nmfd::solvers::iter_solver_base<space_type, monitor_type, log_type, operator_type, prec_type>;
    auto                         space = std::make_shared<space_type>( 2 );
    auto                         op    = std::make_shared<operator_type>( *space, std::vector<double>{ 4, 1, -1, 5 } );
    auto                         prec  = std::make_shared<prec_type>( *space, std::vector<double>{ .25, .2 } );
    typename solver_type::params p;
    p.monitor.rel_tol            = 1e-12;
    p.monitor.max_iters_num      = 100;
    p.monitor.out_min_resid_norm = true;
    solver_type         solver( op, space, nullptr, p, prec );
    std::vector<double> rhs{ 1, 2 }, x( 2 );
    op->fail_after( 0 );
    test.require( !solver.solve( rhs, x ) && x[0] == 0 && x[1] == 0, "Failed operator preserves initial state" );
    op->allow_all_calls();
    prec->fail_after( 0 );
    test.require( !solver.solve( rhs, x ) && x[0] == 0 && x[1] == 0, "Failed preconditioner is propagated" );
    prec->allow_all_calls();
    test.require( solver.solve( rhs, x ), "Recovery after numerical failure" );
    rhs[0] = std::numeric_limits<double>::infinity();
    test.require( !solver.solve( rhs, x ), "Nonfinite RHS" );
    rhs[0] = 1;
    x[0]   = std::numeric_limits<double>::quiet_NaN();
    test.require( !solver.solve( rhs, x ), "Nonfinite initial state" );
    space->assign_scalar( 0, x );
    auto invalid_prec =
        std::make_shared<prec_type>( *space, std::vector<double>( 2, std::numeric_limits<double>::infinity() ) );
    solver_type invalid( op, space, nullptr, p, invalid_prec );
    test.require( !invalid.solve( rhs, x ) && x[0] == 0 && x[1] == 0, "Nonfinite correction is not applied" );

    p.monitor.rel_tol            = 0;
    p.monitor.abs_tol            = 0;
    p.monitor.max_iters_num      = 2;
    p.monitor.out_min_resid_norm = false;
    solver_type limited( op, space, nullptr, p, prec );
    test.require( !limited.solve( rhs, x ) && limited.monitor().iters_performed() == 2, "Iteration exhaustion" );
    test.require_close( x[0], .15, 1e-15, "Original Jacobi recurrence, first component" );
    test.require_close( x[1], .45, 1e-15, "Original Jacobi recurrence, second component" );
    p.monitor.max_iters_num = 0;
    solver_type zero_budget( op, space, nullptr, p, prec );
    space->assign_scalar( 0, x );
    test.require( !zero_budget.solve( rhs, x ) && x[0] == 0 && x[1] == 0, "Zero iteration budget" );
    require_invalid( test, [p] { solver_type bad( nullptr, nullptr, p ); } );
    for ( int kind = 0; kind < 5; ++kind )
        require_invalid( test, [space, prec, p, kind] {
            auto bad = p;
            if ( kind == 0 )
                bad.monitor.rel_tol = -1;
            if ( kind == 1 )
                bad.monitor.abs_tol = std::numeric_limits<double>::quiet_NaN();
            if ( kind == 2 )
                bad.monitor.max_iters_num = -1;
            if ( kind == 3 )
                bad.monitor.min_iters_num = -1;
            if ( kind == 4 )
                bad.monitor.min_iters_num = 1;
            solver_type invalid( space, nullptr, bad, prec );
        } );
    auto        identity  = std::make_shared<operator_type>( *space, std::vector<double>{ 1, 0, 0, 1 } );
    auto        unit_prec = std::make_shared<prec_type>( *space, std::vector<double>{ 1, 1 } );
    solver_type rebound( space, nullptr, {}, unit_prec );
    bool        unbound = false;
    try
    {
        rebound.solve( rhs, x );
    }
    catch ( const std::logic_error & )
    {
        unbound = true;
    }
    test.require( unbound, "Unbound solver has a clear diagnostic" );
    rebound.set_operator( op );
    rebound.update_operator( identity );
    base_type &base = rebound;
    test.require( base.solve( rhs, x ), "Canonical base with operator rebinding" );
    space->assign_scalar( 0, x );
    test.require( base.solve( *identity, rhs, x ), "Borrowed operator entry point" );
    rhs.assign( 2, 0 );
    x.assign( 2, 0 );
    test.require( rebound.solve( rhs, x ), "Zero RHS and state" );
    typename solver_type::params minimum;
    minimum.monitor.min_iters_num = 2;
    solver_type minimum_solver( identity, space, nullptr, minimum, unit_prec );
    test.require(
        minimum_solver.solve( rhs, x ) && minimum_solver.monitor().iters_performed() == 2,
        "Minimum iterations on exact solution"
    );
    solver_type no_preconditioner( identity, space );
    rhs.assign( 2, 1 );
    x.assign( 2, 0 );
    test.require( no_preconditioner.solve( rhs, x ), "Absent preconditioner acts as identity" );
    minimum.monitor.rel_tol       = std::numeric_limits<double>::max();
    minimum.monitor.min_iters_num = 0;
    solver_type overflowing_tolerance( identity, space, nullptr, minimum, unit_prec );
    rhs.assign( 2, 2 );
    x.assign( 2, 0 );
    test.require( !overflowing_tolerance.solve( rhs, x ), "Overflowing effective tolerance is not convergence" );
}

struct counted_space : nmfd::tests::cpu_reference_vector_space<double>
{
    using parent_t = nmfd::tests::cpu_reference_vector_space<double>;
    using parent_t::parent_t;
    mutable int allocations = 0, frees = 0, starts = 0, stops = 0;
    bool        fail_start = false;
    void        init_vector( vector_type &x ) const
    {
        parent_t::init_vector( x );
        ++allocations;
    }
    void free_vector( vector_type &x ) const
    {
        parent_t::free_vector( x );
        ++frees;
    }
    void start_use_vector( vector_type &x ) const
    {
        if ( fail_start )
            throw std::runtime_error( "Injected activation failure" );
        parent_t::start_use_vector( x );
        ++starts;
    }
    void stop_use_vector( vector_type &x ) const
    {
        parent_t::stop_use_vector( x );
        ++stops;
    }
};

void test_ownership( nmfd::tests::test_harness &test )
{
    using operator_type = nmfd::tests::analytical_operator<counted_space>;
    using log_type      = scfd::utils::log_std;
    using monitor_type  = nmfd::solvers::default_monitor<counted_space, log_type>;
    using prec_type     = nmfd::preconditioners::dummy<counted_space, operator_type>;
    using solver_type   = nmfd::solvers::jacobi<counted_space, operator_type, prec_type, monitor_type, log_type>;
    auto space          = std::make_shared<counted_space>( 2 );
    space->fail_start   = true;
    bool caught         = false;
    try
    {
        solver_type solver( space );
    }
    catch ( const std::runtime_error & )
    {
        caught = true;
    }
    test.require( caught && space->allocations == space->frees, "Partial construction cleanup" );
    space->fail_start = false;
    {
        auto                         op = std::make_shared<operator_type>( *space, std::vector<double>{ 1, 0, 0, 1 } );
        typename solver_type::params p;
        p.monitor.out_min_resid_norm = true;
        solver_type         solver( op, space, nullptr, p );
        std::vector<double> rhs{ 1, 2 }, x( 2 );
        const int           allocations = space->allocations;
        for ( int attempt = 0; attempt < 3; ++attempt )
        {
            x.assign( 2, 0 );
            test.require( solver.solve( rhs, x ), "Repeated monitored solve" );
        }
        test.require( allocations == space->allocations, "No solve-time workspace allocation" );
        op->fail_after( op->operator_calls() + 1 );
        x.assign( 2, 0 );
        test.require( !solver.solve( rhs, x ), "Failure inside active monitor scope" );
    }
    test.require(
        space->allocations == space->frees && space->starts == space->stops,
        "Balanced allocation and monitor activation"
    );
    struct throwing_prec
    {
        bool fail = true;
        void set_operator( std::shared_ptr<const operator_type> )
        {
        }
        void apply( counted_space::vector_type & ) const
        {
            if ( fail )
                throw std::runtime_error( "Injected backend failure" );
        }
    };
    {
        auto op   = std::make_shared<operator_type>( *space, std::vector<double>{ 1, 0, 0, 1 } );
        auto prec = std::make_shared<throwing_prec>();
        using throwing_solver =
            nmfd::solvers::jacobi<counted_space, operator_type, throwing_prec, monitor_type, log_type>;
        typename throwing_solver::params p;
        p.monitor.out_min_resid_norm = true;
        throwing_solver     solver( op, space, nullptr, p, prec );
        std::vector<double> rhs{ 1, 2 }, x( 2 );
        const int           starts = space->starts, stops = space->stops;
        caught = false;
        try
        {
            solver.solve( rhs, x );
        }
        catch ( const std::runtime_error & )
        {
            caught = true;
        }
        test.require(
            caught && space->starts - starts == space->stops - stops, "Exception releases the active monitor"
        );
        prec->fail = false;
        test.require( solver.solve( rhs, x ), "Solver remains usable after a backend exception" );
    }
    std::weak_ptr<counted_space> weak   = space;
    auto                         solver = std::make_unique<solver_type>( space );
    space.reset();
    test.require( !weak.expired(), "Solver retains its space" );
    solver.reset();
    test.require( weak.expired(), "Sole-owner cleanup without cycles" );

    struct backend_type
    {
        log_type  logger;
        log_type &log()
        {
            return logger;
        }
    } backend;
    space = std::make_shared<counted_space>( 2 );
    typename solver_type::utils_hierarchy  utilities( backend, space );
    typename solver_type::params_hierarchy parameters;
    auto recursive = nmfd::detail::algo_hierarchy_creator<solver_type>::get( utilities, parameters );
    auto op        = std::make_shared<operator_type>( *space, std::vector<double>{ 1, 0, 0, 1 } );
    recursive->set_operator( op );
    std::vector<double> rhs{ 1, 2 }, x( 2 );
    test.require(
        recursive->solve( rhs, x ) && utilities.log == &backend.log(), "Hierarchy construction and borrowed logger"
    );
#ifdef NMFD_ENABLE_NLOHMANN
    auto                                   json = parameters.to_json();
    typename solver_type::params_hierarchy restored;
    restored.from_json( json );
    test.require( restored.to_json() == json, "Jacobi hierarchy JSON round trip" );
    typename solver_type::params partial;
    partial.from_json( nlohmann::json::object() );
    test.require(
        partial.monitor.max_iters_num == parameters.monitor.max_iters_num, "Optional JSON fields preserve defaults"
    );
#endif
}

}

int main()
{
    nmfd::tests::test_harness test;
    test_solve<float, nmfd::solvers::default_monitor>( test );
    test_solve<double, nmfd::solvers::default_monitor>( test );
    test_solve<long double, nmfd::solvers::default_monitor>( test );
    test_solve<std::complex<double>, nmfd::solvers::default_monitor>( test );
    test_solve<double, nmfd::solvers::monitor_krylov>( test );
    test_failures( test );
    test_ownership( test );
    return test.finish( "NMFD Jacobi" );
}
