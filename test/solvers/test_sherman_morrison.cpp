#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <scfd/utils/log.h>
#if defined( TEST_OPERATIONS_CUDA )
#    include <scfd/backend/cuda.h>
#    include <scfd/external_libraries/cublas_wrap_singleton_impl.h>
#    include <nmfd/operations/dense_operations_cuda_hip.h>
#elif defined( TEST_OPERATIONS_HIP )
#    include <scfd/backend/hip.h>
#    include <scfd/external_libraries/hipblas_wrap_singleton_impl.h>
#    include <nmfd/operations/dense_operations_cuda_hip.h>
#else
#    include <scfd/backend/serial_cpu.h>
#    include <scfd/backend/omp.h>
#    include <nmfd/operations/dense_operations_base.h>
#endif
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/matrix_operator.h>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/sherman_morrison.h>
#include "common/test_harness.h"

namespace
{

template <class Operations>
struct test_operator : nmfd::operations::matrix_operator<Operations>
{
    using parent_type = nmfd::operations::matrix_operator<Operations>;
    using vector_type = typename parent_type::vector_type;
    using parent_type::parent_type;
    bool fail = false;
    bool apply( const vector_type &x, vector_type &out ) const
    {
        if ( fail )
            return false;
        parent_type::apply( x, out );
        return true;
    }
};

template <class Space, class Operator>
struct preconditioner
{
    using vector_type = typename Space::vector_type;
    std::shared_ptr<Space> space;
    bool                   fail           = false;
    bool                   throw_on_apply = false;
    void                   set_operator( std::shared_ptr<const Operator> )
    {
    }
    bool apply( vector_type &x ) const
    {
        if ( throw_on_apply )
            throw std::runtime_error( "injected preconditioner exception" );
        if ( fail )
            return false;
        space->scale( typename Space::scalar_type( 0.25 ), x );
        return true;
    }
};

template <class Vector>
void fill( Vector &v, std::initializer_list<double> values )
{
    auto view = v.create_view( false );
    int  i    = 0;
    for ( double value : values )
        view( i++ ) = value;
    view.release( true );
}

template <class Backend, template <class, class, class, class, class> class Solver>
void test_solver( nmfd::tests::test_harness &test )
{
#if defined( TEST_OPERATIONS_CUDA ) || defined( TEST_OPERATIONS_HIP )
    using operations_type = nmfd::operations::dense_operations_cuda_hip<double, Backend>;
#else
    using operations_type = nmfd::operations::dense_operations<double, Backend>;
#endif
    using operator_type = test_operator<operations_type>;
    using space_type    = typename operator_type::vector_space_type;
    using log_type      = scfd::utils::log_std;
    using monitor_type  = nmfd::solvers::default_monitor<space_type, log_type>;
    using prec_type     = preconditioner<space_type, operator_type>;
    using solver_type =
        nmfd::solvers::sherman_morrison<space_type, monitor_type, log_type, operator_type, prec_type, Solver>;
    auto                                  operations = std::make_shared<operations_type>();
    typename operations_type::matrix_type matrix;
    operations->init_matrix( 3, 3, matrix );
    auto         view          = matrix.create_view( false );
    const double entries[3][3] = { { 4, 1, 0 }, { -1, 5, 1 }, { 0, -1, 6 } };
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            view( i, j ) = entries[i][j];
    view.release( true );
    auto op                                    = std::make_shared<operator_type>( operations, std::move( matrix ) );
    auto space                                 = op->get_dom_space();
    auto prec                                  = std::make_shared<prec_type>();
    prec->space                                = space;
    std::weak_ptr<space_type>    weak_space    = space;
    std::weak_ptr<operator_type> weak_operator = op;
    std::weak_ptr<prec_type>     weak_prec     = prec;
    {
        nmfd::detail::vector_wrap<space_type, false, false> c( *space ), d( *space ), b( *space ), x( *space ),
            exact( *space ), residual( *space );
        for ( auto *v : { &c, &d, &b, &x, &exact, &residual } )
        {
            v->init();
            v->start_use();
        }
        fill( *c, { .5, -1, 2 } );
        fill( *d, { 1, .25, -.5 } );
        fill( *exact, { 1, -2, 3 } );
        solver_type                   solver( space, op, prec );
        typename monitor_type::params p;
        p.rel_tol            = 1e-9;
        p.abs_tol            = 1e-11;
        p.max_iters_num      = 100;
        p.out_min_resid_norm = true;
        solver.get_linsolver_handle()->monitor().configure( p );
        solver.get_linsolver_handle()->set_use_precond_resid( false );
        solver.get_linsolver_handle_original()->monitor().configure( p );
        const auto owners = space.use_count();
        op->apply( *exact, *b );
        space->assign_scalar( 0., *x );
        test.require( solver.solve( *op, *b, *x ), "ordinary solve converges" );
        space->assign_lin_comb( 1., *x, -1., *exact, *residual );
        test.require( space->norm( *residual ) < 1e-8, "ordinary solution matches assembled reference" );
        solver.is_small_alpha( true );
        for ( double alpha : { 2., 1e-4, -1e-4, 2. } )
        {
            // Exercise pivot scaling without making the reference system ill conditioned.
            const double scale = std::abs( alpha ) < 1e-3 ? alpha : 1.;
            fill( *d, { scale, .25 * scale, -.5 * scale } );
            const double gamma = 1.3;
            op->apply( *exact, *b );
            space->add_lin_comb( -space->scalar_prod( *c, *exact ) / alpha, *d, gamma, *b );
            space->assign_scalar( 0., *x );
            test.require( solver.solve( gamma, *op, alpha, *c, *d, *b, *x ), "rank-one solve converges" );
            space->assign_lin_comb( 1., *x, -1., *exact, *residual );
            test.require( space->norm( *residual ) < 1e-7, "rank-one solution matches assembled reference" );
            op->apply( *x, *residual );
            space->add_lin_comb( -space->scalar_prod( *c, *x ) / alpha, *d, gamma, *residual );
            space->add_lin_comb( -1., *b, 1., *residual );
            test.require( space->norm( *residual ) < 1e-8 * ( 1. + space->norm( *b ) ), "rank-one true residual" );

            op->apply( *exact, *b );
            space->add_lin_comb( 2., *d, 1., *b );
            const double beta = space->scalar_prod( *c, *exact ) + alpha * 2.;
            double       v    = 99.;
            space->assign_scalar( 0., *x );
            auto &monitor = solver.get_linsolver_handle()->monitor();
            monitor.set_temp_tolerance( p.rel_tol * .5 );
            const bool converged = solver.solve( *op, *c, *d, alpha, *b, beta, *x, v );
            monitor.restore_tolerance();
            test.require(
                monitor.rel_tol() == p.rel_tol && monitor.abs_tol() == p.abs_tol,
                "border solve preserves caller tolerance overrides"
            );
            test.require( converged, "scalar-bordered solve converges" );
            test.require_close( v, 2., 1e-7, "scalar-border component matches assembled reference" );
            space->assign_lin_comb( 1., *x, -1., *exact, *residual );
            test.require( space->norm( *residual ) < 1e-7, "bordered vector matches assembled reference" );
            op->apply( *x, *residual );
            space->add_lin_comb( v, *d, 1., *residual );
            space->add_lin_comb( -1., *b, 1., *residual );
            test.require(
                std::hypot( space->norm( *residual ), space->scalar_prod( *c, *x ) + alpha * v - beta ) < 1e-7,
                "bordered true residual"
            );
            auto input_view = ( *d ).create_view( true );
            test.require(
                input_view( 0 ) == scale && input_view( 1 ) == .25 * scale && input_view( 2 ) == -.5 * scale,
                "rank-one and bordered solves preserve d at every pivot"
            );
            input_view.release( false );
        }
        auto d_view = ( *d ).create_view( true );
        test.require_close( d_view( 0 ), 1., 0., "input d preserved" );
        test.require_close( d_view( 1 ), .25, 0., "input d preserved after small-alpha preconditioning" );
        d_view.release( false );
        prec->fail = true;
        double v   = 99.;
        space->assign_scalar( 0., *x );
        test.require( !solver.solve( *op, *c, *d, 2., *b, 1., *x, v ), "preconditioner failure propagated" );
        test.require( v == 99., "failed bordered solve does not publish scalar" );
        solver.set_preconditioner_bypass( true );
        op->apply( *exact, *b );
        space->assign_scalar( 0., *x );
        test.require( solver.solve( *op, *b, *x ), "unpreconditioned recovery succeeds" );
        solver.set_preconditioner_bypass( false );
        test.require( !solver.preconditioner_bypass_enabled(), "preconditioner restored" );
        prec->fail = false;
        op->fail   = true;
        test.require( !solver.solve( *op, *b, *x ), "operator failure propagated" );
        op->fail             = false;
        auto &monitor        = solver.get_linsolver_handle()->monitor();
        prec->throw_on_apply = true;
        bool threw           = false;
        try
        {
            solver.solve( *op, *c, *d, 1e-4, *b, 1., *x, v );
        }
        catch ( const std::runtime_error & )
        {
            threw = true;
        }
        test.require(
            threw && monitor.rel_tol() == p.rel_tol && monitor.abs_tol() == p.abs_tol,
            "exception restores internal tolerances"
        );
        prec->throw_on_apply = false;
        // A strongly scaled elimination may fail, but must never certify a bad full residual.
        op->apply( *exact, *b );
        space->add_lin_comb( 2., *d, 1., *b );
        space->assign_scalar( 0., *x );
        v                      = 99.;
        const double beta      = space->scalar_prod( *c, *exact ) + 2e-4;
        const bool   difficult = solver.solve( *op, *c, *d, 1e-4, *b, beta, *x, v );
        if ( difficult )
        {
            op->apply( *x, *residual );
            space->add_lin_comb( v, *d, 1., *residual );
            space->add_lin_comb( -1., *b, 1., *residual );
            test.require(
                space->norm( *residual ) <= p.abs_tol + p.rel_tol * std::hypot( space->norm( *b ), beta ),
                "difficult elimination certifies the original residual"
            );
        }
        else
            test.require( v == 99., "unresolved elimination does not publish a false success" );
        test.require( !solver.solve( 1., *op, 0., *c, *d, *b, *x ), "zero elimination pivot reports failure" );
        test.require(
            !solver.solve( 1., *op, std::numeric_limits<double>::quiet_NaN(), *c, *d, *b, *x ),
            "nonfinite pivot reports failure"
        );
        test.require( space.use_count() == owners, "solves do not retain additional shared handles" );
    }
    prec.reset();
    op.reset();
    space.reset();
    test.require(
        weak_space.expired() && weak_operator.expired() && weak_prec.expired(),
        "dependency graph has no ownership cycle"
    );
}

template <class Backend>
void test_backend( nmfd::tests::test_harness &test )
{
    test_solver<Backend, nmfd::solvers::cgs>( test );
    test_solver<Backend, nmfd::solvers::bicgstabl>( test );
}

}

int main()
{
    nmfd::tests::test_harness test;
#if defined( TEST_OPERATIONS_CUDA )
    scfd::backend::cuda<std::ptrdiff_t>::init_device();
    scfd::cublas_wrap blas( false, true );
    test_backend<scfd::backend::cuda<std::ptrdiff_t>>( test );
    scfd::cublas_wrap::set_inst( nullptr );
#elif defined( TEST_OPERATIONS_HIP )
    scfd::backend::hip<std::ptrdiff_t>::init_device();
    scfd::hipblas_wrap blas( false, true );
    test_backend<scfd::backend::hip<std::ptrdiff_t>>( test );
    scfd::hipblas_wrap::set_inst( nullptr );
#else
    test_backend<scfd::backend::serial_cpu<std::ptrdiff_t>>( test );
    test_backend<scfd::backend::omp<std::ptrdiff_t>>( test );
#endif
    return test.finish( "Sherman-Morrison integration" );
}
