#include <limits>
#include <memory>
#include <stdexcept>
#include <scfd/utils/log.h>
#if defined( TEST_OPERATIONS_CUDA )
#    include <scfd/backend/cuda.h>
#    include <scfd/external_libraries/cublas_wrap_singleton_impl.h>
#    include <scfd/external_libraries/cusolver_wrap_singleton_impl.h>
#    include <nmfd/operations/dense_operations_cuda_hip.h>
#elif defined( TEST_OPERATIONS_HIP )
#    include <scfd/backend/hip.h>
#    include <scfd/external_libraries/hipblas_wrap_singleton_impl.h>
#    include <scfd/external_libraries/hipsolver_wrap_singleton_impl.h>
#    include <nmfd/operations/dense_operations_cuda_hip.h>
#else
#    include <scfd/backend/serial_cpu.h>
#    include <scfd/backend/omp.h>
#    include <nmfd/operations/dense_operations_base.h>
#endif
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/matrix_operator.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/direct_solver.h>
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

template <class Operations, class Operator>
struct dense_inverse
{
    using vector_type = typename Operations::vector_type;
    using T           = typename Operations::scalar_type;
    std::shared_ptr<Operations>     operations;
    std::shared_ptr<const Operator> op;
    bool                            fail           = false;
    bool                            inaccurate     = false;
    bool                            nonfinite      = false;
    bool                            throw_on_apply = false;
    T                               relative_error = T( 0 );
    void                            set_operator( std::shared_ptr<const Operator> A )
    {
        op = std::move( A );
    }
    bool apply( vector_type &x ) const
    {
        if ( throw_on_apply )
            throw std::runtime_error( "injected inverse exception" );
        if ( fail )
            return false;
        if ( nonfinite )
        {
            operations->assign_scalar( std::numeric_limits<T>::infinity(), x );
            return true;
        }
        if ( inaccurate )
            return true;
        if ( !nmfd::solvers::krylov::invoke_status( &Operations::solve, *operations, *op, x, x ) )
            return false;
        if ( relative_error != T( 0 ) )
            operations->scale( T( 1 ) + relative_error, x );
        return true;
    }
};

template <class T, class Backend>
void test_solver( nmfd::tests::test_harness &test )
{
#if defined( TEST_OPERATIONS_CUDA ) || defined( TEST_OPERATIONS_HIP )
    using operations_type = nmfd::operations::dense_operations_cuda_hip<T, Backend>;
#else
    using operations_type = nmfd::operations::dense_operations<T, Backend>;
#endif
    using operator_type = test_operator<operations_type>;
    using space_type    = typename operator_type::vector_space_type;
    using monitor_type  = nmfd::solvers::default_monitor<space_type, scfd::utils::log_std>;
    using inverse_type  = dense_inverse<operations_type, operator_type>;
    using solver_type =
        nmfd::solvers::direct_solver<space_type, monitor_type, scfd::utils::log_std, operator_type, inverse_type>;
    using extended_type = nmfd::solvers::sherman_morrison<
        space_type, monitor_type, scfd::utils::log_std, operator_type, inverse_type, nmfd::solvers::direct_solver>;
    auto                                  operations = std::make_shared<operations_type>();
    typename operations_type::matrix_type matrix;
    matrix.init( 3, 3 );
    auto    view          = matrix.create_view( false );
    const T entries[3][3] = { { 0, 2, 1 }, { 1, -2, -3 }, { -1, 1, 2 } };
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            view( i, j ) = entries[i][j];
    view.release( true );
    auto op                                   = std::make_shared<operator_type>( operations, std::move( matrix ) );
    auto space                                = op->get_dom_space();
    auto inverse                              = std::make_shared<inverse_type>();
    inverse->operations                       = operations;
    std::weak_ptr<operator_type> weak_op      = op;
    std::weak_ptr<space_type>    weak_space   = space;
    std::weak_ptr<inverse_type>  weak_inverse = inverse;
    {
        nmfd::detail::vector_wrap<space_type, false, false> b( *space ), x( *space ), exact( *space ), c( *space ),
            d( *space ), residual( *space );
        for ( auto *vector : { &b, &x, &exact, &c, &d, &residual } )
        {
            vector->init();
            vector->start_use();
        }
        auto reference = ( *exact ).create_view( false );
        reference( 0 ) = T( 1 );
        reference( 1 ) = T( -2 );
        reference( 2 ) = T( 3 );
        reference.release( true );
        typename solver_type::params parameters;
        parameters.monitor.rel_tol            = T( 200 ) * std::numeric_limits<T>::epsilon();
        parameters.monitor.abs_tol            = parameters.monitor.rel_tol;
        parameters.monitor.out_min_resid_norm = true;
        solver_type solver( space, nullptr, parameters, inverse );
        test.require( solver.status() == solver_type::result_status::idle, "initial status is idle" );
        solver.set_operator( op );
        const auto owners = space.use_count();
        for ( int repeat = 0; repeat < 3; ++repeat )
        {
            op->apply( *exact, *b );
            space->assign_scalar( T( 7 ), *x );
            test.require( solver.solve( *b, *x ), "bound direct solve" );
            space->assign_lin_comb( T( 1 ), *x, T( -1 ), *exact, *residual );
            test.require( space->norm( *residual ) < parameters.monitor.rel_tol * T( 10 ), "known direct solution" );
            test.require( solver.monitor().iters_performed() == 1, "one inverse per solve" );
            test.require( solver.status() == solver_type::result_status::converged, "success status" );
            space->assign_lin_comb( T( 1 ), solver.monitor().min_resid_norm_x(), T( -1 ), *x, *residual );
            test.require( space->norm( *residual ) == T( 0 ), "minimum-residual state initialized on first inverse" );
        }
        for ( int mode = 0; mode < 5; ++mode )
        {
            inverse->fail       = mode == 0;
            inverse->nonfinite  = mode == 1;
            inverse->inaccurate = mode == 2;
            op->fail            = mode == 3;
            solver.set_preconditioner_bypass( mode == 4 );
            space->assign_scalar( T( 7 ), *x );
            test.require( !solver.solve( *b, *x ), "inverse/operator/residual failure propagated" );
            auto unchanged = ( *x ).create_view( true );
            for ( int i = 0; i < 3; ++i )
                test.require( unchanged( i ) == T( 7 ), "failed solve preserves output" );
        }
        inverse->fail = inverse->nonfinite = inverse->inaccurate = op->fail = false;
        solver.set_preconditioner_bypass( false );
        space->assign_scalar( std::numeric_limits<T>::infinity(), *b );
        test.require(
            !solver.solve( *b, *x ) && solver.status() == solver_type::result_status::invalid_input,
            "nonfinite RHS has an explicit failure status"
        );
        test.require( !solver.monitor().converged(), "failed attempt cannot retain previous convergence" );
        op->apply( *exact, *b );
        test.require( solver.solve( *b, *b ), "RHS/output aliasing" );
        // Changing values in the same bound matrix must not reuse stale inverse data.
        auto changed = op->create_view( true );
        changed( 0, 0 ) += T( 2 );
        changed.release( true );
        solver.update_operator( op );
        op->apply( *exact, *b );
        test.require( solver.solve( *op, *b, *x ), "updated matrix values" );
        test.require( space.use_count() == owners, "no shared handle copies retained by solve" );
        inverse->throw_on_apply = true;
        bool threw              = false;
        try
        {
            solver.solve( *b, *x );
        }
        catch ( const std::runtime_error & )
        {
            threw = true;
        }
        inverse->throw_on_apply = false;
        test.require( threw && solver.solve( *b, *x ), "backend exceptions propagate and allow recovery" );

        extended_type extended( space, op, inverse );
        extended.get_linsolver_handle()->monitor().configure( parameters.monitor );
        extended.get_linsolver_handle_original()->monitor().configure( parameters.monitor );
        space->assign_scalar( T( 0.1 ), *c );
        space->assign_scalar( T( 0.2 ), *d );
        test.require( extended.solve( *op, *b, *x ), "ordinary supplied-inverse integration" );
        op->apply( *exact, *b );
        space->add_lin_comb( -space->scalar_prod( *c, *exact ) / T( 2 ), *d, T( 1 ), *b );
        test.require( extended.solve( T( 1 ), *op, T( 2 ), *c, *d, *b, *x ), "rank-one direct solve" );
        space->assign_lin_comb( T( 1 ), *x, T( -1 ), *exact, *residual );
        test.require( space->norm( *residual ) < parameters.monitor.rel_tol * T( 10 ), "rank-one assembled reference" );
        op->apply( *exact, *b );
        space->add_lin_comb( T( 2 ), *d, T( 1 ), *b );
        T       v    = T( 99 );
        const T beta = space->scalar_prod( *c, *exact ) + T( 4 );
        test.require( extended.solve( *op, *c, *d, T( 2 ), *b, beta, *x, v ), "scalar-bordered direct solve" );
        test.require_close( v, T( 2 ), double( parameters.monitor.rel_tol ) * 10, "border scalar reference" );
        space->assign_lin_comb( T( 1 ), *x, T( -1 ), *exact, *residual );
        test.require( space->norm( *residual ) < parameters.monitor.rel_tol * T( 10 ), "border vector reference" );

        // A controlled inverse error fits the original budget but not an extra small-pivot tightening.
        auto identity = op->create_view( false );
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                identity( i, j ) = i == j ? T( 1 ) : T( 0 );
        identity.release( true );
        space->assign_scalar( T( 0 ), *c );
        auto c_view = ( *c ).create_view( true );
        c_view( 0 ) = T( 1 );
        c_view.release( true );
        space->assign_scalar( T( 0 ), *d );
        auto d_view = ( *d ).create_view( true );
        d_view( 1 ) = T( 1 );
        d_view.release( true );
        inverse->relative_error    = T( 32 ) * std::numeric_limits<T>::epsilon();
        parameters.monitor.rel_tol = T( 256 ) * std::numeric_limits<T>::epsilon();
        parameters.monitor.abs_tol = T( 0 );
        auto &monitor              = extended.get_linsolver_handle()->monitor();
        monitor.configure( parameters.monitor );
        for ( bool scaled : { false, true } )
        {
            extended.is_small_alpha( scaled );
            for ( T alpha : { T( 1e-4 ), T( -1e-4 ) } )
            {
                space->assign( *d, *b );
                space->assign_scalar( T( 0 ), *x );
                monitor.set_temp_tolerance( parameters.monitor.rel_tol / T( 2 ) );
                const T tolerance = monitor.rel_tol() * space->norm( *b );
                v                 = T( 99 );
                test.require(
                    extended.solve( *op, *c, *d, alpha, *b, T( 0 ), *x, v ),
                    "small-pivot direct solve uses the original bordered tolerance"
                );
                test.require(
                    monitor.rel_tol() == parameters.monitor.rel_tol / T( 2 ) && monitor.abs_tol() == T( 0 ),
                    "small-pivot solve restores caller tolerance override"
                );
                monitor.restore_tolerance();
                test.require( monitor.rel_tol() == parameters.monitor.rel_tol, "caller restores baseline tolerance" );
                op->apply( *x, *residual );
                space->add_lin_comb( v, *d, T( 1 ), *residual );
                space->add_lin_comb( T( -1 ), *b, T( 1 ), *residual );
                const T norm = std::hypot( space->norm( *residual ), space->scalar_prod( *c, *x ) + alpha * v );
                test.require( norm <= tolerance, "accepted small-pivot correction meets the full residual target" );
                test.require(
                    norm > std::abs( alpha ) * tolerance, "obsolete pivot tightening would reject correction"
                );

                space->assign_scalar( T( 0 ), *b );
                v = T( 99 );
                test.require(
                    !extended.solve( *op, *c, *d, alpha, *b, T( 1 ), *x, v ),
                    "large eliminated RHS cannot inflate the original residual budget"
                );
                test.require( v == T( 99 ), "inaccurate bordered correction does not publish scalar" );
            }
        }
        inverse->relative_error = T( 0 );
        space->assign_scalar( T( 0 ), *b );
        test.require(
            extended.solve( *op, *c, *d, T( 1e-4 ), *b, T( 0 ), *x, v ),
            "zero bordered RHS supports an exactly zero tolerance"
        );
        space->assign_scalar( std::numeric_limits<T>::infinity(), *b );
        test.require(
            !extended.solve( *op, *c, *d, T( 1e-4 ), *b, T( 0 ), *x, v ),
            "nonfinite original residual budget reports failure"
        );
    }
    inverse.reset();
    op.reset();
    space.reset();
    test.require( weak_op.expired() && weak_space.expired() && weak_inverse.expired(), "no ownership cycles" );
}

template <class Backend>
void test_backend( nmfd::tests::test_harness &test )
{
    test_solver<float, Backend>( test );
    test_solver<double, Backend>( test );
}
}

int main()
{
    nmfd::tests::test_harness test;
#if defined( TEST_OPERATIONS_CUDA )
    scfd::backend::cuda<std::ptrdiff_t>::init_device();
    scfd::cublas_wrap   blas( false, true );
    scfd::cusolver_wrap solver( &blas, true );
    test_backend<scfd::backend::cuda<std::ptrdiff_t>>( test );
#elif defined( TEST_OPERATIONS_HIP )
    scfd::backend::hip<std::ptrdiff_t>::init_device();
    scfd::hipblas_wrap   blas( false, true );
    scfd::hipsolver_wrap solver( &blas, true );
    test_backend<scfd::backend::hip<std::ptrdiff_t>>( test );
#else
    test_backend<scfd::backend::serial_cpu<std::ptrdiff_t>>( test );
    test_backend<scfd::backend::omp<std::ptrdiff_t>>( test );
#endif
    return test.finish( "Direct solver and Sherman-Morrison" );
}
