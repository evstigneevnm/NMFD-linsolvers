#include <cmath>
#include <limits>
#include <memory>
#include <scfd/utils/log.h>
#ifdef TEST_OPERATIONS_CUDA
#    include <scfd/backend/cuda.h>
#    include <scfd/external_libraries/cublas_wrap_singleton_impl.h>
#    include <nmfd/operations/dense_operations_cuda_hip.h>
#else
#    include <scfd/backend/serial_cpu.h>
#    include <scfd/backend/omp.h>
#    include <nmfd/operations/dense_operations_base.h>
#endif
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/matrix_operator.h>
#include <nmfd/operations/backend/complex.h>
#include <nmfd/solvers/bicgstab.h>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/monitor_krylov.h>
#include "common/test_harness.h"

#ifndef NMFD_TEST_SOLVER
#    define NMFD_TEST_SOLVER bicgstab
#endif

namespace
{

template <class Backend, class T>
void test_backend( nmfd::tests::test_harness &test )
{
#ifdef TEST_OPERATIONS_CUDA
    using operations_type = nmfd::operations::dense_operations_cuda_hip<T, Backend>;
#else
    using operations_type = nmfd::operations::dense_operations<T, Backend>;
#endif
    using operator_type = nmfd::operations::matrix_operator<operations_type>;
    using space_type    = typename operator_type::vector_space_type;
    using log_type      = scfd::utils::log_std;
    using monitor_type  = nmfd::solvers::monitor_krylov<space_type, log_type>;
    using solver_type   = nmfd::solvers::NMFD_TEST_SOLVER<space_type, monitor_type, log_type, operator_type>;
    using norm_type     = typename space_type::norm_type;
    auto                                  operations = std::make_shared<operations_type>();
    typename operations_type::matrix_type matrix;
    operations->init_matrix( 8, 8, matrix );
    auto view         = matrix.create_view( false );
    using traits_type = typename operations_type::scalar_traits;
    for ( int i = 0; i < 8; ++i )
        for ( int j = 0; j < 8; ++j )
            view( i, j ) =
                traits_type::make( i == j ? 4 + i : ( i == j + 1 ? -1 : ( j == i + 1 ? 1 : 0 ) ), i == j ? 0.2 : 0 );
    view.release( true );
    auto op    = std::make_shared<operator_type>( operations, std::move( matrix ) );
    auto space = op->get_dom_space();
    nmfd::detail::vector_wrap<space_type, false, false> exact( *space ), rhs( *space ), x( *space ), residual( *space );
    exact.init();
    exact.start_use();
    rhs.init();
    rhs.start_use();
    x.init();
    x.start_use();
    residual.init();
    residual.start_use();
    space->assign_scalar( T( 2 ), *exact );
    op->apply( *exact, *rhs );
    const auto                   tolerance = norm_type( 500 ) * std::numeric_limits<norm_type>::epsilon();
    typename solver_type::params p;
    p.monitor.rel_tol       = tolerance;
    p.monitor.abs_tol       = tolerance * norm_type( 0.01 );
    p.monitor.max_iters_num = 100;
    for ( bool high_precision : { false, true } )
    {
        if ( high_precision )
            space->set_high_precision();
        else
            space->set_regular_precision();
        for ( bool recompute : { false, true } )
        {
            p.use_real_resid = recompute;
            solver_type solver( op, space, nullptr, p );
            space->assign_scalar( T( 0 ), *x );
            for ( int attempt = 0; attempt < 2; ++attempt )
            {
                test.require( solver.solve( *rhs, *x ), "Canonical backend solve" );
                op->apply( *x, *residual );
                space->add_lin_comb( T( 1 ), *rhs, T( -1 ), *residual );
                test.require(
                    space->norm( *residual ) <= norm_type( 2 ) * tolerance * space->norm( *rhs ),
                    "Canonical backend true residual"
                );
                space->assign_lin_comb( T( 1 ), *x, T( -1 ), *exact, *residual );
                test.require(
                    space->norm( *residual ) <= norm_type( 10 ) * tolerance * space->norm( *exact ),
                    "Canonical backend solution error"
                );
                space->scale( T( 0.9 ), *x );
            }
        }
    }
}

template <class Backend>
void test_backend( nmfd::tests::test_harness &test )
{
    using complex_type = nmfd::operations::backend::complex_t<Backend, double>;
    test_backend<Backend, float>( test );
    test_backend<Backend, double>( test );
    test_backend<Backend, complex_type>( test );
}

}

int main()
{
    nmfd::tests::test_harness test;
#ifdef TEST_OPERATIONS_CUDA
    using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
    backend_type::init_device();
    scfd::cublas_wrap blas( false, true );
    test_backend<backend_type>( test );
    scfd::cublas_wrap::set_inst( nullptr );
#else
    test_backend<scfd::backend::serial_cpu<std::ptrdiff_t>>( test );
    test_backend<scfd::backend::omp<std::ptrdiff_t>>( test );
#endif
    return test.finish( "Short-recurrence canonical backends" );
}
