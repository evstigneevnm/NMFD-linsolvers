#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef TEST_OPERATIONS_CUDA
#    include <scfd/backend/cuda.h>
#    include <scfd/external_libraries/cublas_wrap_singleton_impl.h>
#    include <nmfd/operations/dense_operations_cuda_hip.h>
#else
#    include <scfd/backend/serial_cpu.h>
#    include <scfd/backend/omp.h>
#    include <nmfd/operations/dense_operations_base.h>
#endif
#include <scfd/utils/log.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/glued_matrix_operator.h>
#include <nmfd/operations/matrix_operator.h>
#include <nmfd/operations/scfd_vector_space.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>

#include "common/test_harness.h"

namespace
{

template <class VectorSpace>
struct scaled_operator
{
    using vector_type = typename VectorSpace::vector_type;
    std::shared_ptr<VectorSpace> space;

    void apply( const vector_type &in, vector_type &out ) const
    {
        space->assign_lin_comb( typename VectorSpace::scalar_type( 3 ), in, out );
    }
};

template <class Operator, class VectorSpace>
void check_solve(
    const std::shared_ptr<Operator> &op, const std::shared_ptr<VectorSpace> &space, nmfd::tests::test_harness &test
)
{
    using T                    = typename VectorSpace::scalar_type;
    using norm_type            = typename VectorSpace::norm_type;
    using log_type             = scfd::utils::log_std;
    using monitor_type         = nmfd::solvers::monitor_krylov<VectorSpace, log_type>;
    using solver_type          = nmfd::solvers::gmres<VectorSpace, monitor_type, log_type, Operator>;
    using operations_base_type = typename VectorSpace::operations_base_type;
    using factory_base_type    = nmfd::operations::vector_factory_base<
           T, typename VectorSpace::vector_type, typename VectorSpace::multivector_type,
           typename VectorSpace::ordinal_type>;
    static_assert( std::is_floating_point_v<norm_type> );
    static_assert( std::is_base_of_v<operations_base_type, VectorSpace> );
    static_assert( std::is_base_of_v<factory_base_type, VectorSpace> );

    nmfd::detail::vector_wrap<VectorSpace, false, false> exact( *space ), rhs( *space ), solution( *space ),
        residual( *space );
    exact.init();
    exact.start_use();
    rhs.init();
    rhs.start_use();
    solution.init();
    solution.start_use();
    residual.init();
    residual.start_use();
    space->assign_scalar( T( 2 ), *exact );
    op->apply( *exact, *rhs );
    const norm_type tolerance = norm_type( 100 ) * std::numeric_limits<norm_type>::epsilon();
    log_type        log;
    log.set_verbosity( -1 );
    for ( bool high_precision : { false, true } )
    {
        operations_base_type &operations = *space;
        if ( high_precision )
            operations.set_high_precision();
        else
            operations.set_regular_precision();
        for ( char side : { 'L', 'R' } )
        {
            typename solver_type::params parameters;
            parameters.basis_size                   = 4;
            parameters.batch_size                   = 1;
            parameters.preconditioner_side          = side;
            parameters.orthogonalization            = "mgs";
            parameters.reorthogonalization_policy   = "always";
            parameters.max_orthogonalization_passes = 2;
            parameters.monitor.rel_tol              = tolerance;
            parameters.monitor.abs_tol              = tolerance * norm_type( 0.01 );
            parameters.monitor.max_iters_num        = 40;
            solver_type solver( op, space, &log, parameters );
            space->assign_scalar( T( 0 ), *solution );
            for ( int attempt = 0; attempt < 2; ++attempt )
            {
                test.require( solver.solve( *rhs, *solution ), "Canonical space GMRES converges" );
                op->apply( *solution, *residual );
                space->add_lin_comb( T( 1 ), *rhs, T( -1 ), *residual );
                test.require(
                    space->norm( *residual ) <= norm_type( 10 ) * tolerance * space->norm( *rhs ),
                    "Canonical space true residual"
                );
                space->assign_lin_comb( T( 1 ), *solution, T( -1 ), *exact, *residual );
                test.require(
                    space->norm( *residual ) <= norm_type( 10 ) * tolerance * space->norm( *exact ),
                    "Canonical space solution error"
                );
                space->scale( T( 0.9 ), *solution );
            }
        }
    }
}

template <class Operations>
void fill_matrix(
    Operations &operations, typename Operations::matrix_type &matrix, int rows, int columns, int row_offset = 0,
    int column_offset = 0
)
{
    using traits_type = typename Operations::scalar_traits;
    operations.init_matrix( rows, columns, matrix );
    auto view = matrix.create_view( false );
    for ( int column = 0; column < columns; ++column )
        for ( int row = 0; row < rows; ++row )
        {
            const int i = row + row_offset, j = column + column_offset;
            const int real      = i == j ? 4 + i : ( i == j + 1 ? -1 : ( j == i + 1 ? 1 : 0 ) );
            view( row, column ) = traits_type::make( real, i == j ? 0.25 : 0 );
        }
    view.release( true );
}

template <class Backend, class T>
void check_backend( nmfd::tests::test_harness &test )
{
#ifdef TEST_OPERATIONS_CUDA
    using operations_type = nmfd::operations::dense_operations_cuda_hip<T, Backend>;
#else
    using operations_type = nmfd::operations::dense_operations<T, Backend>;
#endif
    using matrix_operator_type                       = nmfd::operations::matrix_operator<operations_type>;
    using glued_operator_type                        = nmfd::operations::glued_matrix_operator<operations_type, 2>;
    auto                                  operations = std::make_shared<operations_type>();
    typename operations_type::matrix_type matrix;
    fill_matrix( *operations, matrix, 4, 4 );
    auto op = std::make_shared<matrix_operator_type>( operations, std::move( matrix ) );
    check_solve( op, op->get_dom_space(), test );

    constexpr std::array<int, 2>                   sizes{ 1, 3 }, offsets{ 0, 1 };
    typename glued_operator_type::block_array_type blocks;
    for ( std::size_t i = 0; i < 2; ++i )
        for ( std::size_t j = 0; j < 2; ++j )
        {
            blocks[i][j] = std::make_shared<typename operations_type::matrix_type>();
            fill_matrix( *operations, *blocks[i][j], sizes[i], sizes[j], offsets[i], offsets[j] );
        }
    auto glued = std::make_shared<glued_operator_type>( operations, std::move( blocks ) );
    check_solve( glued, glued->get_dom_space(), test );

    using space_type = nmfd::operations::scfd_vector_space<Backend, T>;
    auto space       = std::make_shared<space_type>( 4 );
    auto scaled      = std::make_shared<scaled_operator<space_type>>( scaled_operator<space_type>{ space } );
    check_solve( scaled, space, test );
}

template <class Backend>
void check_backend( nmfd::tests::test_harness &test )
{
    using complex_type = nmfd::operations::detail::scfd_backend_ext::complex_t<Backend, double>;
    check_backend<Backend, float>( test );
    check_backend<Backend, double>( test );
    check_backend<Backend, complex_type>( test );
}

}

int main()
{
    try
    {
        nmfd::tests::test_harness test;
#ifdef TEST_OPERATIONS_CUDA
        using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
        backend_type::init_device();
        scfd::cublas_wrap blas( false, true );
        check_backend<backend_type>( test );
        scfd::cublas_wrap::set_inst( nullptr );
#else
        check_backend<scfd::backend::serial_cpu<std::ptrdiff_t>>( test );
        check_backend<scfd::backend::omp<std::ptrdiff_t>>( test );
#endif
        return test.finish( "NMFD-operations integration" );
    }
    catch ( const std::exception &error )
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
