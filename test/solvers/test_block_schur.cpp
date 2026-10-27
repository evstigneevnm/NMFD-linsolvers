#include <complex>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

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
#include <nmfd/operations/block_operator.h>
#include <nmfd/operations/matrix_operator.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/preconditioners/block_schur.h>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <scfd/utils/log.h>

#include "common/test_harness.h"

namespace
{

template <class Operations>
struct checked_matrix : nmfd::operations::matrix_operator<Operations>
{
    using parent_type = nmfd::operations::matrix_operator<Operations>;
    using vector_type = typename parent_type::vector_type;
    using parent_type::parent_type;
    bool fail = false;

    bool apply( const vector_type &in, vector_type &out ) const
    {
        if ( fail )
            return false;
        parent_type::apply( in, out );
        return true;
    }
};

// Test-only inverse: sample a small operator and solve its assembled matrix, never form an inverse.
// Production H/Q may instead use fixed stationary sweeps, factorizations, or scalable field approximations.
template <class Operations, class Operator>
class sampled_inverse
{
    using space_type = typename Operations::vector_space_type;
    using T          = typename Operations::scalar_type;

public:
    using vector_type = typename space_type::vector_type;

    sampled_inverse( std::shared_ptr<Operations> operations, std::shared_ptr<space_type> space )
        : operations_( std::move( operations ) ), space_( std::move( space ) ), basis_( *space_ ), image_( *space_ ),
          candidate_( *space_ )
    {
        for ( auto *vector : { &basis_, &image_, &candidate_ } )
        {
            vector->init();
            vector->start_use();
        }
        const auto n = space_->size();
        operations_->init_matrix( n, n, matrix_ );
    }

    bool set_operator( std::shared_ptr<const Operator> op )
    {
        ++setups;
        ready_ = false;
        op_    = std::move( op );
        if ( throw_setup )
            throw std::runtime_error( "injected setup exception" );
        if ( fail_setup || !op_ )
            return false;
        auto       matrix = matrix_.create_view( false );
        const auto n      = space_->size();
        for ( std::size_t j = 0; j < n; ++j )
        {
            space_->assign_scalar( T( 0 ), *basis_ );
            space_->set_value_at_point( T( 1 ), j, *basis_ );
            if ( !nmfd::solvers::krylov::apply_operator( *op_, *basis_, *image_ ) )
            {
                matrix.release( false );
                return false;
            }
            auto column = ( *image_ ).create_view( true );
            for ( std::size_t i = 0; i < n; ++i )
                matrix( i, j ) = column( i ) + ( i == j ? shift : T( 0 ) );
            column.release( false );
        }
        matrix.release( true );
        space_->assign_scalar( T( 0 ), *basis_ );
        ready_ = operations_->solve( matrix_, *basis_, *candidate_ );
        return ready_;
    }

    bool apply( vector_type &in_out ) const
    {
        ++applications;
        if ( !ready_ || fail_apply || ( fail_on_application != 0 && applications == fail_on_application ) )
        {
            space_->assign_scalar( T( 99 ), in_out );
            return false;
        }
        if ( nonfinite )
        {
            space_->assign_scalar( T( std::numeric_limits<typename space_type::norm_type>::infinity() ), in_out );
            return true;
        }
        if ( !operations_->solve( matrix_, in_out, *candidate_ ) )
            return false;
        space_->scale( scale, *candidate_ );
        space_->assign( *candidate_, in_out );
        return true;
    }

    T           shift      = T( 0 );
    T           scale      = T( 1 );
    bool        fail_setup = false, fail_apply = false, throw_setup = false, nonfinite = false;
    int         setups = 0, fail_on_application = 0;
    mutable int applications = 0;

private:
    std::shared_ptr<Operations>                                 operations_;
    std::shared_ptr<space_type>                                 space_;
    std::shared_ptr<const Operator>                             op_;
    typename Operations::matrix_type                            matrix_;
    nmfd::detail::vector_wrap<space_type, false, false>         basis_, image_;
    mutable nmfd::detail::vector_wrap<space_type, false, false> candidate_;
    bool                                                        ready_ = false;
};

template <class Operations>
std::shared_ptr<checked_matrix<Operations>> make_matrix(
    const std::shared_ptr<Operations> &operations, int rows, int columns,
    const std::vector<typename Operations::scalar_type> &values
)
{
    typename Operations::matrix_type matrix;
    operations->init_matrix( rows, columns, matrix );
    auto view = matrix.create_view( false );
    for ( int i = 0; i < rows; ++i )
        for ( int j = 0; j < columns; ++j )
            view( i, j ) = values[i * columns + j];
    view.release( true );
    return std::make_shared<checked_matrix<Operations>>( operations, std::move( matrix ) );
}

template <class Space, class Operator, class Preconditioner, class Solver>
void check_solver(
    const std::shared_ptr<Space> &space, const std::shared_ptr<Operator> &op,
    const std::shared_ptr<Preconditioner> &prec, const typename Space::vector_type &rhs,
    const typename Space::vector_type &exact, nmfd::tests::test_harness &test
)
{
    using T                           = typename Space::scalar_type;
    using norm_type                   = typename Space::norm_type;
    const norm_type         tolerance = norm_type( 1000 ) * std::numeric_limits<norm_type>::epsilon();
    typename Solver::params p;
    p.basis_size            = 5;
    p.monitor.rel_tol       = tolerance;
    p.monitor.abs_tol       = tolerance;
    p.monitor.max_iters_num = 100;
    Solver                                         solver( space, nullptr, p, prec );
    nmfd::detail::vector_wrap<Space, false, false> solution( *space ), residual( *space );
    solution.init();
    solution.start_use();
    residual.init();
    residual.start_use();
    for ( int attempt = 0; attempt < 2; ++attempt )
    {
        solver.set_operator( op );
        space->assign_scalar( T( 0 ), *solution );
        test.require( prec->ready() && solver.solve( rhs, *solution ), "Augmented Krylov convergence" );
        test.require( op->apply( *solution, *residual ), "Complete original operator action" );
        space->add_lin_comb( T( -1 ), rhs, T( 1 ), *residual );
        test.require(
            space->norm( *residual ) <= tolerance * ( norm_type( 1 ) + space->norm( rhs ) ), "Complete true residual"
        );
        space->assign_lin_comb( T( 1 ), *solution, T( -1 ), exact, *residual );
        test.require( space->norm( *residual ) <= norm_type( 10 ) * tolerance, "Both solution blocks match reference" );
    }
    prec->set_operator( nullptr );
    space->assign_scalar( T( 0 ), *solution );
    test.require( !solver.solve( rhs, *solution ), "Krylov reports unavailable preconditioner" );
    solver.update_operator( op );
    test.require( prec->ready() && solver.solve( rhs, *solution ), "Krylov recovers after explicit reprepare" );
}

// Cases: nonsymmetric k=1/k=2 borders; gauge-fixed velocity/pressure saddle point;
// and singular A with nonsingular K, using a shift only in H, not in the physical operator.
template <class Backend, class T>
void check_case( int k, bool stokes, bool singular_a, nmfd::tests::test_harness &test )
{
#if defined( TEST_OPERATIONS_CUDA ) || defined( TEST_OPERATIONS_HIP )
    using operations_type = nmfd::operations::dense_operations_cuda_hip<T, Backend>;
#else
    using operations_type = nmfd::operations::dense_operations<T, Backend>;
#endif
    using component_type = typename operations_type::vector_space_type;
    using space_type     = nmfd::operations::product_vector_space<component_type, component_type>;
    using block_type     = checked_matrix<operations_type>;
    using operator_type  = nmfd::operations::block_operator<space_type, block_type, block_type, block_type, block_type>;
    using first_inverse_type = sampled_inverse<operations_type, block_type>;
    using schur_type = nmfd::preconditioners::schur_complement_operator<space_type, operator_type, first_inverse_type>;
    using schur_inverse_type = sampled_inverse<operations_type, schur_type>;
    using prec_type =
        nmfd::preconditioners::block_schur<space_type, operator_type, first_inverse_type, schur_inverse_type>;
    using monitor_type = nmfd::solvers::monitor_krylov<space_type, scfd::utils::log_std>;
    using gmres_type   = nmfd::solvers::gmres<space_type, monitor_type, scfd::utils::log_std, operator_type, prec_type>;
    using bicgstabl_type =
        nmfd::solvers::bicgstabl<space_type, monitor_type, scfd::utils::log_std, operator_type, prec_type>;
    const int      n = 3, total = n + k;
    auto           operations = std::make_shared<operations_type>();
    std::vector<T> entries( total * total, T( 0 ) );
    for ( int i = 0; i < n; ++i )
        entries[i * total + i] = T( stokes ? 3 : i + 2 );
    if ( stokes )
    {
        entries[1] = entries[total] = entries[total + 2] = entries[2 * total + 1] = T( -1 );
        const int divergence[2][3]                                                = { { 1, -1, 0 }, { 0, 1, -1 } };
        for ( int i = 0; i < n; ++i )
            for ( int j = 0; j < k; ++j )
                entries[i * total + n + j] = entries[( n + j ) * total + i] = T( divergence[j][i] );
    }
    else
    {
        entries[1]           = T( .25 );
        const double b[3][2] = { { 1, 0 }, { 0, 1 }, { 1, .5 } };
        const double c[2][3] = { { .5, 1, -.25 }, { 1, -.5, 1 } };
        for ( int i = 0; i < n; ++i )
            for ( int j = 0; j < k; ++j )
            {
                entries[i * total + n + j]     = T( b[i][j] );
                entries[( n + j ) * total + i] = T( c[j][i] );
            }
    }
    if ( singular_a )
        entries[0] = T( 0 );
    auto                                     full = make_matrix( operations, total, total, entries );
    std::vector<std::shared_ptr<block_type>> blocks;
    for ( int row = 0; row < 2; ++row )
        for ( int col = 0; col < 2; ++col )
        {
            const int      rows = row == 0 ? n : k, cols = col == 0 ? n : k;
            std::vector<T> values( rows * cols );
            for ( int i = 0; i < rows; ++i )
                for ( int j = 0; j < cols; ++j )
                    values[i * cols + j] = entries[( i + row * n ) * total + j + col * n];
            blocks.push_back( make_matrix( operations, rows, cols, values ) );
        }
    auto first_space = blocks[0]->get_dom_space(), second_space = blocks[3]->get_dom_space();
    auto space = std::make_shared<space_type>( first_space, second_space );
    auto op    = std::make_shared<operator_type>( space, blocks[0], blocks[1], blocks[2], blocks[3] );
    auto h     = std::make_shared<first_inverse_type>( operations, first_space );
    h->shift   = singular_a ? T( 1 ) : T( 0 );
    auto q     = std::make_shared<schur_inverse_type>( operations, second_space );
    auto prec  = std::make_shared<prec_type>( space, h, q );
    for ( int missing = 0; missing < 3; ++missing )
    {
        bool rejected = false;
        try
        {
            prec_type invalid( missing == 0 ? nullptr : space, missing == 1 ? nullptr : h, missing == 2 ? nullptr : q );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        test.require( rejected, "Invalid retained dependencies fail construction without leaking workspace" );
    }
    std::weak_ptr<space_type>         weak_space = space;
    std::weak_ptr<operator_type>      weak_op    = op;
    std::weak_ptr<first_inverse_type> weak_h     = h;
    std::weak_ptr<schur_inverse_type> weak_q     = q;
    {
        nmfd::detail::vector_wrap<space_type, false, false> exact( *space ), rhs( *space ), result( *space ),
            error( *space );
        for ( auto *vector : { &exact, &rhs, &result, &error } )
        {
            vector->init();
            vector->start_use();
        }
        std::vector<T> expected( total ), reference( total );
        for ( int i = 0; i < total; ++i )
            expected[i] = T( ( i % 2 == 0 ? 1 : -1 ) * ( i + 1 ) );
        space->set( expected.data(), *exact, total );
        for ( int i = 0; i < total; ++i )
            for ( int j = 0; j < total; ++j )
                reference[i] += entries[i * total + j] * expected[j];
        space->set( reference.data(), *rhs, total );
        test.require( op->apply( *exact, *result ), "Block action succeeds" );
        space->assign_lin_comb( T( 1 ), *result, T( -1 ), *rhs, *error );
        const auto tolerance =
            typename space_type::norm_type( 1000 ) * std::numeric_limits<typename space_type::norm_type>::epsilon();
        test.require( space->norm( *error ) < tolerance, "Block action matches separately assembled matrix" );
        test.require( !prec->ready() && !prec->apply( *rhs, *result ), "Unprepared preconditioner fails" );
        prec->set_operator( op );
        test.require( prec->ready(), "Zero E permits Schur setup" );
        if ( !singular_a )
        {
            test.require( prec->apply( *rhs, *result ), "Full Schur inverse action" );
            space->assign_lin_comb( T( 1 ), *result, T( -1 ), *exact, *error );
            test.require( space->norm( *error ) < tolerance, "Exact factors recover assembled solution" );
            space->assign( *rhs, *result );
            test.require( prec->apply( *result ), "In-place Schur action" );
            space->assign_lin_comb( T( 1 ), *result, T( -1 ), *exact, *error );
            test.require( space->norm( *error ) < tolerance, "In-place action preserves RHS until commit" );
        }
        check_solver<space_type, operator_type, prec_type, gmres_type>( space, op, prec, *rhs, *exact, test );
        check_solver<space_type, operator_type, prec_type, bicgstabl_type>( space, op, prec, *rhs, *exact, test );

        // Rebuild numerical factors for changed values even though the block/operator handles did not change.
        auto e_view    = blocks[3]->create_view( true );
        e_view( 0, 0 ) = T( -.5 );
        e_view.release( true );
        test.require( op->apply( *exact, *rhs ), "Changed lower-right block action" );
        const int setups = q->setups;
        prec->update_operator( op );
        test.require( prec->ready() && q->setups == setups + 1, "Same-handle coefficient changes refresh Schur setup" );
        check_solver<space_type, operator_type, prec_type, gmres_type>( space, op, prec, *rhs, *exact, test );

        if ( singular_a )
        {
            auto zero_e    = blocks[3]->create_view( false );
            zero_e( 0, 0 ) = T( 0 );
            zero_e.release( true );
            h->scale = T( 0 );
            prec->set_operator( op );
            test.require(
                !prec->ready(), "Unusable H produces singular approximate Schur, not a singularity verdict on K"
            );
            h->scale = T( 1 );
            prec->set_operator( op );
            test.require( prec->ready(), "Explicit change of inverse approximation recovers setup" );
            test.require( op->apply( *exact, *rhs ), "Original singular-A system remains unchanged" );
        }

        // Any partially modified scratch must not reach the caller, including the second H application.
        space->assign( *exact, *result );
        for ( int failure = 0; failure < 5; ++failure )
        {
            h->fail_apply          = failure == 0;
            q->fail_apply          = failure == 1;
            blocks[1]->fail        = failure == 2;
            q->nonfinite           = failure == 3;
            h->fail_on_application = failure == 4 ? h->applications + 2 : 0;
            test.require( !prec->apply( *rhs, *result ), "Inverse, block, and nonfinite failures propagate" );
            space->assign_lin_comb( T( 1 ), *result, T( -1 ), *exact, *error );
            test.require( space->norm( *error ) == 0., "Failed preconditioner preserves output" );
        }
        h->fail_apply = q->fail_apply = q->nonfinite = blocks[1]->fail = false;
        h->fail_on_application                                         = 0;
        h->fail_setup                                                  = true;
        prec->update_operator( op );
        test.require(
            !prec->ready() && !prec->apply( *rhs, *result ), "First-factor setup failure disables stale factors"
        );
        h->fail_setup = false;
        q->fail_setup = true;
        prec->set_operator( op );
        test.require( !prec->ready(), "Schur-factor setup failure disables preconditioner" );
        q->fail_setup = false;
        prec->set_operator( op );
        test.require( prec->ready() && prec->apply( *rhs, *result ), "Repreparation recovers after failed setup" );
        q->throw_setup = true;
        bool threw     = false;
        try
        {
            prec->set_operator( op );
        }
        catch ( const std::runtime_error & )
        {
            threw = true;
        }
        test.require( threw && !prec->ready(), "Setup exceptions propagate and disable stale factors" );
        q->throw_setup = false;
        prec->set_operator( nullptr );
        test.require( !prec->ready() && !prec->apply( *rhs, *result ), "Null binding invalidates setup" );
        prec->set_operator( op );
        test.require( prec->ready(), "Valid binding after invalidation" );
        const auto owners = space.use_count();
        test.require( prec->apply( *rhs, *result ), "Prepared action after recovery" );
        test.require( space.use_count() == owners, "Repeated actions do not retain extra space handles" );
    }
    h.reset();
    q.reset();
    op.reset();
    space.reset();
    test.require( !weak_space.expired() && !weak_op.expired(), "Preconditioner retains prepared dependencies" );
    prec.reset();
    test.require(
        weak_space.expired() && weak_op.expired() && weak_h.expired() && weak_q.expired(),
        "Schur dependency graph has no cycle"
    );
}

template <class Backend>
void check_backend( nmfd::tests::test_harness &test )
{
    check_case<Backend, double>( 1, false, false, test );
    check_case<Backend, double>( 2, false, false, test );
    check_case<Backend, double>( 2, true, false, test );
    check_case<Backend, double>( 1, false, true, test );
    check_case<Backend, float>( 1, false, false, test );
}

}

int main()
{
    try
    {
        nmfd::tests::test_harness test;
#if defined( TEST_OPERATIONS_CUDA )
        using backend_type = scfd::backend::cuda<>;
        backend_type::init_device();
        scfd::cublas_wrap   blas( false, true );
        scfd::cusolver_wrap dense_solver( &blas, true );
        check_backend<backend_type>( test );
#elif defined( TEST_OPERATIONS_HIP )
        using backend_type = scfd::backend::hip<>;
        backend_type::init_device();
        scfd::hipblas_wrap   blas( false, true );
        scfd::hipsolver_wrap dense_solver( &blas, true );
        check_backend<backend_type>( test );
#else
        check_backend<scfd::backend::serial_cpu<>>( test );
        check_backend<scfd::backend::omp<>>( test );
        check_case<scfd::backend::serial_cpu<>, std::complex<double>>( 2, false, false, test );
#endif
        return test.finish( "Block Schur" );
    }
    catch ( const std::exception &error )
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
