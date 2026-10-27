#include <complex>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
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
#include <nmfd/operations/glued_matrix_operator.h>
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

template <class Base, bool Checked>
struct test_operations : Base
{
    using T           = typename Base::scalar_type;
    using vector_type = typename Base::vector_type;
    using matrix_type = typename Base::matrix_type;
    bool fail         = false;

    std::conditional_t<Checked, bool, void>
    add_matrix_vector_prod( T alpha, const matrix_type &matrix, const vector_type &in, T beta, vector_type &out ) const
    {
        if constexpr ( Checked )
            if ( fail )
                return false;
        Base::add_matrix_vector_prod( alpha, matrix, in, beta, out );
        if constexpr ( Checked )
            return true;
    }
};

// Both access paths must pass the original matrix handle to matrix-aware setup.
template <class Operations>
class matrix_inverse
{
    using space_type = typename Operations::vector_space_type;
    using T          = typename Operations::scalar_type;

public:
    using vector_type = typename Operations::vector_type;
    using matrix_type = typename Operations::matrix_type;

    matrix_inverse( std::shared_ptr<Operations> operations, std::shared_ptr<space_type> space )
        : operations_( std::move( operations ) ), space_( std::move( space ) ), zero_( *space_ ), candidate_( *space_ ),
          shifted_( std::make_shared<matrix_type>() )
    {
        zero_.init();
        zero_.start_use();
        candidate_.init();
        candidate_.start_use();
        operations_->init_matrix( space_->size(), space_->size(), *shifted_ );
        space_->assign_scalar( T( 0 ), *zero_ );
    }

    bool set_operator( std::shared_ptr<const matrix_type> op )
    {
        ready_   = false;
        original = std::move( op );
        if ( !original || fail_setup )
            return false;
        matrix_ = original;
        if ( shift != T( 0 ) )
        {
            operations_->assign_matrix( *original, *shifted_ );
            auto view = shifted_->create_view( true );
            for ( std::size_t i = 0; i < space_->size(); ++i )
                view( i, i ) += shift;
            view.release( true );
            matrix_ = shifted_;
        }
        ready_ = operations_->solve( *matrix_, *zero_, *candidate_ );
        return ready_;
    }

    bool apply( vector_type &in_out ) const
    {
        if ( !ready_ || fail_apply )
        {
            space_->assign_scalar( T( 99 ), in_out );
            return false;
        }
        if ( !operations_->solve( *matrix_, in_out, *candidate_ ) )
            return false;
        space_->assign( *candidate_, in_out );
        return true;
    }

    std::shared_ptr<const matrix_type> original;
    T                                  shift      = T( 0 );
    bool                               fail_setup = false, fail_apply = false;

private:
    std::shared_ptr<Operations>                                 operations_;
    std::shared_ptr<space_type>                                 space_;
    nmfd::detail::vector_wrap<space_type, false, false>         zero_;
    mutable nmfd::detail::vector_wrap<space_type, false, false> candidate_;
    std::shared_ptr<matrix_type>                                shifted_;
    std::shared_ptr<const matrix_type>                          matrix_;
    bool                                                        ready_ = false;
};

// A small assembled reference Q; production Q may inspect original_system for sparse pressure setup.
template <class Operations, class SchurOperator>
class schur_inverse : public matrix_inverse<Operations>
{
    using parent_type = matrix_inverse<Operations>;
    using space_type  = typename Operations::vector_space_type;
    using T           = typename Operations::scalar_type;

public:
    schur_inverse( std::shared_ptr<Operations> operations, std::shared_ptr<space_type> space )
        : parent_type( operations, space ), operations_( std::move( operations ) ), space_( std::move( space ) ),
          matrix_( std::make_shared<typename Operations::matrix_type>() ), basis_( *space_ ), image_( *space_ )
    {
        basis_.init();
        basis_.start_use();
        image_.init();
        image_.start_use();
        operations_->init_matrix( space_->size(), space_->size(), *matrix_ );
    }

    bool set_operator( std::shared_ptr<const SchurOperator> op )
    {
        parent_type::set_operator( nullptr );
        original_system = op ? op->get_operator() : nullptr;
        if ( !original_system )
            return false;
        auto matrix = matrix_->create_view( false );
        for ( std::size_t j = 0; j < space_->size(); ++j )
        {
            space_->assign_scalar( T( 0 ), *basis_ );
            space_->set_value_at_point( T( 1 ), j, *basis_ );
            if ( !op->apply( *basis_, *image_ ) )
            {
                matrix.release( false );
                return false;
            }
            auto image = ( *image_ ).create_view( true );
            for ( std::size_t i = 0; i < space_->size(); ++i )
                matrix( i, j ) = image( i );
            image.release( false );
        }
        matrix.release( true );
        return parent_type::set_operator( matrix_ );
    }

    std::shared_ptr<const typename SchurOperator::operator_type> original_system;

private:
    std::shared_ptr<Operations>                         operations_;
    std::shared_ptr<space_type>                         space_;
    std::shared_ptr<typename Operations::matrix_type>   matrix_;
    nmfd::detail::vector_wrap<space_type, false, false> basis_, image_;
};

template <class Solver, class Space, class Operator, class Preconditioner>
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
    p.basis_size      = 5;
    p.monitor.rel_tol = p.monitor.abs_tol = tolerance;
    p.monitor.max_iters_num               = 100;
    Solver                                         solver( space, nullptr, p, prec );
    nmfd::detail::vector_wrap<Space, false, false> x( *space ), error( *space );
    x.init();
    x.start_use();
    error.init();
    error.start_use();
    solver.set_operator( op );
    space->assign_scalar( T( 0 ), *x );
    test.require( prec->ready() && solver.solve( rhs, *x ), "Glued augmented Krylov converges" );
    op->apply( *x, *error );
    space->add_lin_comb( T( -1 ), rhs, T( 1 ), *error );
    test.require( space->norm( *error ) <= tolerance * ( norm_type( 1 ) + space->norm( rhs ) ), "Glued true residual" );
    space->assign_lin_comb( T( 1 ), *x, T( -1 ), exact, *error );
    test.require( space->norm( *error ) < norm_type( 20 ) * tolerance, "Glued solution matches independent reference" );
}

enum class system_kind
{
    nonsymmetric,
    stokes,
    singular_a
};

template <class Backend, class T, bool Checked = false>
void check_case( int k, system_kind kind, nmfd::tests::test_harness &test )
{
#if defined( TEST_OPERATIONS_CUDA ) || defined( TEST_OPERATIONS_HIP )
    using base_operations = nmfd::operations::dense_operations_cuda_hip<T, Backend>;
#else
    using base_operations = nmfd::operations::dense_operations<T, Backend>;
#endif
    using operations_type = test_operations<base_operations, Checked>;
    using block_type      = nmfd::operations::matrix_operator<operations_type>;
    using component_type  = typename operations_type::vector_space_type;
    using product_space   = nmfd::operations::product_vector_space<component_type, component_type>;
    using product_operator =
        nmfd::operations::block_operator<product_space, block_type, block_type, block_type, block_type>;
    using glued_operator = nmfd::operations::glued_matrix_operator<operations_type, 2>;
    using glued_space    = typename glued_operator::vector_space_type;
    using access_type    = nmfd::operations::two_block_access<glued_space, glued_operator>;
    using h_type         = matrix_inverse<operations_type>;
    using product_schur  = nmfd::preconditioners::schur_complement_operator<product_space, product_operator, h_type>;
    using glued_schur    = nmfd::preconditioners::schur_complement_operator<glued_space, glued_operator, h_type>;
    using product_q      = schur_inverse<operations_type, product_schur>;
    using glued_q        = schur_inverse<operations_type, glued_schur>;
    using product_prec   = nmfd::preconditioners::block_schur<product_space, product_operator, h_type, product_q>;
    using glued_prec     = nmfd::preconditioners::block_schur<glued_space, glued_operator, h_type, glued_q>;
    using monitor_type   = nmfd::solvers::monitor_krylov<glued_space, scfd::utils::log_std>;
    using gmres_type =
        nmfd::solvers::gmres<glued_space, monitor_type, scfd::utils::log_std, glued_operator, glued_prec>;
    using bicgstabl_type =
        nmfd::solvers::bicgstabl<glued_space, monitor_type, scfd::utils::log_std, glued_operator, glued_prec>;
    static_assert( !std::is_default_constructible_v<access_type> );
    static_assert(
        std::is_same_v<
            decltype( access_type::template component<0>( std::declval<const typename glued_space::vector_type &>() ) ),
            const typename operations_type::vector_type &>
    );
    std::weak_ptr<operations_type> weak_operations;
    {
        auto operations  = std::make_shared<operations_type>();
        weak_operations  = operations;
        const int      n = 3, total = n + k;
        std::vector<T> matrix( total * total, T( 0 ) );
        for ( int i = 0; i < n; ++i )
            matrix[i * total + i] = T( kind == system_kind::stokes ? 3 : i + 2 );
        const double b[3][2]          = { { 1, 0 }, { 0, 1 }, { 1, .5 } };
        const double c[2][3]          = { { .5, 1, -.25 }, { 1, -.5, 1 } };
        const double divergence[2][3] = { { 1, -1, 0 }, { 0, 1, -1 } };
        for ( int i = 0; i < n; ++i )
            for ( int j = 0; j < k; ++j )
            {
                matrix[i * total + n + j]     = T( kind == system_kind::stokes ? divergence[j][i] : b[i][j] );
                matrix[( n + j ) * total + i] = T( kind == system_kind::stokes ? divergence[j][i] : c[j][i] );
            }
        if ( kind == system_kind::stokes )
            matrix[1] = matrix[total] = matrix[total + 2] = matrix[2 * total + 1] = T( -1 );
        else
            matrix[1] = T( .25 );
        if ( kind == system_kind::singular_a )
            matrix[0] = T( 0 );
        std::vector<std::shared_ptr<block_type>>  blocks;
        typename glued_operator::block_array_type raw_blocks;
        for ( int row = 0; row < 2; ++row )
            for ( int col = 0; col < 2; ++col )
            {
                typename operations_type::matrix_type block;
                const int                             rows = row == 0 ? n : k, cols = col == 0 ? n : k;
                operations->init_matrix( rows, cols, block );
                auto view = block.create_view( false );
                for ( int i = 0; i < rows; ++i )
                    for ( int j = 0; j < cols; ++j )
                        view( i, j ) = matrix[( i + row * n ) * total + j + col * n];
                view.release( true );
                blocks.push_back( std::make_shared<block_type>( operations, std::move( block ) ) );
                raw_blocks[row][col] = blocks.back();
            }
        auto first = blocks[0]->get_dom_space(), second = blocks[3]->get_dom_space();
        auto ps = std::make_shared<product_space>( first, second );
        auto pk = std::make_shared<product_operator>( ps, blocks[0], blocks[1], blocks[2], blocks[3] );
        auto gk = std::make_shared<glued_operator>( operations, std::move( raw_blocks ) );
        auto gs = gk->get_dom_space();
        auto ph = std::make_shared<h_type>( operations, first ), gh = std::make_shared<h_type>( operations, first );
        ph->shift = gh->shift = kind == system_kind::singular_a ? T( 1 ) : T( 0 );
        auto pq               = std::make_shared<product_q>( operations, second );
        auto gq               = std::make_shared<glued_q>( operations, second );
        auto pp               = std::make_shared<product_prec>( ps, ph, pq );
        auto gp               = std::make_shared<glued_prec>( gs, gh, gq, access_type( operations ) );
        nmfd::detail::vector_wrap<product_space, false, false> px( *ps ), prhs( *ps ), pout( *ps );
        nmfd::detail::vector_wrap<glued_space, false, false>   gx( *gs ), grhs( *gs ), gout( *gs ), error( *gs );
        for ( auto *v : { &px, &prhs, &pout } )
        {
            v->init();
            v->start_use();
        }
        for ( auto *v : { &gx, &grhs, &gout, &error } )
        {
            v->init();
            v->start_use();
        }
        const auto tolerance =
            typename glued_space::norm_type( 1000 ) * std::numeric_limits<typename glued_space::norm_type>::epsilon();
        std::vector<T> exact( total ), rhs( total, T( 0 ) );
        for ( int i = 0; i < total; ++i )
            exact[i] = T( ( i % 2 == 0 ? 1 : -1 ) * ( i + 1 ) );
        for ( int i = 0; i < total; ++i )
            for ( int j = 0; j < total; ++j )
                rhs[i] += matrix[i * total + j] * exact[j];
        ps->set( exact.data(), *px, total );
        ps->set( rhs.data(), *prhs, total );
        operations->assign( ( *px ).first, ( *gx ).comp( 0 ) );
        operations->assign( ( *px ).second, ( *gx ).comp( 1 ) );
        operations->assign( ( *prhs ).first, ( *grhs ).comp( 0 ) );
        operations->assign( ( *prhs ).second, ( *grhs ).comp( 1 ) );
        gk->apply( *gx, *gout );
        gs->assign_lin_comb( T( 1 ), *gout, T( -1 ), *grhs, *error );
        test.require( gs->norm( *error ) < tolerance, "Glued action matches independently assembled system" );
        pp->set_operator( pk );
        gp->set_operator( gk );
        test.require( pp->ready() && gp->ready(), "Both representations prepare the same Schur factorization" );
        test.require( gh->original == gk->comp_ptr( 0, 0 ), "Glued setup receives original A without copying it" );
        test.require(
            pq->original_system == pk && gq->original_system == gk, "Q can inspect original matrix/operator blocks"
        );
        const auto owners = operations.use_count();
        test.require( pp->apply( *prhs, *pout ) && gp->apply( *grhs, *gout ), "Both Schur actions succeed" );
        operations->assign_lin_comb( T( 1 ), ( *pout ).first, T( -1 ), ( *gout ).comp( 0 ), ( *error ).comp( 0 ) );
        operations->assign_lin_comb( T( 1 ), ( *pout ).second, T( -1 ), ( *gout ).comp( 1 ), ( *error ).comp( 1 ) );
        test.require( gs->norm( *error ) < tolerance, "Product and glued Schur actions agree" );
        test.require( operations.use_count() == owners, "Block applications retain no additional operation handles" );
        gs->assign( *grhs, *gout );
        test.require( gp->apply( *gout ), "In-place glued preconditioning succeeds" );
        check_solver<gmres_type>( gs, gk, gp, *grhs, *gx, test );
        check_solver<bicgstabl_type>( gs, gk, gp, *grhs, *gx, test );

        gs->assign( *gx, *gout );
        gh->fail_apply = true;
        test.require( !gp->apply( *grhs, *gout ), "Failed first inverse propagates" );
        gs->assign_lin_comb( T( 1 ), *gout, T( -1 ), *gx, *error );
        test.require( gs->norm( *error ) == 0., "Failed glued action preserves output" );
        gh->fail_apply = false;
        gq->fail_apply = true;
        test.require( !gp->apply( *gout ), "Failed Schur inverse propagates for in-place action" );
        gs->assign_lin_comb( T( 1 ), *gout, T( -1 ), *gx, *error );
        test.require( gs->norm( *error ) == 0., "Failed in-place action preserves input" );
        gq->fail_apply = false;
        if constexpr ( Checked )
        {
            operations->fail = true;
            test.require( !gp->apply( *grhs, *gout ), "Boolean matrix-action failure propagates through access" );
            gs->assign_lin_comb( T( 1 ), *gout, T( -1 ), *gx, *error );
            test.require( gs->norm( *error ) == 0., "Matrix-action failure preserves caller output" );
            operations->fail = false;
        }
        gq->fail_setup = true;
        gp->update_operator( gk );
        test.require( !gp->ready() && !gp->apply( *grhs, *gout ), "Failed glued setup disables stale factors" );
        gq->fail_setup  = false;
        auto changed    = blocks[3]->create_view( true );
        changed( 0, 0 ) = T( -.5 );
        changed.release( true );
        gp->update_operator( gk );
        gk->apply( *gx, *grhs );
        test.require( gp->ready(), "Same-handle coefficient changes rebuild setup" );
        check_solver<gmres_type>( gs, gk, gp, *grhs, *gx, test );
        gp->set_operator( nullptr );
        test.require( !gp->ready(), "Null binding invalidates glued setup" );
        bool rejected = false;
        try
        {
            access_type invalid( nullptr );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        test.require( rejected, "Access rejects null operations" );
    }
    test.require( weak_operations.expired(), "Stateful access and Schur views release operations without cycles" );
}

template <class Backend>
void check_backend( nmfd::tests::test_harness &test )
{
    check_case<Backend, double>( 1, system_kind::nonsymmetric, test );
    check_case<Backend, double>( 2, system_kind::nonsymmetric, test );
    check_case<Backend, double>( 2, system_kind::stokes, test );
    check_case<Backend, double>( 1, system_kind::singular_a, test );
    check_case<Backend, float>( 1, system_kind::nonsymmetric, test );
    check_case<Backend, double, true>( 2, system_kind::nonsymmetric, test );
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
        scfd::cusolver_wrap solver( &blas, true );
        check_backend<backend_type>( test );
#elif defined( TEST_OPERATIONS_HIP )
        using backend_type = scfd::backend::hip<>;
        backend_type::init_device();
        scfd::hipblas_wrap   blas( false, true );
        scfd::hipsolver_wrap solver( &blas, true );
        check_backend<backend_type>( test );
#else
        check_backend<scfd::backend::serial_cpu<>>( test );
        check_backend<scfd::backend::omp<>>( test );
        check_case<scfd::backend::serial_cpu<>, std::complex<double>>( 2, system_kind::nonsymmetric, test );
#endif
        return test.finish( "Glued/product Schur parity" );
    }
    catch ( const std::exception &error )
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
