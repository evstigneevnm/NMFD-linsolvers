#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/block_operator.h>
#include <nmfd/operations/linalg/small_dense.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/operations/rect_vector_space.h>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/preconditioners/block_schur.h>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <scfd/backend/serial_cpu.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/log.h>

#include "common/test_harness.h"

namespace
{

using parameter_type   = nmfd::operations::static_vector_space<double, 2, scfd::static_vec::vec<double, 2>>;
using parameter_vector = typename parameter_type::vector_type;

template <class Space>
struct diagonal_block
{
    using vector_type = typename Space::vector_type;
    std::shared_ptr<Space> space;
    double                 coefficient = 2.;

    void apply( const vector_type &in, vector_type &out ) const
    {
        space->assign_lin_comb( coefficient, in, out );
    }
};

template <class Space>
class coupling
{
    using vector_type = typename Space::vector_type;

public:
    coupling( std::shared_ptr<Space> space, int rank ) : space_( std::move( space ) ), weights_( *space_ )
    {
        weights_.init();
        weights_.start_use();
        auto view = ( *weights_ ).create_view( false );
        for ( int i = 0; i < rank + 2; ++i )
            view( i, 0 ) = rank * ( rank + 3 ) / 2 + i + 1.;
        view.release( true );
    }

    void apply( const parameter_vector &in, vector_type &out ) const
    {
        space_->assign_scalar( in[0], out );
        space_->add_lin_comb( in[1], *weights_, 1., out );
    }

    void apply( const vector_type &in, parameter_vector &out ) const
    {
        out[0] = space_->sum( in );
        out[1] = space_->scalar_prod( *weights_, in );
    }

private:
    std::shared_ptr<Space>                         space_;
    nmfd::detail::vector_wrap<Space, false, false> weights_;
};

struct zero_block
{
    void apply( const parameter_vector &, parameter_vector &out ) const
    {
        out[0] = out[1] = 0.;
    }
};

template <class Space>
struct diagonal_inverse
{
    std::shared_ptr<Space> space;
    double                 coefficient = 0.;

    void set_operator( std::shared_ptr<const diagonal_block<Space>> op )
    {
        coefficient = 1. / op->coefficient;
    }
    void apply( typename Space::vector_type &in_out ) const
    {
        space->scale( coefficient, in_out );
    }
};

template <class SchurOperator>
struct parameter_inverse
{
    nmfd::operations::linalg::small_matrix<double, 2, 2> lu{ 2, 2 };
    std::array<std::size_t, 2>                           pivots{};

    bool set_operator( std::shared_ptr<const SchurOperator> op )
    {
        for ( int column = 0; column < 2; ++column )
        {
            parameter_vector basis{ 0., 0. }, image;
            basis[column] = 1.;
            if ( !op->apply( basis, image ) )
                return false;
            for ( int row = 0; row < 2; ++row )
                lu( row, column ) = image[row];
        }
        return nmfd::operations::linalg::factor_lu( lu, pivots ).ok();
    }
    void apply( parameter_vector &in_out ) const
    {
        nmfd::operations::linalg::small_vector<double, 2> candidate{ in_out[0], in_out[1] };
        nmfd::operations::linalg::solve_lu_in_place( lu, pivots, candidate );
        for ( int i = 0; i < 2; ++i )
            in_out[i] = candidate[i];
    }
};

template <class Solver, class Space, class Operator, class Preconditioner>
void check_solver(
    const std::shared_ptr<Space> &space, const std::shared_ptr<Operator> &op,
    const std::shared_ptr<Preconditioner> &prec, diagonal_block<typename Space::first_space_type> &a,
    nmfd::tests::test_harness &test
)
{
    typename Solver::params p;
    p.basis_size      = 3;
    p.monitor.rel_tol = p.monitor.abs_tol = 1.e-11;
    p.monitor.max_iters_num               = 40;
    Solver                                         solver( space, nullptr, p, prec );
    nmfd::detail::vector_wrap<Space, false, false> exact( *space ), rhs( *space ), x( *space ), residual( *space );
    for ( auto *vector : { &exact, &rhs, &x, &residual } )
    {
        vector->init();
        vector->start_use();
    }
    space->assign_scalar( 2., *exact );
    for ( double coefficient : { 2., 3. } )
    {
        // A changes in place; the true system and its approximate factors must both be refreshed.
        a.coefficient = coefficient;
        test.require( op->apply( *exact, *rhs ), "Distributed block action" );
        solver.update_operator( op );
        space->assign_scalar( 0., *x );
        test.require( prec->ready() && solver.solve( *rhs, *x ), "MPI augmented Krylov convergence" );
        test.require( op->apply( *x, *residual ), "Distributed true operator" );
        space->add_lin_comb( -1., *rhs, 1., *residual );
        test.require( space->norm( *residual ) <= 1.e-10 * space->norm( *rhs ), "MPI complete true residual" );
        space->assign_lin_comb( 1., *x, -1., *exact, *residual );
        test.require( space->norm( *residual ) < 1.e-10, "MPI field and replicated parameters" );
    }
}

}

int main( int argc, char *argv[] )
{
    try
    {
        scfd::communication::mpi_wrap platform( argc, argv );
        auto                          comm = platform.comm_world();
        using field_type    = nmfd::rect_vector_space<double, 1, 1, scfd::backend::serial_cpu<>, decltype( comm )>;
        using space_type    = nmfd::operations::product_vector_space<field_type, parameter_type>;
        using operator_type = nmfd::operations::block_operator<
            space_type, diagonal_block<field_type>, coupling<field_type>, coupling<field_type>, zero_block>;
        using h_type     = diagonal_inverse<field_type>;
        using schur_type = nmfd::preconditioners::schur_complement_operator<space_type, operator_type, h_type>;
        using prec_type =
            nmfd::preconditioners::block_schur<space_type, operator_type, h_type, parameter_inverse<schur_type>>;
        using monitor_type = nmfd::solvers::monitor_krylov<space_type, scfd::utils::log_std>;
        using gmres_type =
            nmfd::solvers::gmres<space_type, monitor_type, scfd::utils::log_std, operator_type, prec_type>;
        using bicgstabl_type =
            nmfd::solvers::bicgstabl<space_type, monitor_type, scfd::utils::log_std, operator_type, prec_type>;
        auto field        = std::make_shared<field_type>( typename field_type::idx_nd_type{ comm.myid + 2 }, comm );
        auto space        = std::make_shared<space_type>( field, std::make_shared<parameter_type>() );
        auto a            = std::make_shared<diagonal_block<field_type>>();
        a->space          = field;
        auto off_diagonal = std::make_shared<coupling<field_type>>( field, comm.myid );
        auto op =
            std::make_shared<operator_type>( space, a, off_diagonal, off_diagonal, std::make_shared<zero_block>() );
        auto h    = std::make_shared<h_type>();
        h->space  = field;
        auto prec = std::make_shared<prec_type>( space, h, std::make_shared<parameter_inverse<schur_type>>() );
        nmfd::tests::test_harness test;
        check_solver<gmres_type>( space, op, prec, *a, test );
        check_solver<bicgstabl_type>( space, op, prec, *a, test );
        const int failures = comm.all_reduce_sum( int( test.failures() ) );
        if ( comm.myid == 0 )
            std::cout << "Block Schur MPI: " << test.checks() << " checks per rank on " << comm.num_procs << " ranks, "
                      << failures << " failures\n";
        return failures == 0 ? 0 : 1;
    }
    catch ( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
