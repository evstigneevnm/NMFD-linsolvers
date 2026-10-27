#include <memory>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/pair_vector_space.h>
#include <nmfd/operations/scfd_vector_space.h>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <scfd/backend/serial_cpu.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/log.h>

#include "common/test_harness.h"

namespace
{

// Two different eigenvalues exercise product-vector recurrences without a Schur implementation.
template <class Space>
struct block_diagonal
{
    using vector_type = typename Space::vector_type;
    std::shared_ptr<Space> space;
    double                 first_coefficient = 2.;
    bool                   fail              = false;

    bool apply( const vector_type &x, vector_type &out ) const
    {
        if ( fail )
            return false;
        space->first_space().assign_lin_comb( first_coefficient, x.first, out.first );
        space->second_space().assign_lin_comb( 3., x.second, out.second );
        return true;
    }
};

template <class Space, class Solver>
void check_solver( const std::shared_ptr<Space> &space, nmfd::tests::test_harness &test )
{
    auto op   = std::make_shared<block_diagonal<Space>>();
    op->space = space;
    typename Solver::params parameters;
    parameters.basis_size            = 2;
    parameters.monitor.rel_tol       = 1.e-12;
    parameters.monitor.abs_tol       = 1.e-12;
    parameters.monitor.max_iters_num = 50;
    Solver                                         solver( space, nullptr, parameters );
    nmfd::detail::vector_wrap<Space, false, false> exact( *space ), rhs( *space ), x( *space ), residual( *space );
    exact.init();
    exact.start_use();
    rhs.init();
    rhs.start_use();
    x.init();
    x.start_use();
    residual.init();
    residual.start_use();
    space->assign_scalar( 2., *exact );
    for ( double coefficient : { 2., 4. } )
    {
        op->first_coefficient = coefficient;
        op->apply( *exact, *rhs );
        solver.set_operator( op );
        space->assign_scalar( 0., *x );
        test.require( solver.solve( *rhs, *x ), "Product-space Krylov convergence and rebinding" );
        op->apply( *x, *residual );
        space->add_lin_comb( -1., *rhs, 1., *residual );
        test.require( space->norm( *residual ) <= 1.e-11 * space->norm( *rhs ), "Full true residual" );
        space->assign_lin_comb( 1., *exact, -1., *x, *residual );
        test.require( space->norm( *residual ) <= 1.e-11, "Both solution components" );
    }
    op->fail = true;
    space->assign_scalar( 0., *x );
    test.require( !solver.solve( *rhs, *x ), "Product operator failure propagates" );
}

template <class Space>
void check_space( const std::shared_ptr<Space> &space, nmfd::tests::test_harness &test )
{
    using log_type      = scfd::utils::log_std;
    using monitor_type  = nmfd::solvers::monitor_krylov<Space, log_type>;
    using operator_type = block_diagonal<Space>;
    check_solver<Space, nmfd::solvers::gmres<Space, monitor_type, log_type, operator_type>>( space, test );
    check_solver<Space, nmfd::solvers::bicgstabl<Space, monitor_type, log_type, operator_type>>( space, test );
}

}

int main()
{
    using state_type     = nmfd::operations::scfd_vector_space<scfd::backend::serial_cpu<>, double>;
    using parameter_type = nmfd::operations::static_vector_space<double, 2, scfd::static_vec::vec<double, 2>>;
    auto                      state     = std::make_shared<state_type>( 3 );
    auto                      parameter = std::make_shared<parameter_type>();
    nmfd::tests::test_harness test;
    check_space(
        std::make_shared<nmfd::operations::product_vector_space<state_type, parameter_type>>( state, parameter ), test
    );
    check_space(
        std::make_shared<nmfd::operations::pair_vector_space<state_type, parameter_type>>( state, parameter ), test
    );
    return test.finish( "Product-space Krylov" );
}
