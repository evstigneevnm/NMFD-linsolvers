#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <scfd/utils/log.h>

#include <nmfd/preconditioners/mg.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>

#include "common/cpu_reference_vector_space.h"
#include "common/test_harness.h"
#include "solvers/coarsening.h"
#include "solvers/ident_op.h"
#include "solvers/linear_operator_elliptic.h"
#include "solvers/prolongator.h"
#include "solvers/residual_regularization_test.h"
#include "solvers/restrictor.h"
#include "solvers/smoother_elliptic.h"

namespace
{

template<class VectorSpace, class Operator>
typename VectorSpace::norm_type relative_residual(
    const VectorSpace& vector_space,
    const Operator& matrix_operator,
    const typename VectorSpace::vector_type& rhs,
    const typename VectorSpace::vector_type& solution)
{
    typename VectorSpace::vector_type residual(
        static_cast<std::size_t>(vector_space.size()),
        typename VectorSpace::scalar_type{});
    matrix_operator.apply(solution, residual);
    vector_space.add_lin_comb(1.0, rhs, -1.0, residual);
    return vector_space.norm(residual)/
        std::max(vector_space.norm(rhs), 1e-300);
}

} // namespace

int main()
{
    using log_type = scfd::utils::log_std;
    using vector_space_type =
        nmfd::tests::cpu_reference_vector_space<double>;
    using prolongator_type = tests::prolongator<vector_space_type, log_type>;
    using restrictor_type = tests::restrictor<vector_space_type, log_type>;
    using coarse_solver_type = tests::ident_op<vector_space_type, log_type>;
    using operator_type =
        tests::linear_operator_elliptic<vector_space_type, log_type>;
    using coarsening_type = tests::coarsening<operator_type, log_type>;
    using smoother_type =
        tests::smoother_elliptic<vector_space_type, log_type>;
    using multigrid_type =
        nmfd::preconditioners::mg<
            operator_type,
            restrictor_type,
            prolongator_type,
            smoother_type,
            coarse_solver_type,
            coarsening_type,
            log_type>;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using regularization_type =
        nmfd::solvers::detail::residual_regularization_test<
            vector_space_type,
            log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type,
            multigrid_type,
            regularization_type>;

    nmfd::tests::test_harness test;
    constexpr std::size_t dimension = 128;
    constexpr double physical_residual_tolerance = 5e-8;
    auto vector_space =
        std::make_shared<vector_space_type>(
            static_cast<vector_space_type::ordinal_type>(dimension));
    auto matrix_operator =
        std::make_shared<operator_type>(*vector_space);
    auto regularization =
        std::make_shared<regularization_type>(vector_space);

    vector_space_type::vector_type rhs(dimension, 0.0);
    const double two_pi = 2.0*std::acos(-1.0);
    for(std::size_t index = 0; index < dimension; ++index)
    {
        const double coordinate =
            static_cast<double>(index)/static_cast<double>(dimension);
        rhs[index] = std::sin(two_pi*coordinate);
    }

    typename multigrid_type::utils_hierarchy multigrid_utils;
    multigrid_utils.log = nullptr;
    typename multigrid_type::params_hierarchy multigrid_parameters;
    multigrid_parameters.direct_coarse = false;
    multigrid_parameters.num_sweeps_pre = 3;
    multigrid_parameters.num_sweeps_post = 3;

    typename solver_type::params parameters;
    parameters.basis_size = 25;
    parameters.batch_size = 5;
    parameters.orthogonalization = "mgs";
    parameters.reorthogonalization_policy = "always";
    parameters.max_orthogonalization_passes = 2;
    parameters.monitor.rel_tol = 1e-10;
    parameters.monitor.abs_tol = 1e-12;
    parameters.monitor.max_iters_num = 300;
    parameters.monitor.divide_out_norms_by_rel_base = false;

    for(const char side : {'L', 'R'})
    {
        auto multigrid =
            std::make_shared<multigrid_type>(
                multigrid_utils,
                multigrid_parameters);
        parameters.preconditioner_side = side;
        solver_type solver(
            matrix_operator,
            vector_space,
            nullptr,
            parameters,
            multigrid,
            regularization);
        vector_space_type::vector_type solution(dimension, 0.0);
        const bool converged = solver.solve(rhs, solution);
        const std::string label =
            "multigrid " + std::string(1, side);
        test.require(converged, label + " converges");
        const auto initial_residual = relative_residual(
            *vector_space,
            *matrix_operator,
            rhs,
            solution);
        if(initial_residual > physical_residual_tolerance)
            std::cout << label
                      << " physical residual = " << initial_residual << '\n';
        test.require(
            initial_residual <= physical_residual_tolerance,
            label + " physical residual");
        test.require(
            std::abs(vector_space->sum(solution)) <= 2e-8,
            label + " zero-mean solution");

        vector_space->scale(0.99999, solution);
        test.require(
            solver.solve(rhs, solution),
            label + " reused initial guess converges");
        const auto reused_residual = relative_residual(
            *vector_space,
            *matrix_operator,
            rhs,
            solution);
        if(reused_residual > physical_residual_tolerance)
            std::cout << label
                      << " reused physical residual = " << reused_residual
                      << '\n';
        test.require(
            reused_residual <= physical_residual_tolerance,
            label + " reused physical residual");
    }

    return test.finish("NMFD GMRES multigrid tests");
}
