#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <scfd/utils/log.h>

#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>

#include "common/cpu_reference_vector_space.h"
#include "common/test_harness.h"
#include "solvers/linear_operator_advection.h"
#include "solvers/linear_operator_diffusion.h"
#include "solvers/linear_operator_elliptic.h"
#include "solvers/preconditioner_advection.h"
#include "solvers/preconditioner_diffusion.h"
#include "solvers/preconditioner_elliptic.h"
#include "solvers/residual_regularization_test.h"

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

template<class Solver, class VectorSpace, class Operator, class Preconditioner>
void solve_both_sides(
    const std::string& label,
    const std::shared_ptr<VectorSpace>& vector_space,
    const std::shared_ptr<Operator>& matrix_operator,
    const std::shared_ptr<Preconditioner>& preconditioner,
    typename Solver::params parameters,
    const typename VectorSpace::vector_type& rhs,
    nmfd::tests::test_harness& test)
{
    for(const char side : {'L', 'R'})
    {
        parameters.preconditioner_side = side;
        typename VectorSpace::vector_type solution(
            static_cast<std::size_t>(vector_space->size()),
            typename VectorSpace::scalar_type{});
        Solver solver(
            matrix_operator,
            vector_space,
            nullptr,
            parameters,
            preconditioner);
        const bool converged = solver.solve(rhs, solution);
        const std::string case_name =
            label + " " + std::string(1, side);
        test.require(converged, case_name + " converges");
        test.require(
            relative_residual(
                *vector_space,
                *matrix_operator,
                rhs,
                solution) <= 2e-8,
            case_name + " physical residual");
        test.require(
            solver.monitor().iters_performed() <=
                parameters.monitor.max_iters_num,
            case_name + " iteration bound");
    }
}

void test_diffusion(nmfd::tests::test_harness& test)
{
    using vector_space_type =
        nmfd::tests::cpu_reference_vector_space<double>;
    using log_type = scfd::utils::log_std;
    using operator_type =
        tests::linear_operator_diffusion<vector_space_type, log_type>;
    using preconditioner_type =
        tests::preconditioner_diffusion<
            vector_space_type,
            operator_type,
            log_type>;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type,
            preconditioner_type>;

    constexpr std::size_t dimension = 128;
    auto vector_space =
        std::make_shared<vector_space_type>(
            static_cast<vector_space_type::ordinal_type>(dimension));
    auto matrix_operator =
        std::make_shared<operator_type>(*vector_space, 0.05);
    auto preconditioner =
        std::make_shared<preconditioner_type>(vector_space, 12);
    vector_space_type::vector_type reference(dimension, 0.0);
    const double pi = std::acos(-1.0);
    for(std::size_t index = 0; index < dimension; ++index)
    {
        reference[index] = std::sin(
            pi*static_cast<double>(index)/
            static_cast<double>(dimension - 1));
    }
    reference.front() = reference.back() = 0.0;
    vector_space_type::vector_type rhs(dimension, 0.0);
    matrix_operator->apply(reference, rhs);

    typename solver_type::params parameters;
    parameters.basis_size = 30;
    parameters.batch_size = 5;
    parameters.reorthogonalization_policy = "dgks";
    parameters.monitor.rel_tol = 1e-10;
    parameters.monitor.abs_tol = 1e-13;
    parameters.monitor.max_iters_num = 300;
    parameters.monitor.divide_out_norms_by_rel_base = false;
    solve_both_sides<solver_type>(
        "diffusion",
        vector_space,
        matrix_operator,
        preconditioner,
        parameters,
        rhs,
        test);
}

void test_advection(nmfd::tests::test_harness& test)
{
    using vector_space_type =
        nmfd::tests::cpu_reference_vector_space<double>;
    using log_type = scfd::utils::log_std;
    using operator_type =
        tests::linear_operator_advection<vector_space_type, log_type>;
    using preconditioner_type =
        tests::preconditioner_advection<
            vector_space_type,
            operator_type,
            log_type>;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type,
            preconditioner_type>;

    constexpr std::size_t dimension = 128;
    auto vector_space =
        std::make_shared<vector_space_type>(
            static_cast<vector_space_type::ordinal_type>(dimension));
    auto matrix_operator =
        std::make_shared<operator_type>(*vector_space, 1.0, 0.05);
    auto preconditioner =
        std::make_shared<preconditioner_type>(vector_space, 3);
    vector_space_type::vector_type reference(dimension, 0.0);
    const double two_pi = 2.0*std::acos(-1.0);
    for(std::size_t index = 0; index < dimension; ++index)
    {
        reference[index] = std::sin(
            two_pi*static_cast<double>(index)/
            static_cast<double>(dimension));
    }
    vector_space_type::vector_type rhs(dimension, 0.0);
    matrix_operator->apply(reference, rhs);

    typename solver_type::params parameters;
    parameters.basis_size = 24;
    parameters.batch_size = 4;
    parameters.reorthogonalization_policy = "dgks";
    parameters.monitor.rel_tol = 1e-10;
    parameters.monitor.abs_tol = 1e-13;
    parameters.monitor.max_iters_num = 250;
    parameters.monitor.divide_out_norms_by_rel_base = false;
    solve_both_sides<solver_type>(
        "advection",
        vector_space,
        matrix_operator,
        preconditioner,
        parameters,
        rhs,
        test);
}

void test_elliptic_regularization(nmfd::tests::test_harness& test)
{
    using vector_space_type =
        nmfd::tests::cpu_reference_vector_space<double>;
    using log_type = scfd::utils::log_std;
    using operator_type =
        tests::linear_operator_elliptic<vector_space_type, log_type>;
    using preconditioner_type =
        tests::preconditioner_elliptic<
            vector_space_type,
            operator_type,
            log_type>;
    using regularization_type =
        nmfd::solvers::detail::residual_regularization_test<
            vector_space_type,
            log_type>;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type,
            preconditioner_type,
            regularization_type>;

    constexpr std::size_t dimension = 96;
    auto vector_space =
        std::make_shared<vector_space_type>(
            static_cast<vector_space_type::ordinal_type>(dimension));
    auto matrix_operator =
        std::make_shared<operator_type>(*vector_space);
    auto preconditioner =
        std::make_shared<preconditioner_type>(vector_space, 16);
    auto regularization =
        std::make_shared<regularization_type>(vector_space);
    vector_space_type::vector_type reference(dimension, 0.0);
    const double two_pi = 2.0*std::acos(-1.0);
    for(std::size_t index = 0; index < dimension; ++index)
    {
        reference[index] = std::sin(
            two_pi*static_cast<double>(index)/
            static_cast<double>(dimension));
    }
    vector_space_type::vector_type rhs(dimension, 0.0);
    matrix_operator->apply(reference, rhs);

    typename solver_type::params parameters;
    parameters.basis_size = 30;
    parameters.batch_size = 5;
    parameters.reorthogonalization_policy = "dgks";
    parameters.monitor.rel_tol = 1e-10;
    parameters.monitor.abs_tol = 1e-12;
    parameters.monitor.max_iters_num = 400;
    parameters.monitor.divide_out_norms_by_rel_base = false;

    for(const char side : {'L', 'R'})
    {
        parameters.preconditioner_side = side;
        vector_space_type::vector_type solution(dimension, 0.0);
        solver_type solver(
            matrix_operator,
            vector_space,
            nullptr,
            parameters,
            preconditioner,
            regularization);
        const bool converged = solver.solve(rhs, solution);
        const std::string case_name =
            "elliptic " + std::string(1, side);
        test.require(converged, case_name + " converges");
        test.require(
            relative_residual(
                *vector_space,
                *matrix_operator,
                rhs,
                solution) <= 2e-8,
            case_name + " physical residual");
        test.require(
            std::abs(vector_space->sum(solution)) <= 2e-8,
            case_name + " remains in zero-mean subspace");
    }
}

} // namespace

int main()
{
    nmfd::tests::test_harness test;
    test_diffusion(test);
    test_advection(test);
    test_elliptic_regularization(test);
    return test.finish("NMFD legacy GMRES tests");
}
