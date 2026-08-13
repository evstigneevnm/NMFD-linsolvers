#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <scfd/utils/log.h>

#include <nmfd/detail/algo_utils_hierarchy_factory.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>

#include "common/analytical_operator.h"
#include "common/cpu_reference_vector_space.h"
#include "common/test_harness.h"

namespace
{

class hierarchy_test_backend
{
public:
    scfd::utils::log_std& log()
    {
        return log_;
    }

private:
    scfd::utils::log_std log_;
};

template<class VectorSpace>
struct backend_aware_hierarchy_algorithm
{
    struct utils_hierarchy
    {
        std::shared_ptr<VectorSpace> vector_space;
        const void* log_address = nullptr;

        utils_hierarchy() = default;

        template<class Backend>
        utils_hierarchy(
            Backend& backend,
            std::shared_ptr<VectorSpace> vector_space_)
            : vector_space(std::move(vector_space_)),
              log_address(
                  static_cast<const void*>(std::addressof(backend.log())))
        {
        }
    };
};

template<class VectorSpace>
struct vector_space_hierarchy_algorithm
{
    struct utils_hierarchy
    {
        std::shared_ptr<VectorSpace> vector_space;

        utils_hierarchy() = default;

        explicit utils_hierarchy(
            std::shared_ptr<VectorSpace> vector_space_)
            : vector_space(std::move(vector_space_))
        {
        }
    };
};

template<class LinearSolver>
struct recursive_hierarchy_consumer
{
    struct utils_hierarchy
    {
        typename LinearSolver::utils_hierarchy linear_solver;

        template<class Backend, class VectorSpace>
        utils_hierarchy(
            Backend& backend,
            std::shared_ptr<VectorSpace> vector_space)
            : linear_solver(backend, std::move(vector_space))
        {
        }
    };
};

struct explicitly_constructed_residual_regularization
{
    explicitly_constructed_residual_regularization() = delete;

    explicit explicitly_constructed_residual_regularization(int)
    {
    }
};

template<class Scalar>
std::vector<Scalar> multiply(
    const std::vector<Scalar>& matrix,
    const std::vector<Scalar>& vector)
{
    const std::size_t dimension = vector.size();
    std::vector<Scalar> result(dimension, Scalar{});
    for(std::size_t row = 0; row < dimension; ++row)
        for(std::size_t col = 0; col < dimension; ++col)
            result[row] += matrix[row*dimension + col]*vector[col];
    return result;
}

template<class VectorSpace, class Operator>
typename VectorSpace::norm_type physical_relative_residual(
    const VectorSpace& vector_space,
    const Operator& matrix_operator,
    const typename VectorSpace::vector_type& rhs,
    const typename VectorSpace::vector_type& solution)
{
    typename VectorSpace::vector_type residual(
        static_cast<std::size_t>(vector_space.size()),
        typename VectorSpace::scalar_type{});
    matrix_operator.apply(solution, residual);
    vector_space.add_lin_comb(
        typename VectorSpace::scalar_type(1),
        rhs,
        typename VectorSpace::scalar_type(-1),
        residual);
    return vector_space.norm(residual)/
        std::max(
            vector_space.norm(rhs),
            std::numeric_limits<typename VectorSpace::norm_type>::min());
}

template<class Scalar>
void run_unpreconditioned_case(
    const std::string& label,
    const std::vector<Scalar>& matrix,
    const std::vector<Scalar>& reference,
    const std::string& orthogonalization,
    const std::string& reorthogonalization,
    nmfd::tests::test_harness& test)
{
    using vector_space_type =
        nmfd::tests::cpu_reference_vector_space<Scalar>;
    using operator_type = nmfd::tests::analytical_operator<vector_space_type>;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type>;

    auto vector_space =
        std::make_shared<vector_space_type>(
            static_cast<typename vector_space_type::ordinal_type>(
                reference.size()));
    auto matrix_operator =
        std::make_shared<operator_type>(*vector_space, matrix);
    const auto rhs_host = multiply(matrix, reference);
    typename vector_space_type::vector_type rhs = rhs_host;
    typename vector_space_type::vector_type solution(reference.size(), Scalar{});

    typename solver_type::params parameters;
    parameters.basis_size = static_cast<unsigned>(reference.size());
    parameters.batch_size = 1;
    parameters.orthogonalization = orthogonalization;
    parameters.reorthogonalization_policy = reorthogonalization;
    parameters.max_orthogonalization_passes =
        reorthogonalization == "none" ? 1 : 2;
    parameters.monitor.rel_tol = 1e-12;
    parameters.monitor.abs_tol = 1e-14;
    parameters.monitor.max_iters_num = 30;
    parameters.monitor.save_convergence_history = true;
    parameters.monitor.divide_out_norms_by_rel_base = false;

    solver_type solver(matrix_operator, vector_space, nullptr, parameters);
    const bool converged = solver.solve(rhs, solution);
    test.require(converged, label + " converges");
    for(std::size_t index = 0; index < reference.size(); ++index)
    {
        test.require_close(
            solution[index],
            reference[index],
            3e-10,
            label + " solution component " + std::to_string(index));
    }
    test.require(
        physical_relative_residual(
            *vector_space,
            *matrix_operator,
            rhs,
            solution) <= 2e-11,
        label + " physical residual");
    test.require(
        !solver.monitor().convergence_history().empty(),
        label + " records convergence history");
    test.require(
        solver.monitor().iters_performed() <=
            parameters.monitor.max_iters_num,
        label + " respects iteration limit");

    vector_space->scale(Scalar(0.999), solution);
    const bool reused_converged = solver.solve(rhs, solution);
    test.require(reused_converged, label + " reused guess converges");
    test.require(
        physical_relative_residual(
            *vector_space,
            *matrix_operator,
            rhs,
            solution) <= 2e-11,
        label + " reused guess physical residual");
}

void test_real_solvers(nmfd::tests::test_harness& test)
{
    const std::vector<double> matrix{
        4.0, 1.0, 0.0, 0.0,
        0.0, 3.0, -1.0, 0.0,
        0.0, 0.0, 2.0, 0.5,
        0.2, 0.0, 0.0, 1.0
    };
    const std::vector<double> reference{0.5, -2.0, 3.0, -1.5};
    run_unpreconditioned_case(
        "real MGS",
        matrix,
        reference,
        "mgs",
        "none",
        test);
    run_unpreconditioned_case(
        "real MGS/DGKS",
        matrix,
        reference,
        "mgs",
        "dgks",
        test);
    run_unpreconditioned_case(
        "real CGS2",
        matrix,
        reference,
        "cgs",
        "always",
        test);
}

void test_complex_solver(nmfd::tests::test_harness& test)
{
    using scalar_type = std::complex<double>;
    const std::vector<scalar_type> matrix{
        {2.0, 1.0}, {0.5, -0.25}, {0.0, 0.0},
        {0.0, 0.0}, {-1.0, 0.5}, {0.25, 0.1},
        {0.1, -0.2}, {0.0, 0.0}, {3.0, -2.0}
    };
    const std::vector<scalar_type> reference{
        {0.5, -0.25},
        {-2.0, 0.75},
        {3.0, 1.5}
    };
    run_unpreconditioned_case(
        "complex MGS/DGKS",
        matrix,
        reference,
        "mgs",
        "dgks",
        test);
}

void test_preconditioners(nmfd::tests::test_harness& test)
{
    using scalar_type = double;
    using vector_space_type =
        nmfd::tests::cpu_reference_vector_space<scalar_type>;
    using operator_type = nmfd::tests::analytical_operator<vector_space_type>;
    using preconditioner_type =
        nmfd::tests::diagonal_preconditioner<
            vector_space_type,
            operator_type>;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type,
            preconditioner_type>;

    const std::vector<double> matrix{
        4.0, 1.0, 0.0, 0.0,
        0.0, 3.0, -1.0, 0.0,
        0.0, 0.0, 2.0, 0.5,
        0.2, 0.0, 0.0, 1.0
    };
    const std::vector<double> reference{0.5, -2.0, 3.0, -1.5};
    auto vector_space = std::make_shared<vector_space_type>(4);
    auto matrix_operator =
        std::make_shared<operator_type>(*vector_space, matrix);
    auto preconditioner =
        std::make_shared<preconditioner_type>(
            *vector_space,
            std::vector<double>{0.25, 1.0/3.0, 0.5, 1.0});
    const auto rhs = multiply(matrix, reference);

    typename solver_type::params parameters;
    parameters.basis_size = 4;
    parameters.batch_size = 1;
    parameters.reorthogonalization_policy = "dgks";
    parameters.monitor.rel_tol = 1e-12;
    parameters.monitor.abs_tol = 1e-14;
    parameters.monitor.max_iters_num = 30;
    parameters.monitor.divide_out_norms_by_rel_base = false;

    for(const char side : {'L', 'R'})
    {
        parameters.preconditioner_side = side;
        typename vector_space_type::vector_type solution(4, 0.0);
        solver_type solver(
            matrix_operator,
            vector_space,
            nullptr,
            parameters,
            preconditioner);
        const bool converged = solver.solve(rhs, solution);
        test.require(
            converged,
            std::string("preconditioned ") + side + " converges");
        test.require(
            physical_relative_residual(
                *vector_space,
                *matrix_operator,
                rhs,
                solution) <= 2e-11,
            std::string("preconditioned ") + side + " residual");
        test.require(
            preconditioner->apply_calls() > 0,
            std::string("preconditioned ") + side + " is applied");
        preconditioner->reset_calls();
    }
}

void test_failure_paths(nmfd::tests::test_harness& test)
{
    using vector_space_type =
        nmfd::tests::cpu_reference_vector_space<double>;
    using operator_type = nmfd::tests::analytical_operator<vector_space_type>;
    using preconditioner_type =
        nmfd::tests::diagonal_preconditioner<
            vector_space_type,
            operator_type>;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type,
            preconditioner_type>;

    auto vector_space = std::make_shared<vector_space_type>(2);
    auto matrix_operator =
        std::make_shared<operator_type>(
            *vector_space,
            std::vector<double>{2.0, 1.0, 0.0, 3.0});
    auto preconditioner =
        std::make_shared<preconditioner_type>(
            *vector_space,
            std::vector<double>{0.5, 1.0/3.0});
    vector_space_type::vector_type rhs{1.0, 2.0};
    vector_space_type::vector_type solution(2, 0.0);

    typename solver_type::params parameters;
    parameters.basis_size = 2;
    parameters.batch_size = 1;
    parameters.monitor.max_iters_num = 10;
    parameters.monitor.divide_out_norms_by_rel_base = false;

    solver_type solver(
        matrix_operator,
        vector_space,
        nullptr,
        parameters,
        preconditioner);

    matrix_operator->fail_after(0);
    test.require(
        !solver.solve(rhs, solution),
        "GMRES propagates operator failure");
    matrix_operator->allow_all_calls();
    matrix_operator->reset_calls();

    preconditioner->fail_after(0);
    vector_space->assign_scalar(0.0, solution);
    test.require(
        !solver.solve(rhs, solution),
        "GMRES propagates preconditioner failure");
    preconditioner->allow_all_calls();
    preconditioner->reset_calls();

    parameters.monitor.max_iters_num = 1;
    parameters.basis_size = 1;
    solver_type short_solver(
        matrix_operator,
        vector_space,
        nullptr,
        parameters,
        preconditioner);
    vector_space->assign_scalar(0.0, solution);
    test.require(
        !short_solver.solve(rhs, solution),
        "GMRES reports iteration-limit non-convergence");
    test.require(
        short_solver.monitor().iters_performed() == 1,
        "GMRES stops at configured iteration limit");
}

void test_initial_solution_and_parameters(nmfd::tests::test_harness& test)
{
    using vector_space_type =
        nmfd::tests::cpu_reference_vector_space<double>;
    using operator_type = nmfd::tests::analytical_operator<vector_space_type>;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type>;

    auto vector_space = std::make_shared<vector_space_type>(2);
    auto matrix_operator =
        std::make_shared<operator_type>(
            *vector_space,
            std::vector<double>{2.0, 0.0, 0.0, 3.0});
    vector_space_type::vector_type exact{0.5, -2.0};
    const auto rhs = multiply(matrix_operator->matrix(), exact);

    typename solver_type::params parameters;
    parameters.basis_size = 2;
    parameters.batch_size = 1;
    parameters.monitor.rel_tol = 1e-12;
    parameters.monitor.abs_tol = 1e-14;
    parameters.monitor.max_iters_num = 10;
    parameters.monitor.divide_out_norms_by_rel_base = false;
    solver_type solver(matrix_operator, vector_space, nullptr, parameters);
    test.require(solver.solve(rhs, exact), "exact initial solution converges");
    test.require(
        solver.monitor().iters_performed() == 0,
        "exact initial solution needs zero iterations");

    auto require_invalid =
        [&](typename solver_type::params invalid, const std::string& label)
        {
            bool threw = false;
            try
            {
                solver_type candidate(
                    matrix_operator,
                    vector_space,
                    nullptr,
                    invalid);
            }
            catch(const std::invalid_argument&)
            {
                threw = true;
            }
            test.require(threw, label);
        };
    auto invalid = parameters;
    invalid.basis_size = 0;
    require_invalid(invalid, "zero basis size rejected");
    invalid = parameters;
    invalid.batch_size = 0;
    require_invalid(invalid, "zero batch size rejected");
    invalid = parameters;
    invalid.preconditioner_side = 'X';
    require_invalid(invalid, "invalid preconditioner side rejected");
    invalid = parameters;
    invalid.orthogonalization = "invalid";
    require_invalid(invalid, "invalid orthogonalization rejected");
    invalid = parameters;
    invalid.reorthogonalization_policy = "invalid";
    require_invalid(invalid, "invalid reorthogonalization rejected");
}

void test_utils_hierarchy_construction(nmfd::tests::test_harness& test)
{
    using vector_space_type =
        nmfd::tests::cpu_reference_vector_space<double>;
    using operator_type = nmfd::tests::analytical_operator<vector_space_type>;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type>;

    hierarchy_test_backend backend;
    auto vector_space = std::make_shared<vector_space_type>(2);

    using backend_algorithm =
        backend_aware_hierarchy_algorithm<vector_space_type>;
    const auto backend_utils =
        nmfd::detail::make_algo_utils_hierarchy<backend_algorithm>(
            backend,
            vector_space);
    test.require(
        backend_utils.vector_space == vector_space,
        "hierarchy factory forwards vector space to backend-aware utilities");
    test.require(
        backend_utils.log_address ==
            static_cast<const void*>(std::addressof(backend.log())),
        "hierarchy factory prefers backend-aware construction");

    using vector_space_algorithm =
        vector_space_hierarchy_algorithm<vector_space_type>;
    const auto vector_space_utils =
        nmfd::detail::make_algo_utils_hierarchy<vector_space_algorithm>(
            backend,
            vector_space);
    test.require(
        vector_space_utils.vector_space == vector_space,
        "hierarchy factory falls back to vector-space construction");

    typename solver_type::utils_hierarchy solver_utils(
        backend,
        vector_space);
    test.require(
        solver_utils.vec_ops == vector_space,
        "GMRES hierarchy stores the supplied vector space");
    test.require(
        solver_utils.log == std::addressof(backend.log()),
        "GMRES hierarchy uses the backend logger");
    test.require(
        static_cast<bool>(solver_utils.residual_reg),
        "GMRES hierarchy creates residual regularization utilities");
    test.require(
        static_cast<bool>(solver_utils.dense_ops),
        "GMRES hierarchy creates dense utilities");
    test.require(
        solver_utils.preconditioner.ops == vector_space,
        "GMRES hierarchy recursively creates preconditioner utilities");

    typename recursive_hierarchy_consumer<solver_type>::utils_hierarchy
        recursive_utils(backend, vector_space);
    test.require(
        recursive_utils.linear_solver.vec_ops == vector_space,
        "Newton-style recursive hierarchy constructs GMRES utilities");
    test.require(
        recursive_utils.linear_solver.preconditioner.ops == vector_space,
        "Newton-style recursive hierarchy constructs nested utilities");

    auto explicit_residual =
        std::make_shared<typename solver_type::residual_regulaization_t>();
    auto explicit_dense =
        std::make_shared<typename solver_type::dense_operations_t>();
    typename solver_type::preconditioner_utils_hierarchy_type
        explicit_preconditioner(vector_space);
    typename solver_type::utils_hierarchy explicit_utils(
        explicit_preconditioner,
        vector_space,
        std::addressof(backend.log()),
        explicit_residual,
        explicit_dense);
    test.require(
        explicit_utils.residual_reg == explicit_residual,
        "explicit hierarchy preserves residual regularization resources");
    test.require(
        explicit_utils.dense_ops == explicit_dense,
        "explicit hierarchy preserves dense resources");

    using explicit_resource_solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type,
            nmfd::preconditioners::dummy<
                vector_space_type,
                operator_type>,
            explicitly_constructed_residual_regularization>;
    test.require(
        !std::is_constructible_v<
            typename explicit_resource_solver_type::utils_hierarchy,
            hierarchy_test_backend&,
            std::shared_ptr<vector_space_type>>,
        "backend convenience construction is disabled for explicit-only "
        "resources");
    auto explicit_only_residual =
        std::make_shared<explicitly_constructed_residual_regularization>(1);
    typename explicit_resource_solver_type::preconditioner_utils_hierarchy_type
        explicit_only_preconditioner(vector_space);
    typename explicit_resource_solver_type::utils_hierarchy
        explicit_only_utils(
            explicit_only_preconditioner,
            vector_space,
            std::addressof(backend.log()),
            explicit_only_residual,
            std::make_shared<
                typename explicit_resource_solver_type::dense_operations_t>());
    test.require(
        explicit_only_utils.residual_reg == explicit_only_residual,
        "explicit-only resources remain injectable");

    auto matrix_operator =
        std::make_shared<operator_type>(
            *vector_space,
            std::vector<double>{2.0, 0.0, 0.0, 4.0});
    vector_space_type::vector_type rhs{1.0, -2.0};
    vector_space_type::vector_type solution(2, 0.0);
    typename solver_type::params_hierarchy parameters;
    parameters.basis_size = 2;
    parameters.batch_size = 1;
    parameters.monitor.rel_tol = 1e-12;
    parameters.monitor.abs_tol = 1e-14;
    parameters.monitor.max_iters_num = 10;
    parameters.monitor.divide_out_norms_by_rel_base = false;

    solver_type solver(solver_utils, parameters);
    solver.set_operator(matrix_operator);
    test.require(
        solver.solve(rhs, solution),
        "GMRES created from recursive utilities solves a system");
    test.require_close(
        solution[0],
        0.5,
        1e-12,
        "hierarchy-created GMRES solution component 0");
    test.require_close(
        solution[1],
        -0.5,
        1e-12,
        "hierarchy-created GMRES solution component 1");
}

} // namespace

int main()
{
    nmfd::tests::test_harness test;
    test_real_solvers(test);
    test_complex_solver(test);
    test_preconditioners(test);
    test_failure_paths(test);
    test_initial_solution_and_parameters(test);
    test_utils_hierarchy_construction(test);
    return test.finish("NMFD GMRES core tests");
}
