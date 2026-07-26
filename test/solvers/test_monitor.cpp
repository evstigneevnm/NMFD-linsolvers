#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <scfd/utils/log.h>

#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/monitor_krylov.h>

#include "common/cpu_reference_vector_space.h"
#include "common/test_harness.h"

namespace
{

using log_type = scfd::utils::log_std;
using vector_space_type = nmfd::tests::cpu_reference_vector_space<double>;
using monitor_type =
    nmfd::solvers::default_monitor<vector_space_type, log_type>;
using krylov_monitor_type =
    nmfd::solvers::monitor_krylov<vector_space_type, log_type>;

void test_convergence_and_history(nmfd::tests::test_harness& test)
{
    vector_space_type vector_space(2);
    monitor_type::params parameters;
    parameters.rel_tol = 0.1;
    parameters.abs_tol = 0.0;
    parameters.min_iters_num = 1;
    parameters.max_iters_num = 5;
    parameters.save_convergence_history = true;

    monitor_type monitor(vector_space, nullptr, parameters);
    vector_space_type::vector_type rhs{3.0, 4.0};
    vector_space_type::vector_type solution{0.0, 0.0};
    vector_space_type::vector_type residual{0.25, 0.0};
    monitor.start(rhs);

    test.require(
        !monitor.check_finished(solution, residual),
        "minimum iteration count delays convergence");
    ++monitor;
    test.require(
        monitor.check_finished(solution, residual),
        "monitor converges after minimum iteration count");
    test.require(monitor.converged(), "monitor records convergence");
    test.require_close(
        monitor.rhs_norm(),
        5.0,
        1e-14,
        "monitor stores RHS norm");
    test.require(
        monitor.convergence_history().size() == 2,
        "monitor records convergence history");
    monitor.stop();
}

void test_limits_and_invalid_solution(nmfd::tests::test_harness& test)
{
    vector_space_type vector_space(2);
    monitor_type::params parameters;
    parameters.rel_tol = 0.0;
    parameters.abs_tol = 0.0;
    parameters.max_iters_num = 1;

    monitor_type monitor(vector_space, nullptr, parameters);
    vector_space_type::vector_type rhs{1.0, 0.0};
    vector_space_type::vector_type solution{0.0, 0.0};
    vector_space_type::vector_type residual{1.0, 0.0};
    monitor.start(rhs);
    test.require(
        !monitor.check_finished(solution, residual),
        "monitor continues before iteration limit");
    ++monitor;
    test.require(
        monitor.check_finished(solution, residual),
        "monitor stops at iteration limit");
    test.require(
        !monitor.converged(),
        "iteration-limit stop is not convergence");
    monitor.stop();

    solution[0] = std::numeric_limits<double>::quiet_NaN();
    monitor.start(rhs);
    test.require(
        monitor.check_finished(solution, residual),
        "monitor stops for invalid solution");
    test.require(
        !monitor.is_valid_number(),
        "monitor reports invalid solution");
    monitor.stop();
}

void test_temporary_parameters(nmfd::tests::test_harness& test)
{
    vector_space_type vector_space(1);
    monitor_type::params parameters;
    parameters.rel_tol = 1e-4;
    parameters.max_iters_num = 20;
    monitor_type monitor(vector_space, nullptr, parameters);

    monitor.set_temp_tolerance(2e-2);
    test.require_close(
        monitor.rel_tol(),
        2e-2,
        1e-14,
        "temporary tolerance is applied");
    monitor.restore_tolerance();
    test.require_close(
        monitor.rel_tol(),
        1e-4,
        1e-14,
        "temporary tolerance restores original value");

    monitor.set_temp_max_iterations(3);
    test.require(
        monitor.max_iters_num() == 3,
        "temporary iteration limit is applied");
    monitor.restore_max_iterations();
    test.require(
        monitor.max_iters_num() == 20,
        "temporary iteration limit restores original value");
}

void test_zero_rhs_scaling(nmfd::tests::test_harness& test)
{
    vector_space_type vector_space(2);
    monitor_type::params parameters;
    parameters.rel_tol = 1e-6;
    parameters.abs_tol = 1e-3;
    parameters.divide_out_norms_by_rel_base = true;
    parameters.save_convergence_history = true;

    monitor_type monitor(vector_space, nullptr, parameters);
    vector_space_type::vector_type rhs{0.0, 0.0};
    vector_space_type::vector_type solution{0.0, 0.0};
    vector_space_type::vector_type residual{1e-4, 0.0};
    monitor.start(rhs);
    test.require(
        monitor.check_finished(solution, residual),
        "zero-RHS solve uses absolute tolerance");
    test.require(
        std::isfinite(monitor.resid_norm_out()),
        "zero-RHS output residual is finite");
    test.require_close(
        monitor.resid_norm_out(),
        1e-4,
        1e-14,
        "zero-RHS output residual remains absolute");
    test.require_close(
        monitor.tol_out(),
        1e-3,
        1e-14,
        "zero-RHS output tolerance remains absolute");
    test.require(
        std::isfinite(monitor.convergence_history().front().second),
        "zero-RHS convergence history is finite");
    monitor.stop();
}

void test_minimum_residual_storage(nmfd::tests::test_harness& test)
{
    vector_space_type vector_space(2);
    monitor_type::params parameters;
    parameters.rel_tol = 0.0;
    parameters.abs_tol = 0.0;
    parameters.max_iters_num = 10;
    parameters.out_min_resid_norm = true;

    monitor_type monitor(vector_space, nullptr, parameters);
    vector_space_type::vector_type rhs{1.0, 0.0};
    vector_space_type::vector_type solution{1.0, 1.0};
    vector_space_type::vector_type residual{0.5, 0.0};
    monitor.start(rhs);
    monitor.check_finished(solution, residual);

    ++monitor;
    solution = {2.0, 2.0};
    residual = {0.25, 0.0};
    monitor.check_finished(solution, residual);

    ++monitor;
    solution = {3.0, 3.0};
    residual = {0.75, 0.0};
    monitor.check_finished(solution, residual);

    const auto& minimum_solution = monitor.min_resid_norm_x();
    test.require_close(
        minimum_solution[0],
        2.0,
        1e-14,
        "minimum-residual solution stores first component");
    test.require_close(
        minimum_solution[1],
        2.0,
        1e-14,
        "minimum-residual solution stores second component");
    monitor.stop();
}

void test_krylov_estimate_and_complex_norm_type(
    nmfd::tests::test_harness& test)
{
    vector_space_type vector_space(1);
    krylov_monitor_type::params parameters;
    parameters.rel_tol = 0.1;
    parameters.abs_tol = 0.0;
    krylov_monitor_type monitor(vector_space, nullptr, parameters);
    vector_space_type::vector_type rhs{2.0};
    monitor.start(rhs);
    test.require(
        monitor.check_finished_by_ritz_estimate(0.19),
        "Krylov monitor accepts converged Ritz estimate");
    test.require(
        !monitor.check_finished_by_ritz_estimate(
            std::numeric_limits<double>::infinity()),
        "Krylov monitor rejects non-finite Ritz estimate");
    monitor.stop();

    using complex_space_type =
        nmfd::tests::cpu_reference_vector_space<std::complex<double>>;
    using complex_monitor_type =
        nmfd::solvers::monitor_krylov<complex_space_type, log_type>;
    test.require(
        (std::is_same_v<typename complex_monitor_type::norm_type, double>),
        "complex monitor uses real norm type");
}

} // namespace

int main()
{
    nmfd::tests::test_harness test;
    test_convergence_and_history(test);
    test_limits_and_invalid_solution(test);
    test_temporary_parameters(test);
    test_zero_rhs_scaling(test);
    test_minimum_residual_storage(test);
    test_krylov_estimate_and_complex_norm_type(test);
    return test.finish("NMFD monitor tests");
}
