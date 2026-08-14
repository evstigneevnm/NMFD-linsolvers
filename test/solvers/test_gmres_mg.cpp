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

template<class VectorSpace>
class recording_mg_observer
{
public:
    struct record
    {
        std::size_t level;
        nmfd::preconditioners::mg_cycle_event event;
        typename VectorSpace::norm_type norm;
    };

    recording_mg_observer() : records_(std::make_shared<std::vector<record>>())
    {
    }

    void on_event(
        std::size_t level,
        nmfd::preconditioners::mg_cycle_event event,
        const VectorSpace& vector_space,
        const typename VectorSpace::vector_type& vector) const
    {
        records_->push_back(record{level, event, vector_space.norm(vector)});
    }

    void clear() const
    {
        records_->clear();
    }

    bool saw(nmfd::preconditioners::mg_cycle_event event) const
    {
        return std::any_of(
            records_->begin(), records_->end(),
            [event](const record& value) { return value.event == event; });
    }

    bool all_finite() const
    {
        return std::all_of(
            records_->begin(), records_->end(),
            [](const record& value) { return std::isfinite(value.norm); });
    }

    bool empty() const
    {
        return records_->empty();
    }

private:
    std::shared_ptr<std::vector<record>> records_;
};

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
    using observer_type = recording_mg_observer<vector_space_type>;
    using multigrid_type =
        nmfd::preconditioners::mg<
            operator_type,
            restrictor_type,
            prolongator_type,
            smoother_type,
            coarse_solver_type,
            coarsening_type,
            log_type,
            observer_type>;
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
    observer_type observer;
    multigrid_utils.observer = observer;
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
        *multigrid_utils.coarsening.counters =
            typename coarsening_type::statistics{};
        auto multigrid =
            std::make_shared<multigrid_type>(
                multigrid_utils,
                multigrid_parameters);
        observer.clear();
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

        const auto status = multigrid->get_status();
        test.require(multigrid->is_ready(), label + " multigrid ready");
        test.require(status.levels >= 2, label + " multigrid hierarchy built");
        test.require(status.apply_calls > 0, label + " multigrid apply count");
        test.require(!observer.empty(), label + " multigrid observer called");
        test.require(
            observer.saw(nmfd::preconditioners::mg_cycle_event::cycle_begin_rhs),
            label + " multigrid begin event");
        test.require(
            observer.saw(nmfd::preconditioners::mg_cycle_event::cycle_end_solution),
            label + " multigrid end event");
        test.require(observer.all_finite(), label + " multigrid observer finite norms");

        const auto previous_levels = multigrid->level_count();
        const auto structural_calls_before_update =
            multigrid_utils.coarsening.counters->next_level_calls;
        const auto coarsening_instances_before_update =
            multigrid_utils.coarsening.counters->construction_calls;
        const auto coarse_updates_before =
            multigrid_utils.coarsening.counters->update_coarse_operator_calls;
        solver.update_operator(matrix_operator);
        const auto updated_status = multigrid->get_status();
        test.require(
            multigrid->level_count() == previous_levels,
            label + " numerical update preserves level count");
        test.require(
            updated_status.apply_calls == 0,
            label + " numerical update resets apply count");
        test.require(
            updated_status.operator_updates == 1,
            label + " numerical update count");
        test.require(
            multigrid_utils.coarsening.counters->next_level_calls ==
                structural_calls_before_update,
            label + " numerical update preserves transfer hierarchy");
        test.require(
            multigrid_utils.coarsening.counters->construction_calls ==
                coarsening_instances_before_update,
            label + " numerical update preserves coarsening policy state");
        test.require(
            multigrid_utils.coarsening.counters->update_coarse_operator_calls -
                    coarse_updates_before ==
                previous_levels - 1,
            label + " numerical update refreshes coarse operators");

        vector_space->assign_scalar(0.0, solution);
        test.require(
            solver.solve(rhs, solution),
            label + " converges after numerical update");
        test.require(
            relative_residual(
                *vector_space, *matrix_operator, rhs, solution) <=
                physical_residual_tolerance,
            label + " physical residual after numerical update");

        bool incompatible_update_rejected = false;
        try
        {
            solver.update_operator(
                std::make_shared<operator_type>(dimension/2));
        }
        catch(const std::invalid_argument&)
        {
            incompatible_update_rejected = true;
        }
        test.require(
            incompatible_update_rejected,
            label + " incompatible numerical update rejected");
        vector_space->assign_scalar(0.0, solution);
        test.require(
            solver.solve(rhs, solution),
            label + " retains old operator after rejected update");
        test.require(
            relative_residual(
                *vector_space, *matrix_operator, rhs, solution) <=
                physical_residual_tolerance,
            label + " old operator residual after rejected update");

        const auto structural_calls_before_rebuild =
            multigrid_utils.coarsening.counters->next_level_calls;
        const auto coarsening_instances_before_rebuild =
            multigrid_utils.coarsening.counters->construction_calls;
        solver.set_operator(matrix_operator);
        const auto rebuilt_status = multigrid->get_status();
        test.require(
            multigrid->level_count() == previous_levels,
            label + " transactional rebuild preserves level count");
        test.require(
            rebuilt_status.apply_calls == 0,
            label + " transactional rebuild resets apply count");
        test.require(
            rebuilt_status.operator_updates == 0,
            label + " structural rebuild resets numerical update count");
        test.require(
            multigrid_utils.coarsening.counters->next_level_calls >
                structural_calls_before_rebuild,
            label + " structural rebuild reconstructs transfers");
        test.require(
            multigrid_utils.coarsening.counters->construction_calls ==
                coarsening_instances_before_rebuild + 1,
            label + " structural rebuild replaces coarsening policy state");
    }

    return test.finish("NMFD GMRES multigrid tests");
}
