// Copyright © 2020-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of NMFD.

// NMFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// NMFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with NMFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __NMFD_GMRES_H__
#define __NMFD_GMRES_H__

#include <stdexcept>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef NMFD_ENABLE_NLOHMANN
#include <nlohmann/json.hpp>
#endif
#include "detail/monitor_call_wrap.h"
#include <nmfd/detail/algo_utils_hierarchy.h>
#include <nmfd/detail/algo_utils_hierarchy_factory.h>
#include <nmfd/detail/algo_params_hierarchy.h>
#include <nmfd/detail/algo_hierarchy_creator.h>
#include <nmfd/operations/linalg/host_small_dense_backend.h>
#include "iter_solver_base.h"
#include "detail/residual_regularization_dummy.h"
#include "krylov/arnoldi.h"
#include "krylov/operator_apply.h"
#include "krylov/orthogonalization.h"
#include <nmfd/preconditioners/dummy.h>

namespace nmfd
{
namespace solvers 
{

//demands for template parameters:
//VectorOperations fits VectorOperations concept (see CONCEPTS.txt)
//LinearOperator and Preconditioner fit LinearOperator concept (see CONCEPTS.txt)
//VectorOperations, LinearOperator and Preconditioner have same T_vec

//Monitor concept:
//TODOjj

template
<
     class VectorOperations, class Monitor, class Log, 
     class LinearOperator, class Preconditioner = preconditioners::dummy<VectorOperations,LinearOperator>,
     class ResidualRegulariation = detail::residual_regularization_dummy,
     class DenseOperations = nmfd::operations::linalg::host_small_dense_backend<
         typename VectorOperations::Ord,
         typename VectorOperations::scalar_type>
>
class gmres : public iter_solver_base<VectorOperations, Monitor, Log, LinearOperator, Preconditioner>
{
    using parent_t = iter_solver_base<VectorOperations, Monitor, Log, LinearOperator, Preconditioner>;
    using logged_obj_t = typename parent_t::logged_obj_t;
    using logged_obj_params_t = typename parent_t::logged_obj_params_t;

public:
    using scalar_type =  typename VectorOperations::scalar_type;
    using norm_type = typename VectorOperations::norm_type;
    using vector_type = typename VectorOperations::vector_type;
    using multivector_type = typename VectorOperations::multivector_type;
    using linear_operator_type =  LinearOperator;
    using preconditioner_type = Preconditioner;
    using vector_operations_type = VectorOperations;
    using dense_operations_t = DenseOperations;
    using monitor_type = Monitor;
    using log_type = Log;
    using residual_regulaization_t = ResidualRegulariation;


    struct params : public logged_obj_params_t
    {
        unsigned basis_size; //size of the krylov basis
        unsigned batch_size; //size of the batch size s.t. Ritz vector converhence is checked each batch_size inside the krylov bass size
        char preconditioner_side; //can be L for left and R for right
        bool reorthogonalization; //apply additional reorthogonalization in Gram-Schmidt process
        std::string orthogonalization;
        std::string reorthogonalization_policy;
        norm_type dgks_eta;
        norm_type breakdown_relative_tolerance;
        unsigned max_orthogonalization_passes;
        bool do_restart_on_false_ritz_convergence;
        typename Monitor::params monitor;

        params(const std::string &log_prefix = "", const std::string &log_name = "gmres::") :
            logged_obj_params_t(0, log_prefix+log_name),
            basis_size(20), 
            batch_size(5), 
            preconditioner_side('R'), 
            reorthogonalization(false), 
            orthogonalization("mgs"),
            reorthogonalization_policy("none"),
            dgks_eta(static_cast<norm_type>(0.717)),
            breakdown_relative_tolerance(
                static_cast<norm_type>(10)*std::numeric_limits<norm_type>::epsilon()),
            max_orthogonalization_passes(2),
            do_restart_on_false_ritz_convergence(false),
            monitor( typename Monitor::params(this->log_msg_prefix) )
        {
        }
        #ifdef NMFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
            basis_size = j.value("basis_size", basis_size);
            batch_size = j.value("batch_size", batch_size);
            preconditioner_side = j.value("preconditioner_side", preconditioner_side);
            reorthogonalization = j.value("reorthogonalization", reorthogonalization);
            orthogonalization = j.value("orthogonalization", orthogonalization);
            reorthogonalization_policy = j.value(
                "reorthogonalization_policy",
                reorthogonalization_policy);
            dgks_eta = j.value("dgks_eta", dgks_eta);
            breakdown_relative_tolerance = j.value(
                "breakdown_relative_tolerance",
                breakdown_relative_tolerance);
            max_orthogonalization_passes = j.value(
                "max_orthogonalization_passes",
                max_orthogonalization_passes);
            do_restart_on_false_ritz_convergence = j.value("do_restart_on_false_ritz_convergence", do_restart_on_false_ritz_convergence);
            monitor.from_json(j.at("monitor"));
        }
        nlohmann::json to_json() const
        {
            return
                nlohmann::json
                {
                    {"type", "gmres"},
                    {"basis_size", basis_size},
                    {"batch_size", batch_size},
                    {"preconditioner_side", preconditioner_side},
                    {"reorthogonalization", reorthogonalization},
                    {"orthogonalization", orthogonalization},
                    {"reorthogonalization_policy", reorthogonalization_policy},
                    {"dgks_eta", dgks_eta},
                    {"breakdown_relative_tolerance", breakdown_relative_tolerance},
                    {"max_orthogonalization_passes", max_orthogonalization_passes},
                    {"do_restart_on_false_ritz_convergence", do_restart_on_false_ritz_convergence},
                    {"monitor", monitor.to_json()}
                };
        }
        #endif
    };
    struct utils
    {
        std::shared_ptr<vector_operations_type> vec_ops;
        Log *log;
        std::shared_ptr<residual_regulaization_t> residual_reg;
        std::shared_ptr<dense_operations_t> dense_ops;
        utils() = default;
        utils(
            std::shared_ptr<vector_operations_type> vec_ops_, Log *log_ = nullptr,
            std::shared_ptr<residual_regulaization_t> residual_reg_ = std::make_shared<residual_regulaization_t>(),
            std::shared_ptr<dense_operations_t> dense_ops_ = std::make_shared<dense_operations_t>()
        ) : 
            vec_ops(vec_ops_), log(log_), residual_reg(residual_reg_), dense_ops(dense_ops_) 
        {
        }
    };
    using preconditioner_params_hierarchy_type = typename nmfd::detail::algo_params_hierarchy<Preconditioner>::type;
    struct params_hierarchy : public params
    {
        preconditioner_params_hierarchy_type preconditioner;

        params_hierarchy(const std::string &log_prefix = "", const std::string &log_name = "gmres::") : 
            params(log_prefix, log_name),
            preconditioner(this->log_msg_prefix)
        {
        }
        params_hierarchy(
            const params &prm_, 
            const preconditioner_params_hierarchy_type &preconditioner_
        ) : params(prm_), preconditioner(preconditioner_)
        {
        }
        #ifdef NMFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
            params::from_json(j);
            preconditioner.from_json(j.at("preconditioner"));
        }
        nlohmann::json to_json() const
        {
            nlohmann::json  j = params::to_json(),
                            j_prec = preconditioner.to_json();
            j["preconditioner"] = j_prec;
            return j;
        }
        #endif
    };

    using preconditioner_utils_hierarchy_type = typename nmfd::detail::algo_utils_hierarchy<Preconditioner>::type;
    struct utils_hierarchy : public utils
    {
        preconditioner_utils_hierarchy_type preconditioner;

        utils_hierarchy() = default;
        template<class ...Args>
        utils_hierarchy(
            preconditioner_utils_hierarchy_type preconditioner_,
            Args... args
        ) : 
            utils(args...),
            preconditioner(preconditioner_)
        {        
        }

        template<
            class Backend,
            class ResidualRegularizationForConstruction =
                residual_regulaization_t,
            class DenseOperationsForConstruction = dense_operations_t,
            std::enable_if_t<
                std::is_default_constructible_v<
                    ResidualRegularizationForConstruction> &&
                std::is_default_constructible_v<
                    DenseOperationsForConstruction>,
                int> = 0>
        utils_hierarchy(
            Backend& backend,
            std::shared_ptr<vector_operations_type> vec_space)
            : utils(
                  vec_space,
                  std::addressof(backend.log()),
                  std::make_shared<residual_regulaization_t>(),
                  std::make_shared<dense_operations_t>()),
              preconditioner(
                  nmfd::detail::make_algo_utils_hierarchy<
                      preconditioner_type>(backend, std::move(vec_space)))
        {
        }
    };

private:
    using T = scalar_type;
    using T_vec = vector_type;
    using T_mvec = multivector_type;

    using D_vec = typename dense_operations_t::vector_type;
    using D_mat = typename dense_operations_t::matrix_type;
    using monitor_call_wrap_t = detail::monitor_call_wrap<VectorOperations, Monitor>;
    using orthogonalization_options_type =
        krylov::orthogonalization_options<norm_type>;

    mutable T_mvec V_;
    mutable T_vec r_;
    mutable T_vec y_;
    mutable T_vec x_tmp_;
    
    //parameters:
    params prms_;

    //host dense operations vectors and matrices
    mutable D_mat H_;
    mutable D_vec s_;
    mutable D_vec cs_;
    mutable D_vec sn_;
    mutable D_vec s_h_;
    mutable std::vector<T> h_col_;
    mutable std::vector<T> h_pass_;

    void validate_params() const
    {
        if(prms_.basis_size == 0)
            throw std::invalid_argument("gmres basis_size must be positive");
        if(prms_.batch_size == 0)
            throw std::invalid_argument("gmres batch_size must be positive");
        if(prms_.preconditioner_side != 'L' && prms_.preconditioner_side != 'R')
            throw std::invalid_argument("gmres preconditioner_side must be L or R");
        if(prms_.max_orthogonalization_passes == 0)
            throw std::invalid_argument(
                "gmres max_orthogonalization_passes must be positive");
        (void)krylov::parse_orthogonalization_method(prms_.orthogonalization);
        (void)krylov::parse_reorthogonalization_policy(
            prms_.reorthogonalization_policy);
    }

    orthogonalization_options_type orthogonalization_options() const
    {
        orthogonalization_options_type options;
        options.method =
            krylov::parse_orthogonalization_method(prms_.orthogonalization);
        options.reorthogonalization =
            krylov::parse_reorthogonalization_policy(
                prms_.reorthogonalization_policy);
        if(
            prms_.reorthogonalization &&
            options.reorthogonalization ==
                krylov::reorthogonalization_policy::none)
        {
            options.reorthogonalization =
                krylov::reorthogonalization_policy::always;
        }
        options.dgks_eta = prms_.dgks_eta;
        options.breakdown_relative = prms_.breakdown_relative_tolerance;
        options.max_passes = prms_.max_orthogonalization_passes;
        return options;
    }

    bool apply_regularization(T_vec& vector) const
    {
        return krylov::apply_in_place(*residual_reg_, vector);
    }

    bool calc_left_preconditioned_residual(
        const linear_operator_type &A,
        const T_vec &x,
        const T_vec &b,
        T_vec &r) const
    {
        if(!krylov::apply_operator(A, x, r))
            return false;
        vec_ops_->add_lin_comb(static_cast<T>(1.0), b, static_cast<T>(-1.0), r);

        if ((prec_ != nullptr)&&(prms_.preconditioner_side == 'L'))
            return krylov::apply_in_place(*prec_, r);
        return true;
    }

    bool calc_right_precond_solution(T_vec &x) const
    {
        if ((prec_ != nullptr)&&(prms_.preconditioner_side == 'R'))
            return krylov::apply_in_place(*prec_, x);
        return true;
    }

    bool calc_krylov_vector(
        const linear_operator_type &A,
        const T_vec &x,
        T_vec &r) const
    {
        if(prec_ == nullptr)
            return krylov::apply_operator(A, x, r);

        if(prms_.preconditioner_side == 'L')
        {
            if(!krylov::apply_operator(A, x, r))
                return false;
            return krylov::apply_in_place(*prec_, r);
        }

        vec_ops_->assign(x, y_);
        if(!krylov::apply_in_place(*prec_, y_))
            return false;
        return krylov::apply_operator(A, y_, r);
    }


    void init_host()
    {
        dense_ops_->init_matrix(H_);
        dense_ops_->init_col_vectors(s_, cs_, sn_, s_h_);
    }

    void free_host_()
    {
        dense_ops_->free_matrix(H_);
        dense_ops_->free_col_vectors(s_, cs_, sn_, s_h_);
    }

    void init_all() const
    {
        vec_ops_->init_vector( r_ );
        vec_ops_->init_vector( y_ );
        vec_ops_->init_vector( x_tmp_ );
        vec_ops_->init_multivector( V_, prms_.basis_size+1);
    }

    void start_use_all() const
    {
        vec_ops_->start_use_vector( r_ );
        vec_ops_->start_use_vector( y_ );
        vec_ops_->start_use_vector( x_tmp_ );
        vec_ops_->start_use_multivector( V_, prms_.basis_size+1 );    
    }

    void stop_use_all() const
    {
        vec_ops_->stop_use_vector( r_ );
        vec_ops_->stop_use_vector( y_ );
        vec_ops_->stop_use_vector( x_tmp_ );
        vec_ops_->stop_use_multivector( V_, prms_.basis_size+1 );        
    }
    void free_all() const
    {
        vec_ops_->free_vector( r_ );
        vec_ops_->free_vector( y_ );
        vec_ops_->free_vector( x_tmp_ );
        vec_ops_->free_multivector( V_, prms_.basis_size+1 );
    }

    void zero_host_H() const
    {
        dense_ops_->assign_scalar_matrix(0, H_);
    }
    void zero_host_s() const
    {
        dense_ops_->assign_scalar_col_vector(0, s_);
    }


     //constructs solution of the linear system
    void construct_solution(const int i, const D_vec& s, T_vec& x) const
    {
        vec_ops_->assign_scalar(0, x);
        for (int j = 0; j <= i; j++) 
        {
            vec_ops_->add_lin_comb(
                dense_ops_->vector_at(s, j),
                V_,
                prms_.basis_size + 1,
                j,
                T(1),
                x);
        }
    }


protected:
    using parent_t::monitor_;
    using parent_t::vec_ops_;
    using parent_t::prec_;
    std::shared_ptr<dense_operations_t> dense_ops_;
    std::shared_ptr<residual_regulaization_t> residual_reg_;

public:
    ~gmres()
    {
        free_host_();
        free_all();
    }

    gmres(  
        std::shared_ptr<vector_operations_type> vec_ops, 
        Log *log = nullptr,
        const params& prm = params(),
        std::shared_ptr<preconditioner_type> prec = nullptr,
        std::shared_ptr<residual_regulaization_t> residual_reg = std::make_shared<residual_regulaization_t>(),
        std::shared_ptr<dense_operations_t> dense_ops = std::make_shared<dense_operations_t>()
    ) : 
        parent_t(std::move(vec_ops), log, prm, prm.monitor, std::move(prec) ), 
        prms_(prm),
        dense_ops_(std::move(dense_ops)),
        residual_reg_(std::move(residual_reg))
    {
        validate_params();
        dense_ops_->init(prm.basis_size+1, prm.basis_size);
        init_host();
        init_all();
        h_col_.resize(prm.basis_size + 1, T{});
        h_pass_.resize(prm.basis_size + 1, T{});
    }
    gmres(  
        std::shared_ptr<const linear_operator_type> A,
        std::shared_ptr<vector_operations_type> vec_ops, 
        Log *log = nullptr,
        const params& prm = params(),
        std::shared_ptr<preconditioner_type> prec = nullptr,
        std::shared_ptr<residual_regulaization_t> residual_reg = std::make_shared<residual_regulaization_t>(),
        std::shared_ptr<dense_operations_t> dense_ops = std::make_shared<dense_operations_t>()
    ) : 
        gmres(std::move(vec_ops),log,prm,std::move(prec),std::move(residual_reg),std::move(dense_ops))
    {
        parent_t::set_operator(std::move(A));
    }

    gmres(  
        const utils_hierarchy& utils,
        const params_hierarchy& prm = params_hierarchy()      
    ) : 
        gmres(  
            utils.vec_ops, utils.log, prm,
            nmfd::detail::algo_hierarchy_creator<preconditioner_type>::get(utils.preconditioner,prm.preconditioner),
            utils.residual_reg, utils.dense_ops
        )
    {
    }

    const std::shared_ptr<preconditioner_type> &preconditioner()const
    {
        return prec_;
    }

    virtual bool solve(const linear_operator_type &A, const T_vec &b, T_vec &x)const
    {
        const auto restart_ = prms_.basis_size;
        const auto orth_options = orthogonalization_options();
        start_use_all();
        monitor_call_wrap_t monitor_wrap(monitor_);

        bool operation_ok = true;
        if ((prec_ != nullptr)&&(prms_.preconditioner_side == 'L'))
        {
            vec_ops_->assign(b, r_);
            operation_ok =
                krylov::apply_in_place(*prec_, r_) &&
                apply_regularization(r_);
            if(operation_ok)
                monitor_wrap.start(r_);
        }
        else
        {
            monitor_wrap.start(b);
        }

        if(operation_ok)
        {
            operation_ok =
                calc_left_preconditioned_residual(A, x, b, r_) &&
                apply_regularization(r_);
        }

        bool finished = false;
        std::size_t total_iterations = 0;
        norm_type previous_residual = norm_type{};
        if(operation_ok)
        {
            previous_residual = vec_ops_->norm(r_);
            finished = monitor_.check_finished(x, r_);
        }

        while(operation_ok && !finished)
        {
            zero_host_H();
            zero_host_s();

            const norm_type beta = vec_ops_->norm(r_);
            if(
                !std::isfinite(beta) ||
                !(beta > std::numeric_limits<norm_type>::min()))
            {
                operation_ok = false;
                logged_obj_t::error_f(
                    "solve: invalid residual norm at restart: %e",
                    static_cast<double>(beta));
                break;
            }

            vec_ops_->scale(T(norm_type(1)/beta), r_);
            vec_ops_->assign(r_, V_, restart_ + 1, 0);
            dense_ops_->vector_at(s_, 0) = static_cast<T>(beta);

            int last_column = -1;
            bool accepted_trial_update = false;
            bool request_restart = false;
            for(unsigned column = 0; column < restart_; ++column)
            {
                ++monitor_;
                ++total_iterations;
                last_column = static_cast<int>(column);

                auto apply_krylov_operator =
                    [this, &A](const T_vec& source, T_vec& destination)
                    {
                        return calc_krylov_vector(A, source, destination) &&
                            apply_regularization(destination);
                    };
                const auto arnoldi_result = krylov::arnoldi_step(
                    *vec_ops_,
                    apply_krylov_operator,
                    V_,
                    restart_ + 1,
                    column,
                    y_,
                    r_,
                    h_col_.data(),
                    h_pass_.data(),
                    H_,
                    orth_options);

                if(!arnoldi_result.succeeded())
                {
                    operation_ok = false;
                    logged_obj_t::error_f(
                        "solve: Arnoldi step %u failed with status %s",
                        column,
                        krylov::arnoldi_status_name(arnoldi_result.status));
                    break;
                }

                dense_ops_->plane_rotation_col(
                    H_,
                    cs_,
                    sn_,
                    s_,
                    static_cast<typename VectorOperations::Ord>(column));
                using std::abs;
                const norm_type residual_estimate = static_cast<norm_type>(
                    abs(dense_ops_->vector_at(s_, column + 1)));
                const norm_type reduction =
                    previous_residual > norm_type{}
                        ? residual_estimate/previous_residual
                        : norm_type{};
                if(
                    total_iterations % prms_.batch_size == 0 ||
                    column + 1 == restart_)
                {
                    logged_obj_t::info_f(
                        "iter = %zu(%u), resid_estimate = %e, reduction = %.03f",
                        total_iterations,
                        column + 1,
                        static_cast<double>(monitor_.norm_out(residual_estimate)),
                        static_cast<double>(reduction));
                }
                previous_residual = residual_estimate;

                const bool estimated_convergence =
                    monitor_.check_finished_by_ritz_estimate(
                        residual_estimate);
                if(
                    estimated_convergence &&
                    prms_.do_restart_on_false_ritz_convergence)
                {
                    request_restart = true;
                    break;
                }

                if(estimated_convergence)
                {
                    vec_ops_->assign(x, x_tmp_);
                    try
                    {
                        dense_ops_->solve_upper_triangular_subsystem(
                            H_,
                            s_,
                            s_h_,
                            static_cast<typename VectorOperations::Ord>(
                                column + 1));
                    }
                    catch(const std::exception& error)
                    {
                        operation_ok = false;
                        logged_obj_t::error_f(
                            "solve: projected triangular solve failed: %s",
                            error.what());
                        break;
                    }

                    construct_solution(static_cast<int>(column), s_h_, y_);
                    operation_ok =
                        calc_right_precond_solution(y_) &&
                        apply_regularization(y_);
                    if(operation_ok)
                        vec_ops_->add_lin_comb(T(1), y_, T(1), x);
                    if(operation_ok)
                    {
                        operation_ok =
                            calc_left_preconditioned_residual(A, x, b, r_) &&
                            apply_regularization(r_);
                    }
                    if(!operation_ok)
                        break;

                    const bool trial_finished = monitor_.check_finished(x, r_);
                    if(trial_finished)
                    {
                        accepted_trial_update = true;
                        finished = true;
                        break;
                    }
                    vec_ops_->assign(x_tmp_, x);
                }

                if(
                    arnoldi_result.status == krylov::arnoldi_status::happy_breakdown)
                {
                    request_restart = true;
                    break;
                }
            }

            if(!operation_ok || accepted_trial_update)
                break;

            if(last_column < 0)
            {
                operation_ok = false;
                logged_obj_t::error_f("solve: no Arnoldi column was generated");
                break;
            }

            try
            {
                dense_ops_->solve_upper_triangular_subsystem(
                    H_,
                    s_,
                    s_h_,
                    static_cast<typename VectorOperations::Ord>(
                        last_column + 1));
            }
            catch(const std::exception& error)
            {
                operation_ok = false;
                logged_obj_t::error_f(
                    "solve: projected triangular solve failed: %s",
                    error.what());
                break;
            }

            construct_solution(last_column, s_h_, y_);
            operation_ok =
                calc_right_precond_solution(y_) &&
                apply_regularization(y_);
            if(operation_ok)
                vec_ops_->add_lin_comb(T(1), y_, T(1), x);
            if(operation_ok)
            {
                operation_ok =
                    calc_left_preconditioned_residual(A, x, b, r_) &&
                    apply_regularization(r_);
            }
            if(!operation_ok)
                break;

            finished = monitor_.check_finished(x, r_);
            if(request_restart && !finished)
                previous_residual = vec_ops_->norm(r_);
        }

        const bool converged = operation_ok && monitor_.converged();
        if(!operation_ok)
            logged_obj_t::error_f("solve: operator or preconditioner failed");
        else if(!converged)
            logged_obj_t::error_f("solve: linear solver failed to converge");

        stop_use_all();
        return converged;
    }

    bool solve(const vector_type &b, vector_type &x)const
    {
        if(!parent_t::A_)
            throw std::logic_error("gmres::solve called without an operator");
        return solve(*parent_t::A_, b, x);
    }
};

}
}

#endif //__NMFD_GMRES_H__
