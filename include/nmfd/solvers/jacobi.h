// Copyright (C) 2020-2026 Oleg Ryabkov, Nikolay Evstigneev
// SPDX-License-Identifier: GPL-2.0-only

#ifndef __NMFD_SOLVERS_JACOBI_H__
#define __NMFD_SOLVERS_JACOBI_H__

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#ifdef NMFD_ENABLE_NLOHMANN
#    include <nlohmann/json.hpp>
#endif
#include <nmfd/detail/algo_hierarchy_creator.h>
#include <nmfd/detail/algo_params_hierarchy.h>
#include <nmfd/detail/algo_utils_hierarchy.h>
#include <nmfd/detail/algo_utils_hierarchy_factory.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/iter_solver_base.h>
#include <nmfd/solvers/krylov/operator_apply.h>
#include "detail/monitor_call_wrap.h"

namespace nmfd
{
namespace solvers
{

// Preserve the template order used by the existing Newton and CH clients.
template <class VectorOp, class LaplaceOp, class Precond, class Monitor, class Log>
class jacobi : public iter_solver_base<VectorOp, Monitor, Log, LaplaceOp, Precond>
{
    using parent_t            = iter_solver_base<VectorOp, Monitor, Log, LaplaceOp, Precond>;
    using logged_obj_params_t = typename parent_t::logged_obj_params_t;
    using vector_wrap_type    = nmfd::detail::vector_wrap<VectorOp, false, false>;
    using T                   = typename VectorOp::scalar_type;

public:
    using scalar_type           = typename VectorOp::scalar_type;
    using norm_type             = typename VectorOp::norm_type;
    using vector_type           = typename VectorOp::vector_type;
    using vector_space_type     = VectorOp;
    using vector_operation_type = VectorOp;
    using laplace_operator_type = LaplaceOp;
    using vector_operation_ptr  = std::shared_ptr<VectorOp>;
    using laplace_operator_ptr  = std::shared_ptr<LaplaceOp>;
    using preconditioner_ptr    = std::shared_ptr<Precond>;

    struct params : logged_obj_params_t
    {
        typename Monitor::params monitor;
        params( const std::string &prefix = "", const std::string &name = "jacobi::" )
            : logged_obj_params_t( 0, prefix + name ), monitor( this->log_msg_prefix )
        {
        }
#ifdef NMFD_ENABLE_NLOHMANN
        void from_json( const nlohmann::json &j )
        {
            if ( j.contains( "monitor" ) )
                monitor.from_json( j.at( "monitor" ) );
        }
        nlohmann::json to_json() const
        {
            return { { "type", "jacobi" }, { "monitor", monitor.to_json() } };
        }
#endif
    };
    using preconditioner_params_hierarchy_type = typename nmfd::detail::algo_params_hierarchy<Precond>::type;
    using preconditioner_utils_hierarchy_type  = typename nmfd::detail::algo_utils_hierarchy<Precond>::type;

    struct params_hierarchy : params
    {
        preconditioner_params_hierarchy_type preconditioner;
        params_hierarchy( const std::string &prefix = "", const std::string &name = "jacobi::" )
            : params( prefix, name ), preconditioner( this->log_msg_prefix )
        {
        }
        params_hierarchy( const params &p, const preconditioner_params_hierarchy_type &prec )
            : params( p ), preconditioner( prec )
        {
        }
#ifdef NMFD_ENABLE_NLOHMANN
        void from_json( const nlohmann::json &j )
        {
            params::from_json( j );
            if ( j.contains( "preconditioner" ) )
                preconditioner.from_json( j.at( "preconditioner" ) );
        }
        nlohmann::json to_json() const
        {
            auto j              = params::to_json();
            j["preconditioner"] = preconditioner.to_json();
            return j;
        }
#endif
    };
    struct utils
    {
        vector_operation_ptr vec_ops;
        Log                 *log = nullptr;
        utils()                  = default;
        utils( vector_operation_ptr space, Log *logger = nullptr ) : vec_ops( std::move( space ) ), log( logger )
        {
        }
        template <class Backend>
        utils( Backend &backend, vector_operation_ptr space )
            : utils( std::move( space ), std::addressof( backend.log() ) )
        {
        }
    };
    struct utils_hierarchy : utils
    {
        preconditioner_utils_hierarchy_type preconditioner;
        utils_hierarchy() = default;
        utils_hierarchy( preconditioner_utils_hierarchy_type prec, vector_operation_ptr space, Log *logger = nullptr )
            : utils( std::move( space ), logger ), preconditioner( std::move( prec ) )
        {
        }
        template <class Backend>
        utils_hierarchy( Backend &backend, vector_operation_ptr space )
            : utils( backend, space ),
              preconditioner( nmfd::detail::make_algo_utils_hierarchy<Precond>( backend, std::move( space ) ) )
        {
        }
    };

    jacobi(
        vector_operation_ptr space, Log *log = nullptr, const params &p = params(), preconditioner_ptr prec = nullptr
    )
        : parent_t( checked_space( std::move( space ), p ), log, p, p.monitor, std::move( prec ) ),
          residual_( *this->vec_ops_ )
    {
        residual_.init();
        residual_.start_use();
    }
    jacobi(
        std::shared_ptr<const LaplaceOp> A, vector_operation_ptr space, Log *log = nullptr, const params &p = params(),
        preconditioner_ptr prec = nullptr
    )
        : jacobi( std::move( space ), log, p, std::move( prec ) )
    {
        this->set_operator( std::move( A ) );
    }
    jacobi( const utils_hierarchy &u, const params_hierarchy &p = params_hierarchy() )
        : jacobi(
              u.vec_ops, u.log, p,
              nmfd::detail::algo_hierarchy_creator<Precond>::get( u.preconditioner, p.preconditioner )
          )
    {
    }

    bool solve( const vector_type &rhs, vector_type &x ) const override
    {
        if ( !this->A_ )
            throw std::logic_error( "Jacobi solver has no bound operator" );
        return solve( *this->A_, rhs, x );
    }
    bool solve( const LaplaceOp &A, const vector_type &rhs, vector_type &x ) const override
    {
        if ( !std::isfinite( this->vec_ops_->norm( rhs ) ) || !calculate_residual( A, rhs, x ) )
            return false;
        detail::monitor_call_wrap<VectorOp, Monitor> scope( this->monitor_ );
        scope.start( rhs );
        if ( !std::isfinite( this->monitor_.tol() ) )
            return false;
        while ( true )
        {
            const bool finished = this->monitor_.check_finished( x, *residual_ );
            if ( !this->monitor_.is_valid_number() || !std::isfinite( this->monitor_.resid_norm() ) )
                return false;
            if ( finished )
            {
                const bool converged = this->monitor_.converged();
                if ( this->monitor_.out_min_resid_norm() )
                    this->vec_ops_->assign( this->monitor_.min_resid_norm_x(), x );
                return converged;
            }
            if ( this->prec_ && !krylov::apply_in_place( *this->prec_, *residual_ ) )
                return false;
            if ( !this->vec_ops_->is_valid_number( *residual_ ) )
                return false;
            this->vec_ops_->add_lin_comb( T( -1 ), *residual_, T( 1 ), x );
            ++this->monitor_;
            if ( !calculate_residual( A, rhs, x ) )
                return false;
        }
    }

private:
    mutable vector_wrap_type residual_;

    static vector_operation_ptr checked_space( vector_operation_ptr space, const params &p )
    {
        if ( !space || !std::isfinite( p.monitor.rel_tol ) || !std::isfinite( p.monitor.abs_tol ) ||
             p.monitor.rel_tol < 0 || p.monitor.abs_tol < 0 || p.monitor.min_iters_num < 0 ||
             p.monitor.max_iters_num < p.monitor.min_iters_num )
            throw std::invalid_argument( "Invalid Jacobi solver parameters" );
        return space;
    }
    bool calculate_residual( const LaplaceOp &A, const vector_type &rhs, const vector_type &x ) const
    {
        if ( !this->vec_ops_->is_valid_number( x ) || !krylov::apply_operator( A, x, *residual_ ) )
            return false;
        this->vec_ops_->add_lin_comb( T( -1 ), rhs, T( 1 ), *residual_ );
        return true;
    }
};

}
}

#endif
