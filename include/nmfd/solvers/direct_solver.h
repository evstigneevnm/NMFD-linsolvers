// Copyright (C) 2026 Oleg Ryabkov, Nikolay Evstigneev
// SPDX-License-Identifier: GPL-2.0-only

#ifndef __NMFD_SOLVERS_DIRECT_SOLVER_H__
#define __NMFD_SOLVERS_DIRECT_SOLVER_H__

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/detail/monitor_call_wrap.h>
#include <nmfd/solvers/iter_solver_base.h>
#include <nmfd/solvers/krylov/operator_apply.h>

namespace nmfd
{
namespace solvers
{

// The supplied preconditioner is an inverse, not an iterative approximation.
template <class VectorOperations, class Monitor, class Log, class LinearOperator, class Inverse>
class direct_solver : public iter_solver_base<VectorOperations, Monitor, Log, LinearOperator, Inverse>
{
    using parent_t         = iter_solver_base<VectorOperations, Monitor, Log, LinearOperator, Inverse>;
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;

public:
    using scalar_type = typename VectorOperations::scalar_type;
    using norm_type   = typename VectorOperations::norm_type;
    using vector_type = typename VectorOperations::vector_type;

    struct params : parent_t::logged_obj_params_t
    {
        typename Monitor::params monitor;
        params( const std::string &prefix = "", const std::string &name = "direct::" )
            : parent_t::logged_obj_params_t( 0, prefix + name ), monitor( this->log_msg_prefix )
        {
        }
    };

    enum class result_status
    {
        idle,
        converged,
        invalid_input,
        inverse_failed,
        operator_failed,
        residual_too_large
    };

    direct_solver(
        std::shared_ptr<VectorOperations> operations, Log *log = nullptr, const params &p = params(),
        std::shared_ptr<Inverse> inverse = nullptr
    )
        : parent_t( checked_space( std::move( operations ), p ), log, p, p.monitor, std::move( inverse ) ),
          candidate_( *this->vec_ops_ ), residual_( *this->vec_ops_ )
    {
        candidate_.init();
        candidate_.start_use();
        residual_.init();
        residual_.start_use();
    }

    result_status status() const
    {
        return status_;
    }

    // There is no iterative fallback when the only inverse is bypassed.
    void set_preconditioner_bypass( bool enabled )
    {
        bypass_ = enabled;
    }
    bool preconditioner_bypass_enabled() const
    {
        return bypass_;
    }

    bool solve( const vector_type &b, vector_type &x ) const override
    {
        if ( !this->A_ )
            throw std::logic_error( "Direct solver has no bound operator" );
        return solve( *this->A_, b, x );
    }

    bool solve( const LinearOperator &A, const vector_type &b, vector_type &x ) const override
    {
        if ( this->A_.get() != &A || !this->prec_ )
            throw std::logic_error( "Direct solver requires a bound operator and inverse" );
        status_ = result_status::invalid_input;
        detail::monitor_call_wrap<VectorOperations, Monitor> scope( this->monitor_ );
        scope.start( b );
        if ( !this->vec_ops_->is_valid_number( b ) || !std::isfinite( this->monitor_.tol() ) )
            return false;
        this->vec_ops_->assign( b, *candidate_ );
        status_ = result_status::inverse_failed;
        if ( bypass_ || !krylov::apply_in_place( *this->prec_, *candidate_ ) ||
             !this->vec_ops_->is_valid_number( *candidate_ ) )
            return false;
        status_ = result_status::operator_failed;
        if ( !krylov::apply_operator( A, *candidate_, *residual_ ) )
            return false;
        this->vec_ops_->add_lin_comb( scalar_type( -1 ), b, scalar_type( 1 ), *residual_ );
        ++this->monitor_;
        this->monitor_.check_finished( *candidate_, *residual_ );
        status_ = result_status::residual_too_large;
        if ( !this->monitor_.is_valid_number() || !std::isfinite( this->monitor_.resid_norm() ) ||
             !this->monitor_.converged() )
            return false;
        this->vec_ops_->assign( *candidate_, x );
        status_ = result_status::converged;
        return true;
    }

private:
    static std::shared_ptr<VectorOperations> checked_space( std::shared_ptr<VectorOperations> space, const params &p )
    {
        if ( !space || !std::isfinite( p.monitor.rel_tol ) || !std::isfinite( p.monitor.abs_tol ) ||
             p.monitor.rel_tol < norm_type( 0 ) || p.monitor.abs_tol < norm_type( 0 ) || p.monitor.min_iters_num < 0 ||
             p.monitor.max_iters_num < p.monitor.min_iters_num )
            throw std::invalid_argument( "Invalid direct solver parameters" );
        return space;
    }

    mutable vector_wrap_type candidate_;
    mutable vector_wrap_type residual_;
    mutable result_status    status_ = result_status::idle;
    bool                     bypass_ = false;
};

} // namespace solvers
} // namespace nmfd

#endif
