// Copyright (C) 2016-2026 Oleg Ryabkov, Nikolay Evstigneev
// SPDX-License-Identifier: GPL-2.0-only

#ifndef __NMFD_SHORT_RECURRENCE_SOLVER_H__
#define __NMFD_SHORT_RECURRENCE_SOLVER_H__

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
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
#include "monitor_call_wrap.h"

namespace nmfd
{
namespace solvers
{
namespace detail
{

template <class Monitor, class Log>
struct short_recurrence_params : scfd::utils::logged_obj_base<Log>::params
{
    using logged_params_type                   = typename scfd::utils::logged_obj_base<Log>::params;
    bool                     use_precond_resid = true;
    bool                     use_real_resid    = false;
    int                      resid_recalc_freq = 0;
    typename Monitor::params monitor;

    short_recurrence_params( const std::string &prefix, const std::string &name )
        : logged_params_type( 0, prefix + name ), monitor( this->log_msg_prefix )
    {
    }
#ifdef NMFD_ENABLE_NLOHMANN
    void from_json( const nlohmann::json &j )
    {
        use_precond_resid = j.value( "use_precond_resid", use_precond_resid );
        use_real_resid    = j.value( "use_real_resid", use_real_resid );
        resid_recalc_freq = j.value( "resid_recalc_freq", resid_recalc_freq );
        if ( j.contains( "monitor" ) )
            monitor.from_json( j.at( "monitor" ) );
    }
    nlohmann::json to_json() const
    {
        return {
            { "use_precond_resid", use_precond_resid },
            { "use_real_resid", use_real_resid },
            { "resid_recalc_freq", resid_recalc_freq },
            { "monitor", monitor.to_json() }
        };
    }
#endif
};

template <class VectorOperations, class Monitor, class Log, class LinearOperator, class Preconditioner, class Params>
class short_recurrence_solver : public iter_solver_base<VectorOperations, Monitor, Log, LinearOperator, Preconditioner>
{
    using parent_t         = iter_solver_base<VectorOperations, Monitor, Log, LinearOperator, Preconditioner>;
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;

public:
    using scalar_type                          = typename VectorOperations::scalar_type;
    using norm_type                            = typename VectorOperations::norm_type;
    using vector_type                          = typename VectorOperations::vector_type;
    using vector_space_type                    = VectorOperations;
    using params                               = Params;
    using preconditioner_params_hierarchy_type = typename nmfd::detail::algo_params_hierarchy<Preconditioner>::type;
    using preconditioner_utils_hierarchy_type  = typename nmfd::detail::algo_utils_hierarchy<Preconditioner>::type;

    struct params_hierarchy : params
    {
        preconditioner_params_hierarchy_type preconditioner;
        params_hierarchy( const std::string &prefix = "" ) : params( prefix ), preconditioner( this->log_msg_prefix )
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
        std::shared_ptr<VectorOperations> vec_ops;
        Log                              *log = nullptr;
        utils()                               = default;
        utils( std::shared_ptr<VectorOperations> space, Log *logger = nullptr )
            : vec_ops( std::move( space ) ), log( logger )
        {
        }
        template <class Backend>
        utils( Backend &backend, std::shared_ptr<VectorOperations> space )
            : utils( std::move( space ), std::addressof( backend.log() ) )
        {
        }
    };
    struct utils_hierarchy : utils
    {
        preconditioner_utils_hierarchy_type preconditioner;
        utils_hierarchy() = default;
        utils_hierarchy(
            preconditioner_utils_hierarchy_type prec, std::shared_ptr<VectorOperations> space, Log *logger = nullptr
        )
            : utils( std::move( space ), logger ), preconditioner( std::move( prec ) )
        {
        }
        template <class Backend>
        utils_hierarchy( Backend &backend, std::shared_ptr<VectorOperations> space )
            : utils( backend, space ),
              preconditioner( nmfd::detail::make_algo_utils_hierarchy<Preconditioner>( backend, std::move( space ) ) )
        {
        }
    };

    using parent_t::solve;
    bool solve( const vector_type &rhs, vector_type &x ) const override
    {
        if ( !this->A_ )
            throw std::logic_error( "Linear solver has no bound operator" );
        return this->solve( *this->A_, rhs, x );
    }
    const std::shared_ptr<Preconditioner> &preconditioner() const
    {
        return this->prec_;
    }
    void set_use_precond_resid( bool enabled )
    {
        prms_.use_precond_resid = enabled;
    }
    void set_use_real_resid( bool enabled )
    {
        prms_.use_real_resid = enabled;
    }
    void set_resid_recalc_freq( int frequency )
    {
        if ( frequency < 0 )
            throw std::invalid_argument( "Residual recomputation frequency must be nonnegative" );
        prms_.resid_recalc_freq = frequency;
    }
    void set_preconditioner_bypass( bool enabled )
    {
        bypass_ = enabled;
    }
    bool preconditioner_bypass_enabled() const
    {
        return bypass_;
    }

protected:
    using T = scalar_type;
    using parent_t::monitor_;
    using parent_t::vec_ops_;
    params prms_;

    enum class iteration_state
    {
        running,
        converged,
        failed
    };

    short_recurrence_solver(
        std::shared_ptr<VectorOperations> space, Log *log, const params &p, std::shared_ptr<Preconditioner> prec,
        std::size_t vectors
    )
        : parent_t( checked_space( std::move( space ) ), log, p, p.monitor, std::move( prec ) ), prms_( p ),
          checked_residual_( *vec_ops_ )
    {
        if ( p.resid_recalc_freq < 0 || p.monitor.max_iters_num < 0 || p.monitor.min_iters_num < 0 ||
             p.monitor.min_iters_num > p.monitor.max_iters_num || !std::isfinite( p.monitor.rel_tol ) ||
             !std::isfinite( p.monitor.abs_tol ) || p.monitor.rel_tol < norm_type( 0 ) ||
             p.monitor.abs_tol < norm_type( 0 ) )
            throw std::invalid_argument( "Invalid short-recurrence solver parameters" );
        checked_residual_.init();
        checked_residual_.start_use();
        allocate_vectors( vectors );
    }
    short_recurrence_solver( const utils_hierarchy &u, const params_hierarchy &p, std::size_t vectors )
        : short_recurrence_solver(
              u.vec_ops, u.log, p,
              nmfd::detail::algo_hierarchy_creator<Preconditioner>::get( u.preconditioner, p.preconditioner ), vectors
          )
    {
    }

    void allocate_vectors( std::size_t count )
    {
        vectors_.reserve( count );
        while ( vectors_.size() < count )
        {
            vectors_.emplace_back( *vec_ops_ );
            vectors_.back().init();
            vectors_.back().start_use();
        }
    }
    vector_type &vector( std::size_t index ) const
    {
        return *vectors_[index];
    }

    static bool finite( T value )
    {
        return value == value && value - value == T( 0 );
    }
    static bool quotient( T numerator, T denominator, T &result )
    {
        if ( !finite( numerator ) || !finite( denominator ) || denominator == T( 0 ) )
            return false;
        result = numerator / denominator;
        return finite( result );
    }
    bool breakdown( const char *coefficient ) const
    {
        this->warning_f( "solve: breakdown or nonfinite coefficient in %s", coefficient );
        return false;
    }
    bool precondition( vector_type &v ) const
    {
        return bypass_ || !this->prec_ || krylov::apply_in_place( *this->prec_, v );
    }
    bool apply( const LinearOperator &A, const vector_type &v, vector_type &out ) const
    {
        return krylov::apply_operator( A, v, out ) && precondition( out );
    }
    bool residual( const LinearOperator &A, const vector_type &rhs, const vector_type &x, vector_type &r ) const
    {
        if ( !krylov::apply_operator( A, x, r ) )
            return false;
        vec_ops_->add_lin_comb( T( 1 ), rhs, T( -1 ), r );
        return true;
    }
    bool initialize(
        const LinearOperator &A, const vector_type &rhs, const vector_type &x, vector_type &r,
        bool precondition_residual = true
    ) const
    {
        rhs_norm_ = vec_ops_->norm( rhs );
        if ( !std::isfinite( rhs_norm_ ) || !vec_ops_->is_valid_number( x ) )
            return false;
        if ( !residual( A, rhs, x, r ) || ( precondition_residual && !precondition( r ) ) )
            return false;
        return std::isfinite( vec_ops_->norm( r ) );
    }
    bool candidate_norm(
        const LinearOperator &A, const vector_type &rhs, const vector_type &x, const vector_type &r, norm_type &norm
    ) const
    {
        const vector_type *checked = &r;
        if ( prms_.use_real_resid || !prms_.use_precond_resid )
        {
            if ( !residual( A, rhs, x, *checked_residual_ ) ||
                 ( prms_.use_precond_resid && !precondition( *checked_residual_ ) ) )
                return false;
            checked = &*checked_residual_;
        }
        norm = vec_ops_->norm( *checked );
        return std::isfinite( norm );
    }
    bool start_monitor( const vector_type &rhs, monitor_call_wrap<VectorOperations, Monitor> &scope ) const
    {
        if ( prms_.use_precond_resid )
        {
            vec_ops_->assign( rhs, *checked_residual_ );
            if ( !precondition( *checked_residual_ ) )
                return false;
            scope.start( *checked_residual_ );
        }
        else
            scope.start( rhs );
        return std::isfinite( monitor_.tol() );
    }
    iteration_state check(
        const LinearOperator &A, const vector_type &rhs, const vector_type &x, const vector_type &r,
        bool preconditioned_residual = true
    ) const
    {
        const vector_type *checked       = &r;
        norm_type          physical_norm = norm_type( -1 );
        if ( !preconditioned_residual && prms_.use_precond_resid )
        {
            vec_ops_->assign( r, *checked_residual_ );
            if ( !precondition( *checked_residual_ ) )
                return iteration_state::failed;
            checked = &*checked_residual_;
        }
        else if ( preconditioned_residual && ( prms_.use_real_resid || !prms_.use_precond_resid ) )
        {
            if ( !residual( A, rhs, x, *checked_residual_ ) )
                return iteration_state::failed;
            physical_norm = vec_ops_->norm( *checked_residual_ );
            if ( prms_.use_precond_resid && !precondition( *checked_residual_ ) )
                return iteration_state::failed;
            checked = &*checked_residual_;
        }
        const bool finished = monitor_.check_finished( x, *checked );
        if ( !monitor_.is_valid_number() || !std::isfinite( monitor_.resid_norm() ) )
            return iteration_state::failed;
        if ( !finished )
            return iteration_state::running;
        if ( !monitor_.converged() )
            return iteration_state::failed;
        if ( physical_norm < norm_type( 0 ) )
        {
            if ( !preconditioned_residual )
                physical_norm = prms_.use_precond_resid ? vec_ops_->norm( r ) : monitor_.resid_norm();
            else
            {
                if ( !residual( A, rhs, x, *checked_residual_ ) )
                    return iteration_state::failed;
                physical_norm = vec_ops_->norm( *checked_residual_ );
            }
        }
        if ( !std::isfinite( physical_norm ) )
            return iteration_state::failed;
        if ( physical_norm <= monitor_.abs_tol() + monitor_.rel_tol() * rhs_norm_ )
            return iteration_state::converged;
        // A preconditioned tolerance hit is not a failure while iterations remain.
        return monitor_.iters_performed() < monitor_.max_iters_num() ? iteration_state::running
                                                                     : iteration_state::failed;
    }
    bool finish( const LinearOperator &A, const vector_type &rhs, vector_type &x, iteration_state state ) const
    {
        if ( monitor_.out_min_resid_norm() )
        {
            if ( !residual( A, rhs, monitor_.min_resid_norm_x(), *checked_residual_ ) )
                return false;
            const norm_type norm = vec_ops_->norm( *checked_residual_ );
            if ( std::isfinite( norm ) && ( state != iteration_state::converged ||
                                            norm <= monitor_.abs_tol() + monitor_.rel_tol() * rhs_norm_ ) )
                vec_ops_->assign( monitor_.min_resid_norm_x(), x );
        }
        return state == iteration_state::converged;
    }
    bool recompute(
        const LinearOperator &A, const vector_type &rhs, const vector_type &x, vector_type &r, int completed_steps
    ) const
    {
        return prms_.resid_recalc_freq == 0 || completed_steps % prms_.resid_recalc_freq != 0 ||
               ( residual( A, rhs, x, r ) && precondition( r ) );
    }

private:
    static std::shared_ptr<VectorOperations> checked_space( std::shared_ptr<VectorOperations> space )
    {
        if ( !space )
            throw std::invalid_argument( "Linear solver requires a vector space" );
        return space;
    }
    bool                                  bypass_   = false;
    mutable norm_type                     rhs_norm_ = norm_type( 0 );
    mutable vector_wrap_type              checked_residual_;
    mutable std::vector<vector_wrap_type> vectors_;
};

}
}
}

#endif
