// Copyright (C) 2016-2026 Oleg Ryabkov, Nikolay Evstigneev
// SPDX-License-Identifier: GPL-2.0-only

#ifndef __NMFD_BICGSTAB_H__
#define __NMFD_BICGSTAB_H__

#include <nmfd/preconditioners/dummy.h>
#include "detail/short_recurrence_solver.h"

namespace nmfd
{
namespace solvers
{
namespace detail
{
template <class Monitor, class Log>
struct bicgstab_params : short_recurrence_params<Monitor, Log>
{
    using parent_t = short_recurrence_params<Monitor, Log>;
    bicgstab_params( const std::string &prefix = "", const std::string &name = "bicgstab::" ) : parent_t( prefix, name )
    {
    }
#ifdef NMFD_ENABLE_NLOHMANN
    nlohmann::json to_json() const
    {
        auto j    = parent_t::to_json();
        j["type"] = "bicgstab";
        return j;
    }
#endif
};
}

template <
    class VectorOperations, class Monitor, class Log, class LinearOperator,
    class Preconditioner = preconditioners::dummy<VectorOperations, LinearOperator>>
class bicgstab
    : public detail::short_recurrence_solver<
          VectorOperations, Monitor, Log, LinearOperator, Preconditioner, detail::bicgstab_params<Monitor, Log>>
{
    using parent_t = detail::short_recurrence_solver<
        VectorOperations, Monitor, Log, LinearOperator, Preconditioner, detail::bicgstab_params<Monitor, Log>>;
    using T          = typename parent_t::scalar_type;
    using state_type = typename parent_t::iteration_state;
    using parent_t::monitor_;
    using parent_t::vec_ops_;

public:
    using parent_t::solve;
    using typename parent_t::params;
    using typename parent_t::params_hierarchy;
    using typename parent_t::utils_hierarchy;
    using typename parent_t::vector_type;

    bicgstab(
        std::shared_ptr<VectorOperations> space, Log *log = nullptr, const params &p = params(),
        std::shared_ptr<Preconditioner> prec = nullptr
    )
        : parent_t( std::move( space ), log, p, std::move( prec ), 6 )
    {
    }
    bicgstab(
        std::shared_ptr<const LinearOperator> A, std::shared_ptr<VectorOperations> space, Log *log = nullptr,
        const params &p = params(), std::shared_ptr<Preconditioner> prec = nullptr
    )
        : bicgstab( std::move( space ), log, p, std::move( prec ) )
    {
        this->set_operator( std::move( A ) );
    }
    bicgstab( const utils_hierarchy &u, const params_hierarchy &p = params_hierarchy() ) : parent_t( u, p, 6 )
    {
    }

    bool solve( const LinearOperator &A, const vector_type &rhs, vector_type &x ) const override
    {
        auto                                                &r      = this->vector( 0 );
        auto                                                &shadow = this->vector( 1 );
        auto                                                &p      = this->vector( 2 );
        auto                                                &s      = this->vector( 3 );
        auto                                                &t      = this->vector( 4 );
        auto                                                &v      = this->vector( 5 );
        detail::monitor_call_wrap<VectorOperations, Monitor> scope( monitor_ );
        if ( !this->start_monitor( rhs, scope ) || !this->initialize( A, rhs, x, r ) )
            return false;
        auto state = this->check( A, rhs, x, r );
        vec_ops_->assign( r, shadow );
        vec_ops_->assign_scalar( T( 0 ), p );
        vec_ops_->assign_scalar( T( 0 ), v );
        T rho_previous( 1 ), alpha( 1 ), omega( 1 );
        while ( state == state_type::running )
        {
            const T rho = vec_ops_->scalar_prod( shadow, r );
            T       ratio, alpha_over_omega;
            if ( !this->quotient( rho, rho_previous, ratio ) || !this->quotient( alpha, omega, alpha_over_omega ) ||
                 !this->finite( ratio * alpha_over_omega ) )
                return this->breakdown( "beta" );
            const T beta = ratio * alpha_over_omega;
            vec_ops_->add_lin_comb( T( 1 ), r, -beta * omega, v, beta, p );
            if ( !this->apply( A, p, v ) )
                return false;
            if ( !this->quotient( rho, vec_ops_->scalar_prod( shadow, v ), alpha ) )
                return this->breakdown( "alpha" );
            vec_ops_->assign_lin_comb( T( 1 ), r, -alpha, v, s );
            vec_ops_->add_lin_comb( alpha, p, T( 1 ), x );
            // The BiCG half-step can solve the system before the omega division.
            typename parent_t::norm_type half_step_norm;
            if ( !this->candidate_norm( A, rhs, x, s, half_step_norm ) )
                return false;
            bool counted_half_step = false;
            if ( half_step_norm <= monitor_.tol() )
            {
                ++monitor_;
                counted_half_step = true;
                state             = this->check( A, rhs, x, s );
                if ( state != state_type::running )
                    break;
            }
            if ( !this->apply( A, s, t ) )
                return false;
            if ( !this->quotient( vec_ops_->scalar_prod( t, s ), vec_ops_->scalar_prod( t, t ), omega ) )
                return this->breakdown( "omega" );
            vec_ops_->add_lin_comb( omega, s, T( 1 ), x );
            vec_ops_->assign_lin_comb( T( 1 ), s, -omega, t, r );
            if ( !counted_half_step )
                ++monitor_;
            if ( !this->recompute( A, rhs, x, r, monitor_.iters_performed() ) )
                return false;
            state        = this->check( A, rhs, x, r );
            rho_previous = rho;
        }
        return this->finish( A, rhs, x, state );
    }
};

}
}

#endif
