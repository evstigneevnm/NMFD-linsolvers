// Copyright (C) 2016-2026 Oleg Ryabkov, Nikolay Evstigneev
// SPDX-License-Identifier: GPL-2.0-only

#ifndef __NMFD_CGS_H__
#define __NMFD_CGS_H__

#include <nmfd/preconditioners/dummy.h>
#include "detail/short_recurrence_solver.h"

namespace nmfd
{
namespace solvers
{
namespace detail
{
template <class Monitor, class Log>
struct cgs_params : short_recurrence_params<Monitor, Log>
{
    using parent_t = short_recurrence_params<Monitor, Log>;
    cgs_params( const std::string &prefix = "", const std::string &name = "cgs::" ) : parent_t( prefix, name )
    {
        this->use_precond_resid = false;
    }
#ifdef NMFD_ENABLE_NLOHMANN
    nlohmann::json to_json() const
    {
        auto j    = parent_t::to_json();
        j["type"] = "cgs";
        return j;
    }
#endif
};
}

template <
    class VectorOperations, class Monitor, class Log, class LinearOperator,
    class Preconditioner = preconditioners::dummy<VectorOperations, LinearOperator>>
class cgs : public detail::short_recurrence_solver<
                VectorOperations, Monitor, Log, LinearOperator, Preconditioner, detail::cgs_params<Monitor, Log>>
{
    using parent_t = detail::short_recurrence_solver<
        VectorOperations, Monitor, Log, LinearOperator, Preconditioner, detail::cgs_params<Monitor, Log>>;
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

    cgs( std::shared_ptr<VectorOperations> space, Log *log = nullptr, const params &p = params(),
         std::shared_ptr<Preconditioner> prec = nullptr )
        : parent_t( std::move( space ), log, p, std::move( prec ), 6 )
    {
    }
    cgs( std::shared_ptr<const LinearOperator> A, std::shared_ptr<VectorOperations> space, Log *log = nullptr,
         const params &p = params(), std::shared_ptr<Preconditioner> prec = nullptr )
        : cgs( std::move( space ), log, p, std::move( prec ) )
    {
        this->set_operator( std::move( A ) );
    }
    cgs( const utils_hierarchy &u, const params_hierarchy &p = params_hierarchy() ) : parent_t( u, p, 6 )
    {
    }

    bool solve( const LinearOperator &A, const vector_type &rhs, vector_type &x ) const override
    {
        auto                                                &r      = this->vector( 0 );
        auto                                                &shadow = this->vector( 1 );
        auto                                                &p      = this->vector( 2 );
        auto                                                &u      = this->vector( 3 );
        auto                                                &q      = this->vector( 4 );
        auto                                                &v      = this->vector( 5 );
        detail::monitor_call_wrap<VectorOperations, Monitor> scope( monitor_ );
        if ( !this->start_monitor( rhs, scope ) || !this->initialize( A, rhs, x, r, false ) )
            return false;
        auto state = this->check( A, rhs, x, r, false );
        T    rho_previous( 1 );
        while ( state == state_type::running )
        {
            if ( !this->precondition( r ) )
                return false;
            if ( monitor_.iters_performed() == 0 )
                vec_ops_->assign( r, shadow );
            const T rho = vec_ops_->scalar_prod( shadow, r );
            if ( monitor_.iters_performed() == 0 )
            {
                vec_ops_->assign( r, u );
                vec_ops_->assign( u, p );
            }
            else
            {
                T beta;
                if ( !this->quotient( rho, rho_previous, beta ) || !this->finite( beta * beta ) )
                    return this->breakdown( "beta" );
                vec_ops_->assign_lin_comb( T( 1 ), r, beta, q, u );
                vec_ops_->add_lin_comb( T( 1 ), u, beta, q, beta * beta, p );
            }
            if ( !this->apply( A, p, v ) )
                return false;
            T alpha;
            if ( !this->quotient( rho, vec_ops_->scalar_prod( shadow, v ), alpha ) )
                return this->breakdown( "alpha" );
            vec_ops_->assign_lin_comb( T( 1 ), u, -alpha, v, q );
            vec_ops_->assign_lin_comb( T( 1 ), u, T( 1 ), q, v );
            vec_ops_->add_lin_comb( alpha, v, T( 1 ), x );
            ++monitor_;
            // Preserve the production CGS variant's explicit residual recomputation.
            if ( !this->residual( A, rhs, x, r ) )
                return false;
            state        = this->check( A, rhs, x, r, false );
            rho_previous = rho;
        }
        return this->finish( A, rhs, x, state );
    }
};

}
}

#endif
