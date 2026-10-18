// Copyright (C) 2016-2026 Oleg Ryabkov, Nikolay Evstigneev
// SPDX-License-Identifier: GPL-2.0-only

#ifndef __NMFD_BICGSTABL_H__
#define __NMFD_BICGSTABL_H__

#include <algorithm>
#include <nmfd/preconditioners/dummy.h>
#include "detail/short_recurrence_solver.h"

namespace nmfd
{
namespace solvers
{
namespace detail
{
template <class Monitor, class Log>
struct bicgstabl_params : short_recurrence_params<Monitor, Log>
{
    using parent_t = short_recurrence_params<Monitor, Log>;
    int basis_size = 2;
    bicgstabl_params( const std::string &prefix = "", const std::string &name = "bicgstabl::" )
        : parent_t( prefix, name )
    {
    }
#ifdef NMFD_ENABLE_NLOHMANN
    void from_json( const nlohmann::json &j )
    {
        parent_t::from_json( j );
        if ( j.contains( "basis_size" ) )
        {
            const auto &value = j.at( "basis_size" );
            if ( !value.is_number_integer() || value < 1 || value > 30 )
                throw std::invalid_argument( "BiCGStab(L) basis_size must be between 1 and 30" );
            basis_size = value.template get<int>();
        }
    }
    nlohmann::json to_json() const
    {
        auto j          = parent_t::to_json();
        j["type"]       = "bicgstabl";
        j["basis_size"] = basis_size;
        return j;
    }
#endif
};
}

template <
    class VectorOperations, class Monitor, class Log, class LinearOperator,
    class Preconditioner = preconditioners::dummy<VectorOperations, LinearOperator>>
class bicgstabl
    : public detail::short_recurrence_solver<
          VectorOperations, Monitor, Log, LinearOperator, Preconditioner, detail::bicgstabl_params<Monitor, Log>>
{
    using parent_t = detail::short_recurrence_solver<
        VectorOperations, Monitor, Log, LinearOperator, Preconditioner, detail::bicgstabl_params<Monitor, Log>>;
    using T          = typename parent_t::scalar_type;
    using state_type = typename parent_t::iteration_state;
    using parent_t::monitor_;
    using parent_t::prms_;
    using parent_t::vec_ops_;

public:
    using parent_t::solve;
    using typename parent_t::params;
    using typename parent_t::params_hierarchy;
    using typename parent_t::utils_hierarchy;
    using typename parent_t::vector_type;

    bicgstabl(
        std::shared_ptr<VectorOperations> space, Log *log = nullptr, const params &p = params(),
        std::shared_ptr<Preconditioner> prec = nullptr
    )
        : parent_t( std::move( space ), log, p, std::move( prec ), checked_vector_count( p.basis_size ) ),
          coefficients_( coefficient_count( p.basis_size ) )
    {
    }
    bicgstabl(
        std::shared_ptr<const LinearOperator> A, std::shared_ptr<VectorOperations> space, Log *log = nullptr,
        const params &p = params(), std::shared_ptr<Preconditioner> prec = nullptr
    )
        : bicgstabl( std::move( space ), log, p, std::move( prec ) )
    {
        this->set_operator( std::move( A ) );
    }
    bicgstabl( const utils_hierarchy &u, const params_hierarchy &p = params_hierarchy() )
        : parent_t( u, p, checked_vector_count( p.basis_size ) ), coefficients_( coefficient_count( p.basis_size ) )
    {
    }

    void set_basis_size( int size )
    {
        const auto     count = checked_vector_count( size );
        std::vector<T> coefficients( coefficient_count( size ) );
        this->allocate_vectors( count );
        coefficients_.swap( coefficients );
        prms_.basis_size = size;
    }

    bool solve( const LinearOperator &A, const vector_type &rhs, vector_type &x ) const override
    {
        auto                                                &shadow = this->vector( 0 );
        detail::monitor_call_wrap<VectorOperations, Monitor> scope( monitor_ );
        if ( !this->start_monitor( rhs, scope ) || !this->initialize( A, rhs, x, r( 0 ) ) )
            return false;
        auto state = this->check( A, rhs, x, r( 0 ) );
        if ( state != state_type::running )
            return this->finish( A, rhs, x, state );
        const auto shadow_norm = vec_ops_->norm( r( 0 ) );
        if ( !std::isfinite( shadow_norm ) || shadow_norm == 0 )
            return this->breakdown( "shadow residual" );
        vec_ops_->assign_lin_comb( T( 1 ) / shadow_norm, r( 0 ), shadow );
        vec_ops_->assign_scalar( T( 0 ), u( 0 ) );
        T   rho( 1 ), alpha( 0 ), omega( 1 );
        int completed_cycles = 0;
        while ( state == state_type::running )
        {
            const int degree = std::min( prms_.basis_size, monitor_.max_iters_num() - monitor_.iters_performed() );
            rho              = -omega * rho;
            for ( int j = 0; j < degree; ++j )
            {
                const T rho_next = vec_ops_->scalar_prod( shadow, r( j ) );
                T       ratio;
                if ( !this->quotient( rho_next, rho, ratio ) || !this->finite( alpha * ratio ) )
                    return this->breakdown( "beta" );
                const T beta = alpha * ratio;
                rho          = rho_next;
                for ( int i = 0; i <= j; ++i )
                    vec_ops_->add_lin_comb( T( 1 ), r( i ), -beta, u( i ) );
                if ( !this->apply( A, u( j ), u( j + 1 ) ) )
                    return false;
                if ( !this->quotient( rho, vec_ops_->scalar_prod( shadow, u( j + 1 ) ), alpha ) )
                    return this->breakdown( "alpha" );
                for ( int i = 0; i <= j; ++i )
                    vec_ops_->add_lin_comb( -alpha, u( i + 1 ), T( 1 ), r( i ) );
                vec_ops_->add_lin_comb( alpha, u( 0 ), T( 1 ), x );
                ++monitor_;
                typename parent_t::norm_type half_step_norm;
                if ( !this->candidate_norm( A, rhs, x, r( 0 ), half_step_norm ) )
                    return false;
                if ( half_step_norm <= monitor_.tol() )
                {
                    state = this->check( A, rhs, x, r( 0 ) );
                    return this->finish( A, rhs, x, state );
                }
                if ( !this->apply( A, r( j ), r( j + 1 ) ) )
                    return false;
            }
            // Minimum-residual polynomial, following the existing BiCGStab(L) recurrence.
            for ( int j = 1; j <= degree; ++j )
            {
                for ( int i = 1; i < j; ++i )
                {
                    if ( !this->quotient( vec_ops_->scalar_prod( r( i ), r( j ) ), sigma( i ), tau( i, j ) ) )
                        return this->breakdown( "tau" );
                    vec_ops_->add_lin_comb( -tau( i, j ), r( i ), T( 1 ), r( j ) );
                }
                sigma( j ) = vec_ops_->scalar_prod( r( j ), r( j ) );
                if ( !this->quotient( vec_ops_->scalar_prod( r( j ), r( 0 ) ), sigma( j ), gamma_p( j ) ) )
                    return this->breakdown( "sigma" );
            }
            gamma( degree ) = gamma_p( degree );
            omega           = gamma( degree );
            for ( int j = degree - 1; j >= 1; --j )
            {
                gamma( j ) = gamma_p( j );
                for ( int i = j + 1; i <= degree; ++i )
                    gamma( j ) -= tau( j, i ) * gamma( i );
            }
            for ( int j = 1; j < degree; ++j )
            {
                gamma_pp( j ) = gamma( j + 1 );
                for ( int i = j + 1; i < degree; ++i )
                    gamma_pp( j ) += tau( j, i ) * gamma( i + 1 );
            }
            vec_ops_->add_lin_comb( gamma( 1 ), r( 0 ), T( 1 ), x );
            vec_ops_->add_lin_comb( -gamma_p( degree ), r( degree ), T( 1 ), r( 0 ) );
            vec_ops_->add_lin_comb( -gamma( degree ), u( degree ), T( 1 ), u( 0 ) );
            for ( int j = 1; j < degree; ++j )
            {
                vec_ops_->add_lin_comb( gamma_pp( j ), r( j ), T( 1 ), x );
                vec_ops_->add_lin_comb( -gamma_p( j ), r( j ), T( 1 ), r( 0 ) );
                vec_ops_->add_lin_comb( -gamma( j ), u( j ), T( 1 ), u( 0 ) );
            }
            if ( !this->recompute( A, rhs, x, r( 0 ), ++completed_cycles ) )
                return false;
            state = this->check( A, rhs, x, r( 0 ) );
        }
        return this->finish( A, rhs, x, state );
    }

private:
    mutable std::vector<T> coefficients_;
    static std::size_t     checked_vector_count( int size )
    {
        if ( size < 1 || size > 30 )
            throw std::invalid_argument( "BiCGStab(L) basis_size must be between 1 and 30" );
        return static_cast<std::size_t>( 2 * ( size + 1 ) + 1 );
    }
    static std::size_t coefficient_count( int size )
    {
        return static_cast<std::size_t>( ( size + 1 ) * ( size + 5 ) );
    }
    vector_type &r( int i ) const
    {
        return this->vector( 1 + static_cast<std::size_t>( i ) );
    }
    vector_type &u( int i ) const
    {
        return this->vector( 2 + static_cast<std::size_t>( prms_.basis_size + i ) );
    }
    T &tau( int i, int j ) const
    {
        return coefficients_[static_cast<std::size_t>( i * ( prms_.basis_size + 1 ) + j )];
    }
    T &coefficient( int block, int i ) const
    {
        const int width = prms_.basis_size + 1;
        return coefficients_[static_cast<std::size_t>( width * width + block * width + i )];
    }
    T &sigma( int i ) const
    {
        return coefficient( 0, i );
    }
    T &gamma( int i ) const
    {
        return coefficient( 1, i );
    }
    T &gamma_p( int i ) const
    {
        return coefficient( 2, i );
    }
    T &gamma_pp( int i ) const
    {
        return coefficient( 3, i );
    }
};

}
}

#endif
