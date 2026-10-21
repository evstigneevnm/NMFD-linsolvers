// Copyright (C) 2016-2026 Oleg Ryabkov, Nikolay Evstigneev
// SPDX-License-Identifier: GPL-2.0-only

#ifndef __NMFD_SHERMAN_MORRISON_H__
#define __NMFD_SHERMAN_MORRISON_H__

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <limits>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/backend/math.h>
#include <nmfd/solvers/krylov/operator_apply.h>

namespace nmfd
{
namespace solvers
{

// Existing scalar-border elimination, not a solver for a zero elimination pivot.
template <
    class VectorOperations, class Monitor, class Log, class LinearOperator, class Preconditioner,
    template <class, class, class, class, class> class LinearSolver>
class sherman_morrison
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    static_assert(
        !nmfd::operations::backend::scalar_traits<void, scalar_type>::is_complex &&
            !std::numeric_limits<scalar_type>::is_integer,
        "Scalar-bordered elimination requires an ordered real field scalar"
    );

private:
    using T                = scalar_type;
    using math_type        = nmfd::operations::backend::math<void, T>;
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;

    struct reduced_operator
    {
        std::shared_ptr<VectorOperations>     operations;
        std::shared_ptr<const LinearOperator> original;
        const vector_type                    *c    = nullptr;
        const vector_type                    *d    = nullptr;
        T                                     mass = T( 1 );
        T                                     rank = T( -1 );

        bool apply( const vector_type &x, vector_type &out ) const
        {
            if ( !krylov::apply_operator( *original, x, out ) )
                return false;
            operations->add_lin_comb( rank * operations->scalar_prod( *c, x ), *d, mass, out );
            return true;
        }
    };

    struct reduced_preconditioner
    {
        std::shared_ptr<VectorOperations>       operations;
        std::shared_ptr<Preconditioner>         original;
        std::shared_ptr<const reduced_operator> op;
        mutable vector_wrap_type                work;

        reduced_preconditioner( std::shared_ptr<VectorOperations> space, std::shared_ptr<Preconditioner> prec )
            : operations( std::move( space ) ), original( std::move( prec ) ), work( *operations )
        {
            work.init();
            work.start_use();
        }
        void set_operator( std::shared_ptr<const reduced_operator> operation )
        {
            op = std::move( operation );
        }
        bool apply( vector_type &x ) const
        {
            if ( !original )
                return true;
            operations->assign( *op->d, *work );
            if ( !krylov::apply_in_place( *original, x ) || !krylov::apply_in_place( *original, *work ) )
                return false;
            const T denominator = op->mass + op->rank * operations->scalar_prod( *op->c, *work );
            if ( !math_type::is_finite( denominator ) || denominator == T( 0 ) || op->mass == T( 0 ) )
                return false;
            const T correction = -op->rank * operations->scalar_prod( *op->c, x ) / denominator;
            operations->add_lin_comb( correction / op->mass, *work, T( 1 ) / op->mass, x );
            return operations->is_valid_number( x );
        }
    };

public:
    using linear_solver_type = LinearSolver<VectorOperations, Monitor, Log, reduced_operator, reduced_preconditioner>;
    using linear_solver_original_type = LinearSolver<VectorOperations, Monitor, Log, LinearOperator, Preconditioner>;

    sherman_morrison(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<const LinearOperator> op,
        std::shared_ptr<Preconditioner> prec, Log *log = nullptr
    )
        : operations_(
              operations ? std::move( operations )
                         : throw std::invalid_argument( "Sherman-Morrison: null vector space" )
          ),
          operator_( std::make_shared<reduced_operator>() ),
          preconditioner_( std::make_shared<reduced_preconditioner>( operations_, prec ) ),
          reduced_( operations_, log, typename linear_solver_type::params(), preconditioner_ ),
          original_( operations_, log, typename linear_solver_original_type::params(), std::move( prec ) ),
          rhs_( *operations_ )
    {
        if ( !op )
            throw std::invalid_argument( "Sherman-Morrison: null operator" );
        operator_->operations = operations_;
        original_.set_operator( op );
        operator_->original = std::move( op );
        reduced_.set_operator( operator_ );
        rhs_.init();
        rhs_.start_use();
    }
    linear_solver_type *get_linsolver_handle()
    {
        return &reduced_;
    }
    linear_solver_original_type *get_linsolver_handle_original()
    {
        return &original_;
    }
    void is_small_alpha( bool enabled )
    {
        use_small_alpha_ = enabled;
    }
    void set_preconditioner_bypass( bool enabled )
    {
        reduced_.set_preconditioner_bypass( enabled );
        original_.set_preconditioner_bypass( enabled );
    }
    bool preconditioner_bypass_enabled() const
    {
        return reduced_.preconditioner_bypass_enabled();
    }

    bool solve( const LinearOperator &A, const vector_type &b, vector_type &u ) const
    {
        check_operator( A );
        return original_.solve( A, b, u );
    }

    // (gamma A - d c^T / alpha) u = b.
    bool solve(
        const T &gamma, const LinearOperator &A, const T &alpha, const vector_type &c, const vector_type &d,
        const vector_type &b, vector_type &u
    ) const
    {
        if ( !configure( A, gamma, alpha, c, d ) )
            return false;
        if ( scaled( alpha ) )
        {
            operations_->assign_lin_comb( alpha, b, *rhs_ );
            if ( !reduced_.solve( *operator_, *rhs_, u ) )
                return false;
        }
        else if ( !reduced_.solve( *operator_, b, u ) )
            return false;
        if ( !krylov::apply_operator( A, u, *rhs_ ) )
            return false;
        operations_->add_lin_comb( -operations_->scalar_prod( c, u ) / alpha, d, gamma, *rhs_ );
        operations_->add_lin_comb( T( -1 ), b, T( 1 ), *rhs_ );
        return operations_->norm( *rhs_ ) <=
               reduced_.monitor().abs_tol() + reduced_.monitor().rel_tol() * operations_->norm( b );
    }

    // [A d; c^T alpha] [u; v] = [b; beta].
    bool solve(
        const LinearOperator &A, const vector_type &c, const vector_type &d, const T &alpha, const vector_type &b,
        const T &beta, vector_type &u, T &v
    ) const
    {
        if ( !configure( A, T( 1 ), alpha, c, d ) || !math_type::is_finite( beta ) )
            return false;
        const T tolerance = reduced_.monitor().abs_tol() +
                            reduced_.monitor().rel_tol() * math_type::hypot( operations_->norm( b ), beta );
        if ( !math_type::is_finite( tolerance ) )
            return false;
        if ( scaled( alpha ) )
            operations_->assign_lin_comb( alpha, b, -beta, d, *rhs_ );
        else
            operations_->assign_lin_comb( T( 1 ), b, -beta / alpha, d, *rhs_ );
        if ( !solve_border_reduced( alpha, tolerance, u ) )
            return false;
        const T candidate = ( beta - operations_->scalar_prod( c, u ) ) / alpha;
        if ( !math_type::is_finite( candidate ) )
            return false;
        if ( !krylov::apply_operator( A, u, *rhs_ ) )
            return false;
        operations_->add_lin_comb( candidate, d, T( 1 ), *rhs_ );
        operations_->add_lin_comb( T( -1 ), b, T( 1 ), *rhs_ );
        const T scalar_residual = operations_->scalar_prod( c, u ) + alpha * candidate - beta;
        if ( !( math_type::hypot( operations_->norm( *rhs_ ), scalar_residual ) <= tolerance ) )
            return false;
        v = candidate;
        return true;
    }

private:
    bool solve_border_reduced( T alpha, T tolerance, vector_type &u ) const
    {
        auto &monitor = reduced_.monitor();
        // The reduced residual equals the vector block residual, scaled only in the scaled equation.
        const T target = scaled( alpha ) ? math_type::abs( alpha ) * tolerance : tolerance;
        if ( !math_type::is_finite( target ) )
            return false;
        struct restore_guard
        {
            Monitor &monitor;
            T        relative;
            T        absolute;
            ~restore_guard()
            {
                monitor.set_tolerances( relative, absolute );
            }
        } guard{ monitor, monitor.rel_tol(), monitor.abs_tol() };
        monitor.set_tolerances( T( 0 ), target );
        return reduced_.solve( *operator_, *rhs_, u );
    }
    bool scaled( T alpha ) const
    {
        return use_small_alpha_ && math_type::abs( alpha ) < T( 1e-3 );
    }
    void check_operator( const LinearOperator &A ) const
    {
        if ( &A != operator_->original.get() )
            throw std::invalid_argument( "Sherman-Morrison: solve must use the bound operator" );
    }
    bool configure( const LinearOperator &A, T gamma, T alpha, const vector_type &c, const vector_type &d ) const
    {
        check_operator( A );
        if ( !math_type::is_finite( alpha ) || !math_type::is_finite( gamma ) || alpha == T( 0 ) || gamma == T( 0 ) )
            return false;
        operator_->c    = &c;
        operator_->d    = &d;
        operator_->mass = scaled( alpha ) ? alpha * gamma : gamma;
        operator_->rank = scaled( alpha ) ? T( -1 ) : -T( 1 ) / alpha;
        return true;
    }

    std::shared_ptr<VectorOperations>       operations_;
    std::shared_ptr<reduced_operator>       operator_;
    std::shared_ptr<reduced_preconditioner> preconditioner_;
    mutable linear_solver_type              reduced_;
    linear_solver_original_type             original_;
    mutable vector_wrap_type                rhs_;
    bool                                    use_small_alpha_ = false;
};

} // namespace solvers
} // namespace nmfd

#endif
