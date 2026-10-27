#ifndef NMFD_PRECONDITIONERS_BLOCK_SCHUR_H
#define NMFD_PRECONDITIONERS_BLOCK_SCHUR_H

#include <memory>
#include <stdexcept>
#include <utility>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/two_block_access.h>
#include <nmfd/solvers/krylov/operator_apply.h>

namespace nmfd
{
namespace preconditioners
{

// H is a fixed linear approximate inverse of A; the complete operator stays unchanged.
template <
    class VectorSpace, class BlockOperator, class FirstInverse,
    class Access = operations::two_block_access<VectorSpace, BlockOperator>>
class schur_complement_operator
{
    using first_space_type  = typename Access::first_space_type;
    using second_space_type = typename Access::second_space_type;

public:
    using scalar_type   = typename VectorSpace::scalar_type;
    using vector_type   = typename second_space_type::vector_type;
    using operator_type = BlockOperator;
    using access_type   = Access;

    schur_complement_operator(
        std::shared_ptr<VectorSpace> space, std::shared_ptr<FirstInverse> inverse, Access access = Access()
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Schur operator: null space" ) ),
          inverse_( std::move( inverse ) ), access_( std::move( access ) ),
          first_( access_.template space<0>( *space_ ) ), second_( access_.template space<1>( *space_ ) )
    {
        if ( !inverse_ )
            throw std::invalid_argument( "Schur operator: null first inverse" );
        first_.init();
        first_.start_use();
        second_.init();
        second_.start_use();
    }

    void set_operator( std::shared_ptr<const BlockOperator> op )
    {
        op_ = std::move( op );
    }

    const std::shared_ptr<const operator_type> &get_operator() const
    {
        return op_;
    }

    // S*y = E*y - C*H*(B*y). Input and output must not alias.
    bool apply( const vector_type &in, vector_type &out ) const
    {
        if ( &in == &out )
            throw std::invalid_argument( "Schur operator requires distinct input and output" );
        if ( !op_ || !access_.template apply_block<0, 1>( *op_, in, *first_ ) ||
             !solvers::krylov::apply_in_place( *inverse_, *first_ ) ||
             !access_.template apply_block<1, 0>( *op_, *first_, *second_ ) ||
             !access_.template apply_block<1, 1>( *op_, in, out ) )
            return false;
        const auto &second_space = access_.template space<1>( *space_ );
        second_space.add_lin_comb( scalar_type( -1 ), *second_, scalar_type( 1 ), out );
        return second_space.is_valid_number( out );
    }

private:
    std::shared_ptr<VectorSpace>                                       space_;
    std::shared_ptr<FirstInverse>                                      inverse_;
    std::shared_ptr<const BlockOperator>                               op_;
    Access                                                             access_;
    mutable nmfd::detail::vector_wrap<first_space_type, false, false>  first_;
    mutable nmfd::detail::vector_wrap<second_space_type, false, false> second_;
};

// Full block factorization with supplied H ~= A^-1 and Q ~= (E-C*H*B)^-1.
// H and Q expose set_operator(shared_ptr<const Operator>) and apply(vector&), returning void or bool.
// They must remain fixed and linear within a GMRES/BiCGStab(L) solve.
template <
    class VectorSpace, class BlockOperator, class FirstInverse, class SchurInverse,
    class Access = operations::two_block_access<VectorSpace, BlockOperator>>
class block_schur
{
    using first_space_type = typename Access::first_space_type;

public:
    using scalar_type         = typename VectorSpace::scalar_type;
    using vector_type         = typename VectorSpace::vector_type;
    using operator_type       = BlockOperator;
    using access_type         = Access;
    using schur_operator_type = schur_complement_operator<VectorSpace, BlockOperator, FirstInverse, Access>;

    block_schur(
        std::shared_ptr<VectorSpace> space, std::shared_ptr<FirstInverse> first_inverse,
        std::shared_ptr<SchurInverse> schur_inverse, Access access = Access()
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Block Schur: null space" ) ),
          first_inverse_( std::move( first_inverse ) ), schur_inverse_( std::move( schur_inverse ) ),
          access_( std::move( access ) ),
          schur_( std::make_shared<schur_operator_type>( space_, first_inverse_, access_ ) ), candidate_( *space_ ),
          work_( access_.template space<0>( *space_ ) )
    {
        if ( !schur_inverse_ )
            throw std::invalid_argument( "Block Schur: null Schur inverse" );
        candidate_.init();
        candidate_.start_use();
        work_.init();
        work_.start_use();
    }

    // Always rebuild dependent setup, including when the operator handle is unchanged.
    // Numerical setup failures disable apply(); configuration exceptions still propagate.
    void set_operator( std::shared_ptr<const operator_type> op )
    {
        ready_ = false;
        op_    = std::move( op );
        schur_->set_operator( nullptr );
        if ( !op_ || !solvers::krylov::invoke_status( [&]() -> decltype( auto ) {
                 return first_inverse_->set_operator( access_.template block<0, 0>( *op_ ) );
             } ) )
            return;
        schur_->set_operator( op_ );
        std::shared_ptr<const schur_operator_type> prepared = schur_;
        ready_ = solvers::krylov::invoke_status( [&]() -> decltype( auto ) {
            return schur_inverse_->set_operator( std::move( prepared ) );
        } );
    }

    void update_operator( std::shared_ptr<const operator_type> op )
    {
        set_operator( std::move( op ) );
    }

    bool ready() const
    {
        return ready_;
    }

    // Commit only a complete finite action; aliasing rhs/out is supported.
    bool apply( const vector_type &rhs, vector_type &out ) const
    {
        if ( !ready_ )
            return false;
        auto       &candidate    = *candidate_;
        auto       &first        = access_.template component<0>( candidate );
        auto       &second       = access_.template component<1>( candidate );
        const auto &first_space  = access_.template space<0>( *space_ );
        const auto &second_space = access_.template space<1>( *space_ );
        first_space.assign( access_.template component<0>( rhs ), first );
        if ( !solvers::krylov::apply_in_place( *first_inverse_, first ) ||
             !access_.template apply_block<1, 0>( *op_, first, second ) )
            return false;
        second_space.assign_lin_comb(
            scalar_type( 1 ), access_.template component<1>( rhs ), scalar_type( -1 ), second, second
        );
        if ( !solvers::krylov::apply_in_place( *schur_inverse_, second ) ||
             !access_.template apply_block<0, 1>( *op_, second, *work_ ) ||
             !solvers::krylov::apply_in_place( *first_inverse_, *work_ ) )
            return false;
        first_space.add_lin_comb( scalar_type( -1 ), *work_, scalar_type( 1 ), first );
        if ( !space_->is_valid_number( candidate ) )
            return false;
        space_->assign( candidate, out );
        return true;
    }

    bool apply( vector_type &in_out ) const
    {
        return apply( in_out, in_out );
    }

private:
    std::shared_ptr<VectorSpace>                                      space_;
    std::shared_ptr<FirstInverse>                                     first_inverse_;
    std::shared_ptr<SchurInverse>                                     schur_inverse_;
    Access                                                            access_;
    std::shared_ptr<schur_operator_type>                              schur_;
    std::shared_ptr<const operator_type>                              op_;
    mutable nmfd::detail::vector_wrap<VectorSpace, false, false>      candidate_;
    mutable nmfd::detail::vector_wrap<first_space_type, false, false> work_;
    bool                                                              ready_ = false;
};

}
}

#endif
