#ifndef NMFD_PRECONDITIONERS_SMALL_SCHUR_INVERSE_H
#define NMFD_PRECONDITIONERS_SMALL_SCHUR_INVERSE_H

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/linalg/small_dense.h>
#include <nmfd/solvers/krylov/operator_apply.h>

namespace nmfd
{
namespace preconditioners
{

// A small, locally addressable or replicated parameter block. Capacity is a setup bound,
// not the field dimension. Scalar math follows the operations backend traits.
template <class VectorSpace, class Operator, std::size_t Capacity = 16>
class small_schur_inverse
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using math_type   = nmfd::operations::backend::math<void, scalar_type>;
    using vector_type = typename VectorSpace::vector_type;

    explicit small_schur_inverse( std::shared_ptr<VectorSpace> space )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing small Schur space" ) ),
          basis_( *space_ ), image_( *space_ ), size_( space_->size() )
    {
        if ( size_ == 0 || size_ > Capacity )
            throw std::invalid_argument( "Small Schur dimension exceeds its configured capacity" );
        basis_.init();
        basis_.start_use();
        image_.init();
        image_.start_use();
        lu_.resize( size_, size_ );
    }

    // Sample and factor S once per preparation; never assemble the large state block.
    bool set_operator( std::shared_ptr<const Operator> op )
    {
        ready_ = false;
        if ( !op )
            return false;
        std::array<scalar_type, Capacity> column{};
        for ( std::size_t j = 0; j < size_; ++j )
        {
            space_->assign_scalar( scalar_type( 0 ), *basis_ );
            space_->set_value_at_point( scalar_type( 1 ), j, *basis_ );
            if ( !solvers::krylov::apply_operator( *op, *basis_, *image_ ) || !space_->is_valid_number( *image_ ) )
                return false;
            space_->get( *image_, column.data(), size_ );
            for ( std::size_t i = 0; i < size_; ++i )
                lu_( i, j ) = column[i];
        }
        if ( !operations::linalg::factor_lu( lu_, pivots_ ).ok() )
            return false;
        for ( std::size_t i = 0; i < size_; ++i )
            for ( std::size_t j = 0; j < size_; ++j )
                if ( !math_type::is_finite( lu_( i, j ) ) )
                    return false;
        ready_ = true;
        return true;
    }

    bool ready() const
    {
        return ready_;
    }

    // Solve with the prepared factors and publish only a complete finite result.
    bool apply( vector_type &in_out ) const
    {
        if ( !ready_ || !space_->is_valid_number( in_out ) )
            return false;
        std::array<scalar_type, Capacity> values{};
        space_->get( in_out, values.data(), size_ );
        operations::linalg::small_vector<scalar_type, Capacity> candidate( size_ );
        for ( std::size_t i = 0; i < size_; ++i )
            candidate[i] = values[i];
        operations::linalg::solve_lu_in_place( lu_, pivots_, candidate );
        for ( std::size_t i = 0; i < size_; ++i )
        {
            if ( !math_type::is_finite( candidate[i] ) )
                return false;
            values[i] = candidate[i];
        }
        space_->set( values.data(), in_out, size_ );
        return true;
    }

private:
    std::shared_ptr<VectorSpace>                                      space_;
    nmfd::detail::vector_wrap<VectorSpace, false, false>              basis_, image_;
    operations::linalg::small_matrix<scalar_type, Capacity, Capacity> lu_;
    std::array<std::size_t, Capacity>                                 pivots_{};
    std::size_t                                                       size_;
    bool                                                              ready_ = false;
};

}
}

#endif
