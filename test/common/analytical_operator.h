#ifndef __NMFD_TEST_COMMON_ANALYTICAL_OPERATOR_H__
#define __NMFD_TEST_COMMON_ANALYTICAL_OPERATOR_H__

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nmfd
{
namespace tests
{

template <class VectorSpace>
class analytical_operator
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type       = typename vector_space_type::scalar_type;
    using vector_type       = typename vector_space_type::vector_type;

    analytical_operator( const vector_space_type &vector_space, std::vector<scalar_type> matrix )
        : vector_space_( vector_space ), matrix_( std::move( matrix ) ),
          dimension_( static_cast<std::size_t>( vector_space_.size() ) )
    {
        if ( matrix_.size() != dimension_ * dimension_ )
            throw std::invalid_argument( "analytical_operator matrix size mismatch" );
    }

    bool apply( const vector_type &source, vector_type &destination ) const
    {
        ++operator_calls_;
        if ( operator_calls_ > fail_after_ )
            return false;
        vector_space_.assign_scalar( scalar_type{}, destination );
        for ( std::size_t row = 0; row < dimension_; ++row )
        {
            for ( std::size_t col = 0; col < dimension_; ++col )
                destination[row] += matrix_[row * dimension_ + col] * source[col];
        }
        return true;
    }

    void fail_after( std::size_t successful_calls ) const
    {
        fail_after_ = successful_calls;
    }

    void allow_all_calls() const
    {
        fail_after_ = std::numeric_limits<std::size_t>::max();
    }

    void reset_calls() const
    {
        operator_calls_ = 0;
    }

    std::size_t operator_calls() const
    {
        return operator_calls_;
    }

    const std::vector<scalar_type> &matrix() const
    {
        return matrix_;
    }

private:
    const vector_space_type &vector_space_;
    std::vector<scalar_type> matrix_;
    std::size_t              dimension_;
    mutable std::size_t      operator_calls_ = 0;
    mutable std::size_t      fail_after_     = std::numeric_limits<std::size_t>::max();
};

template <class VectorSpace, class Operator>
class diagonal_preconditioner
{
public:
    using vector_space_type = VectorSpace;
    using operator_type     = Operator;
    using scalar_type       = typename vector_space_type::scalar_type;
    using vector_type       = typename vector_space_type::vector_type;

    diagonal_preconditioner( const vector_space_type &vector_space, std::vector<scalar_type> inverse_diagonal )
        : vector_space_( vector_space ), inverse_diagonal_( std::move( inverse_diagonal ) )
    {
        if ( inverse_diagonal_.size() != static_cast<std::size_t>( vector_space_.size() ) )
        {
            throw std::invalid_argument( "diagonal_preconditioner size mismatch" );
        }
    }

    void set_operator( std::shared_ptr<const operator_type> )
    {
    }

    bool apply( vector_type &vector ) const
    {
        ++apply_calls_;
        if ( apply_calls_ > fail_after_ )
            return false;
        for ( std::size_t index = 0; index < inverse_diagonal_.size(); ++index )
            vector[index] *= inverse_diagonal_[index];
        return true;
    }

    bool apply( const vector_type &source, vector_type &destination ) const
    {
        vector_space_.assign( source, destination );
        return apply( destination );
    }

    bool solve( const vector_type &source, vector_type &destination ) const
    {
        return apply( source, destination );
    }

    void fail_after( std::size_t successful_calls ) const
    {
        fail_after_ = successful_calls;
    }

    void allow_all_calls() const
    {
        fail_after_ = std::numeric_limits<std::size_t>::max();
    }

    void reset_calls() const
    {
        apply_calls_ = 0;
    }

    std::size_t apply_calls() const
    {
        return apply_calls_;
    }

private:
    const vector_space_type &vector_space_;
    std::vector<scalar_type> inverse_diagonal_;
    mutable std::size_t      apply_calls_ = 0;
    mutable std::size_t      fail_after_  = std::numeric_limits<std::size_t>::max();
};

} // namespace tests
} // namespace nmfd

#endif
