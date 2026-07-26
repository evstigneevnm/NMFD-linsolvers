#ifndef __NMFD_SOLVERS_KRYLOV_BASIS_STORAGE_H__
#define __NMFD_SOLVERS_KRYLOV_BASIS_STORAGE_H__

#include <cstddef>
#include <stdexcept>

namespace nmfd
{
namespace solvers
{
namespace krylov
{

template<class VectorSpace>
class basis_storage
{
public:
    using vector_space_type = VectorSpace;
    using multivector_type = typename vector_space_type::multivector_type;
    using ordinal_type = typename vector_space_type::ordinal_type;

    basis_storage(
        const vector_space_type& vector_space,
        std::size_t capacity)
        : vector_space_(vector_space),
          capacity_(capacity)
    {
        if(capacity_ == 0)
            throw std::invalid_argument("krylov::basis_storage requires nonzero capacity");
        vector_space_.init_multivector(data_, ordinal_capacity());
        vector_space_.start_use_multivector(data_, ordinal_capacity());
    }

    basis_storage(const basis_storage&) = delete;
    basis_storage& operator=(const basis_storage&) = delete;
    basis_storage(basis_storage&&) = delete;
    basis_storage& operator=(basis_storage&&) = delete;

    ~basis_storage()
    {
        vector_space_.stop_use_multivector(data_, ordinal_capacity());
        vector_space_.free_multivector(data_, ordinal_capacity());
    }

    multivector_type& data()
    {
        return data_;
    }

    const multivector_type& data() const
    {
        return data_;
    }

    std::size_t capacity() const
    {
        return capacity_;
    }

private:
    ordinal_type ordinal_capacity() const
    {
        return static_cast<ordinal_type>(capacity_);
    }

    const vector_space_type& vector_space_;
    std::size_t capacity_;
    multivector_type data_;
};

} // namespace krylov
} // namespace solvers
} // namespace nmfd

#endif
