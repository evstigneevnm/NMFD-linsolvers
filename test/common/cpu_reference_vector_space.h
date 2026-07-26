#ifndef __NMFD_TEST_COMMON_CPU_REFERENCE_VECTOR_SPACE_H__
#define __NMFD_TEST_COMMON_CPU_REFERENCE_VECTOR_SPACE_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace nmfd
{
namespace tests
{

template<class T>
struct reference_real_type
{
    using type = T;
};

template<class T>
struct reference_real_type<std::complex<T>>
{
    using type = T;
};

template<class T>
class cpu_reference_vector_space
{
public:
    using scalar_type = T;
    using norm_type = typename reference_real_type<T>::type;
    using ordinal_type = std::ptrdiff_t;
    using big_ordinal_type = ordinal_type;
    using Ord = ordinal_type;
    using vector_type = std::vector<scalar_type>;
    using multivector_type = std::vector<vector_type>;
    using T_vec = vector_type;
    using T_mvec = multivector_type;

    explicit cpu_reference_vector_space(ordinal_type size)
        : size_(checked_size(size))
    {
    }

    ordinal_type size() const
    {
        return static_cast<ordinal_type>(size_);
    }

    void init_vector(vector_type& vector) const
    {
        vector.clear();
    }

    void start_use_vector(vector_type& vector) const
    {
        vector.assign(size_, scalar_type{});
    }

    void stop_use_vector(vector_type&) const
    {
    }

    void free_vector(vector_type& vector) const
    {
        vector.clear();
    }

    void init_multivector(
        multivector_type& multivector,
        ordinal_type count) const
    {
        multivector.assign(as_size(count), vector_type{});
    }

    void start_use_multivector(
        multivector_type& multivector,
        ordinal_type count) const
    {
        const std::size_t vector_count = as_size(count);
        if(multivector.size() != vector_count)
            multivector.assign(vector_count, vector_type{});
        for(auto& vector : multivector)
            start_use_vector(vector);
    }

    void stop_use_multivector(
        multivector_type&,
        ordinal_type) const
    {
    }

    void free_multivector(
        multivector_type& multivector,
        ordinal_type) const
    {
        multivector.clear();
    }

    vector_type& at(
        multivector_type& multivector,
        ordinal_type count,
        ordinal_type index) const
    {
        check_multivector(multivector, count, index);
        return multivector[as_size(index)];
    }

    const vector_type& at(
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index) const
    {
        check_multivector(multivector, count, index);
        return multivector[as_size(index)];
    }

    void assign(
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index,
        vector_type& vector) const
    {
        vector = at(multivector, count, index);
    }

    void assign(
        const vector_type& vector,
        multivector_type& multivector,
        ordinal_type count,
        ordinal_type index) const
    {
        check_vector(vector);
        at(multivector, count, index) = vector;
    }

    scalar_type scalar_prod(
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index,
        const vector_type& vector) const
    {
        return scalar_prod(at(multivector, count, index), vector);
    }

    scalar_type scalar_prod_l2(
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index,
        const vector_type& vector) const
    {
        return scalar_prod(multivector, count, index, vector);
    }

    void add_lin_comb(
        scalar_type source_coefficient,
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index,
        scalar_type destination_coefficient,
        vector_type& destination) const
    {
        add_lin_comb(
            source_coefficient,
            at(multivector, count, index),
            destination_coefficient,
            destination);
    }

    bool is_valid_number(const vector_type& vector) const
    {
        check_vector(vector);
        for(const auto& value : vector)
            if(!finite(value))
                return false;
        return true;
    }

    scalar_type scalar_prod(
        const vector_type& left,
        const vector_type& right) const
    {
        check_vector(left);
        check_vector(right);
        scalar_type result{};
        for(std::size_t index = 0; index < size_; ++index)
            result += conjugate(left[index])*right[index];
        return result;
    }

    scalar_type scalar_prod_l2(
        const vector_type& left,
        const vector_type& right) const
    {
        return scalar_prod(left, right);
    }

    scalar_type sum(const vector_type& vector) const
    {
        check_vector(vector);
        scalar_type result{};
        for(const auto& value : vector)
            result += value;
        return result;
    }

    norm_type asum(const vector_type& vector) const
    {
        check_vector(vector);
        norm_type result{};
        for(const auto& value : vector)
        {
            using std::abs;
            result += abs(value);
        }
        return result;
    }

    norm_type norm(const vector_type& vector) const
    {
        return std::sqrt(norm_sq(vector));
    }

    norm_type norm_l2(const vector_type& vector) const
    {
        return norm(vector)/std::sqrt(static_cast<norm_type>(size_));
    }

    norm_type norm2(const vector_type& vector) const
    {
        return norm_l2(vector);
    }

    norm_type norm_sq(const vector_type& vector) const
    {
        using std::real;
        const norm_type value =
            static_cast<norm_type>(real(scalar_prod(vector, vector)));
        if(!std::isfinite(value))
            return value;
        return std::max(norm_type{}, value);
    }

    norm_type norm_l2_sq(const vector_type& vector) const
    {
        return norm_sq(vector)/static_cast<norm_type>(size_);
    }

    norm_type norm2_sq(const vector_type& vector) const
    {
        return norm_l2_sq(vector);
    }

    void assign_scalar(
        scalar_type value,
        vector_type& vector) const
    {
        check_vector(vector);
        std::fill(vector.begin(), vector.end(), value);
    }

    void add_mul_scalar(
        scalar_type value,
        scalar_type vector_coefficient,
        vector_type& vector) const
    {
        check_vector(vector);
        for(auto& entry : vector)
            entry = vector_coefficient*entry + value;
    }

    void scale(
        scalar_type coefficient,
        vector_type& vector) const
    {
        check_vector(vector);
        for(auto& value : vector)
            value *= coefficient;
    }

    void assign(
        const vector_type& source,
        vector_type& destination) const
    {
        check_vector(source);
        destination = source;
    }

    void assign_lin_comb(
        scalar_type coefficient,
        const vector_type& source,
        vector_type& destination) const
    {
        check_vector(source);
        check_vector(destination);
        for(std::size_t index = 0; index < size_; ++index)
            destination[index] = coefficient*source[index];
    }

    void assign_lin_comb(
        scalar_type left_coefficient,
        const vector_type& left,
        scalar_type right_coefficient,
        const vector_type& right,
        vector_type& destination) const
    {
        check_vector(left);
        check_vector(right);
        check_vector(destination);
        for(std::size_t index = 0; index < size_; ++index)
        {
            destination[index] =
                left_coefficient*left[index] +
                right_coefficient*right[index];
        }
    }

    void add_lin_comb(
        scalar_type source_coefficient,
        const vector_type& source,
        scalar_type destination_coefficient,
        vector_type& destination) const
    {
        check_vector(source);
        check_vector(destination);
        for(std::size_t index = 0; index < size_; ++index)
        {
            destination[index] =
                source_coefficient*source[index] +
                destination_coefficient*destination[index];
        }
    }

    void add_lin_comb(
        scalar_type first_coefficient,
        const vector_type& first,
        scalar_type second_coefficient,
        const vector_type& second,
        scalar_type destination_coefficient,
        vector_type& destination) const
    {
        check_vector(first);
        check_vector(second);
        check_vector(destination);
        for(std::size_t index = 0; index < size_; ++index)
        {
            destination[index] =
                first_coefficient*first[index] +
                second_coefficient*second[index] +
                destination_coefficient*destination[index];
        }
    }

private:
    static std::size_t checked_size(ordinal_type value)
    {
        if(value <= ordinal_type{})
            throw std::invalid_argument(
                "cpu_reference_vector_space requires positive size");
        return static_cast<std::size_t>(value);
    }

    static std::size_t as_size(ordinal_type value)
    {
        if(value < ordinal_type{})
            throw std::out_of_range("negative vector-space index");
        return static_cast<std::size_t>(value);
    }

    void check_vector(const vector_type& vector) const
    {
        if(vector.size() != size_)
            throw std::invalid_argument(
                "cpu_reference_vector_space vector size mismatch");
    }

    void check_multivector(
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index) const
    {
        if(
            multivector.size() != as_size(count) ||
            index < ordinal_type{} ||
            index >= count)
        {
            throw std::out_of_range(
                "cpu_reference_vector_space multivector index");
        }
        check_vector(multivector[as_size(index)]);
    }

    static scalar_type conjugate(const scalar_type& value)
    {
        if constexpr(std::is_floating_point<scalar_type>::value)
            return value;
        else
        {
            using std::conj;
            return conj(value);
        }
    }

    static bool finite(const scalar_type& value)
    {
        if constexpr(std::is_floating_point<scalar_type>::value)
            return std::isfinite(value);
        else
            return std::isfinite(value.real()) && std::isfinite(value.imag());
    }

    std::size_t size_;
};

} // namespace tests
} // namespace nmfd

#endif
