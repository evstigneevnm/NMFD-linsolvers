#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <nmfd/operations/linalg/host_small_dense_backend.h>
#include <nmfd/solvers/krylov/arnoldi.h>
#include <nmfd/solvers/krylov/basis_storage.h>

#include "common/cpu_reference_vector_space.h"
#include "common/test_harness.h"

namespace
{

using vector_space_type =
    nmfd::tests::cpu_reference_vector_space<double>;
using matrix_type =
    nmfd::operations::linalg::host_dense_matrix<double>;

class matrix_apply
{
public:
    matrix_apply(std::vector<double> matrix, std::size_t dimension)
        : matrix_(std::move(matrix)),
          dimension_(dimension)
    {
    }

    bool operator()(
        const vector_space_type::vector_type& source,
        vector_space_type::vector_type& destination) const
    {
        ++calls_;
        if(calls_ > fail_after_)
            return false;
        std::fill(destination.begin(), destination.end(), 0.0);
        for(std::size_t row = 0; row < dimension_; ++row)
            for(std::size_t col = 0; col < dimension_; ++col)
                destination[row] += matrix_[row*dimension_ + col]*source[col];
        return true;
    }

    void fail_after(std::size_t calls)
    {
        fail_after_ = calls;
    }

    std::size_t calls() const
    {
        return calls_;
    }

private:
    std::vector<double> matrix_;
    std::size_t dimension_;
    mutable std::size_t calls_ = 0;
    std::size_t fail_after_ = std::numeric_limits<std::size_t>::max();
};

void test_factorization(nmfd::tests::test_harness& test)
{
    vector_space_type vector_space(4);
    nmfd::solvers::krylov::basis_storage<vector_space_type> basis(
        vector_space,
        4);
    vector_space_type::vector_type initial{1.0, -2.0, 0.5, 3.0};
    vector_space_type::vector_type basis_vector(4, 0.0);
    vector_space_type::vector_type candidate(4, 0.0);
    matrix_type hessenberg(4, 3);
    double coefficients[4]{};
    double pass_coefficients[4]{};
    matrix_apply apply(
        {
            4.0, 1.0, 0.0, 0.0,
            0.0, 3.0, 2.0, 0.0,
            0.0, 0.0, 2.0, 1.0,
            1.0, 0.0, 0.0, 1.0
        },
        4);
    nmfd::solvers::krylov::orthogonalization_options<double> options;
    options.reorthogonalization =
        nmfd::solvers::krylov::reorthogonalization_policy::always;
    options.compute_orthogonality_error = true;

    const auto result = nmfd::solvers::krylov::build_arnoldi_factorization(
        vector_space,
        apply,
        initial,
        basis.data(),
        basis.capacity(),
        3,
        basis_vector,
        candidate,
        coefficients,
        pass_coefficients,
        hessenberg,
        options);

    test.require(result.succeeded(), "Arnoldi factorization succeeds");
    test.require(result.completed_steps == 3, "Arnoldi completes three steps");
    test.require(result.operator_calls == 3, "Arnoldi counts operator calls");
    test.require(apply.calls() == 3, "operator observes three calls");
    test.require_close(
        result.initial_norm,
        vector_space.norm(initial),
        1e-14,
        "Arnoldi initial norm");

    for(std::size_t col = 0; col < 3; ++col)
    {
        for(std::size_t row = 0; row <= col; ++row)
        {
            const double inner = vector_space.scalar_prod(
                basis.data()[row],
                basis.data()[col]);
            test.require_close(
                inner,
                row == col ? 1.0 : 0.0,
                2e-13,
                "Arnoldi basis orthogonality");
        }

        vector_space_type::vector_type left(4, 0.0);
        vector_space_type::vector_type right(4, 0.0);
        apply(basis.data()[col], left);
        for(std::size_t row = 0; row <= col + 1; ++row)
        {
            vector_space.add_lin_comb(
                hessenberg(row, col),
                basis.data()[row],
                1.0,
                right);
        }
        vector_space.add_lin_comb(-1.0, right, 1.0, left);
        test.require(
            vector_space.norm(left) <= 5e-13,
            "Arnoldi relation A*V=V*H");
    }
}

void test_breakdown(nmfd::tests::test_harness& test)
{
    vector_space_type vector_space(3);
    vector_space_type::multivector_type basis;
    vector_space.init_multivector(basis, 3);
    vector_space.start_use_multivector(basis, 3);
    vector_space_type::vector_type initial{1.0, 0.0, 0.0};
    vector_space_type::vector_type basis_vector(3, 0.0);
    vector_space_type::vector_type candidate(3, 0.0);
    matrix_type hessenberg(3, 2);
    double coefficients[3]{};
    double pass_coefficients[3]{};
    matrix_apply diagonal(
        {
            2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 4.0
        },
        3);
    const auto result = nmfd::solvers::krylov::build_arnoldi_factorization(
        vector_space,
        diagonal,
        initial,
        basis,
        3,
        2,
        basis_vector,
        candidate,
        coefficients,
        pass_coefficients,
        hessenberg);
    test.require(
        result.status == nmfd::solvers::krylov::arnoldi_status::happy_breakdown,
        "eigenvector start produces happy breakdown");
    test.require(result.completed_steps == 1, "breakdown completes one step");
}

void test_failures(nmfd::tests::test_harness& test)
{
    vector_space_type vector_space(2);
    vector_space_type::multivector_type basis;
    vector_space.init_multivector(basis, 2);
    vector_space.start_use_multivector(basis, 2);
    vector_space_type::vector_type initial{1.0, 1.0};
    vector_space_type::vector_type basis_vector(2, 0.0);
    vector_space_type::vector_type candidate(2, 0.0);
    matrix_type hessenberg(2, 1);
    double coefficients[2]{};
    double pass_coefficients[2]{};
    matrix_apply apply({1.0, 0.0, 0.0, 2.0}, 2);
    apply.fail_after(0);
    const auto failed = nmfd::solvers::krylov::build_arnoldi_factorization(
        vector_space,
        apply,
        initial,
        basis,
        2,
        1,
        basis_vector,
        candidate,
        coefficients,
        pass_coefficients,
        hessenberg);
    test.require(
        failed.status == nmfd::solvers::krylov::arnoldi_status::operator_failure,
        "Arnoldi propagates operator failure");
    test.require(failed.operator_calls == 1, "failed call is counted");

    initial = {0.0, 0.0};
    double initial_norm = 1.0;
    const auto invalid =
        nmfd::solvers::krylov::initialize_arnoldi_basis(
            vector_space,
            initial,
            basis,
            2,
            basis_vector,
            initial_norm);
    test.require(
        invalid ==
            nmfd::solvers::krylov::arnoldi_status::invalid_initial_vector,
        "Arnoldi rejects zero initial vector");
}

} // namespace

int main()
{
    nmfd::tests::test_harness test;
    test_factorization(test);
    test_breakdown(test);
    test_failures(test);
    return test.finish("NMFD Arnoldi tests");
}
