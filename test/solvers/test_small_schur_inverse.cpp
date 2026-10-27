#include <limits>
#include <memory>
#include <stdexcept>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/operations/block_operator.h>
#include <nmfd/operations/pair_vector_space.h>
#include <nmfd/preconditioners/block_schur.h>
#include <nmfd/preconditioners/small_schur_inverse.h>
#include <scfd/static_vec/vec.h>
#include "common/test_harness.h"

namespace
{

// K = [2I I; I 0], H = I, S = -I. This tests the production Q through
// the block-Schur setup/action pipeline for one and multiple parameters.
template <int N>
void test_block_factor( nmfd::tests::test_harness &report )
{
    using leaf_type        = nmfd::operations::static_vector_space<double, N, scfd::static_vec::vec<double, N>>;
    using leaf_vector_type = typename leaf_type::vector_type;
    using space_type       = nmfd::operations::pair_vector_space<leaf_type, leaf_type>;
    struct block_type
    {
        double scale;
        void   apply( const leaf_vector_type &in, leaf_vector_type &out ) const
        {
            for ( int i = 0; i < N; ++i )
                out[i] = scale * in[i];
        }
    };
    struct inverse_type
    {
        bool fail = false;
        void set_operator( std::shared_ptr<const block_type> )
        {
        }
        void apply( leaf_vector_type &in_out ) const
        {
            if ( fail )
                for ( int i = 0; i < N; ++i )
                    in_out[i] = 0;
        }
    };
    using operator_type = nmfd::operations::block_operator<space_type, block_type, block_type, block_type, block_type>;
    using schur_type    = nmfd::preconditioners::schur_complement_operator<space_type, operator_type, inverse_type>;
    using q_type        = nmfd::preconditioners::small_schur_inverse<leaf_type, schur_type, N>;
    auto leaf           = std::make_shared<leaf_type>();
    auto space          = std::make_shared<space_type>( leaf, leaf );
    auto a              = std::make_shared<block_type>( block_type{ 2 } );
    auto b              = std::make_shared<block_type>( block_type{ 1 } );
    auto c              = std::make_shared<block_type>( block_type{ 1 } );
    auto e              = std::make_shared<block_type>( block_type{ 0 } );
    auto op             = std::make_shared<operator_type>( space, a, b, c, e );
    auto h              = std::make_shared<inverse_type>();
    auto q              = std::make_shared<q_type>( leaf );
    nmfd::preconditioners::block_schur<space_type, operator_type, inverse_type, q_type> prec( space, h, q );
    typename space_type::vector_type                                                    rhs, result;
    leaf->assign_scalar( 3, rhs.first );
    leaf->assign_scalar( 1, rhs.second );
    prec.set_operator( op );
    report.require( prec.ready() && prec.apply( rhs, result ), "Production Schur setup and action" );
    for ( int i = 0; i < N; ++i )
    {
        report.require_close( result.first[i], 1, 1e-12, "Block-Schur first component" );
        report.require_close( result.second[i], 2, 1e-12, "Block-Schur parameter component" );
    }
    h->fail = true;
    prec.set_operator( op );
    report.require( !prec.ready(), "Singular approximate Schur factor disables preconditioner" );
    report.require( !prec.apply( rhs, result ), "No stale Schur factors after preparation failure" );
    h->fail = false;
    prec.set_operator( op );
    report.require( prec.ready() && prec.apply( rhs, result ), "Schur factor recovery" );
}

// Small parameter Schur blocks: scalar and pivoted two-parameter systems.
// No field matrix or inverse is assembled; repeated actions reuse the LU factors.
template <class T, int N>
void test_inverse( nmfd::tests::test_harness &report )
{
    using space_type  = nmfd::operations::static_vector_space<T, N, scfd::static_vec::vec<T, N>>;
    using vector_type = typename space_type::vector_type;
    struct operator_type
    {
        bool fail = false, nonfinite = false, singular = false;
        T    scale = T( 1 );
        bool apply( const vector_type &in, vector_type &out ) const
        {
            if ( fail )
                return false;
            if constexpr ( N == 1 )
                out[0] = T( 2 ) * in[0];
            else
            {
                out[0] = T( 2 ) * in[1];
                out[1] = T( 3 ) * in[0] + T( 4 ) * in[1];
            }
            for ( int i = 0; i < N; ++i )
                out[i] *= singular ? T( 0 ) : scale;
            if ( nonfinite )
                out[0] = std::numeric_limits<T>::infinity();
            return true;
        }
    };
    auto                                                                     space = std::make_shared<space_type>();
    auto                                                                     op    = std::make_shared<operator_type>();
    nmfd::preconditioners::small_schur_inverse<space_type, operator_type, N> inverse( space );
    vector_type                                                              exact, rhs, candidate;
    space->assign_scalar( T( 1 ), exact );
    space->assign_scalar( T( 0 ), rhs );
    op->apply( exact, rhs );
    candidate = rhs;
    report.require( !inverse.apply( candidate ), "Unprepared Schur inverse must fail" );
    report.require( inverse.set_operator( op ) && inverse.ready(), "Schur factorization" );
    for ( int repeat = 0; repeat < 4; ++repeat )
    {
        candidate = rhs;
        report.require( inverse.apply( candidate ), "Repeated Schur action" );
        for ( int i = 0; i < N; ++i )
            report.require_close( candidate[i], exact[i], 1e-6, "Schur solution" );
    }
    op->scale = T( 2 );
    report.require( inverse.set_operator( op ), "Same-handle Schur update" );
    op->apply( exact, candidate );
    report.require( inverse.apply( candidate ), "Updated Schur action" );
    report.require_close( candidate[0], T( 1 ), 1e-6, "Updated Schur factors" );
    for ( int failure = 0; failure < 4; ++failure )
    {
        op->fail      = failure == 0;
        op->nonfinite = failure == 1;
        op->singular  = failure == 2;
        report.require( !inverse.set_operator( failure == 3 ? nullptr : op ), "Reject invalid Schur setup" );
        candidate = rhs;
        report.require( !inverse.ready() && !inverse.apply( candidate ), "Failed setup disables stale factors" );
        report.require_close( candidate[0], rhs[0], 0, "Failed action leaves input unchanged" );
    }
    op->fail = op->nonfinite = op->singular = false;
    report.require( inverse.set_operator( op ), "Recover after invalid setup" );
    candidate    = rhs;
    candidate[0] = std::numeric_limits<T>::quiet_NaN();
    report.require( !inverse.apply( candidate ), "Reject nonfinite Schur input" );
    op->scale = T( .25 );
    report.require( inverse.set_operator( op ), "Prepare finite factors before overflow probe" );
    space->assign_scalar( std::numeric_limits<T>::max(), candidate );
    report.require( !inverse.apply( candidate ), "Reject nonfinite triangular solution" );
    report.require( candidate[0] == std::numeric_limits<T>::max(), "Overflow action leaves input unchanged" );
}

}

int main()
{
    nmfd::tests::test_harness report;
    test_inverse<float, 1>( report );
    test_inverse<float, 2>( report );
    test_inverse<double, 1>( report );
    test_inverse<double, 2>( report );
    test_inverse<long double, 2>( report );
    test_block_factor<1>( report );
    test_block_factor<2>( report );
    return report.finish( "Small Schur inverse" );
}
