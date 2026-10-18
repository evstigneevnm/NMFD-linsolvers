#ifndef __NMFD_TEST_COMMON_TEST_HARNESS_H__
#define __NMFD_TEST_COMMON_TEST_HARNESS_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

namespace nmfd
{
namespace tests
{

class test_harness
{
public:
    void require( bool condition, const std::string &message )
    {
        ++checks_;
        if ( !condition )
        {
            ++failures_;
            std::cout << "FAIL " << message << '\n';
        }
    }

    template <class Actual, class Expected>
    void require_close( const Actual &actual, const Expected &expected, double tolerance, const std::string &message )
    {
        using std::abs;
        const auto error = abs( actual - expected );
        const auto scale =
            std::max( { 1.0, static_cast<double>( abs( actual ) ), static_cast<double>( abs( expected ) ) } );
        require( static_cast<double>( error ) <= tolerance * scale, message );
    }

    int finish( const std::string &label ) const
    {
        std::cout << "Checks: " << checks_ << ", failures: " << failures_ << '\n';
        if ( failures_ != 0 )
        {
            std::cout << label << ": FAILED\n";
            return EXIT_FAILURE;
        }
        std::cout << label << ": PASSED\n";
        return EXIT_SUCCESS;
    }

    std::size_t checks() const
    {
        return checks_;
    }

    std::size_t failures() const
    {
        return failures_;
    }

private:
    std::size_t checks_   = 0;
    std::size_t failures_ = 0;
};

} // namespace tests
} // namespace nmfd

#endif
