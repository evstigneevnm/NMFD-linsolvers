#include <nmfd/solvers/bicgstab.h>
#include "common/short_recurrence_tests.h"

int main()
{
    return nmfd::tests::run_short_recurrence_tests<nmfd::solvers::bicgstab, false>( "NMFD BiCGStab" );
}
