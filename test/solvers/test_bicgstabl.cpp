#include <nmfd/solvers/bicgstabl.h>
#include "common/short_recurrence_tests.h"

int main()
{
    return nmfd::tests::run_short_recurrence_tests<nmfd::solvers::bicgstabl, true>( "NMFD BiCGStab(L)" );
}
