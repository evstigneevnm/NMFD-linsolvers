#include <nmfd/detail/algo_hierarchy_creator.h>
#include <nmfd/detail/algo_params_hierarchy.h>
#include <nmfd/detail/algo_utils_hierarchy.h>
#include <nmfd/detail/algo_utils_hierarchy_factory.h>
#include <nmfd/detail/params_hierarchy_dummy.h>
#include <nmfd/detail/solver_traits.h>
#include <nmfd/detail/utils_hierarchy_dummy.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/glued_matrix.h>
#include <nmfd/operations/glued_matrix_operator.h>
#include <nmfd/operations/linalg/host_small_dense_backend.h>
#include <nmfd/operations/matrix_operator.h>
#include <nmfd/operations/vector_factory_base.h>
#include <nmfd/operations/vector_operations_base.h>
#include <nmfd/operations/vector_space_base.h>
#include <nmfd/preconditioners/dummy.h>
#include <nmfd/preconditioners/mg.h>
#include <nmfd/preconditioners/preconditioner_interface.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/bicgstab.h>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/detail/default_prec_creator.h>
#include <nmfd/solvers/detail/dense_operations.h>
#include <nmfd/solvers/detail/monitor_call_wrap.h>
#include <nmfd/solvers/detail/residual_regularization_dummy.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/iter_solver_base.h>
#include <nmfd/solvers/jacobi.h>
#include <nmfd/solvers/krylov/arnoldi.h>
#include <nmfd/solvers/krylov/basis_storage.h>
#include <nmfd/solvers/krylov/operator_apply.h>
#include <nmfd/solvers/krylov/orthogonalization.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/sherman_morrison.h>

#include <iostream>

int main()
{
    std::cout << "NMFD public headers: PASSED\n";
    return 0;
}
