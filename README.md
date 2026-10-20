# NMFD Linear Solvers

Template-based linear solvers, Krylov building blocks, preconditioners, and
linear-space interfaces. The pinned `contrib/nmfd-operations` dependency owns vector
spaces, operations, block operators, small dense algebra, and `vector_wrap`.
SCFD supplies backend primitives and utilities; standalone builds use the direct
`contrib/scfd` pin, so only one SCFD include tree is used.

## Building And Testing

Initialize direct dependencies after cloning:

```bash
git submodule update --init
```

Build and run the complete CPU suite with the default configuration:

```bash
make run_tests_cpu TEST_OMP_NUM_THREADS=8
```

The build is controlled by Make include files. A machine-specific or CI
configuration can be selected without changing the Makefile:

```bash
make check_gmres_core_cpu \
  CONFIG_FILE=build_configs/config_github_ubuntu_gcc.inc \
  BUILD_DIR=build/gcc
```

External checkouts can be selected with `NMFD_OPERATIONS_ROOT=/path/to/nmfd-operations`
and `SCFD_ROOT=/path/to/scfd`. Dependency repositories' own submodules are not
needed for this build. Available standalone checks are:

- `check_public_headers_cpu`
- `check_orthogonalization_cpu`
- `check_arnoldi_cpu`
- `check_monitor_cpu`
- `check_sherman_morrison_cpu`
- `check_direct_solver_cpu`
- `check_gmres_core_cpu`
- `check_gmres_legacy_cpu`
- `check_gmres_mg_cpu`
- `check_operations_cpu`
- `check_bicgstab_cpu`
- `check_bicgstabl_cpu`
- `check_cgs_cpu`
- `check_jacobi_cpu`

Jacobi is also owned by this library. Its historical template argument order
`jacobi<VectorOp, Operator, Preconditioner, Monitor, Log>` and existing
constructors remain valid for Newton and Cahn-Hilliard clients. The iteration is
unchanged: `r = A*x-b`, followed by `x -= P*r`. A missing preconditioner means
identity. Every convergence check uses the original residual; failed operations
or nonfinite values return `false`. The residual workspace is allocated once,
and monitor activation is scoped across success, failure, and exceptions.
`check_jacobi_json_cpu` verifies optional JSON/hierarchy support;
`check_jacobi_cuda` exercises canonical CUDA spaces.

The short-recurrence tests cover real/complex arithmetic, serial/OMP storage,
residual recomputation, breakdown, iteration limits, ownership, and hierarchy
construction. Optional `check_bicgstab_json_cpu`, `check_bicgstabl_json_cpu`, and
`check_cgs_json_cpu` require nlohmann/json headers (system-installed or selected
with `JSON_INCLUDE=/path/to/include`). CUDA equivalents are `check_bicgstab_cuda`,
`check_bicgstabl_cuda`, and `check_cgs_cuda`.

BiCGStab, BiCGStab(L), and CGS retain the production left-preconditioned
recurrences. They use the same template argument order, constructors, solver
base, and shared dependencies as GMRES; loggers remain borrowed `Log*` pointers.
`set_operator(shared_ptr)` binds operator-dependent preconditioners. The
`solve(A, rhs, x)` overload borrows its operator and does not rebind a retained
preconditioner; configure it first when its setup depends on `A`.

`use_precond_resid` selects the monitored norm, `use_real_resid` explicitly
recomputes that residual, and `resid_recalc_freq` periodically replaces the
recurrence residual (counted in iterations for BiCGStab, outer polynomial cycles
for BiCGStab(L)). CGS already recomputes its residual every iteration and defaults
to monitoring the original residual.
Success additionally requires the original-system residual to meet the
unpreconditioned tolerance. Numerical breakdown and failed operator or
preconditioner calls return `false`; invalid configuration throws.
BiCGStab(L) defaults to degree 2 and accepts degrees 1 through 30. Workspace and
small polynomial coefficients are allocated at construction or degree changes,
not in the iteration loop.

`sherman_morrison<Space, Monitor, Log, Operator, Preconditioner, Solver>` ports the
existing ordinary, rank-one, and scalar-bordered elimination. Construct it with
shared `(space, operator, preconditioner)` and an optional borrowed `Log*`.
Its `solve(A, ...)` overloads require that same bound operator; coefficient
updates in place are supported. Zero elimination pivots return `false`; this is
not a general bordered solver. Optional small-pivot scaling is selected anew on
each solve and never modifies input vectors. Success requires the original
system residual, not just the reduced Krylov residual. A small-pivot bordered
solve temporarily tightens inner tolerances and restores the caller's settings,
including on exceptions. Workspaces are allocated once; solves do not copy
shared handles. `check_sherman_morrison_cuda` exercises the same analytical tests
on CUDA.

`direct_solver<Space, Monitor, Log, Operator, Inverse>` uses the same solver base
and shared bindings. The supplied inverse exposes `set_operator(shared_ptr)` and
in-place `apply(x)`, returning `void` or `bool`. It may call dense operations or
an analytical inverse; the solver itself contains no factorization arithmetic.
Bind the operator before either solve overload. One inverse application is
checked against the original-system residual, and `x` is published only on
success. `status()` distinguishes input, inverse, operator, and residual failures.
Bypassing the only inverse returns failure, not an identity solve. CPU and CUDA
direct-solver tests also verify rank-one/scalar-border composition and ownership.

The operations-integration test uses canonical dense, activation-time SCFD, and
glued spaces with real and complex scalars. It verifies true GMRES residuals,
repeated solves, precision modes, and left/right preconditioning settings.

CUDA is optional and never required for CPU targets:

```bash
make check_operations_cuda BUILD_DIR=build/cuda CUDA_ARCH=75
```

GitHub Actions runs every check as an independent GCC and Clang matrix job.
These jobs use committed pins. A separate `latest-dependencies` job tests the
branches declared in `.gitmodules`, logs resolved revisions, and never commits
updated pins. It runs on pushes, pull requests, manual dispatch, and daily from
the repository's default branch. Both modes initialize direct dependencies only.
Integrating projects pass canonical dependency roots instead of using nested copies.
Local builds never fetch or switch dependency checkouts.
AddressSanitizer and UndefinedBehaviorSanitizer run after the CPU matrix passes.
The public-header check compiles each solver header in isolation, without any
legacy exceptions. The combined header test also includes shared operations and
the migrated glued-matrix operator. No parent-project include paths are required.
An optional workflow dispatch runs the CUDA integration test on a self-hosted
runner labeled `cuda`.

# Basic interfaces

## VectorOperations + VectorSpace

TODO

VectorOperations is basically BLAS1 interface.

```
/// Nested types
using scalar_type = ...;
using vector_type = ...;

/// Methods

/// y<-x
void assign(const vector_type& x, vector_type& y) const

void assign_scalar(const scalar_type val, vector_type& x)const 
/// x<-mul_x*x+scalar
void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x)

void assign_random(vector_type& x)const 
/// x<-x*scale
void scale(const scalar_type scale, vector_type& x)
/// returns sum of all elements
[[nodiscard]] scalar_type sum(const vector_type& x)const 
/// returns sum of all elements absolute values
[[nodiscard]] scalar_type asum(const vector_type &x)const
/// returns either all elements are valid numbers (not nans or infs)
[[nodiscard]] bool is_valid_number(const vector_type &x)const
/// returns inner product of vectors (sum of elements products)
[[nodiscard]] scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
/// returns some weighted inner product of vectors (problem dependent, for example L2 integral norm)
[[nodiscard]] scalar_type scalar_prod_l2(const vector_type &x, const vector_type &y)const
/// Returns ANY norm that fits problem convergence
[[nodiscard]] scalar_type norm(const vector_type &x) const
/// norm(x)^2
[[nodiscard]] scalar_type norm_sq(const vector_type &x) const
/// scalar_prod(x,x)
[[nodiscard]] scalar_type norm2_sq(const vector_type &x) const
/// sqrt(scalar_prod(x,x))
[[nodiscard]] scalar_type norm2(const vector_type &x) const
/// scalar_prod_l2(x,x)
[[nodiscard]] scalar_type norm_l2_sq(const vector_type &x) const
/// sqrt(scalar_prod_l2(x,x))
[[nodiscard]] scalar_type norm_l2(const vector_type &x) const
/// May be ommited for now
[[nodiscard]] vector_type at(multivector_type& x, ordinal_type m, ordinal_type k_)
/// y<-x*mul_x+y*mul_y
void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const
/// z<-x*mul_x+y*mul_y+z*mul_z
void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, const scalar_type mul_z, vector_type& z) const
/// y<-x*mul_x
void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y) const
/// z<-x*mul_x+y*mul_y
void assign_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z) const
```

VectorSpace has additional methods that allocate/deallocate vectors meaning that VectorSpace 'knows' 'size' of the vector space:

```
void init_vector(vector_type &v);
void start_use_vector(vector_type &v);
void stop_use_vector(vector_type &v);
void free_vector(vector_type &v);
```

## Operator 

```
class OperatorName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
public:
    void apply(const vector_type &in, vector_type &out);
};
```

In Operator there is a garantee that initial value of out doesnot affects result.

## InplaceOperator 

```
class InplaceOperatorName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
public:
    void apply(vector_type &in_out);
};
```

in_out is both input and output.

## OperatorWithSpaces

```
class OperatorWithSpacesName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
    using vector_space_type = ...; /// VectorSpace
public:
    void apply(const vector_type &in, vector_type &out);
    std::shared_ptr<vector_space_type> get_im_space() const;
    std::shared_ptr<vector_space_type> get_dom_space() const;
};
```

In Operator there is a garantee that initial value of out doesnot affects result.

## InplaceOperatorWithSpaces 

```
class InplaceOperatorWithSpacesName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
    using vector_space_type = ...; /// VectorSpace
public:
    void apply(vector_type &in_out);
    std::shared_ptr<vector_space_type> get_im_space() const;
    std::shared_ptr<vector_space_type> get_dom_space() const;
};
```

in_out is both input and output.

## Preconditioner

Preconditioner is restriction of Operator and InplaceOperator. Plus it has set_operator method for inicialization/reinitialization.

```
class PreconditionerName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
    using operator_type = ...;
    using vector_space_type = ...; /// VectorSpace
public:
    void set_operator(std::shared_ptr<const operator_type> operator);
    void apply(const vector_type &in, vector_type &out);
    void apply(vector_type &in_out);
    std::shared_ptr<vector_space_type> get_im_space() const;
    std::shared_ptr<vector_space_type> get_dom_space() const;
};
```

## Solver 

```
class SolverName
{
public:
    using vector_type = ...;
    using operator_type = ...;
public:
    void set_operator(std::shared_ptr<const operator_type> operator);
    bool solve(const vector_type &rhs, vector_type &res);
};
```

In Solver there is no strict garantee that initial value of res doesnot affects result. For example Solver can use it initial guess.

## PreconditionerWithSpaces, SolverWithSpaces

TODO seems not be often used though

## Coarsening

Creates Restrictor,Prolongator and SystemOperator for next level based on previous SystemOperator.

```
class CoarseningName
{
public:
    using operator_type = ...;
    using restrictor_type = ...;
    using prolongator_type = ...;
public:
    std::tuple<std::shared_ptr<restrictor_type>,std::shared_ptr<prolongator_type>> next_level(const operator_type &op);
    std::shared_ptr<operator_type> coarse_operator(const operator_type &op, const restrictor_type &restrictor, const prolongator_type &prolongator);
    bool coarse_enough(const operator_type &op)const;
};
```

This structure is almost like in amgcl.
Rationale: why do we need two methods for level creation? because of possible rebuild procedure where only operators are updated.

TODO

## HierarchicAlgorithm

Used for automized nested algorithms construction.

```
class HierarchicAlgorithmName
{
public:
    struct params
    {
        params(const std::string &log_prefix = "", const std::string &log_name = "HierarchicAlgorithmName::")
        {
        }
    };
    using params_hierarchy = params;
    struct utils
    {
    };
    using utils_hierarchy = utils;

    HierarchicAlgorithmName(const utils_hierarchy &u, const params_hierarchy &p)
};
```

TODO

# MG class template

mg is Preconditioner.

```
/// SystemOperator is OperatorWithSpaces.
/// Restrictor, Prolongator are Operator.
/// Smoother, CoarseSolver are Preconditioner
/// Coarsening is Coarsening
/// Smoother, CoarseSolver, Coarsening are HierarchicAlgorithm
/// All vector_space_type, vector_type, scalar_type are the same
template
<
    class SystemOperator,
    class Restrictor,
    class Prolongator,
    class Smoother,
    class CoarseSolver,
    class Coarsening
>
class mg
{
};
```

`set_operator(op)` and `update_operator(op)` have intentionally different
contracts:

- `set_operator` accepts a new discrete structure and rebuilds the complete
  hierarchy, including restriction and prolongation operators.
- `update_operator` requires the same structure and preserves hierarchy
  topology. It refreshes the fine and coarse operator values, smoothers, and
  coarse solver numerical data.

A coarsening policy can provide
`same_structure(current, updated)` for validation and
`update_coarse_operator(fine, restrictor, prolongator, current_coarse)` for an
efficient numerical Galerkin update. If the update hook is absent, MG falls
back to `coarse_operator` while retaining the existing transfer operators.
This fallback preserves source compatibility; AMG implementations should
provide the update hook to reuse symbolic sparsity and interpolation data.
The coarsening policy object is retained across numerical updates, allowing it
to own symbolic Galerkin maps and other AMG setup metadata.

Smoothers and coarse solvers may likewise provide `update_operator`. Components
without it fall back to `set_operator`.
