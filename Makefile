CONFIG_FILE ?= build_configs/config_default.inc

ifeq (,$(wildcard $(CONFIG_FILE)))
$(error Config file $(CONFIG_FILE) does not exist)
endif

include $(CONFIG_FILE)

BUILD_DIR ?= build
NMFD_OPERATIONS_ROOT ?= contrib/nmfd-operations
SCFD_ROOT ?= contrib/scfd
CUDA_ROOT ?= /usr/local/cuda
NVCC ?= $(CUDA_ROOT)/bin/nvcc
CUDA_ARCH ?= 75
NVCCFLAGS ?= -std=$(CPPSTD) -O2 --extended-lambda -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)

NMFD_INCLUDE := include
NMFD_OPERATIONS_INCLUDE := $(NMFD_OPERATIONS_ROOT)/include
SCFD_INCLUDE := $(SCFD_ROOT)/include
INCLUDES := -I$(NMFD_INCLUDE) -I$(NMFD_OPERATIONS_INCLUDE) -Itest -isystem $(SCFD_INCLUDE)
CXXFLAGS := -std=$(CPPSTD) $(TARGET_CXX) $(WARNING_FLAGS) \
	$(SANITIZER_FLAGS) $(INCLUDES)
LDFLAGS := $(SANITIZER_FLAGS)
PUBLIC_HEADERS := $(shell find include/nmfd -type f -name '*.h')
OPERATIONS_HEADERS := $(shell find $(NMFD_OPERATIONS_INCLUDE) -type f \( -name '*.h' -o -name '*.cuh' \) 2>/dev/null)
TEST_HEADERS := $(PUBLIC_HEADERS) $(OPERATIONS_HEADERS)

TEST_PUBLIC_HEADERS := $(BUILD_DIR)/test_public_headers_cpu.bin
TEST_ORTHOGONALIZATION := $(BUILD_DIR)/test_orthogonalization_cpu.bin
TEST_ARNOLDI := $(BUILD_DIR)/test_arnoldi_cpu.bin
TEST_MONITOR := $(BUILD_DIR)/test_monitor_cpu.bin
TEST_SHERMAN_MORRISON := $(BUILD_DIR)/test_sherman_morrison_cpu.bin
TEST_DENSE1_EXTENDED := $(BUILD_DIR)/test_dense1_extended_solver_cpu.bin
TEST_DIRECT := $(BUILD_DIR)/test_direct_solver_cpu.bin
TEST_GMRES_CORE := $(BUILD_DIR)/test_gmres_core_cpu.bin
TEST_GMRES_LEGACY := $(BUILD_DIR)/test_gmres_legacy_cpu.bin
TEST_GMRES_MG := $(BUILD_DIR)/test_gmres_mg_cpu.bin
TEST_OPERATIONS := $(BUILD_DIR)/test_operations_integration_cpu.bin
TEST_OPERATIONS_CUDA := $(BUILD_DIR)/test_operations_integration_cuda.bin
TEST_JACOBI := $(BUILD_DIR)/test_jacobi_cpu.bin
TEST_JACOBI_BACKEND := $(BUILD_DIR)/test_jacobi_backend_cpu.bin
SHORT_SOLVERS := bicgstab bicgstabl cgs
SHORT_CPU_TESTS := $(addprefix $(BUILD_DIR)/test_,$(addsuffix _cpu.bin,$(SHORT_SOLVERS)))
SHORT_BACKEND_TESTS := $(addprefix $(BUILD_DIR)/test_,$(addsuffix _backend_cpu.bin,$(SHORT_SOLVERS)))
JSON_INCLUDE ?=
JSON_FLAGS := $(if $(JSON_INCLUDE),-I$(JSON_INCLUDE))
BOOST_ROOT ?=
BOOST_FLAGS := $(if $(BOOST_ROOT),-I$(BOOST_ROOT)/include)

CPU_TESTS := \
	$(TEST_PUBLIC_HEADERS) \
	$(TEST_ORTHOGONALIZATION) \
	$(TEST_ARNOLDI) \
	$(TEST_MONITOR) $(TEST_SHERMAN_MORRISON) $(TEST_DENSE1_EXTENDED) $(TEST_DIRECT) \
	$(TEST_GMRES_CORE) \
	$(TEST_GMRES_LEGACY) \
	$(TEST_GMRES_MG) \
	$(TEST_OPERATIONS) $(SHORT_CPU_TESTS) $(SHORT_BACKEND_TESTS) $(TEST_JACOBI) $(TEST_JACOBI_BACKEND)

.PHONY: \
	all \
	tests_cpu \
	run_tests_cpu \
	check_public_headers_cpu \
	check_orthogonalization_cpu \
	check_arnoldi_cpu \
	check_monitor_cpu \
	check_gmres_core_cpu \
	check_gmres_legacy_cpu \
	check_gmres_mg_cpu \
	check_operations_cpu \
	check_operations_cuda \
	clean \
	check_dependencies

all: tests_cpu

$(BUILD_DIR)/test_multiprecision_cpu.bin: test/solvers/test_multiprecision.cpp $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) $(BOOST_FLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

.PHONY: check_multiprecision_cpu
check_multiprecision_cpu: $(BUILD_DIR)/test_multiprecision_cpu.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $<

$(BUILD_DIR)/test_%_cpu.bin: test/solvers/test_%.cpp test/common/short_recurrence_tests.h $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(BUILD_DIR)/test_%_json_cpu.bin: test/solvers/test_%.cpp test/common/short_recurrence_tests.h $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) $(JSON_FLAGS) -DNMFD_ENABLE_NLOHMANN $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(SHORT_BACKEND_TESTS): $(BUILD_DIR)/test_%_backend_cpu.bin: test/solvers/test_short_recurrence_backend.cpp $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) -DNMFD_TEST_SOLVER=$* $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_JACOBI_BACKEND): test/solvers/test_short_recurrence_backend.cpp $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) -DNMFD_TEST_JACOBI $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

check_jacobi_cpu: $(TEST_JACOBI) $(TEST_JACOBI_BACKEND)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_JACOBI)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_JACOBI_BACKEND)

check_jacobi_json_cpu: $(BUILD_DIR)/test_jacobi_json_cpu.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $<

check_jacobi_cuda: $(BUILD_DIR)/test_jacobi_cuda.bin
	$<

$(addprefix check_,$(addsuffix _cpu,$(SHORT_SOLVERS))): check_%_cpu: $(BUILD_DIR)/test_%_cpu.bin $(BUILD_DIR)/test_%_backend_cpu.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(BUILD_DIR)/test_$*_cpu.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(BUILD_DIR)/test_$*_backend_cpu.bin

$(addprefix check_,$(addsuffix _json_cpu,$(SHORT_SOLVERS))): check_%_json_cpu: $(BUILD_DIR)/test_%_json_cpu.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $<

$(BUILD_DIR)/test_%_cuda.bin: test/solvers/test_short_recurrence_backend.cpp $(TEST_HEADERS) $(BUILD_DIR)/operations_cuda_reductions.o | check_dependencies
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DTEST_OPERATIONS_CUDA -DNMFD_TEST_SOLVER=$* $(if $(filter jacobi,$*),-DNMFD_TEST_JACOBI) -x cu -c $< -o $(BUILD_DIR)/test_$*_cuda.o
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/test_$*_cuda.o $(BUILD_DIR)/operations_cuda_reductions.o -lcublas -o $@

$(addprefix check_,$(addsuffix _cuda,$(SHORT_SOLVERS))): check_%_cuda: $(BUILD_DIR)/test_%_cuda.bin
	$<

check_dependencies:
	@test -f "$(NMFD_OPERATIONS_INCLUDE)/nmfd/operations/dense_vector_space.h" || \
		{ echo "NMFD-operations headers not found under $(NMFD_OPERATIONS_INCLUDE); initialize recursive submodules or set NMFD_OPERATIONS_ROOT"; exit 1; }
	@test -f "$(SCFD_INCLUDE)/scfd/utils/log.h" || \
		{ echo "SCFD headers not found under $(SCFD_INCLUDE)"; exit 1; }

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_PUBLIC_HEADERS): test/test_public_headers.cpp $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_ORTHOGONALIZATION): test/krylov/test_orthogonalization.cpp test/common/cpu_reference_vector_space.h $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_ARNOLDI): test/krylov/test_arnoldi.cpp test/common/cpu_reference_vector_space.h $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_MONITOR): test/solvers/test_monitor.cpp test/common/cpu_reference_vector_space.h $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_GMRES_CORE): test/solvers/test_gmres_core.cpp test/common/cpu_reference_vector_space.h test/common/analytical_operator.h $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_GMRES_LEGACY): test/solvers/test_gmres_legacy.cpp test/common/cpu_reference_vector_space.h $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_GMRES_MG): test/solvers/test_gmres_mg.cpp test/common/cpu_reference_vector_space.h $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_OPERATIONS): test/solvers/test_operations_integration.cpp test/common/test_harness.h $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(BUILD_DIR)/operations_cuda_reductions.o: $(NMFD_OPERATIONS_INCLUDE)/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_kernels.cu $(TEST_HEADERS) | $(BUILD_DIR) check_dependencies
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(TEST_OPERATIONS_CUDA): test/solvers/test_operations_integration.cpp test/common/test_harness.h $(TEST_HEADERS) $(BUILD_DIR)/operations_cuda_reductions.o | check_dependencies
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DTEST_OPERATIONS_CUDA -x cu -c $< -o $(BUILD_DIR)/test_operations_integration_cuda.o
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/test_operations_integration_cuda.o $(BUILD_DIR)/operations_cuda_reductions.o -lcublas -o $@

tests_cpu: $(CPU_TESTS)

check_public_headers_cpu: $(TEST_PUBLIC_HEADERS)
	@set -e; \
	for header in $(PUBLIC_HEADERS); do \
		include_name=$${header#include/}; \
		printf '#include <%s>\nint main() { return 0; }\n' "$$include_name" | \
			$(HOST_CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -x c++ -fsyntax-only -; \
	done
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_PUBLIC_HEADERS)

check_orthogonalization_cpu: $(TEST_ORTHOGONALIZATION)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_ORTHOGONALIZATION)

check_arnoldi_cpu: $(TEST_ARNOLDI)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_ARNOLDI)

check_monitor_cpu: $(TEST_MONITOR)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_MONITOR)

check_sherman_morrison_cpu: $(TEST_SHERMAN_MORRISON)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $<

check_dense1_extended_solver_cpu: $(TEST_DENSE1_EXTENDED)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $<

check_direct_solver_cpu: $(TEST_DIRECT)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $<

$(BUILD_DIR)/test_direct_solver_cuda.bin: test/solvers/test_direct_solver.cpp $(TEST_HEADERS) $(BUILD_DIR)/operations_cuda_reductions.o | check_dependencies
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DTEST_OPERATIONS_CUDA -x cu -c $< -o $(BUILD_DIR)/test_direct_solver_cuda.o
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/test_direct_solver_cuda.o $(BUILD_DIR)/operations_cuda_reductions.o -lcublas -lcusolver -o $@

check_direct_solver_cuda: $(BUILD_DIR)/test_direct_solver_cuda.bin
	$<

$(BUILD_DIR)/test_sherman_morrison_cuda.bin: test/solvers/test_sherman_morrison.cpp $(TEST_HEADERS) $(BUILD_DIR)/operations_cuda_reductions.o | check_dependencies
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DTEST_OPERATIONS_CUDA -x cu -c $< -o $(BUILD_DIR)/test_sherman_morrison_cuda.o
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/test_sherman_morrison_cuda.o $(BUILD_DIR)/operations_cuda_reductions.o -lcublas -o $@

check_sherman_morrison_cuda: $(BUILD_DIR)/test_sherman_morrison_cuda.bin
	$<

check_gmres_core_cpu: $(TEST_GMRES_CORE)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_CORE)

check_gmres_legacy_cpu: $(TEST_GMRES_LEGACY)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_LEGACY)

check_gmres_mg_cpu: $(TEST_GMRES_MG)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_MG)

check_operations_cpu: $(TEST_OPERATIONS)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_OPERATIONS)

check_operations_cuda: $(TEST_OPERATIONS_CUDA)
	$(TEST_OPERATIONS_CUDA)

run_tests_cpu: tests_cpu
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_DIRECT)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_SHERMAN_MORRISON)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_DENSE1_EXTENDED)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_PUBLIC_HEADERS)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_ORTHOGONALIZATION)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_ARNOLDI)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_MONITOR)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_CORE)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_LEGACY)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_MG)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_OPERATIONS)
	@set -e; for test in $(SHORT_CPU_TESTS) $(SHORT_BACKEND_TESTS); do OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $$test; done
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_JACOBI)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_JACOBI_BACKEND)

.PHONY: $(addprefix check_,$(addsuffix _cpu,$(SHORT_SOLVERS))) $(addprefix check_,$(addsuffix _json_cpu,$(SHORT_SOLVERS))) $(addprefix check_,$(addsuffix _cuda,$(SHORT_SOLVERS)))
.PHONY: check_jacobi_cpu check_jacobi_json_cpu check_jacobi_cuda
.PHONY: check_sherman_morrison_cpu check_sherman_morrison_cuda
.PHONY: check_dense1_extended_solver_cpu
.PHONY: check_direct_solver_cpu check_direct_solver_cuda

clean:
	rm -rf $(BUILD_DIR)
