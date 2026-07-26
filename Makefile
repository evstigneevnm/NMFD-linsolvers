CONFIG_FILE ?= build_configs/config_default.inc

ifeq (,$(wildcard $(CONFIG_FILE)))
$(error Config file $(CONFIG_FILE) does not exist)
endif

include $(CONFIG_FILE)

BUILD_DIR ?= build
SCFD_ROOT ?= contrib/scfd

NMFD_INCLUDE := include
SCFD_INCLUDE := $(SCFD_ROOT)/include
INCLUDES := -I$(NMFD_INCLUDE) -Itest -isystem $(SCFD_INCLUDE)
CXXFLAGS := -std=$(CPPSTD) $(TARGET_CXX) $(WARNING_FLAGS) \
	$(SANITIZER_FLAGS) $(INCLUDES)
LDFLAGS := $(SANITIZER_FLAGS)
PUBLIC_HEADERS := $(shell find include/nmfd -type f -name '*.h')
# This legacy adapter still depends on parent-project common/ headers. It will
# move with the glued-vector operations during the later operations migration.
STANDALONE_PUBLIC_HEADERS := $(filter-out \
	include/nmfd/operations/glued_matrix_operator.h,$(PUBLIC_HEADERS))

TEST_PUBLIC_HEADERS := $(BUILD_DIR)/test_public_headers_cpu.bin
TEST_ORTHOGONALIZATION := $(BUILD_DIR)/test_orthogonalization_cpu.bin
TEST_ARNOLDI := $(BUILD_DIR)/test_arnoldi_cpu.bin
TEST_MONITOR := $(BUILD_DIR)/test_monitor_cpu.bin
TEST_GMRES_CORE := $(BUILD_DIR)/test_gmres_core_cpu.bin
TEST_GMRES_LEGACY := $(BUILD_DIR)/test_gmres_legacy_cpu.bin
TEST_GMRES_MG := $(BUILD_DIR)/test_gmres_mg_cpu.bin

CPU_TESTS := \
	$(TEST_PUBLIC_HEADERS) \
	$(TEST_ORTHOGONALIZATION) \
	$(TEST_ARNOLDI) \
	$(TEST_MONITOR) \
	$(TEST_GMRES_CORE) \
	$(TEST_GMRES_LEGACY) \
	$(TEST_GMRES_MG)

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
	clean \
	check_scfd

all: tests_cpu

check_scfd:
	@test -f "$(SCFD_INCLUDE)/scfd/utils/log.h" || \
		{ echo "SCFD headers not found under $(SCFD_INCLUDE)"; exit 1; }

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_PUBLIC_HEADERS): test/test_public_headers.cpp $(PUBLIC_HEADERS) | $(BUILD_DIR) check_scfd
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_ORTHOGONALIZATION): test/krylov/test_orthogonalization.cpp test/common/cpu_reference_vector_space.h $(PUBLIC_HEADERS) | $(BUILD_DIR) check_scfd
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_ARNOLDI): test/krylov/test_arnoldi.cpp test/common/cpu_reference_vector_space.h $(PUBLIC_HEADERS) | $(BUILD_DIR) check_scfd
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_MONITOR): test/solvers/test_monitor.cpp test/common/cpu_reference_vector_space.h $(PUBLIC_HEADERS) | $(BUILD_DIR) check_scfd
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_GMRES_CORE): test/solvers/test_gmres_core.cpp test/common/cpu_reference_vector_space.h test/common/analytical_operator.h $(PUBLIC_HEADERS) | $(BUILD_DIR) check_scfd
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_GMRES_LEGACY): test/solvers/test_gmres_legacy.cpp test/common/cpu_reference_vector_space.h $(PUBLIC_HEADERS) | $(BUILD_DIR) check_scfd
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

$(TEST_GMRES_MG): test/solvers/test_gmres_mg.cpp test/common/cpu_reference_vector_space.h $(PUBLIC_HEADERS) | $(BUILD_DIR) check_scfd
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) $(LDFLAGS) -o $@

tests_cpu: $(CPU_TESTS)

check_public_headers_cpu: $(TEST_PUBLIC_HEADERS)
	@set -e; \
	for header in $(STANDALONE_PUBLIC_HEADERS); do \
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

check_gmres_core_cpu: $(TEST_GMRES_CORE)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_CORE)

check_gmres_legacy_cpu: $(TEST_GMRES_LEGACY)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_LEGACY)

check_gmres_mg_cpu: $(TEST_GMRES_MG)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_MG)

run_tests_cpu: tests_cpu
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_PUBLIC_HEADERS)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_ORTHOGONALIZATION)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_ARNOLDI)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_MONITOR)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_CORE)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_LEGACY)
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(TEST_GMRES_MG)

clean:
	rm -rf $(BUILD_DIR)
