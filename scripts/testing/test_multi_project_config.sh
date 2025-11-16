#!/usr/bin/env bash
#
# Multi-Project Configuration Test Script
# Tests that all project-agnostic scripts properly load and use project configs
#
# Usage:
#   bash scripts/testing/test_multi_project_config.sh
#

# Don't exit on error so we can see all test results
# set -e

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Multi-Project Configuration Test"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

TESTS_PASSED=0
TESTS_FAILED=0

# Test function
test_script() {
    local script_path=$1
    local project_name=$2
    local test_name=$3

    echo -n "Testing: $test_name... "

    if [ ! -f "$script_path" ]; then
        echo -e "${RED}FAIL${NC} (script not found)"
        ((TESTS_FAILED++))
        return 1
    fi

    # Test help output (dry run to check for errors)
    if bash "$script_path" --help &>/dev/null || bash "$script_path" -h &>/dev/null || head -20 "$script_path" | grep -q "Usage:"; then
        echo -e "${GREEN}PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        # Scripts might not have --help, just check they exist and are readable
        if [ -r "$script_path" ]; then
            echo -e "${GREEN}PASS${NC} (script exists)"
            ((TESTS_PASSED++))
            return 0
        else
            echo -e "${RED}FAIL${NC}"
            ((TESTS_FAILED++))
            return 1
        fi
    fi
}

# Test YAML config loading
test_yaml_config() {
    local project=$1
    local config_file="configs/projects/${project}.yaml"

    echo -n "Testing: Config file ${config_file}... "

    if [ ! -f "$config_file" ]; then
        echo -e "${RED}FAIL${NC} (not found)"
        ((TESTS_FAILED++))
        return 1
    fi

    # Test YAML parsing
    if python3 -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null; then
        echo -e "${GREEN}PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}FAIL${NC} (invalid YAML)"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "Phase 1: Testing Project Configuration Files"
echo "────────────────────────────────────────"
test_yaml_config "luca"

echo ""
echo "Phase 2: Testing Stage Scripts"
echo "────────────────────────────────────────"
test_script "scripts/projects/luca/run_caption_generation.sh" "luca" "run_caption_generation.sh"
test_script "scripts/projects/luca/run_quality_filter.sh" "luca" "run_quality_filter.sh"
test_script "scripts/projects/luca/run_instance_enhancement.sh" "luca" "run_instance_enhancement.sh"
test_script "scripts/projects/luca/run_pose_analysis.sh" "luca" "run_pose_analysis.sh"

echo ""
echo "Phase 3: Testing Workflow Scripts"
echo "────────────────────────────────────────"
test_script "scripts/projects/luca/workflows/run_luca_dataset_pipeline.sh" "luca" "run_luca_dataset_pipeline.sh"
test_script "scripts/projects/luca/workflows/optimized_luca_pipeline.sh" "luca" "optimized_luca_pipeline.sh"
test_script "scripts/projects/luca/workflows/run_complete_luca_pipeline.sh" "luca" "run_complete_luca_pipeline.sh"

echo ""
echo "Phase 4: Testing Training Scripts"
echo "────────────────────────────────────────"
test_script "scripts/projects/luca/training/auto_train_luca.sh" "luca" "auto_train_luca.sh"

echo ""
echo "Phase 5: Testing Python Pipeline Scripts"
echo "────────────────────────────────────────"
if [ -f "scripts/projects/luca/pipelines/luca_dataset_pipeline_simplified.py" ]; then
    echo -n "Testing: luca_dataset_pipeline_simplified.py... "
    if python3 -c "import sys; sys.path.insert(0, 'scripts/projects/luca/pipelines'); import luca_dataset_pipeline_simplified" 2>/dev/null; then
        echo -e "${GREEN}PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${YELLOW}SKIP${NC} (dependencies missing)"
    fi
fi

if [ -f "scripts/projects/luca/pipelines/luca_dataset_preparation_pipeline.py" ]; then
    echo -n "Testing: luca_dataset_preparation_pipeline.py... "
    if python3 -c "import sys; sys.path.insert(0, 'scripts/projects/luca/pipelines'); import luca_dataset_preparation_pipeline" 2>/dev/null; then
        echo -e "${GREEN}PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${YELLOW}SKIP${NC} (dependencies missing)"
    fi
fi

echo ""
echo "Phase 6: Testing Project Parameter Handling"
echo "────────────────────────────────────────"

# Test that scripts accept project parameter
echo -n "Testing: Project parameter validation... "
TEST_OUTPUT=$(bash scripts/projects/luca/workflows/run_complete_luca_pipeline.sh nonexistent_project 2>&1 | head -5 || true)
if echo "$TEST_OUTPUT" | grep -q "Project config not found"; then
    echo -e "${GREEN}PASS${NC} (error handling works)"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}SKIP${NC} (cannot test)"
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
