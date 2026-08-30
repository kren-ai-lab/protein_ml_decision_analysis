#!/usr/bin/env bash

set -euo pipefail

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${ANALYSIS_DIR}/../.." && pwd)"
CONFIG_FILE="${ANALYSIS_DIR}/matched_size_config.json"
SEARCH_SCRIPT="${ANALYSIS_DIR}/find_matched_reductions.py"
BENCHMARK_SCRIPT="${ANALYSIS_DIR}/compare_clusters_to_mmseqs2.py"
ACTION="${1:-all}"

PYTHON_BIN=""
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
for candidate in \
    "/home/nicole/miniconda3/envs/building_ml_models/bin/python" \
    "/home/nicole/.conda/envs/building_ml_models/bin/python" \
    "/home/nicole/miniforge3/envs/building_ml_models/bin/python"
do
    if [[ -z "${PYTHON_BIN}" && -x "${candidate}" ]]; then
        PYTHON_BIN="${candidate}"
    fi
done
if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: no usable Python interpreter was found." >&2
    exit 1
fi

BIOSIEVE_EXEC="$(dirname "${PYTHON_BIN}")/biosieve"
if [[ ! -x "${BIOSIEVE_EXEC}" ]]; then
    BIOSIEVE_EXEC="$(command -v biosieve || true)"
fi

for required_file in \
    "${CONFIG_FILE}" \
    "${SEARCH_SCRIPT}" \
    "${BENCHMARK_SCRIPT}"
do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: missing required file: ${required_file}" >&2
        exit 1
    fi
done

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

validate_inputs() {
    "${PYTHON_BIN}" -c \
        "import numpy, pandas, sklearn, yaml; print('Python dependencies OK')"

    "${PYTHON_BIN}" "${SEARCH_SCRIPT}" \
        --config "${CONFIG_FILE}" \
        --project-root "${PROJECT_DIR}" \
        --validate-only
}

run_reductions() {
    if [[ -z "${BIOSIEVE_EXEC}" || ! -x "${BIOSIEVE_EXEC}" ]]; then
        echo "ERROR: biosieve was not found in the selected Python environment." >&2
        exit 1
    fi

    "${PYTHON_BIN}" "${SEARCH_SCRIPT}" \
        --config "${CONFIG_FILE}" \
        --project-root "${PROJECT_DIR}" \
        --biosieve-exec "${BIOSIEVE_EXEC}"
}

run_one_space() {
    local space_name="$1"
    if [[ -z "${BIOSIEVE_EXEC}" || ! -x "${BIOSIEVE_EXEC}" ]]; then
        echo "ERROR: biosieve was not found in the selected Python environment." >&2
        exit 1
    fi

    "${PYTHON_BIN}" "${SEARCH_SCRIPT}" \
        --config "${CONFIG_FILE}" \
        --project-root "${PROJECT_DIR}" \
        --space "${space_name}" \
        --biosieve-exec "${BIOSIEVE_EXEC}"
}

run_benchmark() {
    "${PYTHON_BIN}" "${BENCHMARK_SCRIPT}" \
        --config "${CONFIG_FILE}" \
        --project-root "${PROJECT_DIR}"
}

echo "Python: ${PYTHON_BIN}"
echo "Action: ${ACTION}"

case "${ACTION}" in
    validate)
        validate_inputs
        ;;
    all)
        validate_inputs
        run_reductions
        run_benchmark
        ;;
    benchmark)
        run_benchmark
        ;;
    ankh2_ext1|esm2_t6_8M_UR50D|esmc_300m|mistral_Prot_v1_134M|prot_bert|prot_t5_xl_uniref50|onehot)
        validate_inputs
        run_one_space "${ACTION}"
        ;;
    *)
        echo "ERROR: unknown action or representation: ${ACTION}" >&2
        echo "Use: validate, all, benchmark, or a configured representation name." >&2
        exit 1
        ;;
esac

echo "Completed action: ${ACTION}"
