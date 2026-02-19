#!/bin/bash
# =============================================================================
# replace_video_paths.sh
#
# Replace the hardcoded dataset root prefix in all three RB-FT data files:
#   - Open-R1-Video/data/smarthome_grpo.jsonl
#   - Qwen-VL-Series-Finetune/rebuttal_scripts/data/reasoning_w_answer.json
#   - Qwen-VL-Series-Finetune/rebuttal_scripts/data/sft_label.json
#
# Usage:
#   bash replace_video_paths.sh --new-root /your/dataset/root
#   bash replace_video_paths.sh --new-root /your/dataset/root --dry-run
#   bash replace_video_paths.sh --new-root /your/dataset/root --suffix _modified
#
# The script can be called from any working directory.
# =============================================================================

# --------------------------------------------------------------------------- #
# Resolve directories relative to this script's location
# --------------------------------------------------------------------------- #
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)   # ARR-2026-RBFT-Rebuttal/

# --------------------------------------------------------------------------- #
# Default configuration
# --------------------------------------------------------------------------- #
OLD_PREFIX="/data/meilong/projects/Rational-Bootstrapped-Finetuning/dataset"
NEW_ROOT=""
DRY_RUN=false
SUFFIX=""

# Paths to the three data files (relative to REPO_DIR)
DATA_FILES=(
    "${REPO_DIR}/Open-R1-Video/data/smarthome_grpo.jsonl"
    "${REPO_DIR}/Qwen-VL-Series-Finetune/rebuttal_scripts/data/reasoning_w_answer.json"
    "${REPO_DIR}/Qwen-VL-Series-Finetune/rebuttal_scripts/data/sft_label.json"
)

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
print_usage() {
    echo "Usage: $(basename "$0") --new-root <path> [--old-prefix <path>] [--dry-run] [--suffix <str>]"
    echo ""
    echo "  --new-root    <path>   New dataset root (required)"
    echo "  --old-prefix  <path>   Old prefix to replace (default: ${OLD_PREFIX})"
    echo "  --dry-run              Show changes without writing files"
    echo "  --suffix      <str>    Write to new file with this suffix instead of in-place"
    echo "  -h, --help             Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --new-root)    NEW_ROOT="$2";    shift 2 ;;
        --old-prefix)  OLD_PREFIX="$2";  shift 2 ;;
        --dry-run)     DRY_RUN=true;     shift   ;;
        --suffix)      SUFFIX="$2";      shift 2 ;;
        -h|--help)     print_usage; exit 0 ;;
        *) echo "[error] Unknown argument: $1"; print_usage; exit 1 ;;
    esac
done

if [[ -z "${NEW_ROOT}" ]]; then
    echo "[error] --new-root is required."
    print_usage
    exit 1
fi

# Strip trailing slashes for clean replacement
NEW_ROOT="${NEW_ROOT%/}"
OLD_PREFIX="${OLD_PREFIX%/}"

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #
echo "Old prefix : ${OLD_PREFIX}"
echo "New prefix : ${NEW_ROOT}"
if ${DRY_RUN}; then
    echo "[dry-run]  No files will be written."
elif [[ -n "${SUFFIX}" ]]; then
    echo "[output]   Writing to *${SUFFIX}.{json,jsonl} copies."
else
    echo "[output]   Modifying files in-place."
fi
echo ""

TOTAL_CHANGED=0

for FILE in "${DATA_FILES[@]}"; do
    if [[ ! -f "${FILE}" ]]; then
        echo "  [SKIP]  $(basename "${FILE}")  (file not found: ${FILE})"
        continue
    fi

    # Count how many occurrences exist
    COUNT=$(grep -cF "${OLD_PREFIX}" "${FILE}" 2>/dev/null || echo 0)

    if [[ "${COUNT}" -eq 0 ]]; then
        echo "  [SKIP]  $(basename "${FILE}")  (0 occurrences of old prefix)"
        continue
    fi

    if ${DRY_RUN}; then
        echo "  [DRY]   $(basename "${FILE}")  —  ${COUNT} occurrence(s) would be replaced"
    else
        # Determine output path
        if [[ -n "${SUFFIX}" ]]; then
            BASENAME=$(basename "${FILE}")
            EXT="${BASENAME##*.}"
            STEM="${BASENAME%.*}"
            OUT_FILE="$(dirname "${FILE}")/${STEM}${SUFFIX}.${EXT}"
            cp "${FILE}" "${OUT_FILE}"
            TARGET="${OUT_FILE}"
        else
            TARGET="${FILE}"
        fi

        # Perform in-place substitution (sed -i compatible with GNU and BSD)
        # Escape special characters in OLD_PREFIX for use in sed
        ESCAPED_OLD=$(printf '%s\n' "${OLD_PREFIX}" | sed 's/[[\.*^$()+?{|]/\\&/g')
        ESCAPED_NEW=$(printf '%s\n' "${NEW_ROOT}"   | sed 's/[[\.*^$()+?{|]/\\&/g; s|/|\\/|g')

        sed -i "s|${ESCAPED_OLD}|${NEW_ROOT}|g" "${TARGET}"

        # Verify
        REMAINING=$(grep -cF "${OLD_PREFIX}" "${TARGET}" 2>/dev/null || echo 0)
        if [[ "${REMAINING}" -eq 0 ]]; then
            echo "  [OK]    $(basename "${FILE}")  —  ${COUNT} occurrence(s) replaced"
            if [[ -n "${SUFFIX}" ]]; then
                echo "          -> written to: ${TARGET}"
            fi
        else
            echo "  [WARN]  $(basename "${FILE}")  —  ${REMAINING} occurrence(s) still remain"
        fi
    fi

    TOTAL_CHANGED=$((TOTAL_CHANGED + COUNT))
done

echo ""
if ${DRY_RUN}; then
    echo "Total occurrences that would be replaced: ${TOTAL_CHANGED}"
else
    echo "Total occurrences replaced: ${TOTAL_CHANGED}"
fi
