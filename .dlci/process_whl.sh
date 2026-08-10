#!/bin/bash

# process_whl.sh - Wheel package processing script
# Process wheel packages by adding SDK timestamp and repackaging
#
# Adapted from the vLLM DengLin CI process_whl.sh. The generic version-pattern
# logic is preserved; a sglang-specific pattern (setuptools-scm dev version with
# a +g<hash> local segment, no +cu117) is added so the sglang wheel also gets
# the .sdk<timestamp> suffix.

set -e
set -o pipefail

# Parameters
OUTPUT_DIR=""
SDK_TAG="${SDK_TAG:-}"
GIT_TAG=""
BUILD_ID="${BUILD_ID:-local}"
VERBOSE=false
DRY_RUN=false

# Display usage
usage() {
    echo "Usage: $0 [options] <wheel_package_path>"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR    Output directory (required)"
    echo "  --sdk-tag TAG       SDK tag (optional, uses env var SDK_TAG by default)"
    echo "  --git-tag TAG       Git tag"
    echo "  --build-id ID       Build ID (optional, uses env var BUILD_ID by default)"
    echo "  --dry-run           Dry run mode, only show what would be done"
    echo "  --verbose           Verbose output"
    echo "  --help              Show help"
    echo ""
    echo "Environment variables:"
    echo "  SDK_TAG             SDK tag from environment"
    echo "  BUILD_ID            Jenkins build ID or custom identifier"
    echo ""
    echo "Examples:"
    echo "  $0 --output-dir ./output sglang-0.5.16.dev560+g692b5c460-py3-none-any.whl"
    echo "  $0 --output-dir /tmp/build_123 --build-id 123 flash_attn-2.8.4+dl5.torch291-cp312-cp312-manylinux_2_28_x86_64.whl"
    exit 1
}

# Logging functions
info() { echo "[INFO] $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }
verbose() { [ "$VERBOSE" = true ] && echo "[VERBOSE] $*" >&2; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --sdk-tag) SDK_TAG="$2"; shift 2 ;;
        --git-tag) GIT_TAG="$2"; shift 2 ;;
        --build-id) BUILD_ID="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --help) usage ;;
        -*) error "Unknown option: $1" ;;
        *) INPUT_WHL="$1"; shift ;;
    esac
done

# Validate parameters
[ -z "$INPUT_WHL" ] && error "Missing wheel package path"
[ -z "$OUTPUT_DIR" ] && error "Missing output directory (--output-dir)"

# DEBUG: Always show the input wheel path for debugging
info "DEBUG: Input wheel path received: $INPUT_WHL"

# Handle wildcard in wheel path - select the most recent file if multiple matches
if [[ "$INPUT_WHL" == *"*"* ]]; then
    info "DEBUG: Wildcard detected in input path: $INPUT_WHL"
    verbose "Wildcard detected in input path: $INPUT_WHL"

    # Get the directory and pattern from the wildcard path
    WHL_DIR=$(dirname "$INPUT_WHL")
    WHL_PATTERN=$(basename "$INPUT_WHL")

    info "DEBUG: Searching in directory: $WHL_DIR for pattern: $WHL_PATTERN"

    # Find all matching files first
    MATCHING_FILES=$(find "$WHL_DIR" -maxdepth 1 -name "$WHL_PATTERN" -type f 2>/dev/null)

    if [ -z "$MATCHING_FILES" ]; then
        error "No files found matching pattern: $INPUT_WHL"
    fi

    # Show all matching files for debugging (using ls -lt for cross-platform compatibility)
    info "DEBUG: All matching files found:"
    for file in $MATCHING_FILES; do
        # Use ls -l for timestamp display (cross-platform compatible)
        ls -l "$file" | awk '{print $6, $7, $8, $9}' | sed 's|^|[INFO] DEBUG:   |' >&2
    done

    # Use ls -t to sort by modification time (newest first), which is cross-platform
    LATEST_FILE=$(ls -t $MATCHING_FILES 2>/dev/null | head -1)

    if [ -z "$LATEST_FILE" ] || [ ! -f "$LATEST_FILE" ]; then
        error "Failed to select the most recent file"
    fi

    info "Selected most recent file: $(basename "$LATEST_FILE")"
    verbose "Selected file: $LATEST_FILE"

    INPUT_WHL="$LATEST_FILE"
else
    info "DEBUG: No wildcard detected, using specific file path: $INPUT_WHL"
fi

[ ! -f "$INPUT_WHL" ] && error "Wheel package not found: $INPUT_WHL"

# Create output directory structure based on BUILD_ID
if [ "$BUILD_ID" != "local" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/build_${BUILD_ID}"
    info "Using build-specific output directory: ${OUTPUT_DIR}"
fi

# Create output directory if it doesn't exist
[ "$DRY_RUN" != true ] && mkdir -p "$OUTPUT_DIR" || info "[DRY-RUN] Would create directory: $OUTPUT_DIR"

# Convert to absolute paths
INPUT_WHL=$(realpath "$INPUT_WHL")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

# Extract basic information
WHL_NAME=$(basename "$INPUT_WHL")
info "Processing wheel package: $WHL_NAME"

# Get current git tag info, extract DL identifier
if [ -z "$GIT_TAG" ]; then
    GIT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "")
fi

# Extract DL identifier - updated patterns
DL_IDENTIFIER=""
if [ -n "$GIT_TAG" ]; then
    info "Detected git tag: $GIT_TAG"
    # Pattern 1: sglang/vLLM style - v0.9.0+dl-main-28 or
    # v0.21.0+dl-release-24 -> dl28 or dl24
    if [[ "$GIT_TAG" =~ v[0-9]+\.[0-9]+\.[0-9]+\+dl-(main|release)-([0-9]+) ]]; then
        DL_NUMBER="${BASH_REMATCH[2]}"
        DL_IDENTIFIER="dl${DL_NUMBER}"
        info "Extracted DL identifier from tag: $DL_IDENTIFIER"
    # Pattern 2: PyTorch style - dl-release-v2.5.1-43 -> dl43
    elif [[ "$GIT_TAG" =~ dl-release-v[0-9]+\.[0-9]+\.[0-9]+-([0-9]+) ]]; then
        DL_NUMBER="${BASH_REMATCH[1]}"
        DL_IDENTIFIER="dl${DL_NUMBER}"
        info "Extracted DL identifier from PyTorch tag: $DL_IDENTIFIER"
    else
        verbose "Git tag doesn't match known DL identifier patterns"
    fi
else
    verbose "No Git tag detected"
fi

# Extract SDK timestamp (supports current and future years, e.g. 202603121743)
SDK_TIMESTAMP=""
if [ -n "$SDK_TAG" ]; then
    SDK_TIMESTAMP=$(echo "$SDK_TAG" | grep -oP '20[0-9]{6,10}' | head -1)
    [ -z "$SDK_TIMESTAMP" ] && error "Cannot extract timestamp from SDK_TAG: $SDK_TAG"
    verbose "Extracted SDK timestamp: $SDK_TIMESTAMP"
fi

# If no SDK timestamp, just copy the original file
if [ -z "$SDK_TIMESTAMP" ]; then
    info "No SDK_TAG provided, using original wheel package"
    if [ "$DRY_RUN" = true ]; then
        info "[DRY-RUN] Would copy $INPUT_WHL to $OUTPUT_DIR/"
    else
        cp "$INPUT_WHL" "$OUTPUT_DIR/"
    fi
    echo "$OUTPUT_DIR/$WHL_NAME"
    exit 0
fi

# Enhanced version pattern matching for different package types
generate_new_wheel_name() {
    local whl_name="$1"
    local new_name="$whl_name"

    # Pattern A (sglang setuptools-scm): pkg-X.Y.Z[.devN]+g<hash>-suffix  (no cu)
    if [[ "$whl_name" =~ ^([^-]+)-([0-9]+\.[0-9]+\.[0-9]+(\.[0-9A-Za-z]+)*)\+g([0-9a-f]+)-(.+)$ ]]; then
        local pkg_name="${BASH_REMATCH[1]}"
        local version="${BASH_REMATCH[2]}"
        local git_hash="${BASH_REMATCH[4]}"
        local suffix="${BASH_REMATCH[5]}"
        new_name="${pkg_name}-${version}+g${git_hash}.sdk${SDK_TIMESTAMP}-${suffix}"

    # Pattern 1: vllm style - xxx+g{hash}.cu{version}
    elif [[ "$whl_name" =~ ^([^-]+)-([0-9]+\.[0-9]+\.[0-9]+[^+]*)\+g([0-9a-f]+)\.cu([0-9]+)-(.+)$ ]]; then
        local pkg_name="${BASH_REMATCH[1]}"
        local version="${BASH_REMATCH[2]}"
        local git_hash="${BASH_REMATCH[3]}"
        local cuda_ver="${BASH_REMATCH[4]}"
        local suffix="${BASH_REMATCH[5]}"
        new_name="${pkg_name}-${version}+g${git_hash}.sdk${SDK_TIMESTAMP}.cu${cuda_ver}-${suffix}"

    # Pattern 2: PyTorch with git hash - xxx+git{hash}
    elif [[ "$whl_name" =~ ^([^-]+)-([0-9]+\.[0-9]+\.[0-9]+[^+]*)\+git([0-9a-f]+)-(.+)$ ]]; then
        local pkg_name="${BASH_REMATCH[1]}"
        local version="${BASH_REMATCH[2]}"
        local git_hash="${BASH_REMATCH[3]}"
        local suffix="${BASH_REMATCH[4]}"
        local dl_part=""
        [ -n "$DL_IDENTIFIER" ] && dl_part=".${DL_IDENTIFIER}"
        new_name="${pkg_name}-${version}+git${git_hash}${dl_part}.sdk${SDK_TIMESTAMP}-${suffix}"

    # Pattern 3: Standard style with cuda - xxx+cu{version}
    elif [[ "$whl_name" =~ ^([^-]+)-([0-9]+\.[0-9]+\.[0-9]+[^+]*)\+cu([0-9]+)-(.+)$ ]]; then
        local pkg_name="${BASH_REMATCH[1]}"
        local version="${BASH_REMATCH[2]}"
        local cuda_ver="${BASH_REMATCH[3]}"
        local suffix="${BASH_REMATCH[4]}"
        local dl_part=""
        [ -n "$DL_IDENTIFIER" ] && dl_part=".${DL_IDENTIFIER}"
        new_name="${pkg_name}-${version}+cu${cuda_ver}${dl_part}.sdk${SDK_TIMESTAMP}-${suffix}"

    # Pattern 4: Dev versions - xxx.dev{date}
    elif [[ "$whl_name" =~ ^([^-]+)-([0-9]+\.[0-9]+\.[0-9]+\.dev[0-9]+)-(.+)$ ]]; then
        local pkg_name="${BASH_REMATCH[1]}"
        local version="${BASH_REMATCH[2]}"
        local suffix="${BASH_REMATCH[3]}"
        new_name="${pkg_name}-${version}.sdk${SDK_TIMESTAMP}-${suffix}"

    # Pattern 5: Post-release versions (e.g., flash_attn-2.7.4.post1)
    elif [[ "$whl_name" =~ ^([^-]+)-([0-9]+\.[0-9]+\.[0-9]+\.post[0-9]+)-(.+)$ ]]; then
        local pkg_name="${BASH_REMATCH[1]}"
        local version="${BASH_REMATCH[2]}"
        local suffix="${BASH_REMATCH[3]}"
        new_name="${pkg_name}-${version}.sdk${SDK_TIMESTAMP}-${suffix}"

    # Pattern 6: Standard versions without cuda
    elif [[ "$whl_name" =~ ^([^-]+)-([0-9]+\.[0-9]+\.[0-9]+)-(.+)$ ]]; then
        local pkg_name="${BASH_REMATCH[1]}"
        local version="${BASH_REMATCH[2]}"
        local suffix="${BASH_REMATCH[3]}"
        new_name="${pkg_name}-${version}.sdk${SDK_TIMESTAMP}-${suffix}"

    else
        verbose "No matching pattern for wheel name format"
    fi

    echo "$new_name"
}

# Generate new package name
NEW_WHL_NAME=$(generate_new_wheel_name "$WHL_NAME")

# Check if processing was successful
if [ "$WHL_NAME" = "$NEW_WHL_NAME" ]; then
    info "Cannot process package name format, using original name"
    if [ "$DRY_RUN" = true ]; then
        info "[DRY-RUN] Would copy $INPUT_WHL to $OUTPUT_DIR/"
    else
        cp "$INPUT_WHL" "$OUTPUT_DIR/"
    fi
    echo "$OUTPUT_DIR/$WHL_NAME"
    exit 0
fi

info "New package name: $NEW_WHL_NAME"

# Create working directory
WORK_DIR="${OUTPUT_DIR}/whl_work_$$"
if [ "$DRY_RUN" = true ]; then
    info "[DRY-RUN] Would create working directory: $WORK_DIR"
    info "[DRY-RUN] Would process and create: $OUTPUT_DIR/$NEW_WHL_NAME"
    echo "$OUTPUT_DIR/$NEW_WHL_NAME"
    exit 0
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cp "$INPUT_WHL" "$WORK_DIR/"

pushd "$WORK_DIR" > /dev/null

# Unzip
verbose "Unzipping wheel package"
unzip -qo "$WHL_NAME"
rm -f "$WHL_NAME"

# Update version info
PKG_VERSION=$(echo $WHL_NAME | sed -E 's/^[^-]+-([^-]+)-.+$/\1/')
verbose "Current version: $PKG_VERSION"

# Generate new version number based on package type
NEW_VERSION="$PKG_VERSION"
if [[ "$PKG_VERSION" =~ ^([0-9]+\.[0-9]+\.[0-9]+(\.[0-9A-Za-z]+)*)\+g([0-9a-f]+)$ ]]; then
    # sglang setuptools-scm
    NEW_VERSION="${BASH_REMATCH[1]}+g${BASH_REMATCH[3]}.sdk${SDK_TIMESTAMP}"
elif [[ "$PKG_VERSION" =~ ^([0-9]+\.[0-9]+\.[0-9]+[^+]*)\+g([0-9a-f]+)\.cu([0-9]+)$ ]]; then
    # vllm style
    NEW_VERSION="${BASH_REMATCH[1]}+g${BASH_REMATCH[2]}.sdk${SDK_TIMESTAMP}.cu${BASH_REMATCH[3]}"
elif [[ "$PKG_VERSION" =~ ^([0-9]+\.[0-9]+\.[0-9]+[^+]*)\+git([0-9a-f]+)$ ]]; then
    # PyTorch with git hash
    DL_PART=""
    [ -n "$DL_IDENTIFIER" ] && DL_PART=".${DL_IDENTIFIER}"
    NEW_VERSION="${BASH_REMATCH[1]}+git${BASH_REMATCH[2]}${DL_PART}.sdk${SDK_TIMESTAMP}"
elif [[ "$PKG_VERSION" =~ ^([0-9]+\.[0-9]+\.[0-9]+[^+]*)\+cu([0-9]+)$ ]]; then
    # Standard with cuda
    DL_PART=""
    [ -n "$DL_IDENTIFIER" ] && DL_PART=".${DL_IDENTIFIER}"
    NEW_VERSION="${BASH_REMATCH[1]}+cu${BASH_REMATCH[2]}${DL_PART}.sdk${SDK_TIMESTAMP}"
elif [[ "$PKG_VERSION" =~ ^([0-9]+\.[0-9]+\.[0-9]+\.dev[0-9]+)$ ]]; then
    # Dev version
    NEW_VERSION="${PKG_VERSION}.sdk${SDK_TIMESTAMP}"
elif [[ "$PKG_VERSION" =~ ^([0-9]+\.[0-9]+\.[0-9]+\.post[0-9]+)$ ]]; then
    # Post-release
    NEW_VERSION="${PKG_VERSION}.sdk${SDK_TIMESTAMP}"
else
    # Standard version
    NEW_VERSION="${PKG_VERSION}.sdk${SDK_TIMESTAMP}"
fi
verbose "New version: $NEW_VERSION"

# Update dist-info directory
DIST_INFO=$(find . -maxdepth 1 -type d -name "*.dist-info")
if [ -n "$DIST_INFO" ]; then
    DIST_NAME=$(basename "$DIST_INFO" | cut -d'-' -f1)
    NEW_DIST_INFO="${DIST_NAME}-${NEW_VERSION}.dist-info"
    mv "$DIST_INFO" "$NEW_DIST_INFO"
    verbose "Updated dist-info: $NEW_DIST_INFO"

    # Update METADATA
    if [ -f "${NEW_DIST_INFO}/METADATA" ]; then
        sed -i "s/Version: ${PKG_VERSION}/Version: ${NEW_VERSION}/" "${NEW_DIST_INFO}/METADATA"
        verbose "Updated METADATA version"
    fi
fi

# Function to generate RECORD entries with proper CSV escaping
make_wheel_record() {
    local FPATH=$1
    if [[ "$FPATH" == *"RECORD" ]]; then
        # RECORD file itself
        echo "$FPATH,,"
    else
        # Calculate hash and size for other files
        if [ -f "$FPATH" ]; then
            HASH=$(openssl dgst -sha256 -binary "$FPATH" | openssl base64 | tr '+/' '-_' | tr -d '=')
            # Use wc -c for cross-platform compatibility (works on both GNU and BSD)
            SIZE=$(wc -c < "$FPATH" | tr -d ' ')

            # If file path contains comma, surround with double quotes (CSV escaping)
            if [[ "$FPATH" == *","* ]]; then
                echo "\"$FPATH\",sha256=$HASH,$SIZE"
            else
                echo "$FPATH",sha256=$HASH,$SIZE
            fi
        fi
    fi
}

# Regenerate RECORD file
RECORD_FILE=$(find . -name "RECORD" || true)
if [ -n "$RECORD_FILE" ]; then
    verbose "Regenerating RECORD file"
    TEMP_RECORD=$(mktemp)

    # Generate new RECORD entries, excluding hidden files
    find . -type f -not -path "./.*" | sort | while read fname; do
        # Remove leading "./"
        fname_clean="${fname#./}"
        make_wheel_record "$fname_clean" >> "$TEMP_RECORD"
    done

    mv "$TEMP_RECORD" "$RECORD_FILE"
    verbose "RECORD file updated with $(wc -l < "$RECORD_FILE") entries"
fi

# Repackage
info "Repackaging: $NEW_WHL_NAME"
zip -rq "$NEW_WHL_NAME" * .[^.]*

# Move to output directory
PROCESSED_PATH="${OUTPUT_DIR}/${NEW_WHL_NAME}"
mv "$NEW_WHL_NAME" "$PROCESSED_PATH"

popd > /dev/null
rm -rf "$WORK_DIR"

# Verify using zip command
if zip -T "$PROCESSED_PATH" >/dev/null 2>&1; then
    verbose "✓ Wheel package format is correct"
else
    error "✗ Wheel package format has issues"
fi

info "Successfully created: $PROCESSED_PATH"
echo "$PROCESSED_PATH"
