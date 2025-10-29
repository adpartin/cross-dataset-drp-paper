#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# Zenodo Test Predictions Downloader
#==============================================================================
# Downloads and extracts test_preds.zip from Zenodo for CSA paper reproduction.
# This script handles the large prediction dataset (~1.6GB) that enables
# running the postprocessing pipeline without re-training models.
#
# Usage:
#   ./fetch_test_preds.sh [DEST_DIR]
#
# Example:
#   ./fetch_test_preds.sh
#   ./fetch_test_preds.sh my_predictions
#
# Prerequisites:
#   - curl or wget installed
#   - unzip utility
#   - Sufficient disk space (~3GB for download + extraction) # Note! The unzipped file is ~2.8GB
#==============================================================================

DEST="${1:-test_preds}"

# Known Zenodo URL for test_preds.zip
# ZENODO_URL="https://zenodo.org/records/15258742/files/test_preds.zip?download=1"
ZENODO_URL="https://zenodo.org/records/15851723/files/test_preds.zip?download=1"

# Skip download if already present and complete
if [ -d "$DEST" ] && [ -f "$DEST/.complete" ]; then
  echo "[info] $DEST already present and complete. Skipping download."
  echo "[info] To re-download, remove: rm -rf $DEST"
  exit 0
fi

# Detect available download tool
if command -v curl >/dev/null 2>&1; then
  DL_TOOL="curl"
  DL_CMD="curl -L --progress-bar -o"
elif command -v wget >/dev/null 2>&1; then
  DL_TOOL="wget"
  DL_CMD="wget --progress=bar -O"
else
  echo "ERROR: Neither curl nor wget found."
  exit 1
fi

TMP_ZIP="test_preds.zip"

# Check available disk space (rough estimate)
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_SPACE" -lt 3000000 ]; then  # 3GB in KB
  echo "WARNING: Less than 3GB free space available. Download will succeed but extraction may fail."
  echo "Available: $(($AVAILABLE_SPACE / 1024 / 1024))GB"
  echo "Required: ~3GB (1.6GB download + 2.8GB extraction)"
fi

echo "[info] Downloading test_preds.zip using $DL_TOOL ..."
echo "[info] Source: $ZENODO_URL"
echo "[info] Destination: $TMP_ZIP"

# Download with progress indication
if ! $DL_CMD "$TMP_ZIP" "$ZENODO_URL"; then
  echo "ERROR: Download failed."
  rm -f "$TMP_ZIP"
  exit 1
fi

FILE_SIZE=$(stat -c%s "$TMP_ZIP" 2>/dev/null || stat -f%z "$TMP_ZIP" 2>/dev/null)
FILE_SIZE_MB=$((FILE_SIZE / 1024 / 1024))
echo "[info] Downloaded: ${FILE_SIZE_MB}MB"

if [ "$FILE_SIZE_MB" -lt 100 ]; then
  echo "WARNING: File size seems too small (${FILE_SIZE_MB}MB). Expected ~1600MB."
  echo "This might not be the correct file."
fi

# Check for required utilities before extraction
if ! command -v unzip >/dev/null 2>&1; then
  echo "ERROR: unzip utility not found. Cannot extract downloaded file."
  echo "Please install unzip and run the script again."
  rm -f "$TMP_ZIP"
  exit 1
fi

# Extract to a temporary directory first, then flatten if needed
TMP_EXTRACT_DIR=".tmp_test_preds_extract"
rm -rf "$TMP_EXTRACT_DIR"
mkdir -p "$TMP_EXTRACT_DIR"

echo "[info] Extracting to temporary directory $TMP_EXTRACT_DIR/..."
if ! unzip -o "$TMP_ZIP" -d "$TMP_EXTRACT_DIR" >/dev/null; then
  echo "ERROR: Extraction failed."
  rm -f "$TMP_ZIP"
  rm -rf "$TMP_EXTRACT_DIR"
  exit 1
fi

# Create destination directory
mkdir -p "$DEST"

# If the zip contains a top-level test_preds/ folder, move its contents up
if [ -d "$TMP_EXTRACT_DIR/test_preds" ]; then
  echo "[info] Detected nested test_preds/ folder. Flattening structure..."
  # Move contents (including hidden files) from nested folder to DEST
  shopt -s dotglob nullglob
  mv "$TMP_EXTRACT_DIR/test_preds"/* "$DEST"/ || true
  shopt -u dotglob nullglob
else
  # Otherwise move everything from temp extract to DEST
  shopt -s dotglob nullglob
  mv "$TMP_EXTRACT_DIR"/* "$DEST"/ || true
  shopt -u dotglob nullglob
fi

# Cleanup temporary extract directory
rm -rf "$TMP_EXTRACT_DIR"

# Enhanced validation: check for expected file structure
echo "[info] Validating extracted content..."

# Count CSV files with expected naming pattern
CSV_COUNT=$(find "$DEST" -type f -name "*_split_*_*.csv" | wc -l | tr -d ' ')
TOTAL_FILES=$(find "$DEST" -type f | wc -l | tr -d ' ')

echo "[info] Found $CSV_COUNT prediction CSV files"

if [ "$CSV_COUNT" -lt 10 ]; then
  echo "WARNING: Fewer than 10 prediction CSVs found in $DEST"
  echo "Expected files like: <SRC>_<TRG>_split_<ID>_<MODEL>.csv"
fi

# Check for common model names
MODELS_FOUND=$(find "$DEST" -name "*.csv" -exec basename {} \; | grep -oE "(deepcdr|deepttc|graphdrp|hidra|lgbm|tcnns|uno)" | sort -u | wc -l)
echo "[info] Found predictions for $MODELS_FOUND different models"

# Mark as complete
touch "$DEST/.complete"

# Cleanup (delete the zip file)
rm -f "$TMP_ZIP"

echo ""
echo "✓ Download and extraction complete!"
echo "✓ Predictions are in: $DEST/"
echo "✓ You can now run: ./quickstart.sh"
echo ""
# echo "Next steps:"
# echo "1. Verify environment: conda env create -f environment.yml"
# echo "2. Run pipeline: ./quickstart.sh"
# echo "3. Check results in: outputs/"