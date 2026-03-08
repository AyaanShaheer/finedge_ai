#!/bin/bash
# build.sh — Creates production artifact: versioned tar.gz + checksum
set -e

VERSION="1.0.0"
ARTIFACT_NAME="finedge_ai_v${VERSION}"
ARTIFACT_DIR="artifacts"
BUILD_DIR="${ARTIFACT_DIR}/${ARTIFACT_NAME}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " FinEdge AI Build Script v${VERSION}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Clean previous artifacts
rm -rf "${ARTIFACT_DIR}"
mkdir -p "${BUILD_DIR}"

# 2. Freeze dependencies
echo "[BUILD] Freezing dependencies..."
pip freeze > requirements.txt
cp requirements.txt "${BUILD_DIR}/requirements.txt"
echo "[BUILD] ✅ requirements.txt frozen ($(wc -l < requirements.txt) packages)"

# 3. Copy source (exclude models, cache, venv)
echo "[BUILD] Copying source..."
rsync -a --exclude='.venv' \
          --exclude='models/' \
          --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='.cache' \
          --exclude='logs/' \
          --exclude='artifacts/' \
          --exclude='.git/' \
          --exclude='*.egg-info' \
          . "${BUILD_DIR}/src/"

# 4. Build Python wheel
echo "[BUILD] Building wheel..."
pip wheel . --no-deps -w "${ARTIFACT_DIR}/wheels/" --quiet 2>/dev/null || \
  echo "[BUILD] ⚠️  Wheel build skipped (no setup.py/pyproject build backend)"

# 5. Create versioned tar.gz
echo "[BUILD] Creating archive..."
cd "${ARTIFACT_DIR}"
tar -czf "${ARTIFACT_NAME}.tar.gz" "${ARTIFACT_NAME}/"
cd ..

# 6. Generate SHA256 checksum
echo "[BUILD] Generating checksum..."
sha256sum "${ARTIFACT_DIR}/${ARTIFACT_NAME}.tar.gz" > "${ARTIFACT_DIR}/${ARTIFACT_NAME}.sha256"
cat "${ARTIFACT_DIR}/${ARTIFACT_NAME}.sha256"

# 7. Summary
ARCHIVE_SIZE=$(du -sh "${ARTIFACT_DIR}/${ARTIFACT_NAME}.tar.gz" | cut -f1)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " ✅ BUILD COMPLETE"
echo " Archive : artifacts/${ARTIFACT_NAME}.tar.gz (${ARCHIVE_SIZE})"
echo " Checksum: artifacts/${ARTIFACT_NAME}.sha256"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
