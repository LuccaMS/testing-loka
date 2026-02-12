#!/bin/bash
set -e
BUILD_DIR="$(pwd)/dist"

# 1. Clean the folder (using sudo since files might be root-owned)
mkdir -p "$BUILD_DIR"
echo "Cleaning dist folder..."
sudo rm -rf "${BUILD_DIR:?}"/*

# 2. Build dependencies as root (the default)
echo "Installing dependencies..."
docker run --rm \
    -v "$PWD":/var/task \
    -w /var/task \
    --entrypoint /bin/bash \
    public.ecr.aws/lambda/python:3.12 \
    -c "pip install -r requirements.txt -t dist --no-cache-dir && \
        rm -rf dist/*/__pycache__ dist/*.pyc"

# 3. Copy code
echo "Copying application files..."
cp *.py *.json dist/ 2>/dev/null || :

# 4. THE FIX: Reclaim ownership of everything in dist
# This changes 'root' back to 'lucca'
echo "Fixing permissions..."
sudo chown -R $(id -u):$(id -g) "$BUILD_DIR"
chmod -R 755 "$BUILD_DIR"

echo "---"
echo "✓ Build complete. LocalStack should see the changes."
echo "---"