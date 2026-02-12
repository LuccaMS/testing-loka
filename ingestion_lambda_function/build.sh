#!/bin/bash
set -e

BUILD_DIR="build"
rm -rf $BUILD_DIR lambda.zip
mkdir -p $BUILD_DIR

echo "Building Lambda package..."

docker run --rm \
    --entrypoint "" \
    -v "$PWD":/var/task \
    -w /var/task \
    public.ecr.aws/lambda/python:3.12 \
    sh -c "
        pip install --upgrade pip && \
        pip install -r requirements.txt -t $BUILD_DIR && \
        cp *.py $BUILD_DIR/
    "

cd $BUILD_DIR
# Don't exclude .dist-info - that's where the metadata is!
zip -r ../lambda.zip . -x "*.pyc" -x "*__pycache__*"
cd ..

echo "✓ Lambda package created: lambda.zip"
echo "  Size: $(du -h lambda.zip | cut -f1)"