#!/bin/bash
set -e

BUILD_DIR="build"
rm -rf $BUILD_DIR lambda.zip
mkdir -p $BUILD_DIR

echo "Building Lambda package..."

docker run --rm \
    -v "$PWD":/var/task \
    -w /var/task \
    public.ecr.aws/lambda/python:3.12 \
    sh -c "
        pip install -r requirements.txt -t $BUILD_DIR && \
        cp *.py *.pkl $BUILD_DIR/ && \
        find $BUILD_DIR -type d -name \"tests\" -exec rm -rf {} + && \
        find $BUILD_DIR -type d -name \"__pycache__\" -exec rm -rf {} + && \
        find $BUILD_DIR -name \"*.pyc\" -delete && \
        rm -rf $BUILD_DIR/*.dist-info && \
        rm -rf $BUILD_DIR/*.egg-info
    "

# Check size before zipping
du -sh $BUILD_DIR

cd $BUILD_DIR
zip -r ../lambda.zip .
cd ..

echo "✓ Lambda package created: lambda.zip"
echo "  Final Zip Size: $(du -h lambda.zip | cut -f1)"
