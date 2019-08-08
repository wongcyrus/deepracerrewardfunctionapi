#!/bin/bash

PKG_DIR="lib"
rm -rf ${PKG_DIR} && mkdir -p ${PKG_DIR}
docker run --rm -v $(pwd):/foo -w /foo lambci/lambda:build-python3.7 \
    pip install -r requirements.txt -t ${PKG_DIR}
