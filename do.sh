#!/bin/bash

set -e
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd ${THIS_DIR}

pip3 install -e .

echo "$@"

"$@"
