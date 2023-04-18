#!/bin/bash

rm -r ./build

export READTHEDOCS_LOC=True
export READTHEDOCS_LANGUAGE=zh
make live
