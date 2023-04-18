#!/bin/bash

rm -r ./build

export READTHEDOCS=False
export READTHEDOCS_LOC=True
export READTHEDOCS_LANGUAGE=en
make live