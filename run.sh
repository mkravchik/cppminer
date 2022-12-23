#!/bin/bash

DATASET=../sources
SNIPPET_SIZE=10
OUT_DIR=./out1
export PYTHONPATH=${PYTHONPATH}:"./cpp_parser"

rm -r $OUT_DIR
mkdir -p $OUT_DIR
python ./src/miner.py $DATASET $OUT_DIR
python ./src/merge.py $OUT_DIR
cd code2seq
./preprocess.sh  .$OUT_DIR .$OUT_DIR
cd ..
