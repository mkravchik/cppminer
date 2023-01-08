#!/bin/bash

DATASET=../sources #../src_tiny
SNIPPET_SIZE=10
OUT_DIR=./out_fixed_func_name_windows # ./out2
export PYTHONPATH=${PYTHONPATH}:"./cpp_parser"

rm -r $OUT_DIR
mkdir -p $OUT_DIR
python ./src/miner.py $DATASET $OUT_DIR -c 200  -w 10 -ws 5 -l 8
python ./src/merge.py $OUT_DIR
cd code2seq
./preprocess.sh  .$OUT_DIR .$OUT_DIR
cd ..
