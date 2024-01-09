#!/usr/bin/env bash
DEVICE=1
SEED=42

# --------- For training ----------
TRAIN_SCRIPT=finetune.py
LOGGER=default # default | wandb

TOKENIZER='google/mt5-large'
MODEL_CLASS=MT5ForConditionalGeneration
MODEL="google/mt5-large"
# SHORT_MODEL_NAME=boundary_encoder_prefix_rev
SHORT_MODEL_NAME=mt5_plain
ATTEMP=1

DATASET_VER=data_bt
# DATASET_CLASS=Seq2SeqDatasetPrefixEncoderBdr
DATASET_CLASS=Seq2SeqDatasetMT5
CONSTRAINT_TYPE=reference
SRC_LANG=en_XX
TGT_LANG=ja_XX
NUM_WORKERS=2

WARMUP_STEPS=2500
# BS=40                   # fp32: GPU2: gpu:32   |   fp16:  48  (4:15/epoch), 80 64(when)  （2:43/epoch）
BS=16
# BS=8
VAL_CHECK_INTERVAL=0.25 # 0.25
EPOCHS=10               # 5
LR=3e-5                 # 3e-5 default
EPS=1e-06
LR_SCHEDULER=linear
DROPOUT=0.0
LABEL_SMOOTHING=0.0

# export PYTHONPATH="../":"${PYTHONPATH}"
MAX_LEN=20 # 50
DATASET_DIR="/raid/ieda/trans_jaen_dataset/Dataset/datasets/${DATASET_VER}/mini"
OUTPUT_DIR="/raid/ieda/lyric_result/${SHORT_MODEL_NAME}_${DATASET_VER}_${ATTEMP}"
# --------- For testing ----------
TEST_SCRIPT=inference.py

LENGTH_TARGET=tgt # src | tgt
TEST_BOS_TOKEN_ID=250025
TEST_SRC_LANG=en_XX # zh_CN
TEST_BS=80
FORCE=no # length | rhyme | no

# change
# TEST_DATASET_DIR="../Dataset/datasets/${DATASET_VER}"
# TEST_DATASET_DIR="/raid/ieda/ChineseDatasets/${DATASET_VER}"
TEST_DATASET_DIR="/raid/ieda/trans_jaen_dataset/Dataset/datasets/${DATASET_VER}/mini"
# TEST_CONSTRAINT_TYPE=source # reference | source | random
TEST_CONSTRAINT_TYPE=reference # reference | source | random
MODEL_NAME_OR_PATH=${OUTPUT_DIR}/best_tfmr
TEST_INPUT_PATH="${TEST_DATASET_DIR}/test.source" # add "" by ieda
REF_PATH="${TEST_DATASET_DIR}/test.target"

TEST_DATASET_VER=trans_jaen_dataset_small # only for the name of output dir
CONSTRAINT_PATH=${TEST_DATASET_DIR}/constraints/${TEST_CONSTRAINT_TYPE}/test.target
TEST_OUTPUT_PATH=${OUTPUT_DIR}/testset\=${TEST_DATASET_VER}/force=${FORCE}/test_constraint\=${TEST_CONSTRAINT_TYPE}_output.txt
TEST_SCORE_PATH=${OUTPUT_DIR}/testset\=${TEST_DATASET_VER}/force=${FORCE}/test_constraint\=${TEST_CONSTRAINT_TYPE}_scores.txt

# ---------- For testing with real English lyrics -----------
REAL_TEST_DATASET_DIR="../Dataset/datasets/test_real_lyrics"
REAL_TEST_INPUT_PATH=${REAL_TEST_DATASET_DIR}/test.source
REAL_CONSTRAINT_PATH=${REAL_TEST_DATASET_DIR}/constraints.txt
REAL_TEST_OUTPUT_PATH=${OUTPUT_DIR}/test_real_output.txt
REAL_TEST_SCORE_PATH=${OUTPUT_DIR}/test_real_scores_.txt
