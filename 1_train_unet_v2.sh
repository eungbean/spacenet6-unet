#!/bin/bash

# source activate solaris

TRAIN_DIR='dataset/train'  # path/to/spacenet6/train/AOI_11_Rotterdam/'
source settings.sh

TRAIN_ARGS="\
    INPUT.TRAIN_VAL_SPLIT_DIR ${DATA_SPLIT_DIR} \
    INPUT.IMAGE_DIR ${TRAIN_DIR} \
    INPUT.BUILDING_DIR ${BUILDING_MASK_DIR} \
    INPUT.SAR_ORIENTATION ${SAR_ORIENTATION_PATH} \
    INPUT.MEAN_STD_DIR ${IMAGE_MEAN_STD_DIR} \
    LOG_ROOT ${TRAIN_LOG_DIR} \
    WEIGHT_ROOT ${MODEL_WEIGHT_DIR} \
    SAVE_CHECKPOINTS ${SAVE_CHECKPOINTS} \
    DUMP_GIT_INFO ${DUMP_GIT_INFO} \
"
# comment out the line below for debug
#TRAIN_ARGS=${TRAIN_ARGS}" SOLVER.EPOCHS 2 EVAL.EPOCH_TO_START_VAL 1"

mkdir -p ${TRAIN_STDOUT_DIR}

echo ''
echo 'training... (U-Net + Double Conv)'
nohup env CUDA_VISIBLE_DEVICES=0 python3 tools/train_unet_v2.py \
    --config 'configs/3_unet-ours.yml' ${TRAIN_ARGS} EXP_ID 2 \
    > train_exp_0002.out 2>&1 &
wait

# echo ''
# echo 'training... (U-Net + Ours)'
# nohup env CUDA_VISIBLE_DEVICES=0 python3 tools/train_uet_v2.py \
#     ${TRAIN_ARGS} \
#     INPUT.TRAIN_VAL_SPLIT_ID 0 \
#     EXP_ID 1 \
#     > train_exp_0002.out 2>&1 &
wait

echo 'done training all models!'