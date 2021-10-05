#!/bin/bash

set -eux

export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased
export DATA_DIR=dataset
export OUTPUT_DIR=output
export BATCH_SIZE=32
export NUM_EPOCHS=6
export SAVE_STEPS=750
export SEED=1

echo "Preprocessing"
python -m onener.transformer.preprocess ${DATA_DIR}/train.raw.txt $BERT_MODEL $MAX_LENGTH > ${DATA_DIR}/train.txt
python -m onener.transformer.preprocess ${DATA_DIR}/test.raw.txt $BERT_MODEL $MAX_LENGTH > ${DATA_DIR}/test.txt
python -m onener.transformer.preprocess ${DATA_DIR}/dev.raw.txt $BERT_MODEL $MAX_LENGTH > ${DATA_DIR}/dev.txt

cat ${DATA_DIR}/train.txt ${DATA_DIR}/dev.txt ${DATA_DIR}/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > ${DATA_DIR}/labels.txt

echo "Train and Evaluation"
python -m onener.transformer.train \
--task_type NER \
--data_dir ${DATA_DIR} \
--labels ${DATA_DIR}/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict \
--overwrite_output_dir