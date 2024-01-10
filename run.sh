export PYTHONPATH=.
TRAIN=/sd0/jelee/datasets/ESD_train_metadata.txt
VALID=/sd0/jelee/datasets/ESD_validation_metadata.txt
TEST=/sd0/jelee/datasets/ESD_test_vocoded.txt
# /sd0/jelee/datasets/ESD_test_metadata.txt, /sd0/jelee/datasets/ESD_test_gen.txt, /sd0/jelee/datasets/ESD_test_vocoded.txt

# TRAIN
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --train_list $TRAIN --valid_list $VALID --test_list $TEST

# TEST
CUDA_VISIBLE_DEVICES=0 python evaluate.py --test_list $TEST