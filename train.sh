# python train.py \
#         --max_source_length 512 \
#         --max_target_length 64 \
#         --preprocessing_num_workers 8 \
#         --batch_size 8 \
#         --weight_decay 0 \
#         --learning_rate 3e-4 \
#         --num_epochs 10 \
#         --output_dir /tmp2/b08902003/output \
#         --gradient_accumulation_steps 8
CUDA_VISIBLE_DEVICES=0 python3 train2.py \
        --model_name_or_path t5-small \
        --train_file data/train.jsonl \
        --text_column maintext \
        --summary_column title \
        --max_source_length 512 \
        --max_target_length 128 \
        --preprocessing_num_workers 4 \
        --per_device_train_batch_size 4 \
        --weight_decay 0 \
        --learning_rate 2e-4 \
        --num_train_epochs 5 \
        --gradient_accumulation_steps 4 \
        --output_dir /tmp2/b08902003/output