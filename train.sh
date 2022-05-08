python train.py \
        --max_source_length 512 \
        --max_target_length 64 \
        --preprocessing_num_workers 8 \
        --batch_size 8 \
        --weight_decay 0 \
        --learning_rate 3e-4 \
        --num_epochs 10 \
        --output_dir /tmp2/b08902003/output \
        --gradient_accumulation_steps 8