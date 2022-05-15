CUDA_VISIBLE_DEVICES=0 python train3.py \
    --wandb \
    --batch_size 2 \
    --validate \
    --accu_step 16 \
    --ckpt_dir output2