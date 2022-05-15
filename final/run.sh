input_file=$1
output_file=$2
CUDA_VISIBLE_DEVICES=0 python3 inference.py \
    --model_path output/ \
    --num_beams 5 \
    --batch_size 16 \ 
    --input_file $input_file \
    --output_file $output_file