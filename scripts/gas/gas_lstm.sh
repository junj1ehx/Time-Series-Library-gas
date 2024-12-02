export CUDA_VISIBLE_DEVICES=0,1

# wells=("苏14区块.csv" "苏59区块.csv" "苏49区块.csv")
wells=("苏14区块.csv")
model_name=LSTM
seq_len=$1
label_len=$2
pred_len=$3
enc_in=$4
features=$5
for well in ${wells[@]}; do
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/gas/ \
        --data_path '/home/gbu-hkx/project/gas/data/outputs/dataset/blocks/'$well \
        --model_id $well'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data gas \
        --features $features \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in $enc_in \
        --dec_in $enc_in \
        --c_out $enc_in \
        --d_model 512 \
        --des 'Exp' \
        --itr 1 \
        --use_multi_gpu \
        --devices 0,1 \
        --freq m \
        --batch_size 2048 \
        --inverse
done
