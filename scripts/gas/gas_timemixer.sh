export CUDA_VISIBLE_DEVICES=0,1

# Block_wells=("桃2区块.csv" "桃7区块.csv" "苏11区块.csv" "苏14区块.csv" "苏19区块.csv" "苏19区块（风险）.csv" "苏20区块.csv" "苏46区块.csv" "苏47区块.csv" "苏48区块.csv" "苏49区块.csv" "苏49区块（风险）.csv" "苏59区块.csv" "苏75区块.csv")
wells=("苏14区块.csv" "苏59区块.csv" "苏49区块.csv")
model_name=TimeMixer
seq_len=$1
label_len=$2
pred_len=$3
enc_in=$4
features=$5
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=256

for well in ${wells[@]}; do
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path  ./dataset/gas/ \
        --data_path '/home/gbu-hkx/project/gas/data/outputs/dataset/blocks/'$well \
        --model_id $well'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data gas \
        --features $features \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --e_layers $e_layers \
        --d_layers 1 \
        --factor 3 \
        --enc_in $enc_in \
        --dec_in $enc_in \
        --c_out $enc_in \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size $batch_size \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --use_multi_gpu \
        --devices 0,1 \
        --down_sampling_window $down_sampling_window \
        --inverse
done
