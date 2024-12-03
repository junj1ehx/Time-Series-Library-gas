# Define the list of well data files
# Block_wells=("桃2区块.csv" "桃7区块.csv" "苏11区块.csv" "苏14区块.csv" "苏19区块.csv" "苏19区块（风险）.csv" "苏20区块.csv" "苏46区块.csv" "苏47区块.csv" "苏48区块.csv" "苏49区块.csv" "苏49区块（风险）.csv" "苏59区块.csv" "苏75区块.csv")

export CUDA_VISIBLE_DEVICES=0,1
# wells=("苏14区块.csv" "苏59区块.csv" "苏49区块.csv")
wells=("苏14区块.csv")
model_name=RNN
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
        --enc_in $enc_in \
        --d_model 32 \
        --itr 1 \
        --use_multi_gpu \
        --devices 0,1 \
        --batch_size 512 \
        --inverse
done
