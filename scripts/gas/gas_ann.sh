# Define the list of well data files
# wells=("桃7-8-9.csv" "桃7-9-9.csv" "桃7-10-9.csv" "桃7-16-14.csv" "桃7-10-10.csv" "桃7-16-16.csv" "桃3.csv" "桃7-10-8.csv" "桃7-1.csv" "桃7-8-8.csv" "苏14-12-41.csv" "桃7-10-11.csv" "苏14-15-41.csv" "桃7-15-14.csv" "桃7.csv" "桃7-8-2.csv" "桃7-9-2.csv" "苏14-14-36.csv" "桃7-19-16.csv" "桃7-21-13.csv")

# Block_wells=("桃2区块.csv" "桃7区块.csv" "苏11区块.csv" "苏14区块.csv" "苏19区块.csv" "苏19区块（风险）.csv" "苏20区块.csv" "苏46区块.csv" "苏47区块.csv" "苏48区块.csv" "苏49区块.csv" "苏49区块（风险）.csv" "苏59区块.csv" "苏75区块.csv")
# wells=("桃2区块.csv" "桃7区块.csv" "苏59区块.csv")
# wells=("苏14区块.csv" "苏59区块.csv" "苏49区块.csv")
wells=("苏14区块.csv")
# /home/gbu-hkx/project/gas/Time-Series-Library/dataset/gas_blocks
export CUDA_VISIBLE_DEVICES=0,1
model_name=ANN
seq_len=$1
label_len=$2
pred_len=$3
enc_in=$4
features=$5
# well='/home/gbu-hkx/project/gas/Time-Series-Library/dataset/gas_all/all_wells_combined.csv'
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
        --d_model 16 \
        --learning_rate 0.001 \
        --train_epochs 20 \
        --patience 10 \
        --des 'Exp' \
        --itr 1 \
        --use_multi_gpu \
        --devices 0,1 \
        --batch_size 128 \
        --inverse
done 


