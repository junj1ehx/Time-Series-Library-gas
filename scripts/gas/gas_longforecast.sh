# Ensure the logs directory exists
mkdir -p /home/gbu-hkx/project/gas/Time-Series-Library-gas/logs

# Define the list of well data files
# wells=("苏14区块.csv" "苏59区块.csv" "苏49区块.csv")
wells=("苏14区块.csv")
# Scripts to skip and custom model scripts
# scripts_to_skip=("gas_longforecast.sh" "gas_TimeXer.sh" "gas_DLinear.sh" "gas_PAttn.sh")
# scripts_to_run=("gas_lstm.sh" "gas_rnn.sh" "gas_svm.sh" "gas_ann.sh" "gas_PatchTST.sh" "gas_timemixer.sh" "gas_iTransformer.sh")
# scripts_to_run=("gas_lstm.sh" "gas_rnn.sh" "gas_svm.sh" "gas_ann.sh" "gas_timemixer.sh")
# scripts_to_run=("gas_ann.sh" "gas_rnn.sh")
scripts_to_run=("gas_timemixer.sh")
scripts_custom_model=("gas_svm.sh" "gas_ann.sh" "gas_lstm.sh" "gas_rnn.sh")
# "gas_PatchTST.sh" "gas_Timemixer.sh"
# scripts_custom_model=("gas_ann.sh")
# Export the wells array so it can be used by other scripts
export wells


# Function to run scripts one by one
run_scripts_in_parallel() {
    local seq_len=$1
    local label_len=$2
    local pred_len=$3
    local enc_in=$4
    local features=$5
    local max_jobs=8  # Set the maximum number of parallel jobs

    for script in /home/gbu-hkx/project/gas/Time-Series-Library-gas/scripts/gas/*.sh; do
        script_name=$(basename "$script")
        if [[ " ${scripts_to_run[@]} " =~ " $script_name " ]]; then
            if [[ " ${scripts_custom_model[@]} " =~ " $script_name " ]]; then
                if [[ $features == 'S' ]]; then
                    echo "Running custom $script_name with single feature..."
                    bash "$script" $seq_len $label_len $pred_len 1 $features > "./logs/${script_name%.sh}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
                else
                    echo "Running custom $script_name..."
                    bash "$script" $seq_len $label_len $pred_len $enc_in $features > "./logs/${script_name%.sh}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
                fi
            else
                echo "Running $script_name..."
                bash "$script" $seq_len $label_len $pred_len $enc_in $features > "./logs/${script_name%.sh}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
            fi

            # Limit the number of parallel jobs
            while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
                sleep 1
            done
        fi
    done
    wait
}

# Run scripts with different parameters
# run_scripts_in_parallel 2 0 1 28 M
# run_scripts_in_parallel 4 0 1 28 M
# run_scripts_in_parallel 6 0 1 28 M
# run_scripts_in_parallel 8 0 1 28 M
# run_scripts_in_parallel 12 0 1 28 M
# run_scripts_in_parallel 2 0 2 28 M
# run_scripts_in_parallel 4 0 2 28 M
# run_scripts_in_parallel 6 0 2 28 M
# run_scripts_in_parallel 8 0 2 28 M
# run_scripts_in_parallel 12 0 2 28 M
# run_scripts_in_parallel 2 0 4 28 M
# run_scripts_in_parallel 4 0 4 28 M
# run_scripts_in_parallel 6 0 4 28 M
# run_scripts_in_parallel 8 0 4 28 M
# run_scripts_in_parallel 12 0 4 28 M
run_scripts_in_parallel 6 0 3 1 S
# run_scripts_in_parallel 6 0 6 1 S
# run_scripts_in_parallel 12 0 3 1 S
# run_scripts_in_parallel 12 0 6 1 S
# run_scripts_in_parallel 12 0 12 1 S
# run_scripts_in_parallel 6 0 12 1 S
# run_scripts_in_parallel 6 0 21 1 S
# run_scripts_in_parallel 6 0 45 1 S
# run_scripts_in_parallel 12 0 24 1 S
# run_scripts_in_parallel 12 0 42 1 S
# run_scripts_in_parallel 12 0 90 1 S