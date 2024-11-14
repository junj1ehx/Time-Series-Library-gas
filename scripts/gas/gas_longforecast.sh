# Ensure the logs directory exists
mkdir -p /home/gbu-hkx/project/gas/Time-Series-Library/logs

# Define the list of well data files
wells=("苏14区块.csv" "苏59区块.csv" "苏49区块.csv")

# Scripts to skip and custom model scripts
scripts_to_skip=("gas_longforecast.sh" "gas_timesnet.sh" "gas_TimeXer.sh" "gas_DLinear.sh" "gas_PAttn.sh" "gas_iTransformer.sh" "gas_PatchTST.sh" "gas_timemixer.sh" "gas_ann.sh" "gas_lstm.sh")
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

    for script in /home/gbu-hkx/project/gas/Time-Series-Library/scripts/gas/*.sh; do
        script_name=$(basename "$script")
        if [[ ! " ${scripts_to_skip[@]} " =~ " $script_name " ]]; then
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
run_scripts_in_parallel 8 8 8 28 M
run_scripts_in_parallel 8 8 4 28 M
run_scripts_in_parallel 8 8 2 28 M
run_scripts_in_parallel 8 8 1 28 M
# run_scripts_in_parallel 8 8 8 28 S
# run_scripts_in_parallel 8 8 4 28 S
run_scripts_in_parallel 8 8 2 28 S
run_scripts_in_parallel 8 8 1 28 S