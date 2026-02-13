#!/bin/bash

########## Modify the path according to your HOME directory ##########
HOME_DIR="/home/yc2367/llm/NVFP4-RaZeR"
######################################################################

batch_size=32
OUTPUT_DIR=${HOME_DIR}/results/acc_gsm8k_llama_bs${batch_size}

model_list=(
    "llama-3.2-3b-ins" "llama-3.1-8b-ins"
)
model_list=(
    "llama-3.2-3b-ins" 
)
task_list="gsm8k_llama"

w_bits=(4)
w_groupsize=(16)

a_bits=(4)
a_groupsize=(16)

dtype_list=("mxfp4" "nvfp4" "nvfp4_4over6")
w_dtype_list=("nvfp4_razer_e3m3")
a_dtype_list=("nvfp4_razer_e4m3")

for model_name in "${model_list[@]}"
do

    python ${HOME_DIR}/run_llama_cot.py --model_name ${model_name} \
        --tasks ${task_list} \
        --output_dir ${OUTPUT_DIR} --use_fp16 \
        --batch_size ${batch_size}

    for w_dtype in "${w_dtype_list[@]}"
    do
        for a_dtype in "${a_dtype_list[@]}"
        do
            python ${HOME_DIR}/run_llama_cot.py --model_name ${model_name} \
                --tasks ${task_list} \
                --output_dir ${OUTPUT_DIR} \
                --w_bits ${w_bits} --w_groupsize ${w_groupsize} --w_dtype ${w_dtype} --w_outlier 8.0 \
                --a_bits ${a_bits} --a_groupsize ${a_groupsize} --a_dtype ${a_dtype} \
                --batch_size ${batch_size}
        done
    done

    for dtype in "${dtype_list[@]}"
    do
        python ${HOME_DIR}/run_llama_cot.py --model_name ${model_name} \
            --tasks ${task_list} \
            --output_dir ${OUTPUT_DIR} \
            --w_bits ${w_bits} --w_groupsize ${w_groupsize} --w_dtype ${dtype} \
            --a_bits ${a_bits} --a_groupsize ${a_groupsize} --a_dtype ${dtype} \
            --batch_size ${batch_size}
    done
done
